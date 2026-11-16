"""models/ma_convlstm.py —— 模型架构（论文附录 2）

包含：
    SpatialAttention / TemporalAttention   两种注意力
    MAConvLSTMCell / MAConvLSTM            带注意力门控的 ConvLSTM
    MAConvLSTMActionRecognizer             本文完整模型（CNN + MA-ConvLSTM）
    CNNLSTM / CNNConvLSTM / CNNBiLSTM      三个消融基线
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from models.cnn_feature_extractor import CNNFeatureExtractor


class SpatialAttention(nn.Module):
    """CBAM 风格的通道维压缩 + 空间注意力，把注意力图乘回特征。"""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(attn))


class TemporalAttention(nn.Module):
    """对 T 个时刻的特征打分并 softmax，得到时间维权重 (B, T, 1)。"""

    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 8),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT_LSTM),
            nn.Linear(feature_dim // 8, 1)
        )

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


class MAConvLSTMCell(nn.Module):
    """多头注意力的 ConvLSTM 单元：遗忘门被空间注意力调制。"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, 3, padding=1)
        self.spatial_attn = SpatialAttention()
        self.forget_attn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        i, f, g, o = torch.split(self.conv(combined), self.hidden_dim, dim=1)
        # 关键改动：遗忘门乘以"空间注意力(细胞状态) + 隐状态"的注意力掩码
        f = f * self.forget_attn(self.spatial_attn(c) + h)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new


class MAConvLSTM(nn.Module):
    """逐帧逐层推进 MAConvLSTMCell，最后对时间维做注意力加权求和。"""

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cells = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            cells.append(MAConvLSTMCell(in_dim, hidden_dim))
        self.cells = nn.ModuleList(cells)
        self.temporal_attn = TemporalAttention(hidden_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        device = x.device

        h = [torch.zeros(B, self.hidden_dim, H, W, device=device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_dim, H, W, device=device)
             for _ in range(self.num_layers)]

        outputs = []
        for t in range(T):
            inp = x[:, t]
            for l in range(self.num_layers):
                h[l], c[l] = self.cells[l](inp, h[l], c[l])
                inp = h[l]
            outputs.append(h[-1])

        out = torch.stack(outputs, dim=1)              # (B, T, hidden_dim, H, W)
        pooled = out.mean(dim=[3, 4])                  # (B, T, hidden_dim)  GAP
        pooled = F.dropout(pooled, cfg.DROPOUT_LSTM, training=self.training)
        weights = self.temporal_attn(pooled)           # (B, T, 1)
        return (pooled * weights).sum(dim=1)           # (B, hidden_dim)


class MAConvLSTMActionRecognizer(nn.Module):
    """本文模型：冻结的 ResNet 特征 + MA-ConvLSTM + 分类头。"""

    def __init__(self, num_classes=101, backbone='resnet18', hidden_dim=64):
        super().__init__()
        self.cnn = CNNFeatureExtractor(backbone=backbone, freeze_backbone=True)
        cnn_dim = self.cnn.feature_dim
        self.lstm = MAConvLSTM(cnn_dim, hidden_dim, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT_CLASSIFIER),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        cnn_feat = self.cnn(x.reshape(-1, C, H, W))
        _, C_cnn, H_feat, W_feat = cnn_feat.shape
        cnn_feat = cnn_feat.reshape(B, T, C_cnn, H_feat, W_feat)
        return self.classifier(self.lstm(cnn_feat))


# ---------------------------------------------------------------------------
# 消融基线
# ---------------------------------------------------------------------------
class CNNLSTM(nn.Module):
    """CNN + 普通 LSTM（全局平均池化后逐帧输入），只取最后时刻。"""

    def __init__(self, num_classes=101, backbone='resnet18', hidden_dim=512):
        super().__init__()
        self.cnn = CNNFeatureExtractor(backbone=backbone, freeze_backbone=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(
            self.cnn.feature_dim, hidden_dim,
            num_layers=2, batch_first=True, dropout=0.5
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        feat = self.cnn(x.reshape(-1, C, H, W))
        feat = self.pool(feat).squeeze(-1).squeeze(-1)
        feat = feat.view(B, T, -1)
        out, _ = self.lstm(feat)
        return self.classifier(out[:, -1, :])


class CNNConvLSTM(nn.Module):
    """CNN + 标准 ConvLSTM（无注意力机制），作为消融基线。"""

    def __init__(self, num_classes=101, backbone='resnet18', hidden_dim=64):
        super().__init__()
        self.cnn = CNNFeatureExtractor(backbone=backbone, freeze_backbone=True)
        self.input_dim = self.cnn.feature_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(self.input_dim + hidden_dim, 4 * hidden_dim, 3, padding=1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT_CLASSIFIER),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        cnn_feat = self.cnn(x.reshape(-1, C, H, W))
        _, C_cnn, H_feat, W_feat = cnn_feat.shape
        cnn_feat = cnn_feat.reshape(B, T, C_cnn, H_feat, W_feat)

        h = torch.zeros(B, self.hidden_dim, H_feat, W_feat, device=x.device)
        c = torch.zeros(B, self.hidden_dim, H_feat, W_feat, device=x.device)

        outputs = []
        for t in range(T):
            combined = torch.cat([cnn_feat[:, t], h], dim=1)
            gates = self.conv(combined)
            i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outputs.append(h)

        out = torch.stack(outputs, dim=1)
        out = F.dropout(out, cfg.DROPOUT_LSTM, training=self.training)
        pooled = out.mean(dim=[3, 4])      # GAP over spatial -> (B, T, hidden_dim)
        pooled = pooled.mean(dim=1)        # mean over time   -> (B, hidden_dim)
        return self.classifier(pooled)


class CNNBiLSTM(nn.Module):
    """CNN + 双向 LSTM 基线。"""

    def __init__(self, num_classes=101, backbone='resnet18', hidden_dim=256):
        super().__init__()
        self.cnn = CNNFeatureExtractor(backbone=backbone, freeze_backbone=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(
            self.cnn.feature_dim, hidden_dim,
            num_layers=2, batch_first=True, dropout=0.5,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        feat = self.cnn(x.reshape(-1, C, H, W))
        feat = self.pool(feat).squeeze(-1).squeeze(-1)
        feat = feat.view(B, T, -1)
        out, _ = self.lstm(feat)
        return self.classifier(out[:, -1, :])
