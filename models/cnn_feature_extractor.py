"""models/cnn_feature_extractor.py —— CNN 特征提取器（论文附录 1）

保留 ResNet 的卷积部分（去掉最后的 avgpool 与 fc），输出带空间维度的特征图，
供后面的 ConvLSTM 使用：

    (B, T, C, H, W) -> (B*T, C_feat, H_feat, W_feat)

``freeze_backbone`` 支持三种粒度：
    True / -1 冻结全部；False / 0 全部参与训练；N>0 冻结主干但解冻最后 N 个 stage。

注：torchvision 从 0.13 起用 ``weights=`` 取代了 ``pretrained=``，这里做了兼容，
两种版本都能跑（行为与论文一致）。

另外：若环境里没装 torchvision（只有 torch），会自动退回到一个**结构等价的
纯 PyTorch ResNet-18**（无预训练权重），保证代码仍能 import / 前向 / 训练，
仅不能用 resnet50 与 ImageNet 预训练。
"""
import torch
import torch.nn as nn

try:                                          # 有 torchvision 就用论文原方案
    from torchvision import models as _tv_models
    _HAS_TORCHVISION = True
except Exception:                             # pragma: no cover - 环境相关
    _tv_models = None
    _HAS_TORCHVISION = False


def _build_resnet(backbone: str, pretrained: bool):
    """构造 ResNet 主干，兼容新旧 torchvision 的预训练参数写法。"""
    if _HAS_TORCHVISION:
        ctor = {"resnet18": _tv_models.resnet18,
                "resnet50": _tv_models.resnet50}.get(backbone)
        if ctor is None:
            raise ValueError(f"Unsupported backbone: {backbone}")
        try:
            return ctor(weights="DEFAULT" if pretrained else None)
        except (TypeError, AttributeError):   # 老版本 torchvision
            return ctor(pretrained=pretrained)

    # ---- 纯 PyTorch 退化实现（无 torchvision 时）----
    if backbone != 'resnet18':
        raise RuntimeError(
            f"未安装 torchvision，只能用纯 PyTorch 版 resnet18（不支持 {backbone}）。"
            " 请先 pip install torchvision")
    if pretrained:
        print("[注意] 未检测到 torchvision，resnet18 使用随机初始化（无 ImageNet 预训练）。")
    return _FallbackResNet18()


class _BasicBlock(nn.Module):
    """ResNet 基本残差块（结构与 torchvision.models.resnet.BasicBlock 一致）。"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _FallbackResNet18(nn.Module):
    """与 torchvision resnet18 同构（conv1..layer4, avgpool, fc），仅无预训练权重。"""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        def _stage(in_ch, out_ch, blocks, stride):
            layers = [_BasicBlock(in_ch, out_ch, stride)]
            layers += [_BasicBlock(out_ch, out_ch, 1) for _ in range(blocks - 1)]
            return nn.Sequential(*layers)

        self.layer1 = _stage(64, 64, 2, 1)
        self.layer2 = _stage(64, 128, 2, 2)
        self.layer3 = _stage(128, 256, 2, 2)
        self.layer4 = _stage(256, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)


class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, freeze_backbone=True):
        super(CNNFeatureExtractor, self).__init__()
        base_model = _build_resnet(backbone, pretrained)

        # 去掉最后的全连接层和平均池化层，保留空间特征
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])

        if backbone == 'resnet18':
            self.feature_dim = 512
        elif backbone == 'resnet50':
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # freeze_backbone=True/-1: 冻结全部; False/0: 全部训练;
        # N>0: 冻结主干但解冻最后 N 个 stage
        if freeze_backbone is True or freeze_backbone == -1:
            for param in self.feature_extractor.parameters():
                param.requires_grad_(False)
        elif isinstance(freeze_backbone, int) and freeze_backbone > 0:
            for param in self.feature_extractor.parameters():
                param.requires_grad_(False)
            # 解冻最后 N 个 stage（按 named_children 逆序取）
            names = [n for n, _ in self.feature_extractor.named_children()]
            for name in names[-freeze_backbone:]:
                for param in getattr(self.feature_extractor, name).parameters():
                    param.requires_grad_(True)

    def forward(self, x):
        # x shape: (B*T, C, H, W) 或 (B, T, C, H, W)
        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)

        features = self.feature_extractor(x)     # (B*T, C_feat, H_feat, W_feat)
        return features
