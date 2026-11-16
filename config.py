"""Config.py 配置文件（论文附录 8）

集中管理路径、模型超参与训练超参，构造 ``cfg = Config()`` 时会自动创建
数据/权重/日志目录。
"""
import os

import torch


class Config:
    # ---------------- 路径 ----------------
    DATA_ROOT = './data/UCF-101'            # 解压后的帧级视频目录
    SPLIT_DIR = './data/ucfTrainTestlist'   # trainlist01.txt / testlist01.txt
    PROCESSED_DATA_DIR = './data/processed'
    MODEL_SAVE_DIR = './checkpoints'
    LOG_DIR = './logs'
    FIGURE_DIR = './figures'

    # ---------------- 模型 ----------------
    CNN_BACKBONE = 'resnet18'
    CNN_FEATURE_DIM = 512
    HIDDEN_DIM = 128
    INPUT_SIZE = (7, 7)

    # ---------------- 训练 ----------------
    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    BACKBONE_LR = 1e-5
    WEIGHT_DECAY = 1e-2
    NUM_FRAMES = 16
    NUM_CLASSES = 101
    RESIZE_SIZE = (160, 160)
    LABEL_SMOOTHING = 0.15
    EARLY_STOP_PATIENCE = 7
    DROPOUT_LSTM = 0.3
    DROPOUT_CLASSIFIER = 0.6
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2                # 多进程加载加速
    USE_AMP = torch.cuda.is_available()   # 混合精度训练
    SEED = 42

    # 调试用：>0 时只取这么多条样本（None/0 表示用全量）
    SUBSET_SIZE = 0

    def __init__(self):
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.FIGURE_DIR, exist_ok=True)


cfg = Config()

# 供 dataset.py 拼划分文件路径用
cfg.SPLIT_TEMPLATE = os.path.join(cfg.SPLIT_DIR, '{}list01.txt')
