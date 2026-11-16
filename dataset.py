"""Dataset.py 数据加载与预处理（论文附录 7）

要点（论文 §3.4 的数据增强设计）：
  * 训练：随机 stride(1~4) + 随机起点 + ±10% 抖动采样 T 帧，越小步长越能覆盖快慢动作；
  * 测试：``np.linspace`` 均匀采样，结果可复现；
  * 训练增广：Resize(256) → RandomResizedCrop(scale 0.3~1.0) → 翻转 →
    ColorJitter → RandomGrayscale → ToTensor → RandomErasing → Normalize。
"""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import transforms
    _HAS_TORCHVISION = True
except Exception:                             # 只有 torch、没装 torchvision 时
    transforms = None
    _HAS_TORCHVISION = False

from config import cfg


class UCF101Dataset(Dataset):
    def __init__(self, data_root, split_path, num_frames=16,
                 resize_size=(224, 224), is_train=True, skip_preprocess=False):
        if not _HAS_TORCHVISION:
            raise RuntimeError(
                '数据加载需要 torchvision，请先安装：pip install torchvision')
        self.data_root = data_root
        self.num_frames = num_frames
        self.resize_size = resize_size
        self.is_train = is_train

        # ============ 1. 扫描类别 ============
        self.classes = sorted([d for d in os.listdir(data_root)
                               if os.path.isdir(os.path.join(data_root, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"[数据集] 发现 {len(self.classes)} 个类别")

        # ============ 2. 读取划分文件 ============
        self.video_list = []
        total_lines = 0
        skipped_format = 0
        skipped_missing = 0
        skipped_class = 0

        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1

                parts = line.split()
                if len(parts) < 1:
                    skipped_format += 1
                    continue

                path_part = parts[0]
                if '/' not in path_part:
                    skipped_format += 1
                    continue

                class_name, video_name = path_part.split('/', 1)
                if class_name not in self.class_to_idx:
                    skipped_class += 1
                    continue

                full_path = os.path.join(data_root, class_name, video_name)
                if not os.path.exists(full_path):
                    skipped_missing += 1
                    continue

                label = self.class_to_idx[class_name]
                self.video_list.append((class_name, video_name, label))

        print(f"[数据集] 读取文件: {split_path}")
        print(f"[数据集] 总行数: {total_lines}")
        print(f"[数据集] 有效样本: {len(self.video_list)}")
        if skipped_format:
            print(f"[数据集] 跳过(格式错误): {skipped_format}")
        if skipped_missing:
            print(f"[数据集] 跳过(文件缺失): {skipped_missing}")
        if skipped_class:
            print(f"[数据集] 跳过(未知类别): {skipped_class}")

        # ============ 3. 可选：子集采样 ============
        if getattr(cfg, 'SUBSET_SIZE', 0) and cfg.SUBSET_SIZE > 0:
            random.shuffle(self.video_list)
            self.video_list = self.video_list[:cfg.SUBSET_SIZE]
            print(f"[数据集] 使用子集，样本数: {len(self.video_list)}")

        # ============ 4. 图像预处理 ============
        # 先 Resize 到 256，再裁剪/随机裁剪到目标尺寸，保证输入尺寸一致
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                # 极端随机裁剪：小到只剩 30% 也能看清动作
                transforms.RandomResizedCrop(resize_size, scale=(0.3, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                # 更强的颜色抖动
                transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.3, hue=0.15),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2),
                                         ratio=(0.3, 3.3)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None

        if self.is_train:
            # 1. 随机采样间隔（1~4 帧取 1 帧，模拟动作快慢变化）
            stride = random.randint(1, 4)
            needed = self.num_frames * stride      # 需要覆盖的帧范围

            if total_frames > needed:
                # 2. 随机起点
                max_start = total_frames - needed
                start = random.randint(0, max_start)
                # 3. 在起点上加随机抖动（±10% 范围），模拟动作起始位置不准
                jitter = int(needed * 0.1)
                start = start + random.randint(-jitter, jitter)
                start = max(0, min(start, total_frames - needed))
            else:
                # 视频太短，从头开始，减小 stride
                start = 0
                stride = max(1, total_frames // self.num_frames)

            indices = [start + i * stride for i in range(self.num_frames)]
        else:
            # 测试时用均匀采样，保证结果可复现
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            idx = min(idx, total_frames - 1)       # 防止越界
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        if len(frames) == 0:
            return None

        while len(frames) < self.num_frames:
            frames.append(frames[-1])              # 用最后一帧补齐

        return np.array(frames)

    def __getitem__(self, idx):
        class_name, video_name, label = self.video_list[idx]
        video_path = os.path.join(self.data_root, class_name, video_name)

        frames = self.load_frames(video_path)
        if frames is None:                        # 坏样本：随机换一个
            return self.__getitem__(random.randint(0, len(self.video_list) - 1))

        transformed_frames = torch.stack([self.transform(frame) for frame in frames])
        return transformed_frames, label

    def __len__(self):
        return len(self.video_list)


def get_dataloader(phase='train'):
    """创建训练或测试的 DataLoader。"""
    split_path = getattr(cfg, 'SPLIT_TEMPLATE',
                         './data/ucfTrainTestlist/{}list01.txt').format(phase)
    dataset = UCF101Dataset(
        data_root=cfg.DATA_ROOT,
        split_path=split_path,
        num_frames=cfg.NUM_FRAMES,
        resize_size=cfg.RESIZE_SIZE,
        is_train=(phase == 'train')
    )
    shuffle = (phase == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,          # 确保这个是 True
        drop_last=shuffle         # 训练集丢掉最后一个不满的 batch
    )
    return dataloader
