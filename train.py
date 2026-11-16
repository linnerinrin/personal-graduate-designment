"""Train.py 训练模块（论文附录 5）

* 损失：CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
* 优化器：AdamW 分层学习率（backbone 用 BACKBONE_LR，LSTM/分类头用 LEARNING_RATE）
* 调度：ReduceLROnPlateau(mode='max', factor=0.5, patience=3, threshold=0.01, min_lr=1e-6)
* 混合精度：cfg.USE_AMP 时用 torch.amp 的 GradScaler / autocast，并 clip_grad_norm_(1.0)
* 早停与最优权重都按 **验证集 macro-F1** 判定（比 accuracy 更稳）
"""
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import cfg
from dataset import get_dataloader
from models.ma_convlstm import (CNNBiLSTM, CNNConvLSTM, CNNLSTM,
                                MAConvLSTMActionRecognizer)


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    epoch, writer, scaler=None):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [Train]')

    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler:
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    epoch_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)

    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Accuracy', acc, epoch)
    writer.add_scalar('Train/F1', f1, epoch)
    return epoch_loss, acc, prec, rec, f1


def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [Val]')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    epoch_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)

    writer.add_scalar('Val/Loss', epoch_loss, epoch)
    writer.add_scalar('Val/Accuracy', acc, epoch)
    writer.add_scalar('Val/F1', f1, epoch)
    return epoch_loss, acc, prec, rec, f1


def build_model(model_name: str) -> nn.Module:
    """按名字构造模型（四个模型共用同一套训练流程）。"""
    if model_name == 'MAConvLSTM':
        return MAConvLSTMActionRecognizer(
            num_classes=cfg.NUM_CLASSES,
            backbone=cfg.CNN_BACKBONE,
            hidden_dim=cfg.HIDDEN_DIM)
    if model_name == 'CNNLSTM':
        return CNNLSTM(num_classes=cfg.NUM_CLASSES,
                       backbone=cfg.CNN_BACKBONE, hidden_dim=512)
    if model_name == 'CNNConvLSTM':
        return CNNConvLSTM(num_classes=cfg.NUM_CLASSES,
                           backbone=cfg.CNN_BACKBONE, hidden_dim=64)
    if model_name == 'CNNBiLSTM':
        return CNNBiLSTM(num_classes=cfg.NUM_CLASSES,
                         backbone=cfg.CNN_BACKBONE, hidden_dim=256)
    raise ValueError(f"Unknown model: {model_name}")


def train_model(model_name='MAConvLSTM'):
    device = cfg.DEVICE
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    train_loader = get_dataloader('train')
    val_loader = get_dataloader('test')

    model = build_model(model_name).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}')
    print(f'Trainable params: {trainable_params:,}')

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    # 分层学习率：backbone（低），LSTM/分类器（高）
    backbone_params, high_lr_params = [], []
    backbone_lr = getattr(cfg, 'BACKBONE_LR', cfg.LEARNING_RATE / 10)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'cnn.' in name or 'feature_extractor' in name:
            backbone_params.append(param)
        else:
            high_lr_params.append(param)

    param_groups = [{'params': high_lr_params, 'lr': cfg.LEARNING_RATE,
                     'weight_decay': cfg.WEIGHT_DECAY}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr,
                             'weight_decay': cfg.WEIGHT_DECAY})
    optimizer = optim.AdamW(param_groups)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
        threshold=0.01, min_lr=1e-6)

    scaler = GradScaler('cuda') if cfg.USE_AMP else None
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, model_name))

    best_val_f1 = 0.0
    best_metrics = {}
    patience_counter = 0
    history = []

    for epoch in range(cfg.NUM_EPOCHS):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, scaler)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, device, epoch, writer)

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'train_f1': round(train_f1, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'val_f1': round(val_f1, 4),
            'lr': current_lr,
        }
        history.append(epoch_result)

        print(f'Epoch {epoch + 1}/{cfg.NUM_EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        print(f'  LR: {current_lr:.2e}')

        # 按 val_f1 保存最优权重（比 accuracy 更稳健）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, os.path.join(cfg.MODEL_SAVE_DIR, f'best_{model_name}.pth'))
            print(f'  [*] New best model saved (val_f1={val_f1:.4f}, '
                  f'val_acc={val_acc:.4f})')
            best_metrics = epoch_result
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOP_PATIENCE:
                print(f'[Early stopping] No improvement for '
                      f'{cfg.EARLY_STOP_PATIENCE} epochs')
                print(f'Best Val F1: {best_val_f1:.4f}')
                break

    writer.close()

    with open(os.path.join(cfg.LOG_DIR, f'{model_name}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\n{"=" * 50}')
    print(f'Training completed for {model_name}')
    print(f'Best Val F1: {best_val_f1:.4f}')
    print(f'Best Val Acc: {best_metrics.get("val_acc", 0):.4f}')
    print(f'{"=" * 50}')

    return model, history, best_metrics


if __name__ == '__main__':
    train_model('MAConvLSTM')
