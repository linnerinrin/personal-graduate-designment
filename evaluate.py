"""Evaluate.py 评估模块（论文附录 4）

输出：
* accuracy / precision(macro) / recall(macro) / f1(macro) / f1(weighted)
* `classification_report` 逐类指标
* `confusion_matrix_{model}.png`（101×101 热力图）
* `evaluation_report_{model}.json` / `.txt`（供 visualize.py 汇总对比）
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from tqdm import tqdm

try:                                          # seaborn 只用于混淆矩阵美化
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False

from config import cfg
from dataset import get_dataloader
from models.ma_convlstm import (CNNBiLSTM, CNNConvLSTM, CNNLSTM,
                                MAConvLSTMActionRecognizer)


def get_model(model_name):
    if model_name == 'MAConvLSTM':
        return MAConvLSTMActionRecognizer(num_classes=cfg.NUM_CLASSES,
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


def evaluate_model(model_path, model_name='MAConvLSTM'):
    device = cfg.DEVICE
    model = get_model(model_name)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    test_loader = get_dataloader('test')

    all_preds, all_labels, all_probs = [], [], []

    print(f'Evaluating {model_name}...')
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    # ---- 指标 ----
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0)

    report = classification_report(all_labels, all_preds, digits=4,
                                   zero_division=0)
    print(f'\n=== {model_name} Evaluation Results ===')
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision (macro): {prec:.4f}')
    print(f'Recall (macro): {rec:.4f}')
    print(f'F1 (macro): {f1:.4f}')
    print(f'F1 (weighted): {f1_w:.4f}')

    # ---- 混淆矩阵 ----
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 16))
    if _HAS_SEABORN:
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    else:                                     # 没装 seaborn 时用 imshow 兜底
        plt.imshow(cm, cmap='Blues', interpolation='nearest', aspect='auto')
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print('Confusion matrix saved.')

    # ---- 保存报告 ----
    metrics = {
        'model': model_name,
        'accuracy': float(acc),
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'f1_macro': float(f1),
        'f1_weighted': float(f1_w),
        'test_samples': len(all_labels),
    }
    with open(f'evaluation_report_{model_name}.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(f'evaluation_report_{model_name}.txt', 'w') as f:
        f.write(f'=== {model_name} Evaluation Results ===\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'Precision (macro): {prec:.4f}\n')
        f.write(f'Recall (macro): {rec:.4f}\n')
        f.write(f'F1 (macro): {f1:.4f}\n')
        f.write(f'F1 (weighted): {f1_w:.4f}\n\n')
        f.write(report)

    print('Report saved.')
    return all_preds, all_labels, metrics


if __name__ == '__main__':
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'MAConvLSTM'
    evaluate_model(f'./checkpoints/best_{model_name}.pth', model_name)
