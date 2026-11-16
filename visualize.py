"""visualize.py 可视化模块（论文附录 3）

1. 训练曲线（loss / accuracy / F1 三联图）
2. 模型对比柱状图（accuracy / precision / recall / F1）
3. 综合报告（终端表格）

用法::

    python visualize.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import cfg


def plot_training_curves(history_path, save_prefix='MAConvLSTM'):
    """绘制训练曲线（loss / accuracy / F1 三联图）。"""
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    train_f1 = [h.get('train_f1', 0) for h in history]
    val_f1 = [h.get('val_f1', 0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(epochs, train_f1, 'b-', label='Train F1', linewidth=2)
    axes[2].plot(epochs, val_f1, 'r-', label='Val F1', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Training and Validation F1')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'{save_prefix}_training_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {out}')


def plot_model_comparison(metrics_files):
    """绘制模型对比柱状图（四个指标 × 四个模型）。"""
    models, accs, f1s, precs, recs = [], [], [], [], []

    for f in metrics_files:
        with open(f, 'r') as fh:
            m = json.load(fh)
        models.append(m['model'])
        accs.append(m['accuracy'])
        f1s.append(m['f1_macro'])
        precs.append(m.get('precision_macro', 0))
        recs.append(m.get('recall_macro', 0))

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width * 1.5, accs, width, label='Accuracy', color='#2E86AB')
    bars2 = ax.bar(x - width / 2, precs, width, label='Precision', color='#A23B72')
    bars3 = ax.bar(x + width / 2, recs, width, label='Recall', color='#F18F01')
    bars4 = ax.bar(x + width * 1.5, f1s, width, label='F1 Score', color='#C73E1D')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison on UCF-101')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)

    for bars in (bars1, bars2, bars3, bars4):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Model comparison saved to model_comparison.png')


def generate_report(metrics_files):
    """打印综合评估报告。"""
    print('\n' + '=' * 60)
    print('         Comprehensive Evaluation Report')
    print('=' * 60)
    print(f'Dataset: UCF-101 ({cfg.NUM_CLASSES} classes)')
    print(f'Backbone: {cfg.CNN_BACKBONE}')
    print(f'Frames: {cfg.NUM_FRAMES}')
    print('-' * 60)
    print(f'{"Model":<16} {"Accuracy":<10} {"Precision":<10} '
          f'{"Recall":<10} {"F1":<10}')
    print('-' * 60)

    results = []
    for f in metrics_files:
        with open(f, 'r') as fh:
            m = json.load(fh)
        results.append(m)
        print(f'{m["model"]:<16} {m["accuracy"]:<10.4f} '
              f'{m.get("precision_macro", 0):<10.4f} '
              f'{m.get("recall_macro", 0):<10.4f} '
              f'{m["f1_macro"]:<10.4f}')

    print('=' * 60)
    return results


if __name__ == '__main__':
    log_dir = Path(cfg.LOG_DIR)

    # 1. 训练曲线
    history_file = log_dir / 'MAConvLSTM_history.json'
    if history_file.exists():
        plot_training_curves(history_file)
    else:
        print(f'[跳过] 未找到 {history_file}')

    # 2. 汇总所有 evaluation_report_*.json
    metrics_files = sorted(Path('.').glob('evaluation_report_*.json'))
    if metrics_files:
        plot_model_comparison(metrics_files)
        generate_report(metrics_files)
    else:
        print('[跳过] 未找到 evaluation_report_*.json，请先运行 evaluate')
