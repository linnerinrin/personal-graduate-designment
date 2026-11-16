"""Main.py 主程序入口（论文附录 6）

用法::

    python main.py --mode train    --model MAConvLSTM
    python main.py --mode evaluate --model MAConvLSTM
    python main.py --mode all      --model CNNLSTM
"""
import argparse

from config import cfg
from evaluate import evaluate_model
from train import train_model


def main():
    parser = argparse.ArgumentParser(
        description='MA-ConvLSTM for UCF101 Action Recognition')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'all'],
                        help='运行模式')
    parser.add_argument('--model', type=str, default='MAConvLSTM',
                        choices=['MAConvLSTM', 'CNNLSTM', 'CNNConvLSTM', 'CNNBiLSTM'],
                        help='模型选择')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='用于评估的模型检查点路径')
    args = parser.parse_args()

    if args.mode in ('train', 'all'):
        print(f'Starting training for {args.model}...')
        print(f'Device: {cfg.DEVICE}')
        print(f'Hidden dim: {cfg.HIDDEN_DIM}, Dropout LSTM: '
              f'{cfg.DROPOUT_LSTM}, Dropout Cls: {cfg.DROPOUT_CLASSIFIER}')
        train_model(model_name=args.model)
        print(f'Training finished for {args.model}.')

    if args.mode in ('evaluate', 'all'):
        checkpoint = args.checkpoint or f'./checkpoints/best_{args.model}.pth'
        print(f'Evaluating {args.model} from {checkpoint}...')
        evaluate_model(checkpoint, args.model)


if __name__ == '__main__':
    main()
