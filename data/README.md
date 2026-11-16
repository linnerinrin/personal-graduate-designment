# 数据放置说明

代码默认读 `./data/`，最终结构应如下（与论文一致）：

```
data/
├── UCF-101/                      # 解压后的视频，每个类别一个子目录
│   ├── ApplyEyeMakeup/
│   │   ├── v_ApplyEyeMakeup_g01_c01.avi
│   │   └── ...
│   ├── ApplyLipstick/
│   └── ... （共 101 个类别）
└── ucfTrainTestlist/
    ├── trainlist01.txt           # 训练划分（每行：类别名/视频名.avi [标签]）
    ├── testlist01.txt            # 测试划分
    ├── classInd.txt              # 类别名 → 编号
    └── ...（其它 split，代码默认用 01）
```

## 获取方式

1. **视频**：UCF-101 官网 `https://www.crcv.ucf.edu/data/UCF101.php`，下载 `UCF101.rar` 与
   `UCF101TrainTestSplits-RecognitionTask.zip`。
   UCF-101 官方只提供 rar（约 6.5 GB）；Linux 用 `unrar x UCF101.rar`，
   Windows 用 WinRAR / 7-Zip 解压即可。
2. **划分文件**：上面的 zip 解压后得到 `ucfTrainTestlist/`，整个目录拷到 `data/` 下。
3. 类别数 101，训练视频 9537 个（论文设定）。

## 小数据调试

没有完整数据集时，可以：

```bash
# 只放几个类别的视频 + 对应的 trainlist/testlist 子集
# 然后在 config.py 里把 SUBSET_SIZE 改成正数（例如 200），只取这么多条样本跑通流程
```

`dataset.py` 会自动跳过划分文件里"文件缺失 / 类别未知 / 格式错误"的行，并在启动时打印统计，
所以先用少数几个类别试跑是安全的。

## 路径改了怎么办

`config.py` 里改这三项即可：

```python
DATA_ROOT = './data/UCF-101'             # 视频根目录
SPLIT_DIR = './data/ucfTrainTestlist'    # 划分文件目录
```

`SPLIT_TEMPLATE` 由 `SPLIT_DIR` 自动拼成 `{split}list01.txt`，想换 split 就改成 `list02`、`list03`。
