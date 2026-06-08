# Cycle 12：7 光束普通 CNN baseline

## 1. 本周期目标

本周期在 Cycle 11 的 7 光束仿真模块基础上，生成首版 7 光束静态训练数据集，并训练普通监督式 CNN baseline。该模型只使用相位标签的 MSE 损失，不加入傅里叶光学物理一致性损失，用于后续物理约束 CNN 的对照。

## 2. 数据集生成

生成命令：

```powershell
python simulation\static\generate_seven_beam_dataset.py --num-samples 1024 --noise-sigma 0 --num-points 256 --window-size 0.01 --waist 0.0005 --beam-distance 0.0015 --crop-size 160 --seed 20260612 --output-dir dataset\seven_beam\main_static --prefix main_clean_seven_beam
```

输出文件：

```text
dataset/seven_beam/main_static/images_main_clean_seven_beam.npy
dataset/seven_beam/main_static/labels_main_clean_seven_beam.npy
dataset/seven_beam/main_static/phases_main_clean_seven_beam.npy
dataset/seven_beam/main_static/config_main_clean_seven_beam.json
```

数据集规模：

| 项目 | 数值 |
| --- | --- |
| 样本数 | `1024` |
| 图像尺寸 | `160 x 160` |
| 标签维度 | `12` |
| 相位通道数 | `6` |
| 噪声强度 | `0` |

## 3. 训练脚本

新增训练脚本：

```text
train/train_seven_beam_baseline.py
```

该脚本主要功能：

- 使用 `train.data_utils.build_dataloaders()` 读取远场图像和 12 维标签。
- 使用 `train.models.SimplePhaseCNN(output_dim=12)` 作为普通 CNN baseline。
- 使用 `MSELoss` 监督学习 6 路相对相位的 `sin/cos` 编码。
- 输出整体 RMSE、MAE 和 6 个通道的逐通道 RMSE。
- 保存训练曲线 CSV、测试汇总 CSV 和结果图。

## 4. 训练命令

```powershell
python train\train_seven_beam_baseline.py --epochs 30 --batch-size 32 --learning-rate 0.001 --seed 20260612 --no-plot --metrics-path result\metrics\baseline_cnn_main_clean_seven_beam_2026-06-08.csv --summary-path result\metrics\baseline_cnn_main_clean_seven_beam_summary_2026-06-08.csv --figure-path result\figures\baseline_cnn_main_clean_seven_beam_2026-06-08.png --model-path models\baseline_cnn_main_clean_seven_beam_2026-06-08.pth
```

训练设备：`cpu`

数据划分：

| split | 样本数 |
| --- | --- |
| train | `716` |
| val | `153` |
| test | `155` |

## 5. 主要结果

| 指标 | 数值 |
| --- | --- |
| test RMSE | `1.0269757509 rad` |
| test RMSE | `58.8413761902 deg` |
| test MAE | `0.8190614581 rad` |
| test MAE | `46.9287647116 deg` |
| test mean error | `0.0406905226 rad` |
| test loss | `0.2888655782` |

逐通道 RMSE：

| 通道 | RMSE(rad) | RMSE(deg) |
| --- | --- | --- |
| channel 1 | `1.0642526150` | `60.9771804810` |
| channel 2 | `0.9943473935` | `56.9719047546` |
| channel 3 | `0.9671255946` | `55.4122123718` |
| channel 4 | `1.1497400999` | `65.8752517700` |
| channel 5 | `0.9450973272` | `54.1500854492` |
| channel 6 | `1.0277509689` | `58.8857879639` |

## 6. 结果文件

```text
result/metrics/baseline_cnn_main_clean_seven_beam_2026-06-08.csv
result/metrics/baseline_cnn_main_clean_seven_beam_summary_2026-06-08.csv
result/figures/baseline_cnn_main_clean_seven_beam_2026-06-08.png
models/baseline_cnn_main_clean_seven_beam_2026-06-08.pth
```

说明：模型权重和数据集文件较大，保留在本地，不提交到 Git；CSV、PNG 和本日志提交到 Git。

## 7. 阶段性判断

7 光束普通 CNN baseline 已经跑通，但误差明显高于此前双光束结果。这是合理的：7 光束远场由 6 个相对相位共同决定，相位耦合更强，反演维度更高，且不同相位组合可能产生相似远场图样。

从结果看，第 4 通道 RMSE 最高，说明当前网络存在一定通道偏差。下一周期应扩展 7 光束傅里叶光学物理一致性损失，比较普通 CNN 与物理约束 CNN 在 7 光束条件下的差异，重点观察物理约束是否能降低整体 RMSE、逐通道偏差或远场重建误差。
