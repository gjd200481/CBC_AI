# Cycle 05 普通 CNN Baseline 实验记录

## 任务目标

训练普通监督式 CNN baseline，只使用相位 `sin/cos` 标签的 MSE 损失，不加入傅里叶光学物理一致性损失。该结果作为后续物理约束 CNN 的直接对照组。

## 数据集

- 数据集名称：`main_clean_two_beam`
- 图像文件：`dataset/two_beam/main_static/images_main_clean_two_beam.npy`
- 标签文件：`dataset/two_beam/main_static/labels_main_clean_two_beam.npy`
- 样本数：2000
- 图像尺寸：`160 x 160`
- 标签格式：`[sin(phi), cos(phi)]`
- 噪声强度：0
- 相位范围：`[-pi, pi]`

## 数据划分

使用固定随机种子 `20260608`：

- 训练集：1400
- 验证集：300
- 测试集：300

## 模型

- 模型名称：`SimplePhaseCNN`
- 输入：`[batch, 1, 160, 160]`
- 输出：`[batch, 2]`
- 输出含义：`[sin(phi), cos(phi)]`
- 损失函数：`MSELoss`
- 优化器：`Adam`
- 学习率：`0.001`
- batch size：`32`
- epoch：`20`

## 训练命令

```powershell
python -m train.evaluate_two_beam `
  --epochs 20 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --seed 20260608 `
  --model-path models\baseline_cnn_main_clean.pth `
  --metrics-path result\metrics\baseline_cnn_main_clean_2026-06-07.csv `
  --summary-path result\metrics\baseline_cnn_main_clean_summary_2026-06-07.csv `
  --figure-path result\figures\baseline_cnn_main_clean_2026-06-07.png `
  --no-plot
```

## 输出文件

- 模型权重：`models/baseline_cnn_main_clean.pth`
- 训练指标：`result/metrics/baseline_cnn_main_clean_2026-06-07.csv`
- 测试摘要：`result/metrics/baseline_cnn_main_clean_summary_2026-06-07.csv`
- 结果图：`result/figures/baseline_cnn_main_clean_2026-06-07.png`

说明：模型权重和结果目录默认被 `.gitignore` 忽略，后续如需提交关键指标，可使用 `git add -f` 强制加入。

## 主要训练过程

| epoch | train_loss | val_loss | val_rmse(rad) | val_rmse(deg) |
|---:|---:|---:|---:|---:|
| 1 | 0.4941168 | 0.3417914 | 0.4119702 | 23.6042 |
| 2 | 0.0745141 | 0.0029823 | 0.0583656 | 3.3441 |
| 3 | 0.0014783 | 0.0004093 | 0.0156818 | 0.8985 |
| 10 | 0.0000400 | 0.0000500 | 0.0070930 | 0.4064 |
| 17 | 0.0000155 | 0.0000137 | 0.0031869 | 0.1826 |
| 20 | 0.0000179 | 0.0000182 | 0.0037113 | 0.2126 |

## 测试集结果

| 指标 | 数值 |
|---|---:|
| RMSE(rad) | 0.0037421337 |
| RMSE(deg) | 0.2144084693 |
| MAE(rad) | 0.0030820819 |
| MAE(deg) | 0.1765902855 |
| Mean error(rad) | 0.0003495065 |
| Mean error(deg) | 0.0200252459 |
| MSE loss | 1.7800666e-05 |

## 结论

在无噪声双光束静态数据集上，普通 CNN baseline 已经能够较准确地完成远场光强到相位误差的反演，测试集相位 RMSE 约为 `0.00374 rad`，即 `0.214 deg`。

该结果说明：在理想干净仿真条件下，远场干涉图中已经包含足够清晰的相位信息，单纯数据驱动 CNN 能学习到稳定映射。后续物理约束 CNN 的重点不应只追求干净数据上的 RMSE，而应重点比较：

- 噪声扰动下的鲁棒性。
- 振幅失配和位置偏移下的泛化能力。
- 预测相位代回 FFT 后的远场重建一致性。
- 主瓣能量占比和 Strehl 比等物理指标。
