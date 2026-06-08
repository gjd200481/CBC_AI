# Cycle 14：7 光束物理损失权重消融

## 1. 本周期目标

本周期对 7 光束物理约束 CNN 中的 `lambda_phy` 进行权重消融，评估物理一致性损失强度对相位 RMSE、逐通道误差和远场重建 MSE 的影响。

损失函数仍为：

```text
L_total = L_phase + lambda_phy * L_farfield
```

其中 `L_phase` 是 12 维相位 sin/cos 标签 MSE，`L_farfield` 是预测相位重建远场与输入远场之间的 MSE。

## 2. 新增脚本

```text
train/sweep_seven_beam_lambda.py
```

该脚本用于固定同一数据集、同一 train/val/test 划分、同一初始化种子，批量训练多个 `lambda_phy` 设置，并输出：

- 每个权重的训练历史 CSV。
- 汇总指标 CSV。
- 权重-误差关系图。
- 本地模型权重文件。

## 3. 快速消融设置

训练命令：

```powershell
python train\sweep_seven_beam_lambda.py --epochs 12 --batch-size 32 --learning-rate 0.001 --seed 20260612 --no-plot
```

测试权重：

```text
lambda_phy = 0, 0.01, 0.05, 0.1, 0.5, 1.0
```

数据集：

```text
dataset/seven_beam/main_static/images_main_clean_seven_beam.npy
dataset/seven_beam/main_static/labels_main_clean_seven_beam.npy
```

数据划分：

| split | 样本数 |
| --- | --- |
| train | `716` |
| val | `153` |
| test | `155` |

## 4. 12 epoch 快速消融结果

| lambda_phy | RMSE(rad) | MAE(rad) | far-field MSE |
| --- | --- | --- | --- |
| `0` | `1.2655979395` | `1.0091397762` | `1.2213083870e-04` |
| `0.01` | `1.2886365652` | `1.0154311657` | `1.2153921238e-04` |
| `0.05` | `1.2822538614` | `1.0190900564` | `1.2290123850e-04` |
| `0.1` | `1.2600396872` | `0.9957970381` | `1.2231583758e-04` |
| `0.5` | `1.2788732052` | `1.0148686171` | `1.2077493584e-04` |
| `1.0` | `1.2690178156` | `1.0020272732` | `1.2155417184e-04` |

快速消融结论：

- `lambda_phy=0.1` 的相位 RMSE 最低。
- `lambda_phy=0.5` 的远场重建 MSE 最低。
- 各权重差异不算很大，说明在当前归一化口径下远场损失量级仍偏小。

## 5. 30 epoch 候选复训

为避免只根据短训练作判断，将 `lambda_phy=0.5` 加长到 30 epoch，并与已有 30 epoch 的普通 CNN 和 `lambda_phy=0.1` 物理约束 CNN 对比。

| 模型 | lambda_phy | epoch | RMSE(rad) | MAE(rad) | far-field MSE |
| --- | --- | --- | --- | --- | --- |
| 普通 CNN | `0` | `30` | `1.0269757509` | `0.8190614581` | `1.1935354043e-04` |
| 物理约束 CNN | `0.1` | `30` | `1.0226855278` | `0.8164239526` | `1.1501365732e-04` |
| 物理约束 CNN | `0.5` | `30` | `1.0502681732` | `0.8294436932` | `1.2102721188e-04` |

30 epoch 复训结论：

- `lambda_phy=0.1` 仍是当前最合适的主实验候选。
- `lambda_phy=0.5` 在 12 epoch 中远场 MSE 较低，但 30 epoch 后相位 RMSE 和远场 MSE 都不如 `lambda_phy=0.1`。
- 当前不建议把 `0.5` 作为 7 光束主实验权重。

## 6. 结果文件

```text
result/metrics/cycle14_seven_beam_lambda_sweep_2026-06-08.csv
result/metrics/cycle14_seven_beam_lambda_sweep/*.csv
result/metrics/cycle14_seven_beam_lambda_sweep_extended_2026-06-08.csv
result/metrics/physics_cnn_lambda_0.5_main_clean_seven_beam_2026-06-08.csv
result/metrics/physics_cnn_lambda_0.5_main_clean_seven_beam_summary_2026-06-08.csv
result/figures/cycle14_seven_beam_lambda_sweep_2026-06-08.png
result/figures/physics_cnn_lambda_0.5_main_clean_seven_beam_2026-06-08.png
```

模型权重保存在本地 `models/`，不提交到 Git。

## 7. 下一步建议

下一周期可进入 7 光束噪声鲁棒性实验。主候选模型建议暂用 `lambda_phy=0.1`，并与普通 CNN 对比 `noise=0, 0.01, 0.03, 0.05, 0.08` 下的相位 RMSE、逐通道 RMSE 和远场重建 MSE。

若后续时间允许，可以围绕 `lambda_phy=0.05, 0.1, 0.2` 做更细的长训练搜索。
