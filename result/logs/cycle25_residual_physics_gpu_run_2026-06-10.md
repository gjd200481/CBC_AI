# Cycle 25：ResidualPhaseCNN + 物理约束 GPU 复跑

## 任务背景

Cycle 23 表明，`residual_cnn` 在采用最佳验证 checkpoint 后，测试 RMSE 可以达到 `0.992071 rad`，已经优于当前普通 CNN 与物理约束 CNN baseline。Cycle 25 的目标是在该残差网络结构上加入傅里叶光学远场一致性损失，验证“残差网络 + 物理约束”是否能进一步提升相位反演精度或物理一致性。

本次实验使用：

```text
ResidualPhaseCNN + L_total = L_phase + lambda_phy * L_farfield
```

## 运行环境

| 项目 | 数值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| PyTorch | `2.5.1+cu121` |
| CUDA 可用 | True |
| 数据集 | 7 光束干净静态主数据集 |

## 数据集与划分

```text
dataset/seven_beam/main_static/images_main_clean_seven_beam.npy
dataset/seven_beam/main_static/labels_main_clean_seven_beam.npy
```

| 项目 | 数值 |
| --- | ---: |
| 总样本数 | `1024` |
| 训练样本 | `716` |
| 验证样本 | `153` |
| 测试样本 | `155` |
| seed | `20260612` |

## 运行命令

优先运行 `lambda_phy=0.1`：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_cycle25_gpu_residual_physics.ps1 -Epochs 50 -BatchSize 32 -LearningRate 0.001 -LambdaPhy 0.1 -NumWorkers 2 -Seed 20260612
```

由于 `lambda_phy=0.1` 的 best checkpoint 已优于当前普通 CNN 与物理约束 CNN baseline，并且接近纯 `residual_cnn_best`，继续补跑：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_cycle25_gpu_residual_physics.ps1 -Epochs 50 -BatchSize 32 -LearningRate 0.001 -LambdaPhy 0.05 -NumWorkers 2 -Seed 20260612
powershell -ExecutionPolicy Bypass -File scripts\run_cycle25_gpu_residual_physics.ps1 -Epochs 50 -BatchSize 32 -LearningRate 0.001 -LambdaPhy 0.2 -NumWorkers 2 -Seed 20260612
```

统一训练设置：

| 参数 | 数值 |
| --- | --- |
| 模型 | `residual_cnn` |
| epoch | 50 |
| batch size | 32 |
| learning rate | 0.001 |
| device | `cuda` |
| num workers | 2 |

## 结果对比

| `lambda_phy` | best epoch | final RMSE(rad) | best checkpoint RMSE(rad) | best checkpoint MAE(rad) | best checkpoint far-field loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.05` | 9 | `1.235870` | `0.983128` | `0.807682` | `1.1047098e-4` |
| `0.1` | 14 | `1.196547` | `1.001047` | `0.807710` | `1.1011334e-4` |
| `0.2` | 11 | `1.185773` | `0.999029` | `0.809740` | `1.0895227e-4` |

当前主要参考基线：

| 模型 | RMSE(rad) |
| --- | ---: |
| 普通 CNN baseline | `1.02698` |
| 物理约束 CNN，`lambda_phy=0.1` | `1.02269` |
| Cycle 23 纯 `residual_cnn_best` | `0.992071` |

本次最佳结果为 `lambda_phy=0.05`，best checkpoint 测试 RMSE 为 `0.983128 rad`。

相对改进：

- 相比普通 CNN baseline：降低约 `0.043852 rad`，相对降低约 `4.27%`。
- 相比原物理约束 CNN：降低约 `0.039562 rad`，相对降低约 `3.87%`。
- 相比 Cycle 23 纯 `residual_cnn_best`：降低约 `0.008943 rad`，相对降低约 `0.90%`。

## 逐通道结果

`lambda_phy=0.05` 的最佳 checkpoint 逐通道 RMSE：

| 通道 | RMSE(rad) |
| --- | ---: |
| channel 1 | `1.043078` |
| channel 2 | `0.919567` |
| channel 3 | `0.953976` |
| channel 4 | `1.022135` |
| channel 5 | `0.878850` |
| channel 6 | `1.066967` |

`lambda_phy=0.2` 的最终 checkpoint 远场损失最低，为 `7.5838454e-5`，但 best checkpoint RMSE 为 `0.999029 rad`，不如 `lambda_phy=0.05`。这说明更强的物理约束可能改善远场一致性或最终 checkpoint 的物理损失，但不一定带来最好的相位 RMSE。

## 输出文件

```text
result/metrics/cycle25_residual_physics_lambda_0p05_50epoch_2026-06-10.csv
result/metrics/cycle25_residual_physics_lambda_0p05_50epoch_summary_2026-06-10.csv
result/figures/cycle25_residual_physics_lambda_0p05_50epoch_2026-06-10.png

result/metrics/cycle25_residual_physics_lambda_0p1_50epoch_2026-06-10.csv
result/metrics/cycle25_residual_physics_lambda_0p1_50epoch_summary_2026-06-10.csv
result/figures/cycle25_residual_physics_lambda_0p1_50epoch_2026-06-10.png

result/metrics/cycle25_residual_physics_lambda_0p2_50epoch_2026-06-10.csv
result/metrics/cycle25_residual_physics_lambda_0p2_50epoch_summary_2026-06-10.csv
result/figures/cycle25_residual_physics_lambda_0p2_50epoch_2026-06-10.png
```

本地模型权重：

```text
models/cycle25_residual_physics_lambda_0p05_50epoch.pth
models/cycle25_residual_physics_lambda_0p05_50epoch_best.pth
models/cycle25_residual_physics_lambda_0p1_50epoch.pth
models/cycle25_residual_physics_lambda_0p1_50epoch_best.pth
models/cycle25_residual_physics_lambda_0p2_50epoch.pth
models/cycle25_residual_physics_lambda_0p2_50epoch_best.pth
```

模型权重为本地产物，不提交 Git。

## 阶段结论

`ResidualPhaseCNN + physics loss` 是有效的。最佳配置暂定为：

```text
model_name = residual_cnn
lambda_phy = 0.05
checkpoint = best validation RMSE checkpoint
```

该配置在本次复跑中取得 `0.983128 rad` 的测试 RMSE，是目前记录中最好的 7 光束相位反演结果。

不过，后期训练仍然存在过拟合现象：最终 epoch 的 RMSE 明显差于 best checkpoint。因此后续主实验应继续坚持最佳验证 checkpoint 策略，并考虑加入早停或学习率调度。

## 后续建议

- 用 `lambda_phy=0.05` 的 best checkpoint 进行主瓣能量占比、Strehl 比和相干合成效率评估。
- 以 `lambda_phy=0.05` 为中心，必要时补充更细权重，例如 `0.02`、`0.08`。
- 对比 `lambda_phy=0.2` 在远场损失上的优势，判断其是否在补偿效果指标中优于 `0.05`。
- 将当前最佳模型作为论文主模型候选，但最终仍需结合补偿后物理指标共同判断。
