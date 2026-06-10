# Cycle 25：残差网络 + 物理约束实验准备

## 背景

当前已有三类模型：

| 模型 | 残差网络 | 物理约束 | 说明 |
| --- | --- | --- | --- |
| 普通 CNN baseline | 否 | 否 | 7 光束基础模型 |
| `physics_cnn_lambda_0.1` | 否 | 是 | 普通 CNN + 傅里叶远场一致性损失 |
| `residual_cnn_best` | 是 | 否 | 残差网络 + 最佳验证 checkpoint |

用户提出是否可以做“残差 + 物理约束”。该建议是合理的，因为它将当前两个有效方向合并：

```text
ResidualPhaseCNN + L_total = L_phase + lambda_phy * L_farfield
```

## 本周期修改

修改训练脚本：

```text
train/train_seven_beam_physics_constrained_cnn.py
```

新增能力：

- 支持 `--model-name residual_cnn`。
- 支持 `--device cuda`。
- 支持 `--num-workers`。
- 保存最终 checkpoint。
- 保存最佳验证 RMSE checkpoint。
- 在 summary CSV 中记录最佳 checkpoint 的测试 RMSE、MAE、远场损失和逐通道 RMSE。

新增 RTX 3060 启动脚本：

```text
scripts/run_cycle25_gpu_residual_physics.ps1
```

## 是否需要 RTX 3060

需要。

该实验包含 7 光束 FFT 物理一致性损失，训练速度明显慢于普通监督模型。当前 CPU 适合做代码检查和小规模 smoke，不适合完整 50 epoch 训练。

## 推荐运行命令

优先运行：

```powershell
.\scripts\run_cycle25_gpu_residual_physics.ps1 -Epochs 50 -BatchSize 32 -LearningRate 0.001 -LambdaPhy 0.1 -NumWorkers 2 -Seed 20260612
```

如果 `lambda_phy=0.1` 有提升，再补跑：

```powershell
.\scripts\run_cycle25_gpu_residual_physics.ps1 -Epochs 50 -BatchSize 32 -LearningRate 0.001 -LambdaPhy 0.05 -NumWorkers 2 -Seed 20260612
.\scripts\run_cycle25_gpu_residual_physics.ps1 -Epochs 50 -BatchSize 32 -LearningRate 0.001 -LambdaPhy 0.2 -NumWorkers 2 -Seed 20260612
```

## 判断标准

优先看：

- `best_checkpoint_test_rmse_rad` 是否低于 `0.992071 rad`。
- `best_checkpoint_farfield_loss` 是否低于现有普通 CNN 和物理约束 CNN。
- 后续补偿评估中主瓣能量占比、Strehl 比、合成效率是否提高。

## 阶段判断

`residual_cnn + physics loss` 是当前最符合论文主题的下一步实验。如果它在相位 RMSE 和物理补偿指标上都优于现有模型，可以作为论文主模型候选；如果只改善远场 MSE 而不改善相位 RMSE，也可作为“物理一致性增强但相位精度受限”的消融结果。
