# Cycle 27：残差物理约束模型补偿指标评估

## 目的

本 Cycle 评估当前相位 RMSE 最优候选：

```text
ResidualPhaseCNN + FarFieldConsistencyLoss
lambda_phy = 0.05
best validation checkpoint
```

目标是判断该模型的相位 RMSE 改善是否能转化为远场补偿物理指标提升。

## 评估设置

- 数据集：`dataset/seven_beam/main_static`
- 样本数：前 `256` 个样本
- 主瓣区域：中心半径 `3 px` 圆形区域
- 对比口径：与 Cycle 19 的补偿综合评估保持一致
- 运行设备：`cuda`，NVIDIA GeForce RTX 3060 Laptop GPU

新增评估模型：

```text
models/cycle23_residual_best_50epoch_residual_cnn_seven_beam_best.pth
models/cycle25_residual_physics_lambda_0p05_50epoch_best.pth
```

旧模型普通 CNN 与首版物理约束 CNN 的补偿指标来自：

```text
result/metrics/cycle19_seven_beam_compensation_effect_summary_2026-06-09.csv
```

## 输出文件

```text
result/metrics/cycle27_residual_physics_compensation_detail_2026-06-11.csv
result/metrics/cycle27_residual_physics_compensation_summary_2026-06-11.csv
result/metrics/cycle27_compensation_comparison_summary_2026-06-11.csv
result/figures/cycle27_residual_physics_compensation_2026-06-11.png
result/figures/cycle27_compensation_comparison_2026-06-11.png
```

## 主要结果

| 状态 | 主瓣能量占比 | Strehl 比 | 合成效率 | 峰值旁瓣比 | 残余相位 RMSE(rad) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 补偿前 | 0.359388 | 0.390687 | 0.532856 | 1.711568 | 1.774909 |
| 普通 CNN 补偿后 | 0.519307 | 0.647172 | 0.786023 | 6.557997 | 0.905907 |
| 首版物理约束 CNN 补偿后 | 0.521546 | 0.653564 | 0.789644 | 6.631886 | 0.894276 |
| `residual_cnn_best` 补偿后 | 0.523614 | 0.663759 | 0.793090 | 6.919307 | 0.862535 |
| `residual_cnn + physics, lambda=0.05` 补偿后 | 0.517471 | 0.653397 | 0.783312 | 6.618823 | 0.880499 |
| 理想相干 | 0.650631 | 1.000000 | 1.000000 | 13.515346 | 0.000000 |

## 阶段判断

1. `residual_cnn_best` 在本次补偿指标中表现最好：主瓣能量占比、Strehl 比、合成效率、峰值旁瓣比和残余相位 RMSE 均优于普通 CNN 与首版物理约束 CNN。
2. `residual_cnn + physics, lambda=0.05` 虽然在 Cycle 25 的测试集相位 RMSE 上达到当前最低值 `0.983128 rad`，但在本次 256 样本补偿评估中没有超过 `residual_cnn_best`。
3. 相比首版物理约束 CNN，`residual_cnn + physics, lambda=0.05` 的残余相位 RMSE 更低：`0.880499 rad` 对 `0.894276 rad`；但主瓣能量、Strehl 比和合成效率略低。
4. 当前不能简单宣布 `residual_cnn + physics` 是最终论文主模型。更稳妥的论文表述是：残差结构在补偿指标上最稳定，物理约束能改善相位 RMSE 和远场一致性，但其权重与补偿目标之间仍需继续优化。

## 对下一 Cycle 的影响

Cycle 28 应在当前最佳残差路线中测试周期相位损失，但需要同时观察：

- 测试集相位 RMSE。
- 远场一致性损失。
- 主瓣能量占比。
- Strehl 比。
- 合成效率。
- 残余相位 RMSE。

如果周期损失只改善相位 RMSE 而不改善补偿指标，应作为消融结果，而不是直接升级为主模型。
