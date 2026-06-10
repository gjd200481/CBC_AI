# Cycle 23：最佳 checkpoint 策略下 residual_cnn 50 epoch GPU 复跑

## 任务背景

Cycle 22 在 RTX 3060 上完成了 `residual_cnn` 完整 7 光束数据集 50 epoch 复跑，但最终 epoch 的测试 RMSE 为 `1.319034 rad`，未优于当前 7 光束普通 CNN baseline `1.02698 rad`。

不过 Cycle 22 的训练曲线显示，验证 RMSE 曾在中途达到 `0.973325 rad`，最终又回升到 `1.219996 rad`。这说明只评估最终 epoch 可能低估了模型能力，因此 Cycle 23 新增“最佳验证 checkpoint”保存和评估逻辑。本次任务是在 RTX 3060 上复跑 50 epoch，并同时比较最终 checkpoint 与最佳 checkpoint 的测试表现。

## 运行环境

| 项目 | 数值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Python | 3.11.7，Anaconda |
| PyTorch | `2.5.1+cu121` |
| CUDA 可用 | True |

## 数据集

使用本地已生成的 7 光束干净静态主数据集：

```text
dataset/seven_beam/main_static/images_main_clean_seven_beam.npy
dataset/seven_beam/main_static/labels_main_clean_seven_beam.npy
```

数据集设置：

| 项目 | 数值 |
| --- | ---: |
| 总样本数 | `1024` |
| 训练样本 | `716` |
| 验证样本 | `153` |
| 测试样本 | `155` |
| seed | `20260612` |

## 运行命令

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_cycle22_gpu_residual.ps1 -Epochs 50 -BatchSize 64 -LearningRate 0.001 -NumWorkers 2 -Seed 20260612
```

等价训练配置：

| 参数 | 数值 |
| --- | --- |
| 模型 | `residual_cnn` |
| epoch | 50 |
| batch size | 64 |
| learning rate | 0.001 |
| device | `cuda` |
| num workers | 2 |
| pin memory | True |
| experiment tag | `cycle23_residual_best_50epoch` |

## 训练过程摘要

训练前期验证 RMSE 快速下降：

| epoch | val RMSE(rad) |
| ---: | ---: |
| 1 | `1.844582` |
| 5 | `1.437498` |
| 8 | `1.021832` |
| 10 | `0.951766` |
| 17 | `0.950714` |

随后训练 loss 继续下降，但验证 RMSE 在后半段回升，最终 epoch 验证 RMSE 为 `1.196506 rad`。这与 Cycle 22 的现象一致，说明该结构在当前训练设置下后期存在明显过拟合或验证性能波动。

## 复跑结果

| 项目 | 最终 checkpoint | 最佳 checkpoint |
| --- | ---: | ---: |
| 测试 RMSE(rad) | `1.269384` | `0.992071` |
| 测试 MAE(rad) | `0.983511` | `0.812456` |
| 测试 loss | `0.503289` | `0.274370` |
| 对应 epoch | 50 | 17 |

最佳验证 checkpoint 的逐通道测试 RMSE：

| 通道 | RMSE(rad) |
| --- | ---: |
| channel 1 | `1.036237` |
| channel 2 | `0.857443` |
| channel 3 | `1.018321` |
| channel 4 | `1.015592` |
| channel 5 | `0.951201` |
| channel 6 | `1.059743` |

## 与当前 baseline 对比

当前 README 中的 7 光束主数据集结果：

| 模型 | RMSE(rad) | MAE(rad) |
| --- | ---: | ---: |
| 普通 CNN | `1.02698` | `0.81906` |
| 物理约束 CNN，`lambda_phy=0.1` | `1.02269` | `0.81642` |

本次最佳 checkpoint 的测试 RMSE 为 `0.992071 rad`：

- 相比普通 CNN baseline，RMSE 降低约 `0.034909 rad`，相对降低约 `3.40%`。
- 相比物理约束 CNN，RMSE 降低约 `0.030619 rad`，相对降低约 `2.99%`。
- 最佳 checkpoint 的 MAE 为 `0.812456 rad`，也略低于普通 CNN 的 `0.81906 rad` 和物理约束 CNN 的 `0.81642 rad`。

因此，若采用最佳验证 checkpoint 作为模型选择策略，`residual_cnn` 在本次复跑中首次超过当前 7 光束主线 baseline。

## 输出文件

```text
result/metrics/cycle23_residual_best_50epoch_2026-06-10.csv
result/metrics/cycle23_residual_best_50epoch/residual_cnn_history.csv
result/metrics/cycle23_residual_best_50epoch/residual_cnn_summary.csv
result/figures/cycle23_residual_best_50epoch_2026-06-10.png
models/cycle23_residual_best_50epoch_residual_cnn_seven_beam.pth
models/cycle23_residual_best_50epoch_residual_cnn_seven_beam_best.pth
```

其中 `models/*.pth` 为本地权重文件，不提交 Git。

## 阶段结论

本次结果说明，`residual_cnn` 的问题主要不是完全无法学习，而是最终 epoch 选择不可靠。最佳验证 checkpoint 将测试 RMSE 从最终 checkpoint 的 `1.269384 rad` 降到 `0.992071 rad`，并优于当前普通 CNN 与物理约束 CNN baseline。

因此后续建议：

- 在论文实验中采用“最佳验证 checkpoint”作为统一模型选择策略，而不是直接使用最终 epoch。
- 用相同策略重新训练并评估 `simple_cnn`，确保 `residual_cnn` 的提升来自结构本身，而不是 checkpoint 选择策略。
- 考虑加入早停、学习率调度或更小学习率，减少后期验证 RMSE 回升。
- 若继续探索 80 epoch，应优先使用较小学习率，例如 `0.0003`，并以最佳 checkpoint 指标作为主要判断标准。

## 注意事项

本次提交应包含日志、CSV 和图，不应包含 `dataset/` 与 `models/*.pth`。模型权重已保留在本地，供后续补偿效果评估使用。
