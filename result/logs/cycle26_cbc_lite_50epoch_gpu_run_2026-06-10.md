# Cycle 26：CBC 自研轻量网络 50 epoch GPU 结果

日期：2026-06-10

## 实验目的

本周期用于验证文献启发后的自研 `cbc_lite_cnn` 是否能在 7 光束相位反演任务中优于当前残差与物理约束路线。

核心问题：

- 周期相位损失是否优于普通 MSE？
- `cbc_lite_cnn` 的轻量条纹特征结构是否优于 `residual_cnn_best`？
- 当前模型是否值得进入后续主瓣能量、Strehl 比和合成效率评估？

## 运行环境

- GPU：NVIDIA GeForce RTX 3060 Laptop GPU
- PyTorch：`2.5.1+cu121`
- CUDA 可用：`True`
- 分支：`cbc-lite-cyclic-phase`
- 数据集：`dataset/seven_beam/main_static`
- 训练/验证/测试划分：`716/153/155`
- 随机种子：`20260612`
- epoch：`50`
- batch size：`64`
- learning rate：`0.001`

## 训练命令

```powershell
.\scripts\run_cycle26_gpu_cbc_lite.ps1 -Epochs 50 -BatchSize 64 -LearningRate 0.001 -NumWorkers 2 -Seed 20260612 -PhaseLoss cyclic
.\scripts\run_cycle26_gpu_cbc_lite.ps1 -Epochs 50 -BatchSize 64 -LearningRate 0.001 -NumWorkers 2 -Seed 20260612 -PhaseLoss cyclic_unit
```

为判断问题来自模型结构还是损失函数，另补跑：

```powershell
python train\sweep_seven_beam_architecture.py --models cbc_lite_cnn --full-dataset --epochs 50 --batch-size 64 --learning-rate 0.001 --seed 20260612 --device cuda --num-workers 2 --pin-memory --phase-loss mse --experiment-tag cycle26_cbc_lite_mse_50epoch --history-dir result\metrics\cycle26_cbc_lite_mse_50epoch --summary-csv result\metrics\cycle26_cbc_lite_mse_50epoch_2026-06-10.csv --figure-path result\figures\cycle26_cbc_lite_mse_50epoch_2026-06-10.png
```

## 结果汇总

| 模型 | 损失 | 参数量 | 最佳 epoch | 最佳验证 RMSE(rad) | 最佳 checkpoint 测试 RMSE(rad) | 测试 MAE(rad) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cbc_lite_cnn` | `mse` | 822,636 | 10 | 1.146974 | 1.219643 | 0.951439 |
| `cbc_lite_cnn` | `cyclic` | 822,636 | 50 | 1.175314 | 1.281704 | 0.999275 |
| `cbc_lite_cnn` | `cyclic_unit` | 822,636 | 38 | 1.158658 | 1.255836 | 0.980571 |

对照结果：

| 对照模型 | 最佳 checkpoint 测试 RMSE(rad) |
| --- | ---: |
| `residual_cnn_best` | 0.992071 |
| `residual_cnn + physics loss, lambda_phy=0.05` | 0.983128 |

## 逐通道 RMSE

`cbc_lite_cnn + mse`：

```text
channel_1: 1.277563 rad
channel_2: 1.195363 rad
channel_3: 1.221823 rad
channel_4: 1.275659 rad
channel_5: 1.183844 rad
channel_6: 1.254285 rad
```

`cbc_lite_cnn + cyclic`：

```text
channel_1: 1.306663 rad
channel_2: 1.231796 rad
channel_3: 1.371714 rad
channel_4: 1.323769 rad
channel_5: 1.171833 rad
channel_6: 1.274655 rad
```

`cbc_lite_cnn + cyclic_unit`：

```text
channel_1: 1.303260 rad
channel_2: 1.174736 rad
channel_3: 1.352801 rad
channel_4: 1.269041 rad
channel_5: 1.195486 rad
channel_6: 1.258348 rad
```

## 结果文件

```text
result/metrics/cycle26_cbc_lite_mse_50epoch_2026-06-10.csv
result/metrics/cycle26_cbc_lite_mse_50epoch/cbc_lite_cnn_history.csv
result/metrics/cycle26_cbc_lite_mse_50epoch/cbc_lite_cnn_summary.csv

result/metrics/cycle26_cbc_lite_cyclic_50epoch_2026-06-10.csv
result/metrics/cycle26_cbc_lite_cyclic_50epoch/cbc_lite_cnn_history.csv
result/metrics/cycle26_cbc_lite_cyclic_50epoch/cbc_lite_cnn_summary.csv

result/metrics/cycle26_cbc_lite_cyclic_unit_50epoch_2026-06-10.csv
result/metrics/cycle26_cbc_lite_cyclic_unit_50epoch/cbc_lite_cnn_history.csv
result/metrics/cycle26_cbc_lite_cyclic_unit_50epoch/cbc_lite_cnn_summary.csv

result/figures/cycle26_cbc_lite_mse_50epoch_2026-06-10.png
result/figures/cycle26_cbc_lite_gpu_comparison_2026-06-10.png
```

本地模型权重已生成，但不提交 Git：

```text
models/cycle26_cbc_lite_mse_50epoch_cbc_lite_cnn_seven_beam.pth
models/cycle26_cbc_lite_mse_50epoch_cbc_lite_cnn_seven_beam_best.pth
models/cycle26_cbc_lite_cyclic_50epoch_cbc_lite_cnn_seven_beam.pth
models/cycle26_cbc_lite_cyclic_50epoch_cbc_lite_cnn_seven_beam_best.pth
models/cycle26_cbc_lite_cyclic_unit_50epoch_cbc_lite_cnn_seven_beam.pth
models/cycle26_cbc_lite_cyclic_unit_50epoch_cbc_lite_cnn_seven_beam_best.pth
```

## 异常说明

`cyclic` 和 `cyclic_unit` 两轮在训练脚本末尾绘图阶段触发 Windows 环境下的 OpenMP 重复库报错，因此原始训练脚本没有生成对应单独图像。但训练、CSV 汇总和模型 checkpoint 已经保存完成。随后使用已保存的 CSV 生成了统一对比图：

```text
result/figures/cycle26_cbc_lite_gpu_comparison_2026-06-10.png
```

## 结论

1. 当前 `cbc_lite_cnn` 不应升级为论文主模型。
2. 同一结构下，普通 `mse` 反而优于 `cyclic` 和 `cyclic_unit`，说明当前周期损失接入方式没有改善泛化。
3. `cbc_lite_cnn + mse` 最佳 checkpoint 测试 RMSE 为 `1.219643 rad`，明显弱于 `residual_cnn_best` 的 `0.992071 rad` 和 `residual_cnn + physics loss` 的 `0.983128 rad`。
4. Cycle 27 不建议把 `cbc_lite_cnn` 直接纳入补偿主评估，应优先继续沿 `residual_cnn + physics loss` 路线，并将 `cbc_lite_cnn` 作为一次负结果记录。

## 下一步建议

- 保留 `cbc_lite_cnn` 代码作为创新探索和消融材料，但不作为当前主模型。
- 若继续探索周期损失，应优先在 `residual_cnn` 上测试 `--phase-loss cyclic`，而不是先更换网络结构。
- 下一轮更值得运行：

```powershell
python train\train_seven_beam_physics_constrained_cnn.py --model-name residual_cnn --phase-loss cyclic --lambda-phy 0.05 --epochs 50 --batch-size 32 --learning-rate 0.001 --seed 20260612 --device cuda --num-workers 2
```

该实验可以判断“残差结构 + 周期相位损失 + 物理约束”是否比当前 `lambda_phy=0.05` 的 MSE 相位监督更好。
