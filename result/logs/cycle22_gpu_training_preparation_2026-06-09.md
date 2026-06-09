# Cycle 22：RTX 3060 长轮次训练准备

## 任务背景

用户补充说明有一台带 RTX 3060 的电脑可以用于多轮次模型训练。因此本周期优先调整计划：暂缓原定泛化实验，先把 Cycle 21 中筛选出的 `residual_cnn` 候选结构整理成可在 GPU 上完整数据长训练的流程。

## 本周期目标

- 增强 `train/sweep_seven_beam_architecture.py`，支持完整数据长训练。
- 支持显式指定 `--device cuda`。
- 支持 `--full-dataset`，避免默认 96 样本 smoke 设置影响长训练。
- 支持 `--experiment-tag`，避免不同实验覆盖模型权重和结果文件。
- 支持 `--num-workers` 和 `--pin-memory`，便于 GPU 训练加速。
- 提供 RTX 3060 训练说明和 PowerShell 启动脚本。
- 在当前 CPU 环境完成最小 smoke 验证。

## 修改文件

```text
train/sweep_seven_beam_architecture.py
GPU_TRAINING_3060.md
scripts/run_cycle22_gpu_residual.ps1
```

## 新增脚本能力

`train/sweep_seven_beam_architecture.py` 新增参数：

| 参数 | 作用 |
| --- | --- |
| `--full-dataset` | 使用完整 7 光束主数据集，而不是默认 96 样本快速筛选 |
| `--device cuda` | 强制使用 CUDA；若 CUDA 不可用会直接报错 |
| `--num-workers` | DataLoader 多进程读取 |
| `--pin-memory` | GPU 训练时加速 CPU 到 GPU 数据传输 |
| `--experiment-tag` | 控制模型权重文件名前缀，避免覆盖旧实验 |
| `--no-save-model` | smoke 验证时不保存模型权重 |

## 本地 smoke 验证

当前机器使用 CPU，仅做最小可运行验证：

```powershell
python train\sweep_seven_beam_architecture.py --models residual_cnn --epochs 1 --batch-size 16 --max-samples 24 --device cpu --no-save-model --experiment-tag cycle22_gpu_smoke --history-dir result\metrics\cycle22_gpu_smoke --summary-csv result\metrics\cycle22_gpu_smoke_2026-06-09.csv --figure-path result\figures\cycle22_gpu_smoke_2026-06-09.png
```

输出结果：

| 项目 | 数值 |
| --- | ---: |
| 使用样本 | 24 |
| 训练/验证/测试 | 16 / 3 / 5 |
| epoch | 1 |
| 模型 | `residual_cnn` |
| 测试 RMSE | `1.812770 rad` |

该结果只用于验证脚本参数、数据读取、训练、评估和画图流程，不用于论文性能结论。

## 推荐 GPU 命令

RTX 3060 上优先运行：

```powershell
.\scripts\run_cycle22_gpu_residual.ps1 -Epochs 50 -BatchSize 64 -LearningRate 0.001 -NumWorkers 2
```

等价 Python 命令：

```powershell
python train\sweep_seven_beam_architecture.py --models residual_cnn --full-dataset --epochs 50 --batch-size 64 --learning-rate 0.001 --device cuda --num-workers 2 --pin-memory --experiment-tag cycle22_residual_full_50epoch --history-dir result\metrics\cycle22_residual_full_50epoch --summary-csv result\metrics\cycle22_residual_full_50epoch_2026-06-09.csv --figure-path result\figures\cycle22_residual_full_50epoch_2026-06-09.png
```

## 长训练完成后的判断标准

优先比较：

- 是否低于当前 7 光束普通 CNN baseline：`RMSE=1.02698 rad`。
- 是否低于当前 7 光束物理约束 CNN：`RMSE=1.02269 rad`。
- 逐通道 RMSE 是否更均衡。
- 验证 RMSE 是否稳定下降，是否出现明显过拟合。

## 阶段结论

本周期已完成 GPU 长训练前的代码和文档准备。当前本地 CPU smoke 验证通过，说明脚本参数、完整/小样本数据切换、设备选择、结果保存路径都能正常工作。下一步应在 RTX 3060 电脑上运行 `residual_cnn` 完整数据 50 epoch 或 80 epoch 长训练，再将结果 CSV 和图带回项目中做正式对比。
