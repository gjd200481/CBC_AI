# Cycle 23：最佳 checkpoint 策略与 3060 后续训练建议

## 任务背景

RTX 3060 已完成 `residual_cnn` 完整 7 光束数据 50 epoch 复跑。结果显示最终 epoch 测试 RMSE 为 `1.319034 rad`，未优于当前普通 CNN baseline `1.02698 rad` 和物理约束 CNN `1.02269 rad`。

但训练过程中最优验证 RMSE 达到 `0.973325 rad`，最终验证 RMSE 回升到 `1.219996 rad`，说明模型后期可能存在过拟合或训练不稳定。因此本周期修正计划：先加入最佳验证 checkpoint 保存与评估，再用 RTX 3060 做公平长训练。

## 本周期修改

修改文件：

```text
train/sweep_seven_beam_architecture.py
GPU_TRAINING_3060.md
scripts/run_cycle22_gpu_residual.ps1
PROJECT_PLAN.md
PROJECT_STATUS.md
README.md
KEY_FILES.md
```

## 训练脚本新增能力

`train/sweep_seven_beam_architecture.py` 已新增：

- 自动记录 `best_epoch`。
- 自动保存最终 epoch 权重：

```text
models/<experiment_tag>_<model>_seven_beam.pth
```

- 自动保存最佳验证 RMSE 权重：

```text
models/<experiment_tag>_<model>_seven_beam_best.pth
```

- 在汇总 CSV 中增加：
  - `best_checkpoint_test_rmse_rad`
  - `best_checkpoint_test_mae_rad`
  - `best_checkpoint_test_loss`
  - `best_checkpoint_channel_i_rmse_rad`
  - `selection_rmse_rad`

## 本地 smoke 验证

在当前 CPU 电脑上运行最小 smoke：

```powershell
python train\sweep_seven_beam_architecture.py --models residual_cnn --epochs 2 --batch-size 16 --max-samples 24 --device cpu --no-save-model --experiment-tag cycle23_best_checkpoint_smoke --history-dir result\metrics\cycle23_best_checkpoint_smoke --summary-csv result\metrics\cycle23_best_checkpoint_smoke_2026-06-10.csv --figure-path result\figures\cycle23_best_checkpoint_smoke_2026-06-10.png
```

验证结果：

| 项目 | 数值 |
| --- | ---: |
| 样本数 | 24 |
| epoch | 2 |
| final epoch 测试 RMSE | `1.746745 rad` |
| best checkpoint 测试 RMSE | `1.812770 rad` |
| best epoch | 1 |

该 smoke 只验证最佳 checkpoint 逻辑可以正常运行，不用于论文结果。

## 是否还需要 RTX 3060

需要。

当前 CPU 可以做代码修改、smoke 和文档整理，但完整 7 光束数据的公平训练需要 3060。下一步不建议继续直接堆 80 epoch，而应优先跑“保存最佳 checkpoint + 固定随机种子”的公平实验。

## 下一次 3060 推荐命令

优先运行：

```powershell
.\scripts\run_cycle22_gpu_residual.ps1 -Epochs 50 -BatchSize 64 -LearningRate 0.001 -NumWorkers 2 -Seed 20260612
```

如需公平对比 `simple_cnn` 和 `residual_cnn`：

```powershell
python train\sweep_seven_beam_architecture.py --models simple_cnn residual_cnn --full-dataset --epochs 50 --batch-size 64 --learning-rate 0.001 --seed 20260612 --device cuda --num-workers 2 --pin-memory --experiment-tag cycle23_arch_fair_50epoch --history-dir result\metrics\cycle23_arch_fair_50epoch --summary-csv result\metrics\cycle23_arch_fair_50epoch_2026-06-10.csv --figure-path result\figures\cycle23_arch_fair_50epoch_2026-06-10.png
```

## 阶段建议

- 当前不建议把 `residual_cnn` 作为论文主模型。
- 暂时保留 `simple_cnn + physics loss` 作为主线。
- `residual_cnn` 下一步只作为候选结构继续验证。
- 若最佳 checkpoint 测试 RMSE 仍无法低于 `1.02698 rad`，应停止残差结构投入，转向数据规模扩充、噪声增强训练或物理约束残差网络。
