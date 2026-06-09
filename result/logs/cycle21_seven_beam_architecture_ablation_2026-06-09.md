# Cycle 21：7 光束网络结构快速消融

## 任务目标

本周期目标是开始优化 7 光束相位反演网络结构，比较原始 CNN、加宽 CNN 和残差 CNN 在同一数据划分下的初步表现，为后续长训练选择候选结构。

由于当前机器使用 CPU 训练，全量 7 光束数据上进行多结构长训练耗时较高。因此本周期定位为**快速结构筛选**，不是最终严格消融结论。

## 新增与修改

新增模型结构：

```text
train/models.py
```

- `WidePhaseCNN`：增加卷积通道数，并使用自适应池化降低全连接层参数量。
- `ResidualBlock`：两层卷积残差块。
- `ResidualPhaseCNN`：残差连接 + 自适应池化，用于提升多路相位特征提取稳定性。
- `build_phase_model()`：按模型名称构建网络，方便后续消融。
- `count_parameters()`：统计模型可训练参数量。

新增结构消融脚本：

```text
train/sweep_seven_beam_architecture.py
```

## 运行命令

```powershell
python train\sweep_seven_beam_architecture.py
```

## 快速消融设置

| 参数 | 数值 |
| --- | --- |
| 数据集 | `dataset/seven_beam/main_static/images_main_clean_seven_beam.npy` |
| 标签 | `dataset/seven_beam/main_static/labels_main_clean_seven_beam.npy` |
| 快速样本数 | 96 |
| 训练/验证/测试 | 67 / 14 / 15 |
| epoch | 2 |
| batch size | 128 |
| learning rate | 0.001 |
| seed | 20260621 |
| device | CPU |

## 对比结果

| 模型 | 参数量 | 训练耗时(s) | 测试 RMSE(rad) | 测试 MAE(rad) | 最优验证 RMSE(rad) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `simple_cnn` | 3,301,772 | `1.17` | `1.815493` | `1.564345` | `1.604577` |
| `wide_cnn` | 2,193,164 | `2.10` | `1.781429` | `1.530938` | `1.600123` |
| `residual_cnn` | 1,008,492 | `10.30` | `1.709031` | `1.477339` | `1.508539` |

## 输出文件

```text
result/metrics/cycle21_seven_beam_architecture_ablation_2026-06-09.csv
result/metrics/cycle21_seven_beam_architecture/simple_cnn_history.csv
result/metrics/cycle21_seven_beam_architecture/wide_cnn_history.csv
result/metrics/cycle21_seven_beam_architecture/residual_cnn_history.csv
result/figures/cycle21_seven_beam_architecture_ablation_2026-06-09.png
```

模型权重保存在本地 `models/`，不提交 Git。

## 阶段结论

在本轮快速筛选中，`residual_cnn` 的测试 RMSE 最低，为 `1.709031 rad`，比 `simple_cnn` 低约 `0.106462 rad`，比 `wide_cnn` 低约 `0.072398 rad`。同时，`residual_cnn` 的参数量最少，约为 `1.01M`，但 CPU 训练耗时最高。

该结果说明残差连接和自适应池化值得继续探索。下一步建议将 `residual_cnn` 作为候选结构，在完整 7 光束数据集上进行更长轮数训练，并与原始 `simple_cnn` 的 30 epoch baseline 进行公平对比。

## 注意事项

本周期结果不应直接与 Cycle 12 的 30 epoch 全量 baseline 数值比较，因为本周期只使用 96 个样本训练 2 轮，目的是快速筛选结构趋势。真正论文结论需要后续使用完整数据、固定划分和相同训练轮数复训候选模型。
