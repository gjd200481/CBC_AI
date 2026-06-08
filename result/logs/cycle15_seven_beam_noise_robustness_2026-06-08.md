# Cycle 15：7 光束探测器噪声鲁棒性实验

## 1. 本周期目标

本周期评估 7 光束普通 CNN 与 7 光束物理约束 CNN 在探测器高斯噪声下的相位反演鲁棒性。重点观察噪声增强时整体相位 RMSE、逐通道 RMSE 和远场重建 MSE 的变化趋势。

## 2. 新增文件

### `simulation/static/generate_seven_beam_noise_robustness_dataset.py`

用于生成共享相位样本的 7 光束噪声鲁棒性数据集。所有噪声等级共用同一组 6 路相位标签，只改变远场图像中的探测器噪声强度。

### `train/evaluate_seven_beam_noise_robustness.py`

用于加载 7 光束普通 CNN 和 `lambda_phy=0.1` 物理约束 CNN，在多个噪声等级数据集上评估：

- 整体相位 RMSE。
- 相位 MAE。
- 逐通道 RMSE。
- 基于 `SevenBeamFourierOptics` 的远场重建 MSE。

## 3. 数据集生成

生成命令：

```powershell
python simulation\static\generate_seven_beam_noise_robustness_dataset.py --num-samples 512 --noise-levels 0 0.01 0.03 0.05 0.08 --num-points 256 --window-size 0.01 --waist 0.0005 --beam-distance 0.0015 --crop-size 160 --seed 20260615 --output-dir dataset\seven_beam\noise_robustness
```

数据集设置：

| 项目 | 数值 |
| --- | --- |
| 样本数 | `512` |
| 图像尺寸 | `160 x 160` |
| 标签维度 | `12` |
| 相位通道数 | `6` |
| 噪声等级 | `0, 0.01, 0.03, 0.05, 0.08` |

输出目录：

```text
dataset/seven_beam/noise_robustness/
```

数据集本体保存在本地，不提交 Git。

## 4. 评估命令

```powershell
python train\evaluate_seven_beam_noise_robustness.py --dataset-dir dataset\seven_beam\noise_robustness --noise-levels 0 0.01 0.03 0.05 0.08 --batch-size 64 --output-csv result\metrics\cycle15_seven_beam_noise_robustness_2026-06-08.csv --improvement-csv result\metrics\cycle15_seven_beam_noise_robustness_improvement_2026-06-08.csv --figure-path result\figures\cycle15_seven_beam_noise_robustness_2026-06-08.png
```

对比模型：

```text
models/baseline_cnn_main_clean_seven_beam_2026-06-08.pth
models/physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.pth
```

## 5. 主要结果

| noise_sigma | 普通 CNN RMSE(rad) | 物理约束 CNN RMSE(rad) | RMSE 变化 |
| --- | --- | --- | --- |
| `0` | `1.0134476423` | `1.0058515072` | `-0.75%` |
| `0.01` | `1.0152368546` | `1.0177407265` | `+0.25%` |
| `0.03` | `1.0526164770` | `1.2601274252` | `+19.71%` |
| `0.05` | `1.1301172972` | `1.5869621038` | `+40.42%` |
| `0.08` | `1.3212444782` | `1.7108837366` | `+29.49%` |

远场重建 MSE：

| noise_sigma | 普通 CNN far-field MSE | 物理约束 CNN far-field MSE | 变化 |
| --- | --- | --- | --- |
| `0` | `1.2190251255e-04` | `1.2020073245e-04` | `-1.40%` |
| `0.01` | `1.7321204359e-04` | `1.7471578212e-04` | `+0.87%` |
| `0.03` | `5.8277812059e-04` | `6.5677396196e-04` | `+12.70%` |
| `0.05` | `1.3962619123e-03` | `1.5789742902e-03` | `+13.09%` |
| `0.08` | `3.3965322073e-03` | `3.5808359389e-03` | `+5.43%` |

## 6. 结果文件

```text
result/metrics/cycle15_seven_beam_noise_robustness_2026-06-08.csv
result/metrics/cycle15_seven_beam_noise_robustness_improvement_2026-06-08.csv
result/figures/cycle15_seven_beam_noise_robustness_2026-06-08.png
```

## 7. 阶段性判断

在干净数据上，`lambda_phy=0.1` 物理约束 CNN 相比普通 CNN 有小幅优势；但随着探测器噪声增加，当前物理约束模型比普通 CNN 更敏感，特别是在 `noise=0.03`、`0.05`、`0.08` 条件下，相位 RMSE 明显升高。

这说明当前 7 光束物理约束模型仍是“干净数据训练 + 干净物理传播约束”，并没有显式学习噪声扰动下的稳定特征。下一步建议：

- 训练噪声增强版 7 光束普通 CNN 和物理约束 CNN。
- 比较训练时加入噪声后，物理约束是否能重新体现优势。
- 后续可考虑将物理损失作用于去噪后的远场目标，而不是直接拟合含噪远场。
