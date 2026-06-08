# Cycle 09 探测器噪声鲁棒性实验记录

## 任务目标

测试普通 CNN 与物理约束 CNN 在不同探测器噪声强度下的相位误差变化，验证物理约束是否能提升噪声条件下的鲁棒性。

## 实验设计

为了保证对比公平，本周期重新生成了一套噪声鲁棒性数据集。所有噪声强度使用同一组真实相位，只改变远场图像中的高斯探测器噪声。

噪声强度：

```text
noise_sigma = 0, 0.01, 0.03, 0.05, 0.08
```

数据集路径：

```text
dataset/two_beam/noise_robustness/
```

每个噪声强度：

- 样本数：1000
- 图像尺寸：`160 x 160`
- 标签格式：`[sin(phi), cos(phi)]`
- 相位范围：`[-pi, pi]`
- phase seed：`20260609`

## 新增脚本

### `simulation/static/generate_two_beam_noise_robustness_dataset.py`

作用：

- 一次性生成多个噪声强度的数据集。
- 保证不同噪声强度共用同一组相位标签。
- 输出 `images_noise_<level>.npy`、`labels_noise_<level>.npy`、`phases_noise_<level>.npy` 和配置文件。

### `train/evaluate_noise_robustness.py`

作用：

- 加载普通 CNN baseline 和物理约束 CNN。
- 在多个噪声数据集上计算相位 RMSE、MAE 和远场重建 MSE。
- 输出 CSV 汇总表和噪声鲁棒性曲线。

## 数据生成命令

```powershell
python simulation\static\generate_two_beam_noise_robustness_dataset.py `
  --num-samples 1000 `
  --noise-levels 0 0.01 0.03 0.05 0.08 `
  --seed 20260609 `
  --output-dir dataset\two_beam\noise_robustness
```

## 评估命令

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -m train.evaluate_noise_robustness `
  --dataset-dir dataset\two_beam\noise_robustness `
  --baseline-model models\baseline_cnn_main_clean.pth `
  --physics-model models\sweep_lambda_0.01_main_clean.pth `
  --noise-levels 0 0.01 0.03 0.05 0.08 `
  --batch-size 64 `
  --output-csv result\metrics\cycle09_noise_robustness_2026-06-08.csv `
  --figure-path result\figures\cycle09_noise_robustness_2026-06-08.png
```

说明：`KMP_DUPLICATE_LIB_OK` 只用于本地 Windows 环境绕过 OpenMP 重复初始化提示，没有写入源码。

## 输出文件

- 指标表：`result/metrics/cycle09_noise_robustness_2026-06-08.csv`
- 曲线图：`result/figures/cycle09_noise_robustness_2026-06-08.png`

## 相位 RMSE 结果

| noise_sigma | 普通 CNN RMSE(rad) | 物理约束 CNN RMSE(rad) | 相对改善 |
|---:|---:|---:|---:|
| 0 | 0.003828659 | 0.004172641 | -8.98% |
| 0.01 | 0.011523650 | 0.009895319 | 14.13% |
| 0.03 | 0.035848718 | 0.030116912 | 15.99% |
| 0.05 | 0.061393619 | 0.054887373 | 10.60% |
| 0.08 | 0.109368250 | 0.111202560 | -1.68% |

## 远场重建 MSE

| noise_sigma | 普通 CNN Far-field MSE | 物理约束 CNN Far-field MSE |
|---:|---:|---:|
| 0 | 3.688689e-09 | 4.788211e-09 |
| 0.01 | 5.044268e-05 | 5.043524e-05 |
| 0.03 | 4.532019e-04 | 4.531200e-04 |
| 0.05 | 1.257019e-03 | 1.256858e-03 |
| 0.08 | 3.214135e-03 | 3.214203e-03 |

## 结论

本周期结果显示：

- 在干净数据 `noise=0` 上，普通 CNN 的相位 RMSE 略低于物理约束 CNN。
- 在中等噪声 `noise=0.01、0.03、0.05` 下，物理约束 CNN 的相位 RMSE 明显低于普通 CNN，相对改善约 `10.60%` 到 `15.99%`。
- 在高噪声 `noise=0.08` 下，物理约束 CNN 略差于普通 CNN，说明当前模型和权重设置存在噪声适用边界。
- 远场 MSE 随噪声增强显著上升，且两种模型差异很小。这是因为远场一致性指标当前直接与带噪输入图比较，指标主要受输入噪声本身支配。

## 阶段性判断

该实验比干净数据实验更能体现物理约束的价值：`lambda_phy=0.01` 物理约束 CNN 在中等噪声下具有更好的相位反演鲁棒性。

后续建议：

- Cycle 10 做振幅失配扰动，继续比较普通 CNN 与物理约束 CNN。
- 后续可增加“与无噪声真值远场比较”的远场物理一致性指标，避免带噪输入主导远场 MSE。
- 若时间允许，可训练“含噪训练集”的普通 CNN 与物理约束 CNN，比较训练增强后的鲁棒性。
