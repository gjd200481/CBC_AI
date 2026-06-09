# Cycle 18：7 光束 Strehl 比评估

## 1. 本周期目标

本周期增加 7 光束 Strehl 比指标，用理想相干合成远场峰值强度作为基准，评估补偿前、普通 CNN 补偿后、物理约束 CNN 补偿后和理想相干状态的峰值强度恢复情况。

## 2. 新增文件

```text
train/evaluate_seven_beam_strehl.py
```

该脚本用于：

- 加载 7 光束主静态数据集。
- 加载普通 CNN 和 `lambda_phy=0.1` 物理约束 CNN。
- 使用网络预测相位进行补偿，得到残余相位：

```text
residual_phase = true_phase - predicted_phase
```

- 重建补偿前、补偿后和理想相干远场。
- 计算 Strehl 比：

```text
Strehl = peak_intensity_current / peak_intensity_ideal
```

- 同时记录每个状态的残余相位 RMSE。

## 3. 运行命令

```powershell
python train\evaluate_seven_beam_strehl.py --max-samples 256 --batch-size 64 --example-index 0 --detail-csv result\metrics\cycle18_seven_beam_strehl_detail_2026-06-09.csv --summary-csv result\metrics\cycle18_seven_beam_strehl_summary_2026-06-09.csv --figure-path result\figures\cycle18_seven_beam_strehl_2026-06-09.png
```

使用数据：

```text
dataset/seven_beam/main_static/images_main_clean_seven_beam.npy
dataset/seven_beam/main_static/labels_main_clean_seven_beam.npy
```

使用模型：

```text
models/baseline_cnn_main_clean_seven_beam_2026-06-08.pth
models/physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.pth
```

样本数：`256`

理想相干远场峰值强度：`12780168.0`

## 4. 主要结果

| 状态 | Strehl 均值 | Strehl 标准差 | 相位 RMSE 均值(rad) | 相对补偿前提升 |
| --- | --- | --- | --- | --- |
| 补偿前 | `0.390687` | `0.112826` | `1.774909` | `0.00%` |
| 普通 CNN 补偿后 | `0.647172` | `0.191321` | `0.905907` | `65.65%` |
| 物理约束 CNN 补偿后 | `0.653564` | `0.194605` | `0.894276` | `67.29%` |
| 理想相干 | `1.000000` | `0.000000` | `0.000000` | `155.96%` |

## 5. 阶段性判断

普通 CNN 和物理约束 CNN 都显著提升了 Strehl 比，说明模型预测相位可以有效恢复远场峰值强度。物理约束 CNN 的平均 Strehl 比略高于普通 CNN：

```text
0.653564 - 0.647172 = 0.006392
```

同时物理约束 CNN 的残余相位 RMSE 更低：

```text
0.894276 rad < 0.905907 rad
```

这说明在当前干净 7 光束数据集上，相位 RMSE、主瓣能量占比和 Strehl 比三个指标的趋势基本一致：物理约束 CNN 相比普通 CNN 有小幅优势，但仍明显低于理想相干状态。

## 6. 结果文件

```text
result/metrics/cycle18_seven_beam_strehl_detail_2026-06-09.csv
result/metrics/cycle18_seven_beam_strehl_summary_2026-06-09.csv
result/figures/cycle18_seven_beam_strehl_2026-06-09.png
```
