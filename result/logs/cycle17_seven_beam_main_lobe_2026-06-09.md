# Cycle 17：7 光束主瓣能量占比与相位补偿效果

## 1. 本周期目标

本周期增加 7 光束相干合成的主瓣能量占比指标，并比较：

- 补偿前远场。
- 普通 CNN 相位补偿后远场。
- 物理约束 CNN 相位补偿后远场。
- 理想相干合成远场。

## 2. 新增文件

```text
train/evaluate_seven_beam_compensation_metrics.py
```

该脚本用于：

- 加载 7 光束主静态数据集。
- 加载普通 CNN 和 `lambda_phy=0.1` 物理约束 CNN。
- 将网络预测相位作为补偿量，计算残余相位：

```text
residual_phase = true_phase - predicted_phase
```

- 根据补偿前、补偿后和理想相位重建 7 光束远场。
- 计算主瓣能量占比。
- 输出典型远场图。

## 3. 主瓣定义

主瓣能量占比定义为：

```text
main_lobe_ratio = E_main_lobe / E_crop
```

其中：

- `E_main_lobe`：远场中心圆形区域内的能量。
- `E_crop`：`160 x 160` 裁剪远场内的总能量。
- 主瓣圆形区域半径：`3 px`。

曾测试 `8 px` 半径，但该半径会把过多中央包络/旁瓣能量计入主瓣，导致理想相干结果不再明显高于随机相位结果。因此本周期采用 `3 px` 作为更严格的主瓣区域。

## 4. 运行命令

```powershell
python train\evaluate_seven_beam_compensation_metrics.py --max-samples 256 --batch-size 64 --main-lobe-radius 3 --example-index 0 --detail-csv result\metrics\cycle17_seven_beam_main_lobe_detail_2026-06-09.csv --summary-csv result\metrics\cycle17_seven_beam_main_lobe_summary_2026-06-09.csv --figure-path result\figures\cycle17_seven_beam_main_lobe_2026-06-09.png
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

## 5. 主要结果

| 状态 | 主瓣能量占比均值 | 标准差 | 相对补偿前提升 |
| --- | --- | --- | --- |
| 补偿前 | `0.359388` | `0.080525` | `0.00%` |
| 普通 CNN 补偿后 | `0.519307` | `0.073300` | `44.50%` |
| 物理约束 CNN 补偿后 | `0.521546` | `0.073165` | `45.12%` |
| 理想相干 | `0.650631` | `0.000000` | `81.04%` |

## 6. 阶段性判断

两类模型都能显著提高主瓣能量占比，说明网络预测相位具有明确的补偿价值。物理约束 CNN 的主瓣能量占比略高于普通 CNN：

```text
0.521546 - 0.519307 = 0.002239
```

相对普通 CNN 提升约 `0.43%`。该提升不大，但与 Cycle 16 中物理约束 CNN 在振幅失配和位置偏移下的小幅优势方向一致。

当前距离理想相干主瓣占比 `0.650631` 仍有差距，说明后续可以继续从模型结构、训练数据规模、噪声增强和物理损失权重等方向提升补偿质量。

## 7. 结果文件

```text
result/metrics/cycle17_seven_beam_main_lobe_detail_2026-06-09.csv
result/metrics/cycle17_seven_beam_main_lobe_summary_2026-06-09.csv
result/figures/cycle17_seven_beam_main_lobe_2026-06-09.png
```
