# Cycle 16：7 光束振幅失配与位置偏移鲁棒性实验

## 1. 本周期目标

本周期评估 7 光束普通 CNN 与 `lambda_phy=0.1` 物理约束 CNN 在复杂非理想扰动下的相位反演鲁棒性。扰动类型包括：

- 外圈 6 路光束振幅失配。
- 7 路光束中心位置随机偏移。

## 2. 新增文件

### `simulation/static/generate_seven_beam_complex_robustness_dataset.py`

用于生成共享相位样本的 7 光束复杂扰动数据集。振幅失配和位置偏移都共用同一批 6 路相位标签。

### `train/evaluate_seven_beam_complex_robustness.py`

用于加载 7 光束普通 CNN 和物理约束 CNN，并在振幅失配、位置偏移数据集上评估：

- 整体相位 RMSE。
- 相位 MAE。
- 逐通道 RMSE。
- 远场重建 MSE。

## 3. 数据集生成

生成命令：

```powershell
python simulation\static\generate_seven_beam_complex_robustness_dataset.py --num-samples 256 --amplitude-levels 0 0.05 0.1 0.2 0.3 --position-levels 0 0.00001 0.00002 0.00005 0.0001 --num-points 256 --window-size 0.01 --waist 0.0005 --beam-distance 0.0015 --crop-size 160 --seed 20260616 --output-dir dataset\seven_beam\complex_robustness
```

数据集设置：

| 项目 | 数值 |
| --- | --- |
| 样本数 | `256` |
| 图像尺寸 | `160 x 160` |
| 标签维度 | `12` |
| 振幅失配范围 | `0, 0.05, 0.1, 0.2, 0.3` |
| 位置偏移范围 | `0, 10um, 20um, 50um, 100um` |

振幅失配设置：中心参考光束振幅固定为 `1.0`，外圈 6 路振幅从 `[1-r, 1+r]` 均匀采样。

位置偏移设置：7 路光束中心位置在 `[-d, d]` 范围内随机偏移，单位为 m。

## 4. 评估命令

```powershell
python train\evaluate_seven_beam_complex_robustness.py --dataset-dir dataset\seven_beam\complex_robustness --amplitude-levels 0 0.05 0.1 0.2 0.3 --position-levels 0 0.00001 0.00002 0.00005 0.0001 --batch-size 64 --output-csv result\metrics\cycle16_seven_beam_complex_robustness_2026-06-08.csv --improvement-csv result\metrics\cycle16_seven_beam_complex_robustness_improvement_2026-06-08.csv --figure-path result\figures\cycle16_seven_beam_complex_robustness_2026-06-08.png
```

对比模型：

```text
models/baseline_cnn_main_clean_seven_beam_2026-06-08.pth
models/physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.pth
```

## 5. 振幅失配结果

| mismatch range | 普通 CNN RMSE(rad) | 物理约束 CNN RMSE(rad) | RMSE 变化 |
| --- | --- | --- | --- |
| `0` | `1.0577218533` | `1.0410439968` | `-1.58%` |
| `0.05` | `1.0580766201` | `1.0377216339` | `-1.92%` |
| `0.1` | `1.0665618181` | `1.0411239862` | `-2.39%` |
| `0.2` | `1.0682966709` | `1.0549601316` | `-1.25%` |
| `0.3` | `1.0798407793` | `1.0650191307` | `-1.37%` |

## 6. 位置偏移结果

| offset range | 普通 CNN RMSE(rad) | 物理约束 CNN RMSE(rad) | RMSE 变化 |
| --- | --- | --- | --- |
| `0um` | `1.0577218533` | `1.0410439968` | `-1.58%` |
| `10um` | `1.0584468842` | `1.0419466496` | `-1.56%` |
| `20um` | `1.0585403442` | `1.0427912474` | `-1.49%` |
| `50um` | `1.0619761944` | `1.0368894339` | `-2.36%` |
| `100um` | `1.0593409538` | `1.0488035679` | `-0.99%` |

## 7. 阶段性判断

与 Cycle 15 的探测器噪声结果不同，当前复杂扰动下物理约束 CNN 基本保持小幅优势：

- 振幅失配范围到 `0.3` 时，物理约束 CNN 的 RMSE 仍比普通 CNN 低约 `1.25%` 到 `2.39%`。
- 位置偏移范围到 `100um` 时，物理约束 CNN 的 RMSE 仍比普通 CNN 低约 `0.99%` 到 `2.36%`。
- 远场 MSE 差异较小，部分等级下物理约束更低，部分等级下略高。

这说明当前物理约束对“光束能量分布和几何位置轻微偏离”比对“探测器噪声”更稳定。后续论文中可将该结果作为物理约束提升非理想光束泛化能力的证据，同时如实说明其对探测噪声并不自动鲁棒。

## 8. 结果文件

```text
result/metrics/cycle16_seven_beam_complex_robustness_2026-06-08.csv
result/metrics/cycle16_seven_beam_complex_robustness_improvement_2026-06-08.csv
result/figures/cycle16_seven_beam_complex_robustness_2026-06-08.png
```
