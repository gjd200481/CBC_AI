# Cycle 13：7 光束物理约束 CNN

## 1. 本周期目标

本周期将傅里叶光学物理一致性损失从双光束扩展到 7 光束，并训练第一版 7 光束物理约束 CNN。目标是在普通相位监督损失之外，引入“预测相位 -> 重建 7 光束近场 -> FFT 远场 -> 与输入远场比较”的物理一致性约束。

## 2. 新增与修改文件

### `train/physics_loss.py`

新增 `SevenBeamFourierOptics`：

- 中心 `beam_0` 为参考光束，相位固定为 0。
- 外圈 `beam_1 ... beam_6` 按六边形排布。
- 输入为 12 维标签 `[sin(phi_1), cos(phi_1), ..., sin(phi_6), cos(phi_6)]`。
- 解码得到 6 路相对相位后，重建 7 光束近场复振幅。
- 使用 `torch.fft.fft2` 计算远场光强。
- 归一化后裁剪中心 `160 x 160` 区域。

### `train/train_seven_beam_physics_constrained_cnn.py`

新增 7 光束物理约束 CNN 训练入口：

```text
L_total = L_phase + lambda_phy * L_farfield
```

其中：

- `L_phase`：预测 12 维 sin/cos 标签与真实标签的 MSE。
- `L_farfield`：预测相位重建远场与输入远场光强之间的 MSE。

## 3. 物理模型一致性验证

使用真实标签输入 `SevenBeamFourierOptics` 重建 7 光束远场，并与 numpy 数据生成结果比较：

| 指标 | 数值 |
| --- | --- |
| reconstructed shape | `(8, 160, 160)` |
| MSE | `1.2022093469e-16` |
| max abs error | `1.0132789612e-06` |
| target mean | `0.0009235210` |
| reconstructed mean | `0.0009235208` |

该结果说明 7 光束 torch 物理模型与 `simulation/common/multi_beam_core.py` 的 numpy 数据生成链路基本一致。

## 4. 训练设置

训练命令：

```powershell
python train\train_seven_beam_physics_constrained_cnn.py --lambda-phy 0.1 --epochs 30 --batch-size 32 --learning-rate 0.001 --seed 20260612 --no-plot --metrics-path result\metrics\physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.csv --summary-path result\metrics\physics_cnn_lambda_0.1_main_clean_seven_beam_summary_2026-06-08.csv --figure-path result\figures\physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.png --model-path models\physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.pth
```

训练设备：`cpu`

数据集与 Cycle 12 保持一致：

```text
dataset/seven_beam/main_static/images_main_clean_seven_beam.npy
dataset/seven_beam/main_static/labels_main_clean_seven_beam.npy
```

数据划分与普通 baseline 保持一致：

| split | 样本数 |
| --- | --- |
| train | `716` |
| val | `153` |
| test | `155` |

## 5. 测试结果

| 指标 | 普通 CNN | 物理约束 CNN |
| --- | --- | --- |
| RMSE(rad) | `1.0269757509` | `1.0226855278` |
| RMSE(deg) | `58.8413761902` | `58.5955645121` |
| MAE(rad) | `0.8190614581` | `0.8164239526` |
| phase loss | `0.2888655782` | `0.2838818210` |
| far-field loss | `1.1935354043e-04` | `1.1501365732e-04` |

相对变化：

- RMSE 降低约 `0.42%`。
- MAE 降低约 `0.32%`。
- 远场重建 MSE 降低约 `3.64%`。

逐通道 RMSE：

| 通道 | 普通 CNN RMSE(rad) | 物理约束 CNN RMSE(rad) |
| --- | --- | --- |
| channel 1 | `1.0642526150` | `1.0718975067` |
| channel 2 | `0.9943473935` | `0.9540684223` |
| channel 3 | `0.9671255946` | `1.0079314709` |
| channel 4 | `1.1497400999` | `1.1117705107` |
| channel 5 | `0.9450973272` | `0.9221594930` |
| channel 6 | `1.0277509689` | `1.0553513765` |

## 6. 结果文件

```text
result/metrics/physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.csv
result/metrics/physics_cnn_lambda_0.1_main_clean_seven_beam_summary_2026-06-08.csv
result/metrics/cycle13_seven_beam_physics_vs_baseline_2026-06-08.csv
result/figures/physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.png
models/physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.pth
```

说明：模型权重保留在本地，不提交到 Git；CSV、PNG、日志和源码提交到 Git。

## 7. 阶段性判断

第一版 7 光束物理约束 CNN 已经跑通，并在相位 RMSE、MAE 和远场重建 MSE 上相对普通 CNN 有小幅改善。不过当前 `lambda_phy=0.1` 时，远场损失量级约为 `1e-4`，乘以权重后对总损失贡献约为 `1e-5`，明显小于相位监督损失。因此该结果说明物理约束链路有效，但权重设置还没有充分发挥作用。

下一周期应进行 7 光束 `lambda_phy` 权重消融，至少比较 `0, 0.01, 0.1, 1.0, 10.0`，重点观察整体相位 RMSE、逐通道 RMSE、远场重建 MSE 和训练稳定性。
