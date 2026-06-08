# Cycle 10 振幅失配鲁棒性实验记录

## 任务目标

测试子光束振幅失配对相位反演结果的影响，比较普通 CNN 和物理约束 CNN 在非理想振幅条件下的泛化能力。

## 实验设计

本周期生成一套振幅失配数据集。所有失配等级使用同一组真实相位，只改变第二束光的振幅。

设置：

```text
amplitude_1 = 1.0
amplitude_2 ~ Uniform(1-r, 1+r)
```

其中 `r` 为振幅失配范围。

测试等级：

```text
r = 0, 0.05, 0.1, 0.2, 0.3
```

数据集路径：

```text
dataset/two_beam/amplitude_mismatch/
```

每个失配等级：

- 样本数：1000
- 图像尺寸：`160 x 160`
- 标签格式：`[sin(phi), cos(phi)]`
- 相位范围：`[-pi, pi]`
- 噪声强度：0
- phase seed：`20260610`

## 新增脚本

### `simulation/static/generate_two_beam_amplitude_mismatch_dataset.py`

作用：

- 生成多个振幅失配等级的数据集。
- 保证不同失配等级共用同一组相位。
- 保存第二束光的实际振幅数组 `amplitude2_<prefix>.npy`。

### `train/evaluate_amplitude_mismatch.py`

作用：

- 加载普通 CNN baseline 和物理约束 CNN。
- 在多个振幅失配数据集上计算相位 RMSE、MAE 和平均误差。
- 输出振幅失配-相位误差曲线。

## 数据生成命令

```powershell
python simulation\static\generate_two_beam_amplitude_mismatch_dataset.py `
  --num-samples 1000 `
  --mismatch-levels 0 0.05 0.1 0.2 0.3 `
  --seed 20260610 `
  --output-dir dataset\two_beam\amplitude_mismatch
```

## 评估命令

```powershell
python -m train.evaluate_amplitude_mismatch `
  --dataset-dir dataset\two_beam\amplitude_mismatch `
  --baseline-model models\baseline_cnn_main_clean.pth `
  --physics-model models\sweep_lambda_0.01_main_clean.pth `
  --mismatch-levels 0 0.05 0.1 0.2 0.3 `
  --batch-size 64 `
  --output-csv result\metrics\cycle10_amplitude_mismatch_2026-06-08.csv `
  --figure-path result\figures\cycle10_amplitude_mismatch_2026-06-08.png
```

## 输出文件

- 指标表：`result/metrics/cycle10_amplitude_mismatch_2026-06-08.csv`
- 曲线图：`result/figures/cycle10_amplitude_mismatch_2026-06-08.png`

## 相位 RMSE 结果

| mismatch_range | 普通 CNN RMSE(rad) | 物理约束 CNN RMSE(rad) | 物理约束相对变化 |
|---:|---:|---:|---:|
| 0 | 0.003884539 | 0.004350358 | -11.99% |
| 0.05 | 0.003883322 | 0.004358020 | -12.22% |
| 0.1 | 0.003883949 | 0.004384364 | -12.88% |
| 0.2 | 0.003936823 | 0.004524674 | -14.93% |
| 0.3 | 0.004277749 | 0.004933744 | -15.34% |

## 结论

本周期结果显示：

- 在当前双光束、无噪声、中心裁剪和最大值归一化设置下，振幅失配对相位反演影响较小。
- 当第二束振幅在 `0.7` 到 `1.3` 范围内随机变化时，普通 CNN 的 RMSE 仅从约 `0.00388 rad` 增加到 `0.00428 rad`。
- 物理约束 CNN 也保持稳定，但整体 RMSE 高于普通 CNN。
- 这说明当前物理约束模型的优势并不体现在振幅失配实验上，而更明显地体现在 Cycle 09 的中等噪声鲁棒性中。

## 阶段性判断

振幅失配实验给出了一个重要边界结论：不是所有非理想扰动都会让物理约束 CNN 优于普通 CNN。论文中应如实表述：

- 普通 CNN 在干净和振幅失配条件下已经具有很强相位反演能力。
- 物理约束 CNN 在中等探测器噪声下表现出更好的鲁棒性。
- 后续需要继续测试位置偏移和混合扰动，判断物理约束在哪些扰动类型下更有价值。

下一步 Cycle 11 建议加入位置偏移扰动，测试子光束中心位置误差对相位反演的影响。
