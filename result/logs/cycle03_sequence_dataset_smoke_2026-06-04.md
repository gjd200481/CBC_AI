# Cycle 03 Smoke Test：双光束远场序列数据

## 目标

验证“远场序列 -> CNN+LSTM -> 预测未来相位”路线所需的数据格式是否能够正确生成。

## 生成命令

```bash
python simulation/generate_two_beam_sequence_dataset.py --num-sequences 6 --input-length 5 --predict-steps 2 --phase-mode mixed --noise-sigma 0.01 --seed 42 --output-dir dataset/two_beam_sequence/cycle03_smoke --prefix smoke_mixed
```

## 数据格式

| 文件 | 形状 | 含义 |
| --- | --- | --- |
| `images_smoke_mixed.npy` | `(6, 5, 160, 160)` | 每个样本包含 5 帧远场光强图 |
| `labels_smoke_mixed.npy` | `(6, 2)` | 未来目标相位 `[sin(phi), cos(phi)]` |
| `input_phases_smoke_mixed.npy` | `(6, 5)` | 输入帧对应真实相位 |
| `target_phases_smoke_mixed.npy` | `(6,)` | 未来目标相位 |
| `all_phases_smoke_mixed.npy` | `(6, 7)` | 完整相位轨迹，长度为 `input_length + predict_steps` |
| `modes_smoke_mixed.npy` | `(6,)` | 每条序列使用的相位扰动模式 |

## 验证结果

- 语法检查通过。
- 序列数据生成成功。
- 标签恢复相位与 `target_phases` 的最大周期误差约为 `1.49e-08 rad`。
- 当前数据形状可作为后续 CNN+LSTM 的输入基础。

## 下一步

- 建立 PyTorch 序列 Dataset。
- 输入格式建议为 `[batch, time, channel, height, width]`。
- CNN 逐帧提取空间特征，LSTM 建模时间演化并预测未来相位。

