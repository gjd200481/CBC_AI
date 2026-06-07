# Cycle 03 静态双光束主数据集记录

## 任务目标

根据新的论文摘要，将项目主线收束为：

```text
单帧远场光强 -> CNN 相位反演 -> sin/cos 相位编码 -> FFT 物理一致性约束
```

本周期先完成论文主路线所需的第一版静态双光束数据集。

## 数据集信息

- 数据集名称：`main_clean_two_beam`
- 数据路径：`dataset/two_beam/main_static/`
- 图像文件：`images_main_clean_two_beam.npy`
- 标签文件：`labels_main_clean_two_beam.npy`
- 原始相位文件：`phases_main_clean_two_beam.npy`
- 配置文件：`config_main_clean_two_beam.json`

## 生成命令

```powershell
python simulation\static\generate_two_beam_dataset.py `
  --num-samples 2000 `
  --noise-sigma 0 `
  --num-points 256 `
  --window-size 0.01 `
  --waist 0.0005 `
  --beam-distance 0.0015 `
  --crop-size 160 `
  --phase-min -3.141592653589793 `
  --phase-max 3.141592653589793 `
  --seed 20260608 `
  --output-dir dataset\two_beam\main_static `
  --prefix main_clean_two_beam `
  --save-phases
```

## 数据形状

- 远场图像：`(2000, 160, 160)`，`float32`
- 相位标签：`(2000, 2)`，`float32`
- 原始相位：`(2000,)`，`float32`

## 数值检查

- 图像最小值：`0.0`
- 图像最大值：`1.0`
- 图像均值：`0.0015072247`
- 图像标准差：`0.0259504467`
- 相位最小值：`-3.1386303902 rad`
- 相位最大值：`3.1414902210 rad`
- 相位均值：`-0.0440238826 rad`
- 相位标准差：`1.8229978085 rad`
- `labels` 与 `[sin(phi), cos(phi)]` 的最大绝对误差：`1.1920928955e-07`

## 结论

该数据集满足新计划 Cycle 03 的核心要求：双光束、单帧远场光强、完整相位周期、`sin/cos` 标签、可复现配置。下一步可进入 Cycle 04，整理 PyTorch Dataset/DataLoader 和周期相位误差计算函数。
