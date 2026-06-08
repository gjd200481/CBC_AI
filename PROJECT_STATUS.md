# CBC_AI 项目任务目标与当前进度

## 项目总目标

本项目面向相干光束合成系统中的相位误差检测问题，目标是建立一种基于深度学习和傅里叶光学物理约束的相位误差反演方法。当前论文主线为：

```text
远场光强图像
↓
CNN 相位反演网络
↓
输出 [sin(phi), cos(phi)]
↓
解码得到相位误差
↓
重建近场复振幅
↓
FFT 得到重建远场
↓
相位监督损失 + 远场物理一致性损失
```

项目希望在 2026 年 7 月底前完成可复现实验代码、仿真数据集、普通 CNN baseline、物理约束 CNN、噪声和复杂扰动鲁棒性实验、关键评价指标与论文写作材料。

## 当前研究定位

此前曾考虑使用 `CNN + LSTM` 做远场序列预测和未来相位预测，但根据新的摘要，当前主线已经调整为单帧远场光强到相位误差的反演。`CNN + LSTM` 动态预测暂时保留为后续拓展，不作为 7 月底前的主任务。

当前重点是证明：

- 远场光斑中包含可用于反演相位误差的信息。
- CNN 可以快速从远场光强预测相位误差。
- `sin/cos` 相位编码可以避免相位周期跳变。
- 傅里叶光学传播约束可以提高模型预测结果的物理一致性。
- 在噪声、振幅失配、位置偏移等非理想条件下，物理约束模型应具有更好的可信度或鲁棒性。

## 已完成工作

### 1. 项目结构整理

已将仿真、训练、模型评估、结果记录和文献资料分开管理。

关键目录：

- `simulation/common/`：公共光学仿真函数。
- `simulation/static/`：静态远场数据生成脚本。
- `simulation/dynamic/`：动态序列数据生成脚本，当前为拓展备用。
- `train/`：训练、数据读取、指标计算和物理损失模块。
- `model/`：模型推理和评估 demo。
- `dataset/`：本地生成数据集，不提交 Git。
- `models/`：本地模型权重，不提交 Git。
- `result/`：实验记录、指标表和结果图。
- `paper/`：期刊论文、学位论文和文献阅读结果。

### 2. 文献资料整理

已下载并整理中英文相关论文，包括：

- 深度学习相位控制。
- 单步相位识别与相干锁相。
- SPGD 与主动相位控制。
- 机器学习自适应光学。
- 中文期刊中关于光纤激光相干合成、相控阵、SPGD 改进的论文。

中文期刊清单见：

```text
paper/journals/chinese/README.md
```

### 3. 静态双光束仿真数据集

已完成双光束高斯相干合成仿真：

- 第一束光作为参考相位。
- 第二束光携带相位误差。
- 近场复振幅通过 FFT 得到远场光强。
- 远场中心区域裁剪为 `160 x 160`。
- 标签使用 `[sin(phi), cos(phi)]`。

当前主数据集：

```text
dataset/two_beam/main_static/
```

数据集信息：

- 样本数：2000
- 图像尺寸：`160 x 160`
- 标签维度：2
- 相位范围：`[-pi, pi]`
- 噪声强度：0
- 随机种子：20260608

记录文件：

```text
result/logs/cycle03_static_dataset_2026-06-07.md
result/metrics/cycle03_static_dataset_2026-06-07.csv
```

### 4. 数据读取与相位指标模块

已完成可复用训练基础模块：

- `train/data_utils.py`
  - `FarFieldPhaseDataset`
  - `split_dataset`
  - `build_dataloaders`

- `train/phase_metrics.py`
  - `decode_sin_cos`
  - `wrap_phase_error`
  - `phase_rmse_from_sin_cos`
  - `phase_metrics_from_sin_cos`

- `train/models.py`
  - `SimplePhaseCNN`

主数据集固定划分：

- 训练集：1400
- 验证集：300
- 测试集：300

记录文件：

```text
result/logs/cycle04_data_loader_metrics_2026-06-07.md
```

### 5. 普通 CNN baseline

已完成普通监督式 CNN baseline 训练，只使用相位标签 MSE 损失。

训练参数：

- 模型：`SimplePhaseCNN`
- epoch：20
- batch size：32
- learning rate：0.001
- 数据集：`main_clean_two_beam`

测试结果：

```text
RMSE(rad) = 0.003742
RMSE(deg) = 0.2144
MAE(rad)  = 0.003082
MAE(deg)  = 0.1766
```

记录文件：

```text
result/logs/cycle05_baseline_cnn_2026-06-07.md
result/metrics/baseline_cnn_main_clean_2026-06-07.csv
result/metrics/baseline_cnn_main_clean_summary_2026-06-07.csv
```

### 6. 傅里叶光学物理一致性损失

已实现可微分物理模块：

```text
train/physics_loss.py
```

核心类：

- `TwoBeamFourierOptics`
- `FarFieldConsistencyLoss`

该模块可以将网络预测的 `[sin(phi), cos(phi)]` 解码为相位，重建近场复振幅，再通过 `torch.fft` 得到远场光强，与输入远场进行 MSE 比较。

验证结果：

```text
真实标签重建远场 MSE = 1.08e-16
最大像素误差 = 4.77e-7
扰动预测下梯度有限 = True
```

记录文件：

```text
result/logs/cycle06_physics_loss_2026-06-07.md
```

### 7. 第一版物理约束 CNN

已完成第一版物理约束 CNN 训练：

```text
L_total = L_phase + lambda_phy * L_farfield
```

新增脚本：

```text
train/train_physics_constrained_cnn.py
```

第一版参数：

- `lambda_phy = 0.1`
- epoch：10
- batch size：32

测试结果：

```text
RMSE(rad) = 0.005782
RMSE(deg) = 0.3313
Far-field MSE = 9.35e-9
```

记录文件：

```text
result/logs/cycle07_physics_constrained_cnn_2026-06-07.md
```

### 8. 物理损失权重消融

已完成 `lambda_phy` 消融实验。

测试权重：

```text
0, 0.01, 0.05, 0.1, 0.5, 1.0
```

统一设置：

- epoch：8
- batch size：32
- seed：20260608

当前最优：

```text
lambda_phy = 0.01
RMSE(rad) = 0.004291
RMSE(deg) = 0.24585
Far-field MSE = 4.82e-9
```

结论：

- 在干净双光束数据集上，`lambda_phy=0.01` 是当前最优候选。
- 过大的物理损失权重会影响相位监督优化。
- 后续噪声鲁棒性实验优先使用 `lambda_phy=0.01`。

记录文件：

```text
result/logs/cycle08_lambda_sweep_2026-06-07.md
result/metrics/cycle08_lambda_sweep_2026-06-07.csv
```

## 当前正在进行的工作

当前已完成 Cycle 09：探测器噪声鲁棒性实验。

目标：

- 生成不同噪声强度下的远场数据。
- 保持各噪声数据集使用同一组真实相位，保证对比公平。
- 比较普通 CNN 和物理约束 CNN 在噪声增强时的误差变化。
- 输出噪声强度与相位 RMSE、远场重建 MSE 的关系曲线。

计划测试噪声：

```text
noise_sigma = 0, 0.01, 0.03, 0.05, 0.08
```

新增脚本：

```text
simulation/static/generate_two_beam_noise_robustness_dataset.py
train/evaluate_noise_robustness.py
```

实验结论：

- 在 `noise=0.01, 0.03, 0.05` 下，物理约束 CNN 的相位 RMSE 低于普通 CNN。
- 中等噪声下相对改善约 `10.60%` 到 `15.99%`。
- 在 `noise=0.08` 下，物理约束 CNN 略差于普通 CNN，说明当前权重和模型存在噪声适用边界。

记录文件：

```text
result/logs/cycle09_noise_robustness_2026-06-08.md
result/metrics/cycle09_noise_robustness_2026-06-08.csv
result/figures/cycle09_noise_robustness_2026-06-08.png
```

## 下一步计划

### Cycle 09

完成探测器噪声鲁棒性实验，得到：

- `result/metrics/cycle09_noise_robustness_2026-06-08.csv`
- `result/figures/cycle09_noise_robustness_2026-06-08.png`
- `result/logs/cycle09_noise_robustness_2026-06-08.md`

### Cycle 10

加入振幅失配扰动，测试子光束能量不一致时的相位反演性能。

### Cycle 11

加入位置偏移扰动，测试子光束中心位置误差对相位反演的影响。

### Cycle 12

生成混合扰动数据集，为论文主实验做准备。

## 当前阶段性判断

当前代码已经形成了较完整的研究闭环：

```text
光学仿真数据生成
↓
CNN 相位反演
↓
普通监督 baseline
↓
傅里叶光学物理约束
↓
物理约束 CNN
↓
权重消融
↓
噪声鲁棒性实验
```

从已有结果看，干净双光束条件下普通 CNN 已经非常强，物理约束模型的优势不应只在干净数据 RMSE 上寻找，而应重点放在：

- 噪声鲁棒性。
- 远场重建一致性。
- 非理想扰动泛化。
- 物理可信度。
- 后续主瓣能量占比和 Strehl 比等物理指标。
