# CBC_AI 项目任务目标与当前进度

## 项目总目标

本项目面向多路相干光束合成系统中的相位误差检测问题，目标是建立一种基于深度学习和傅里叶光学物理约束的相位误差反演方法。当前主目标已从早期双光束验证升级为**7 光束相干合成相位误差反演**。

7 光束主系统采用中心 1 路参考光束 + 外圈 6 路六边形阵列。中心光束相位固定为 0，网络预测外圈 6 路相对相位误差：

```text
label = [sin(phi_1), cos(phi_1), ..., sin(phi_6), cos(phi_6)]
```

双光束系统不废弃，而是作为方法验证、代码基线、物理损失验证和低维对照实验保留。当前论文主线为：

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

项目希望在 2026 年 7 月底前完成 7 光束可复现实验代码、仿真数据集、普通 CNN baseline、物理约束 CNN、噪声和复杂扰动鲁棒性实验、关键评价指标与论文写作材料。

## 当前研究定位

此前曾考虑使用 `CNN + LSTM` 做远场序列预测和未来相位预测，但根据新的摘要，当前主线已经调整为单帧远场光强到相位误差的反演。随后项目目标进一步从双光束升级为 7 光束多路相干合成。`CNN + LSTM` 动态预测暂时保留为后续拓展，不作为 7 月底前的主任务。

当前重点是证明：

- 远场光斑中包含可用于反演相位误差的信息。
- CNN 可以快速从远场光强预测相位误差。
- `sin/cos` 相位编码可以避免相位周期跳变。
- 傅里叶光学传播约束可以提高模型预测结果的物理一致性。
- 在噪声、振幅失配、位置偏移等非理想条件下，物理约束模型应具有更好的可信度或鲁棒性。
- 7 光束系统比双光束更接近多通道 CBC 应用场景，可体现多路相位耦合和高维相位反演难度。

## 已完成工作

### 1. 项目结构整理

已将仿真、训练、模型评估、结果记录和文献资料分开管理。

关键目录：

- `simulation/common/`：公共光学仿真函数。
- `simulation/static/`：静态远场数据生成脚本。
- `simulation/dynamic/`：动态序列数据生成脚本，当前为拓展备用。
- `train/`：训练、数据读取、指标计算和物理损失模块。
- `examples/`：模型推理和评估 demo。
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

已完成双光束高斯相干合成仿真。该部分现在定位为 7 光束主研究之前的低维验证基线：

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

### 3.1 目标升级：7 光束多路相干合成

当前项目目标已升级为 7 光束相干合成。7 光束基础仿真模块已经完成，当前已具备以下能力：

- 7 光束六边形阵列近场复振幅生成。
- 7 光束远场 FFT 传播。
- 6 个相对相位的 `sin/cos` 标签，共 12 维。
- 7 光束 smoke 数据集生成和数值检查。

后续需要继续实现：

- 7 光束普通 CNN baseline。
- 7 光束傅里叶光学物理一致性损失。
- 7 光束噪声、振幅失配、位置偏移和混合扰动鲁棒性实验。

计划中的相位定义：

```text
beam_0: center, phase = 0
beam_1 ... beam_6: outer ring, phase = phi_1 ... phi_6
```

当前已新增文件：

```text
simulation/common/multi_beam_core.py
simulation/static/generate_seven_beam_dataset.py
```

Cycle 11 已生成 32 样本 smoke 数据集，图像形状为 `(32, 160, 160)`，标签形状为 `(32, 12)`，相位形状为 `(32, 6)`。记录文件：

```text
result/logs/cycle11_seven_beam_smoke_2026-06-08.md
result/metrics/cycle11_seven_beam_smoke_2026-06-08.csv
```

Cycle 12 已完成首版 7 光束普通 CNN baseline。使用 1024 个干净样本，训练/验证/测试划分为 `716/153/155`。普通 CNN 测试集整体 RMSE 为 `1.02698 rad`，MAE 为 `0.81906 rad`，逐通道 RMSE 范围约为 `0.94510 rad` 至 `1.14974 rad`。记录文件：

```text
result/logs/cycle12_seven_beam_baseline_2026-06-08.md
result/metrics/baseline_cnn_main_clean_seven_beam_2026-06-08.csv
result/metrics/baseline_cnn_main_clean_seven_beam_summary_2026-06-08.csv
result/figures/baseline_cnn_main_clean_seven_beam_2026-06-08.png
```

Cycle 13 已完成首版 7 光束物理约束 CNN。新增 `SevenBeamFourierOptics`，真实标签重建远场 MSE 约为 `1.20e-16`。在 `lambda_phy=0.1`、30 轮训练下，物理约束 CNN 测试集 RMSE 为 `1.02269 rad`，略低于普通 CNN 的 `1.02698 rad`；远场重建 MSE 从普通 CNN 的 `1.1935e-4` 降至 `1.1501e-4`。记录文件：

```text
result/logs/cycle13_seven_beam_physics_cnn_2026-06-08.md
result/metrics/physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.csv
result/metrics/physics_cnn_lambda_0.1_main_clean_seven_beam_summary_2026-06-08.csv
result/metrics/cycle13_seven_beam_physics_vs_baseline_2026-06-08.csv
result/figures/physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.png
```

Cycle 14 已完成 7 光束 `lambda_phy` 权重消融。快速消融比较 `0, 0.01, 0.05, 0.1, 0.5, 1.0`，其中 `lambda_phy=0.1` 的 12 epoch 相位 RMSE 最低。进一步复训 `lambda_phy=0.5` 到 30 epoch 后，RMSE 为 `1.05027 rad`，不如 `lambda_phy=0.1` 的 `1.02269 rad`。当前主实验候选权重暂定为 `lambda_phy=0.1`。记录文件：

```text
result/logs/cycle14_seven_beam_lambda_sweep_2026-06-08.md
result/metrics/cycle14_seven_beam_lambda_sweep_2026-06-08.csv
result/metrics/cycle14_seven_beam_lambda_sweep_extended_2026-06-08.csv
result/figures/cycle14_seven_beam_lambda_sweep_2026-06-08.png
```

Cycle 15 已完成 7 光束探测器噪声鲁棒性实验。使用 512 个共享相位样本测试 `noise=0, 0.01, 0.03, 0.05, 0.08`。结果显示：`lambda_phy=0.1` 物理约束 CNN 在干净数据上略优于普通 CNN，但在 `noise>=0.03` 时相位 RMSE 明显更高，说明当前干净训练的物理约束模型对探测器噪声较敏感。记录文件：

```text
result/logs/cycle15_seven_beam_noise_robustness_2026-06-08.md
result/metrics/cycle15_seven_beam_noise_robustness_2026-06-08.csv
result/metrics/cycle15_seven_beam_noise_robustness_improvement_2026-06-08.csv
result/figures/cycle15_seven_beam_noise_robustness_2026-06-08.png
```

Cycle 16 已完成 7 光束振幅失配与位置偏移鲁棒性实验。使用 256 个共享相位样本，振幅失配范围为 `0, 0.05, 0.1, 0.2, 0.3`，位置偏移范围为 `0, 10um, 20um, 50um, 100um`。结果显示：`lambda_phy=0.1` 物理约束 CNN 在复杂扰动下基本保持小幅优势，RMSE 相比普通 CNN 降低约 `0.99%` 到 `2.39%`。记录文件：

```text
result/logs/cycle16_seven_beam_complex_robustness_2026-06-08.md
result/metrics/cycle16_seven_beam_complex_robustness_2026-06-08.csv
result/metrics/cycle16_seven_beam_complex_robustness_improvement_2026-06-08.csv
result/figures/cycle16_seven_beam_complex_robustness_2026-06-08.png
```

Cycle 17 已完成 7 光束主瓣能量占比与相位补偿效果评估。采用中心半径 `3 px` 圆形区域作为主瓣，256 个样本上补偿前主瓣能量占比为 `0.35939`，普通 CNN 补偿后为 `0.51931`，物理约束 CNN 补偿后为 `0.52155`，理想相干为 `0.65063`。记录文件：

```text
result/logs/cycle17_seven_beam_main_lobe_2026-06-09.md
result/metrics/cycle17_seven_beam_main_lobe_detail_2026-06-09.csv
result/metrics/cycle17_seven_beam_main_lobe_summary_2026-06-09.csv
result/figures/cycle17_seven_beam_main_lobe_2026-06-09.png
```

Cycle 18 已完成 7 光束 Strehl 比评估。以理想相干远场峰值强度 `12780168.0` 为基准，256 个样本上补偿前 Strehl 均值为 `0.39069`，普通 CNN 补偿后为 `0.64717`，物理约束 CNN 补偿后为 `0.65356`，理想相干为 `1.00000`。记录文件：

```text
result/logs/cycle18_seven_beam_strehl_2026-06-09.md
result/metrics/cycle18_seven_beam_strehl_detail_2026-06-09.csv
result/metrics/cycle18_seven_beam_strehl_summary_2026-06-09.csv
result/figures/cycle18_seven_beam_strehl_2026-06-09.png
```

Cycle 19 已完成 7 光束相位补偿综合效果实验。该周期将主瓣能量、旁瓣能量、Strehl 比、合成效率、峰值旁瓣比和残余相位 RMSE 放在同一脚本中统一评估。256 个样本上，补偿前主瓣能量占比为 `0.35939`，普通 CNN 补偿后为 `0.51931`，物理约束 CNN 补偿后为 `0.52155`；补偿前合成效率为 `0.53286`，普通 CNN 补偿后为 `0.78602`，物理约束 CNN 补偿后为 `0.78964`；补偿前相位 RMSE 为 `1.77491 rad`，普通 CNN 补偿后为 `0.90591 rad`，物理约束 CNN 补偿后为 `0.89428 rad`。记录文件：

```text
result/logs/cycle19_seven_beam_compensation_effect_2026-06-09.md
result/metrics/cycle19_seven_beam_compensation_effect_detail_2026-06-09.csv
result/metrics/cycle19_seven_beam_compensation_effect_summary_2026-06-09.csv
result/figures/cycle19_seven_beam_compensation_effect_2026-06-09.png
```

Cycle 20 已完成双光束/7 光束系统规模对比。7 光束相对双光束，待预测相位数量从 `1` 增加到 `6`，网络输出维度从 `2` 增加到 `12`。普通 CNN 相位 RMSE 从双光束 `0.003742 rad` 增至 7 光束 `1.026976 rad`，物理约束 CNN 相位 RMSE 从双光束 `0.004291 rad` 增至 7 光束 `1.022686 rad`。当前定位为：双光束用于验证方法链路，7 光束用于论文主实验。记录文件：

```text
result/logs/cycle20_system_scale_comparison_2026-06-09.md
result/metrics/cycle20_system_scale_comparison_2026-06-09.csv
result/metrics/cycle20_system_scale_ratio_2026-06-09.csv
result/figures/cycle20_system_scale_comparison_2026-06-09.png
```

Cycle 21 已完成 7 光束网络结构快速消融。新增 `WidePhaseCNN`、`ResidualPhaseCNN` 和结构消融脚本。由于当前 CPU 全量多结构训练耗时较高，本周期采用 96 样本、2 epoch 进行候选筛选。结果显示：`simple_cnn` 测试 RMSE 为 `1.815493 rad`，`wide_cnn` 为 `1.781429 rad`，`residual_cnn` 为 `1.709031 rad`。当前判断是残差 CNN 值得进入完整数据长训练验证。记录文件：

```text
result/logs/cycle21_seven_beam_architecture_ablation_2026-06-09.md
result/metrics/cycle21_seven_beam_architecture_ablation_2026-06-09.csv
result/metrics/cycle21_seven_beam_architecture/
result/figures/cycle21_seven_beam_architecture_ablation_2026-06-09.png
```

Cycle 22 已完成 RTX 3060 长轮次训练准备。根据用户补充的 GPU 资源，本周期优先增强 `train/sweep_seven_beam_architecture.py`，使其支持 `--full-dataset`、`--device cuda`、`--num-workers`、`--pin-memory`、`--experiment-tag` 和 `--no-save-model`。新增 `GPU_TRAINING_3060.md` 和 `scripts/run_cycle22_gpu_residual.ps1`。当前 CPU 环境完成了 24 样本、1 epoch 的 smoke 验证，流程可正常读取数据、训练、评估和保存结果。记录文件：

```text
result/logs/cycle22_gpu_training_preparation_2026-06-09.md
result/metrics/cycle22_gpu_smoke_2026-06-09.csv
result/metrics/cycle22_gpu_smoke/
result/figures/cycle22_gpu_smoke_2026-06-09.png
```

随后已从 RTX 3060 复跑分支合入 `residual_cnn` 完整数据 50 epoch 结果。该模型最终测试 RMSE 为 `1.319034 rad`，未优于当前 7 光束普通 CNN baseline 的 `1.02698 rad` 和物理约束 CNN 的 `1.02269 rad`。不过训练过程中最优验证 RMSE 为 `0.973325 rad`，低于最终验证 RMSE `1.219996 rad`，说明后期可能存在过拟合或训练不稳定，需要保存最佳验证 checkpoint 后重新评估。记录文件：

```text
result/logs/cycle22_residual_full_50epoch_gpu_rerun_2026-06-10.md
result/metrics/cycle22_residual_full_50epoch_2026-06-09.csv
result/metrics/cycle22_residual_full_50epoch/residual_cnn_history.csv
result/metrics/cycle22_residual_full_50epoch/residual_cnn_summary.csv
result/figures/cycle22_residual_full_50epoch_2026-06-10.png
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

已完成振幅失配扰动实验。

测试设置：

```text
amplitude_1 = 1.0
amplitude_2 ~ Uniform(1-r, 1+r)
r = 0, 0.05, 0.1, 0.2, 0.3
```

实验结果：

- 普通 CNN 在 `r=0.3` 时 RMSE 约 `0.004278 rad`。
- 物理约束 CNN 在 `r=0.3` 时 RMSE 约 `0.004934 rad`。
- 当前设置下两类模型都较稳定，但普通 CNN 更优。

记录文件：

```text
result/logs/cycle10_amplitude_mismatch_2026-06-08.md
result/metrics/cycle10_amplitude_mismatch_2026-06-08.csv
result/figures/cycle10_amplitude_mismatch_2026-06-08.png
```

### Cycle 11

已完成 7 光束基础仿真模块和 smoke 数据集。

核心输出：

- `simulation/common/multi_beam_core.py`
- `simulation/static/generate_seven_beam_dataset.py`
- `result/logs/cycle11_seven_beam_smoke_2026-06-08.md`

### Cycle 12

已完成 7 光束普通 CNN baseline。

核心输出：

- `train/train_seven_beam_baseline.py`
- `result/logs/cycle12_seven_beam_baseline_2026-06-08.md`
- `result/metrics/baseline_cnn_main_clean_seven_beam_summary_2026-06-08.csv`

下一步进入 7 光束物理约束 CNN：将 `train/physics_loss.py` 从双光束扩展到 7 光束，根据 6 路预测相位重建 7 光束近场，并把重建远场与输入远场的误差加入总损失。

### Cycle 13

已完成 7 光束物理约束 CNN。

核心输出：

- `train/physics_loss.py` 中的 `SevenBeamFourierOptics`
- `train/train_seven_beam_physics_constrained_cnn.py`
- `result/logs/cycle13_seven_beam_physics_cnn_2026-06-08.md`

当前判断：`lambda_phy=0.1` 可以带来小幅改善，但远场损失对总损失贡献偏小。下一步需要做 `lambda_phy` 权重消融，找到更适合 7 光束系统的物理约束强度。

### Cycle 14

已完成 7 光束物理损失权重消融。

核心输出：

- `train/sweep_seven_beam_lambda.py`
- `result/logs/cycle14_seven_beam_lambda_sweep_2026-06-08.md`
- `result/metrics/cycle14_seven_beam_lambda_sweep_2026-06-08.csv`
- `result/metrics/cycle14_seven_beam_lambda_sweep_extended_2026-06-08.csv`

当前判断：`lambda_phy=0.1` 是当前最合适的 7 光束主实验候选。下一步进入探测器噪声鲁棒性实验。

### Cycle 15

已完成 7 光束探测器噪声鲁棒性实验。

核心输出：

- `simulation/static/generate_seven_beam_noise_robustness_dataset.py`
- `train/evaluate_seven_beam_noise_robustness.py`
- `result/logs/cycle15_seven_beam_noise_robustness_2026-06-08.md`

当前判断：当前物理约束模型在干净数据上略有优势，但对探测器噪声更敏感。后续如要突出鲁棒性，应考虑噪声增强训练或去噪物理一致性目标。

### Cycle 16

已完成 7 光束振幅失配与位置偏移鲁棒性实验。

核心输出：

- `simulation/static/generate_seven_beam_complex_robustness_dataset.py`
- `train/evaluate_seven_beam_complex_robustness.py`
- `result/logs/cycle16_seven_beam_complex_robustness_2026-06-08.md`

当前判断：物理约束对振幅失配和位置偏移这类光束状态扰动有小幅增益，但对探测器噪声不稳定。后续论文应区分“成像噪声”和“光束物理状态扰动”两类鲁棒性。

### Cycle 17

已完成 7 光束主瓣能量占比与相位补偿效果评估。

核心输出：

- `train/evaluate_seven_beam_compensation_metrics.py`
- `result/logs/cycle17_seven_beam_main_lobe_2026-06-09.md`
- `result/metrics/cycle17_seven_beam_main_lobe_summary_2026-06-09.csv`

当前判断：普通 CNN 和物理约束 CNN 都能显著提升主瓣能量占比，物理约束 CNN 略优，但距离理想相干仍有明显空间。

### Cycle 18

已完成 7 光束 Strehl 比评估。

核心输出：

- `train/evaluate_seven_beam_strehl.py`
- `result/logs/cycle18_seven_beam_strehl_2026-06-09.md`
- `result/metrics/cycle18_seven_beam_strehl_summary_2026-06-09.csv`

当前判断：普通 CNN 和物理约束 CNN 都能显著提高 Strehl 比；物理约束 CNN 的 Strehl 比和残余相位 RMSE 略优于普通 CNN。

### Cycle 19

已完成 7 光束相位补偿综合效果实验。

核心输出：

- `train/evaluate_seven_beam_compensation_effect.py`
- `result/logs/cycle19_seven_beam_compensation_effect_2026-06-09.md`
- `result/metrics/cycle19_seven_beam_compensation_effect_summary_2026-06-09.csv`
- `result/figures/cycle19_seven_beam_compensation_effect_2026-06-09.png`

当前判断：普通 CNN 与物理约束 CNN 都能把 7 光束远场能量重新推向主瓣区域。物理约束 CNN 在主瓣能量占比、Strehl 比、合成效率和残余相位 RMSE 上均略优于普通 CNN，可作为论文中“物理约束提升补偿结果物理可信度”的直接支撑。

### Cycle 20

已完成双光束/7 光束系统规模对比。

核心输出：

- `train/compare_system_scale.py`
- `result/logs/cycle20_system_scale_comparison_2026-06-09.md`
- `result/metrics/cycle20_system_scale_comparison_2026-06-09.csv`
- `result/metrics/cycle20_system_scale_ratio_2026-06-09.csv`
- `result/figures/cycle20_system_scale_comparison_2026-06-09.png`

当前判断：双光束任务能验证代码链路和物理损失实现，但任务维度过低，不能充分体现多路 CBC 的通道耦合问题。7 光束系统虽然当前相位 RMSE 明显更高，但更能体现论文研究价值。下一周期应进入网络结构消融，重点降低 7 光束主系统的相位 RMSE。

### Cycle 21

已完成 7 光束网络结构快速消融。

核心输出：

- `train/models.py` 中新增 `WidePhaseCNN` 和 `ResidualPhaseCNN`
- `train/sweep_seven_beam_architecture.py`
- `result/logs/cycle21_seven_beam_architecture_ablation_2026-06-09.md`
- `result/metrics/cycle21_seven_beam_architecture_ablation_2026-06-09.csv`
- `result/figures/cycle21_seven_beam_architecture_ablation_2026-06-09.png`

当前判断：小样本快速筛选中，`residual_cnn` 表现最好，但训练样本数和轮数都很小，只能说明残差结构值得继续验证。下一步应在完整 7 光束数据集上对 `residual_cnn` 做更长训练，并与 `simple_cnn` 30 epoch baseline 进行公平对比。

### Cycle 22

已完成 RTX 3060 长轮次训练准备。

核心输出：

- `GPU_TRAINING_3060.md`
- `scripts/run_cycle22_gpu_residual.ps1`
- `train/sweep_seven_beam_architecture.py` 的 GPU 长训练参数增强
- `result/logs/cycle22_gpu_training_preparation_2026-06-09.md`
- `result/metrics/cycle22_gpu_smoke_2026-06-09.csv`

当前判断：项目已具备在 RTX 3060 电脑上进行 `residual_cnn` 完整数据 50/80 epoch 长训练的入口。长训练完成后，需要把结果 CSV、训练曲线图和本地模型权重带回当前项目，再继续做补偿效果评估和泛化实验。

补充判断：RTX 3060 的 50 epoch 复跑已完成，但最终 epoch 指标不理想。下一步仍然需要 3060，但训练目标应从“继续堆轮数”改为“保存最佳验证 checkpoint + 固定 baseline 随机种子做公平长训练”。当前不建议直接用 `residual_cnn` 替代论文主模型。

最新补充：已合入 `cycle23_residual_best_50epoch` 结果。最佳验证 checkpoint 出现在 epoch 17，测试 RMSE 为 `0.992071 rad`，测试 MAE 为 `0.812456 rad`，首次优于当前普通 CNN baseline 和物理约束 CNN。最终 epoch 测试 RMSE 仍为 `1.269384 rad`，说明后续必须采用最佳验证 checkpoint 或早停策略，而不能使用最终 epoch。下一步应评估 `residual_cnn_best` 的主瓣能量占比、Strehl 比、合成效率和补偿后残余相位 RMSE。

下一阶段建议已更新：可以尝试 `residual_cnn + physics loss`。当前 `residual_cnn_best` 只有残差结构，没有物理约束；`physics_cnn_lambda_0.1` 有物理约束，但没有残差结构。因此新实验应训练 `ResidualPhaseCNN + FarFieldConsistencyLoss`，重点比较 `best_checkpoint_test_rmse_rad`、远场 MSE、主瓣能量占比、Strehl 比和合成效率。该实验需要 RTX 3060。

Cycle 25 补充：`residual_cnn + physics loss` 已完成一轮 GPU 结果回收。当前最佳设置为 `lambda_phy=0.05`，最佳 checkpoint 测试 RMSE 为 `0.983128 rad`，相比 `residual_cnn_best` 的 `0.992071 rad` 有小幅提升，但与 Xie et al. 2024 报道的约 `0.076π ≈ 0.239 rad` 仍有较大差距。

Cycle 26 计划已调整为“文献启发的自研模型创新”。保留 Xie et al. 的周期相位损失思想，新增 `--phase-loss cyclic`；模型结构不照搬 MobileNetV3-Small，而采用项目自研 `cbc_lite_cnn`，包含深度可分离残差块、空间/通道门控和多尺度池化相位回归头。下一步在 RTX 3060 上运行 `scripts/run_cycle26_gpu_cbc_lite.ps1`，验证 `cbc_lite_cnn + cyclic phase loss` 是否优于当前残差与物理约束路线。

Cycle 26 GPU 结果已回收：`cbc_lite_cnn` 分别使用 `mse`、`cyclic`、`cyclic_unit` 训练 50 epoch。最佳结果来自 `mse`，最佳 checkpoint 测试 RMSE 为 `1.219643 rad`；`cyclic` 为 `1.281704 rad`，`cyclic_unit` 为 `1.255836 rad`。三者均弱于当前 `residual_cnn_best` 和 `residual_cnn + physics loss`，因此 `cbc_lite_cnn` 暂不升级为论文主模型。下一步应在当前最优 `residual_cnn + physics loss, lambda_phy=0.05` 路线上测试周期相位损失，而不是继续更换网络结构。

## 当前阶段性判断

当前代码已经形成了较完整的双光束研究闭环：

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

下一阶段的核心变化是：将上述闭环迁移到 7 光束系统。双光束结果将作为“方法可行性证明”，7 光束结果作为论文主实验。
