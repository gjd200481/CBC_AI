# CBC_AI 投稿目标与当前进度

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

项目目标已调整为形成一篇具备一区或二区期刊投稿潜力的研究论文。后续推进不再受固定日期约束；项目恢复 `Cycle` 管理方式，但 `Cycle` 只作为任务分割和实验批次记录，不绑定天数或硬性截止时间。

## 当前研究定位

此前曾考虑使用 `CNN + LSTM` 做远场序列预测和未来相位预测，但根据当前投稿目标，主线已经调整为单帧远场光强到相位误差的反演。随后项目目标进一步从双光束升级为 7 光束多路相干合成。`CNN + LSTM` 动态预测暂时保留为后续拓展，不作为当前论文主任务。

说明：历史记录中的 `Cycle XX` 继续作为实验批次编号和结果索引保留；新的 `Cycle` 表示一个任务包，其完成标准是形成可复现实验、论文图表、指标表或明确负结果。

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

## 无时间约束 Cycle 任务规划

后续恢复 Cycle 管理，但不再把 Cycle 理解为固定天数或截止日期。每个 Cycle 是一个独立任务包，完成后应沉淀日志、指标、图表或论文段落。当前规划如下：

| Cycle | 任务主题 | 核心目标 | 完成标准 |
| --- | --- | --- | --- |
| Cycle 27 | 主模型补偿指标补齐 | 评估 `residual_cnn + physics loss, lambda_phy=0.05` 的主瓣能量、Strehl 比、合成效率、峰值旁瓣比和残余相位 RMSE | 能判断当前最优 RMSE 是否同步带来补偿质量提升 |
| Cycle 28 | 残差物理约束路线的周期损失验证 | 在 `ResidualPhaseCNN + FarFieldConsistencyLoss` 上公平比较 `mse`、`cyclic`、`cyclic_unit` | 决定周期损失进入主模型还是作为负结果消融 |
| Cycle 29 | 七光束数据规模扩展 | 比较 `1024 -> 5000 -> 10000` 样本对 RMSE 和补偿指标的影响 | 得到数据规模是否是主要瓶颈的定量结论 |
| Cycle 30 | 焦前/离焦图像路线 | 生成并训练焦平面、焦前、离焦图像数据 | 判断离焦输入是否成为论文主线创新点 |
| Cycle 31 | 噪声增强与稳健训练 | 解决物理约束模型对探测器噪声不稳定的问题 | 输出统一鲁棒性曲线和可发表结论 |
| Cycle 32 | 论文主图与表格定稿 | 整理 RMSE、补偿指标、鲁棒性、负结果消融图表 | 论文初稿可直接引用主图和主表 |
| Cycle 33 | 论文初稿升级 | 将中文阶段性稿升级为接近期刊格式的论文稿 | 形成可继续翻译、排版或套模板的投稿稿 |
| Cycle 34 | 投稿期刊筛选 | 筛选一区/二区候选期刊并反向检查补实验 | 得到目标期刊列表和补实验清单 |

Cycle 27 已完成。关键输出：

```text
result/logs/cycle27_residual_physics_compensation_2026-06-11.md
result/metrics/cycle27_residual_physics_compensation_summary_2026-06-11.csv
result/metrics/cycle27_compensation_comparison_summary_2026-06-11.csv
result/figures/cycle27_compensation_comparison_2026-06-11.png
```

Cycle 27 结论：`residual_cnn_best` 在 256 样本补偿指标上最优，主瓣能量占比为 `0.523614`，Strehl 比为 `0.663759`，合成效率为 `0.793090`，残余相位 RMSE 为 `0.862535 rad`。`residual_cnn + physics, lambda_phy=0.05` 的补偿指标未超过 `residual_cnn_best`，其主瓣能量占比为 `0.517471`，Strehl 比为 `0.653397`，合成效率为 `0.783312`，残余相位 RMSE 为 `0.880499 rad`。因此后续主模型选择需要同时看相位 RMSE 和补偿物理指标，不能只按 Cycle 25 的测试集 RMSE 排序。

Cycle 28 已完成。关键输出：

```text
result/logs/cycle28_data_scale_10k_2026-06-11.md
result/metrics/cycle28_residual_physics_10k_80epoch_history.csv
result/metrics/cycle28_residual_physics_10k_80epoch_summary.csv
result/metrics/cycle28_10k_compensation_summary.csv
result/metrics/cycle28_1k_vs_10k_summary.csv
result/figures/cycle28_1k_vs_10k_comparison.png
models/cycle28_residual_physics_10k_80epoch_best.pth
```

Cycle 28 结论：将数据规模从1024扩展至10000样本，`residual_cnn + physics loss, lambda_phy=0.05` 最佳checkpoint测试RMSE降至 `0.936 rad`，相比1k数据降低4.8%。但补偿物理指标（主瓣能量0.514，Strehl比0.640，合成效率0.777）略低于1k模型。**关键发现**：数据规模扩展有效但收益有限，相位RMSE与补偿质量矛盾仍未解决。训练过程出现明显过拟合（最优点在epoch 6，最终epoch 80验证RMSE反弹至1.157）。下一步应优先实施补偿质量损失函数重构，而非继续扩大数据。

Cycle 31 已完成。关键输出：

```text
result/logs/cycle31_multiplane_ablation_2026-06-12.md
result/metrics/cycle31_multiplane_ablation_summary_2026-06-12.csv
dataset/seven_beam/multiplane_0_-0.03/
dataset/seven_beam/multiplane_0_-0.05/
dataset/seven_beam/multiplane_0_-0.07/
```

Cycle 31 结论：受Xie et al. 2024启发验证多平面输入策略。Smoke测试（1k数据）显示双平面RMSE比单平面降低 **15.7%**。但完整实验（10k数据）显示多平面改善仅 **0.7-0.8%**。关键发现：多平面收益与数据规模负相关，在当前10k + 11.3M参数配置下，单焦平面已包含足够信息。焦前距离3cm/5cm/7cm差异仅1.4%，推荐5cm对标文献。**阶段性判断**：多平面作为补充实验(supplementary)，证明当前配置已接近单焦平面表示上限，不作为主线创新。项目进入论文写作阶段（Cycle 32-34）。

最新主线判断：

- **Cycle 28 数据规模突破**：10k样本训练的 `residual_cnn + physics loss, lambda_phy=0.05` 最佳 checkpoint 测试 RMSE 为 `0.936 rad`，相比1k数据的 `0.983 rad` 降低 **4.8%**。
- 当前最优相位 RMSE 来自 `cycle28_residual_10k`，测试 RMSE `0.936 rad`。
- **相位RMSE与补偿质量矛盾持续存在**：10k模型相位RMSE更低，但补偿后Strehl比(0.640)、合成效率(0.777)略低于1k模型的Strehl比(0.653)、合成效率(0.783)。
- 数据规模扩展有效但收益有限，说明单纯增加数据量不是唯一解决方案。
- **下一步优先级**：实施补偿质量损失函数重构，直接优化Strehl比和主瓣能量，而非继续盲目扩大数据。
- `cbc_lite_cnn` 与周期损失组合未超过残差物理约束路线，应作为负结果消融保留。
- `cycleXX` 恢复作为任务分割方式，但不包含时间约束；历史与后续 Cycle 均应服务于一区/二区论文证据链。

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
