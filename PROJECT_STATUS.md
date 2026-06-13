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
| Cycle 32 | 六边形对称增强与通道均衡 | 利用 7 光束阵列旋转/镜像对称性做标签感知增强 | 判断是否降低通道不平衡并提升补偿指标 |
| Cycle 33 | 补偿质量损失调度与单位圆约束 | 用 warmup 方式稳定加入 Strehl/主瓣能量损失，并约束 sin/cos 单位圆 | 判断是否在保持 RMSE 的同时提升 Strehl、主瓣能量和合成效率 |
| Cycle 34 | 补偿损失 warmup 与单位圆约束稳定性扫描 | 固定主模型，扫描 `lambda_unit` 与 `comp_warmup_epochs` | 得到当前主训练损失的推荐参数 |
| Cycle 35 | 焦平面/焦前 attribution 解释性分析 | 比较单焦平面与焦前/多平面输入的 saliency/attribution map | 判断焦前图像是否提供更局部、更可分的相位线索 |
| Cycle 36 | 多平面 warmup/unit 融合验证 | 将单平面 warmup5 + unit0.01 迁移到 7cm 多平面训练 | 判断补偿调度是否能与多平面收益叠加 |
| Cycle 37 | 多平面 lambda_comp 扫描 | 固定 7cm 多平面输入，扫描补偿质量损失权重 | 区分补偿质量最优与相位精度最优模型 |
| Cycle 38 | 双主模型证据链整理 | 汇总当前补偿质量主模型与相位精度主模型 | 形成可引用总表和后续选择策略入口 |
| Cycle 39 | checkpoint 选择策略验证 | 同次训练保存 best-RMSE 与 best-comp checkpoint | 判断选择策略能否缓解 RMSE 与补偿质量分歧 |
| Cycle 40 | 显式指标 checkpoint 选择工具验证 | 保存 best-Strehl 与 best-main-lobe checkpoint | 验证训练内显式补偿指标是否可作为选择依据 |
| Cycle 41 | 未归一化 Strehl 指标修复 | 实现与最终评估一致的 torch 远场/Strehl 验证 | 让训练期 checkpoint 选择真正对齐下游补偿质量 |
| Cycle 42 | 焦平面/焦前双分支特征融合 | 用焦平面与焦前图像分别编码再融合 | 判断更聪明的多平面融合是否优于简单通道堆叠 |
| Cycle 43 | 双分支解释性与鲁棒性补强 | 对 Cycle42 做 attribution 与噪声鲁棒性验证 | 判断双分支正结果是否具有物理解释和稳定性 |

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

Cycle 31 结论：受Xie et al. 2024启发验证多平面输入策略。Smoke测试（1k数据）显示双平面RMSE比单平面降低 **15.7%**。但完整实验（10k数据）显示多平面改善仅 **0.7-0.8%**。关键发现：多平面收益与数据规模负相关，在当前10k + 11.3M参数配置下，单焦平面已包含足够信息。焦前距离3cm/5cm/7cm差异仅1.4%，推荐5cm对标文献。**阶段性判断**：多平面作为补充实验(supplementary)，证明当前配置已接近单焦平面表示上限，不作为主线创新。项目暂不进入论文收束阶段，下一阶段继续围绕模型二次改进推进（Cycle 32-34）。

Cycle 32 已完成。关键输出：

```text
result/logs/cycle32_hex_symmetry_augmentation_2026-06-11.md
result/metrics/cycle32_deep_hex_aug_30epoch_history.csv
result/metrics/cycle32_hex_compensation_summary.csv
result/figures/cycle32_hex_compensation_comparison.png
models/cycle32_deep_hex_aug_30epoch.pth
```

Cycle 32 结论：七光束六边形对称增强没有超过 Cycle 30 主模型。`cycle32_hex_aug` 测试 RMSE 为 `0.967701 rad`；统一 256 样本补偿评估中，主瓣能量占比 `0.510868`、Strehl 比 `0.610301`、合成效率 `0.772100`，均略低于 `cycle30_deep_final`。该 Cycle 作为负结果保留：在当前 10k 数据 + 11.3M 深度残差网络配置下，显式几何对称增强没有直接转化为补偿质量收益。

Cycle 33 已完成。关键输出：

```text
result/logs/cycle33_comp_warmup_unit_2026-06-11.md
result/metrics/cycle33_deep_comp_warmup_unit_30epoch_history.csv
result/metrics/cycle33_warmup_unit_compensation_summary.csv
result/figures/cycle33_warmup_unit_compensation_comparison.png
models/cycle33_deep_comp_warmup_unit_30epoch.pth
```

Cycle 33 结论：补偿质量 warmup + 单位圆约束取得小幅正结果。`cycle33_warmup_unit` 测试 RMSE 为 `0.953182 rad`；统一补偿评估中，主瓣能量占比 `0.514104`、合成效率 `0.777395`、残余相位 RMSE `0.909709 rad`，均略优于 Cycle 30 的 `0.513536`、`0.776529`、`0.913416 rad`；Strehl 比 `0.623762` 与 Cycle 30 的 `0.624081` 基本持平。

Cycle 34 已完成：单位圆约束权重扫描 + warmup 节奏扫描。关键输出：

```text
result/logs/cycle34_unit_weight_scan_2026-06-11.md
result/logs/cycle34_unit_weight_scan_stage1_2026-06-11.md
result/logs/cycle34_warmup_scan_2026-06-12.md
result/metrics/cycle34_unit_weight_scan_summary.csv
result/metrics/cycle34_warmup_scan_summary.csv
result/figures/cycle34_unit_weight_scan_comparison.png
result/figures/cycle34_warmup_scan_comparison.png
models/cycle34_unit_0p003_warmup10_30epoch.pth
models/cycle34_unit_0p03_warmup10_30epoch.pth
models/cycle34_warmup5_unit0p01_30epoch.pth
models/cycle34_warmup15_unit0p01_30epoch.pth
```

Cycle 34 结论：扫描 `lambda_unit=0.003/0.01/0.03` 后，`lambda_unit=0.01` 是当前最佳折中点。固定 `lambda_unit=0.01` 后继续扫描 `comp_warmup_epochs=5/10/15`，其中 `warmup5` 在统一 256 样本补偿评估中取得最高主瓣能量占比 `0.514117`、最高 Strehl 比 `0.626688` 和最高合成效率 `0.777425`；`warmup10` 的残余相位 RMSE 最低，为 `0.909709 rad`；`warmup15` 没有带来进一步收益。下一阶段默认推荐参数为 `lambda_comp=0.5, comp_warmup_epochs=5, lambda_unit=0.01, augment_mode=noise`。

Cycle 35 已完成第一批 attribution 工具验证、小样本分析，以及 7cm 多平面正式训练与 paired 补偿评估。关键输出：

```text
train/analyze_phase_attribution.py
result/logs/cycle35_attribution_initial_2026-06-12.md
result/logs/cycle35_multiplane_7cm_attribution_2026-06-12.md
models/cycle35_multiplane_7cm_10k_30epoch.pth
result/metrics/cycle35_multiplane_7cm_10k_30epoch_history.csv
result/metrics/cycle35_attribution_cycle30.csv
result/metrics/cycle35_attribution_warmup5.csv
result/metrics/cycle35_attribution_multiplane_smoke.csv
result/metrics/cycle35_attribution_overview.csv
result/metrics/cycle35_attribution_warmup5_64.csv
result/metrics/cycle35_attribution_multiplane_7cm_64.csv
result/metrics/cycle35_attribution_overview_64.csv
result/metrics/cycle35_multiplane_7cm_paired_compensation_summary.csv
result/figures/cycle35_attribution_cycle30/
result/figures/cycle35_attribution_warmup5/
result/figures/cycle35_attribution_multiplane_smoke/
result/figures/cycle35_attribution_warmup5_64/
result/figures/cycle35_attribution_multiplane_7cm_64/
result/figures/cycle35_multiplane_7cm_paired_compensation.png
```

Cycle 35 结论：7cm 多平面正式模型测试 RMSE 为 `0.940678 rad`，优于当前单平面 warmup5 的 `0.949785 rad`。64 样本 attribution 显示，多平面模型对焦平面/焦前平面的梯度能量约为 `51.3% / 48.7%`，说明焦前通道确实被使用；但其 saliency 平均半径更大（`22.85 px` vs 单平面 `17.92 px`），top 10% 能量集中度更低（`0.803` vs 单平面 `0.909`），因此收益更可能来自额外传播约束或冗余观测，而不是更局部的相位线索。paired 补偿评估中，`multiplane_7cm` 的主瓣能量占比 `0.524718`、Strehl 比 `0.658185`、合成效率 `0.794436`、残余相位 RMSE `0.882901 rad`，均优于 `cycle30_deep_final` 和 `warmup5`。

Cycle 36 已完成：多平面 7cm + warmup5/unit0.01 融合验证。关键输出：

```text
result/logs/cycle36_multiplane_warmup_unit_2026-06-12.md
models/cycle36_multiplane_7cm_warmup5_unit0p01_30epoch.pth
result/metrics/cycle36_multiplane_7cm_warmup5_unit0p01_30epoch_history.csv
result/metrics/cycle36_multiplane_warmup_paired_compensation_summary.csv
result/figures/cycle36_multiplane_warmup_paired_compensation.png
```

Cycle 36 结论：`multiplane_7cm_warmup5_unit` 测试 RMSE 为 `0.939758 rad`，比原始 `multiplane_7cm` 的 `0.940678 rad` 略好；但 paired 补偿指标下降，主瓣能量占比 `0.522417`、Strehl 比 `0.654960`、合成效率 `0.790882`、残余相位 RMSE `0.885590 rad`，均不如原始 `multiplane_7cm` 的 `0.524718`、`0.658185`、`0.794436`、`0.882901 rad`。因此单平面的 warmup/unit 策略不能直接迁移为多平面默认设置，当前综合最佳仍是 Cycle 35 的 `models/cycle35_multiplane_7cm_10k_30epoch.pth`。

Cycle 37 已完成：固定 7cm 多平面输入，扫描 `lambda_comp=0.3/0.4/0.5`。关键输出：

```text
result/logs/cycle37_multiplane_lambda_comp_scan_2026-06-12.md
models/cycle37_multiplane_7cm_lambda_comp0p3_30epoch.pth
models/cycle37_multiplane_7cm_lambda_comp0p4_30epoch.pth
result/metrics/cycle37_lambda_comp_scan_summary.csv
result/figures/cycle37_lambda_comp_scan.png
```

Cycle 37 结论：`lambda_comp=0.3` 测试 RMSE 最低，为 `0.931945 rad`，补偿后残余相位 RMSE 也最低，为 `0.865573 rad`；`lambda_comp=0.5` 的主瓣能量占比 `0.524718`、Strehl 比 `0.658185`、合成效率 `0.794436` 仍为最高；`lambda_comp=0.4` 没有成为理想折中点。后续应保留两个代表模型：补偿质量主模型使用 `models/cycle35_multiplane_7cm_10k_30epoch.pth`，相位精度/残余 RMSE 主模型使用 `models/cycle37_multiplane_7cm_lambda_comp0p3_30epoch.pth`。

Cycle 38 已完成：双主模型证据链整理，并实现多平面训练的双 checkpoint 选择入口。关键输出：

```text
result/logs/cycle38_dual_model_evidence_chain_2026-06-12.md
result/metrics/cycle38_dual_model_evidence_summary.csv
result/figures/cycle38_dual_model_evidence_summary.png
models/cycle38_checkpoint_selection_smoke3_rmse.pth
models/cycle38_checkpoint_selection_smoke3_comp.pth
result/metrics/cycle38_checkpoint_selection_smoke3_history.csv
```

Cycle 38 结论：当前核心矛盾不是模型完全无效，而是 checkpoint 选择指标不同会偏向不同目标。已在 `train/train_multiplane.py` 中增加双 checkpoint 保存：`--model-path` 按 `val_rmse_rad` 最低保存，`--comp-model-path` 按验证集 `comp_loss` 最低保存。3 epoch smoke 已验证保存逻辑正常。下一次正式实验应复跑 30 epoch，并比较 RMSE 选择与补偿选择两种 checkpoint 的 paired 补偿表现。

Cycle 39 已完成：7cm 多平面 `lambda_comp=0.5` 正式复跑，同时保存 best-RMSE 与 best-comp checkpoint。关键输出：

```text
result/logs/cycle39_checkpoint_selection_2026-06-12.md
models/cycle39_multiplane_7cm_comp0p5_best_rmse_30epoch.pth
models/cycle39_multiplane_7cm_comp0p5_best_comp_30epoch.pth
result/metrics/cycle39_checkpoint_selection_paired_summary.csv
result/figures/cycle39_checkpoint_selection_paired.png
```

Cycle 39 结论：`cycle39_best_comp` 相比 `cycle39_best_rmse` 主瓣能量和合成效率略高，但 Strehl 与残余 RMSE 略低；两个 Cycle39 checkpoint 的残余相位 RMSE 都优于 Cycle35 的 `comp0p5_cycle35`，但 Cycle35 仍保持最高主瓣能量、最高 Strehl 和最高合成效率。checkpoint 选择策略有价值，但不足以单独解决“相位精度 vs 补偿能量”的权衡。当前仍保留双主模型：补偿质量主模型 `models/cycle35_multiplane_7cm_10k_30epoch.pth`，相位/残余 RMSE 主模型 `models/cycle37_multiplane_7cm_lambda_comp0p3_30epoch.pth`。

Cycle 40 已完成工具验证：`train_multiplane.py` 已支持保存 best-Strehl 与 best-main-lobe checkpoint，并通过 1 epoch smoke 生成四类 checkpoint。关键输出：

```text
result/logs/cycle40_metric_specific_checkpoint_selection_2026-06-12.md
models/cycle40_metric_selection_smoke1_rmse.pth
models/cycle40_metric_selection_smoke1_comp.pth
models/cycle40_metric_selection_smoke1_strehl.pth
models/cycle40_metric_selection_smoke1_main_lobe.pth
result/metrics/cycle40_metric_selection_smoke1_history.csv
```

Cycle 40 结论：训练内 `SevenBeamFourierOptics.reconstruct_from_phase()` 会按峰值归一化远场，因此训练内 `val_strehl_ratio` 会退化接近 `1.0`，不能等价于最终评估脚本中基于未归一化远场的真实 Strehl。当前不应使用训练内 best-Strehl 做正式 checkpoint 选择；best-main-lobe 仍有参考价值，但不能替代真实 Strehl。下一步若继续模型改进，应先实现与最终评估一致的未归一化 torch 远场/Strehl 验证函数，再启动正式 best-Strehl checkpoint 实验。

2026-06-12 路线修订：结合 Hou 2019、Mills 2022、Xie 2024 以及 Cycle35-40 的结果，后续优化方向从“更大的模型/更多网格搜索”转向“更正确的物理指标 + 更聪明的焦前/焦平面信息融合”。其中 Cycle 41 先修复未归一化 Strehl 和主瓣指标，使训练期 checkpoint 选择与最终补偿评估一致；Cycle 42 再在此基础上设计焦平面/焦前双分支融合模型，避免把多平面输入仅作为普通通道堆叠。

Cycle 41 已完成：修复未归一化 Strehl / 主瓣指标，并完成 7cm 多平面正式 30 epoch 训练与 paired 评估。关键输出：

```text
result/logs/cycle41_unnormalized_strehl_checkpoint_2026-06-12.md
train/plot_cycle41_literature_figure.py
models/cycle41_multiplane_7cm_unorm_best_rmse_30epoch.pth
models/cycle41_multiplane_7cm_unorm_best_comp_30epoch.pth
models/cycle41_multiplane_7cm_unorm_best_strehl_30epoch.pth
models/cycle41_multiplane_7cm_unorm_best_main_lobe_30epoch.pth
result/metrics/cycle41_multiplane_7cm_unorm_30epoch_history.csv
result/metrics/cycle41_unnormalized_strehl_paired_summary.csv
result/figures/cycle41_unnormalized_strehl_paired.png
result/figures/cycle41_literature_style_evidence.png
```

Cycle 41 结论：未归一化 torch 远场/Strehl 验证函数已与最终评估脚本对齐，训练期 `val_strehl_ratio` 不再退化为接近 `1.0`。正式 paired 评估中，`cycle41_best_strehl` 的主瓣能量占比为 `0.524967`、Strehl 为 `0.670898`、合成效率为 `0.795033`，均略优于 Cycle35 的 `0.524718`、`0.658185`、`0.794436`；但残余相位 RMSE 上升到 `0.896828 rad`，不如 Cycle37 comp0.3 的 `0.865573 rad`。因此补偿质量主模型更新为 `models/cycle41_multiplane_7cm_unorm_best_strehl_30epoch.pth`，相位/残余 RMSE 主模型仍为 `models/cycle37_multiplane_7cm_lambda_comp0p3_30epoch.pth`。已新增仿文献综合证据图 `result/figures/cycle41_literature_style_evidence.png`，按方法修正、训练轨迹、下游指标和典型远场图样解释 Cycle41 正结果。

Cycle 42 已完成：实现焦平面/焦前双分支门控融合模型，并完成 7cm 双平面正式 30 epoch 训练、paired 富指标评估和仿文献综合证据图。关键输出：

```text
result/logs/cycle42_dual_plane_fusion_2026-06-13.md
train/plot_cycle42_literature_figure.py
models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth
models/cycle42_dual_plane_fusion_7cm_best_comp_30epoch.pth
models/cycle42_dual_plane_fusion_7cm_best_strehl_30epoch.pth
models/cycle42_dual_plane_fusion_7cm_best_main_lobe_30epoch.pth
result/metrics/cycle42_dual_plane_fusion_7cm_30epoch_history.csv
result/metrics/cycle42_dual_plane_fusion_paired_summary.csv
result/metrics/cycle42_dual_plane_fusion_paired_detail.csv
result/figures/cycle42_dual_plane_fusion_paired.png
result/figures/cycle42_literature_style_fusion_evidence.png
```

Cycle 42 结论：`dual_plane_fusion_cnn` 参数量为 `5.77M`，小于 Cycle41 简单双通道 `deep_residual_cnn` 的 `11.34M`。正式 paired 评估中，`cycle42_best_rmse` 的主瓣能量占比为 `0.525304`、Strehl 为 `0.682690`、合成效率为 `0.795854`、残余相位 RMSE 为 `0.892309 rad`，相较 Cycle41 的 `0.524967`、`0.670898`、`0.795033`、`0.896828 rad` 同时改善。该结果说明焦平面/焦前显式分支融合优于简单通道堆叠，当前补偿质量主模型更新为 `models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth`；相位/残余 RMSE 主模型仍为 `models/cycle37_multiplane_7cm_lambda_comp0p3_30epoch.pth`。

Cycle 43 已完成：双分支解释性与噪声鲁棒性补强。关键输出：

```text
result/logs/cycle43_dual_plane_attribution_noise_2026-06-13.md
result/metrics/cycle43_attribution_cycle41_64.csv
result/metrics/cycle43_attribution_cycle42_64.csv
result/metrics/cycle43_attribution_overview_64.csv
result/metrics/cycle43_dual_plane_noise_robustness_summary.csv
result/figures/cycle43_attribution_overview_64.png
result/figures/cycle43_dual_plane_noise_robustness.png
```

Cycle 43 结论：

1. **Attribution 验证**：64 样本 × 6 通道 attribution 分析显示，Cycle42 双分支模型的焦平面/焦前能量占比为 `48.4% / 51.6%`，标准差约 `0.314`，说明模型在不同样本和通道间动态分配跨平面特征贡献，而非固定偏向某一平面。
2. **噪声鲁棒性验证**：256 样本噪声扫描（σ=0, 0.002, 0.005, 0.01, 0.02, 0.03）显示，Cycle42 在 σ≥0.005 时全面优于 Cycle41。以 σ=0.02 为例，Cycle42 的 Strehl 为 `0.481`（vs Cycle41 `0.407`）、合成效率为 `0.659`（vs `0.554`）、残余 RMSE 为 `1.364 rad`（vs `1.718 rad`）。
3. **阶段判断**：Cycle42 可确认为补偿质量主模型。其干净输入下补偿质量优于 Cycle41，并在中高噪声下保持更强稳定性。Attribution 支持"双分支确实自适应使用两个输入平面"，但未支持"焦前分支绝对主导"的强断言。
4. **技术验证阶段完成**：Cycle 43 是技术验证闭环的最后一环，后续应进入论文收束与投稿准备。

最新主线判断：

- **当前双主模型最终配置**：补偿质量主模型为 `models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth`（Strehl 0.683, 合成效率 0.796）；相位/残余 RMSE 主模型为 `models/cycle37_multiplane_7cm_lambda_comp0p3_30epoch.pth`（残余 RMSE 0.866 rad）。
- **相位 RMSE 与补偿质量权衡是论文特色**：更低 RMSE 不总能带来更高 Strehl 和合成效率，项目通过双主模型和显式补偿质量优化解决该矛盾。
- **焦平面/焦前显式融合成立**：Cycle42 以更小参数量（5.77M）超过 Cycle41（11.34M），收益来自更合理的信息融合而非模型体量。
- **技术验证阶段完成**：Cycle 1-43 完成从双光束到七光束、从简单 CNN 到双分支融合、从相位监督到补偿质量优化、从干净数据到噪声鲁棒性的完整验证链。
- **下一阶段：论文收束与投稿准备**：整理主图主表、撰写 Method 和 Results 章节、补充 Related Work 对标、内部审阅后投稿。
- **关键负结果已记录**：周期损失、轻量网络 cbc_lite、六边形对称增强未带来收益，作为消融分析保留。

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
