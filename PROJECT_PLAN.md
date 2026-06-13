# CBC_AI 投稿目标导向研究路线图

## 项目总目标

本项目的目标不再按固定日期或截止日期推进，而是以形成一篇具备一区或二区期刊投稿潜力的研究论文为导向，围绕多路相干光束合成（Coherent Beam Combining, CBC）中的相位误差反演问题，建立一套可复现、可解释、具有物理约束特征的深度学习方法。

项目恢复 `Cycle` 管理方式，但 `Cycle` 只表示任务分割单元，不再绑定天数、日期或硬性截止时间。每个 Cycle 以是否产出可复现实验、论文图表、指标表或明确负结果为完成标准。

当前论文主线为：

```text
七光束相干合成远场光强图像
-> CNN / Residual CNN 相位反演
-> 六路相对相位 sin/cos 周期编码
-> 傅里叶光学远场重建
-> 相位监督损失 + 远场物理一致性损失
-> 相位 RMSE 与补偿物理指标联合评价
```

研究对象以七光束六边形阵列为主：中心 1 路光束作为参考相位，外圈 6 路光束按六边形排布，网络从远场光强图像中反演 6 个相对相位误差。双光束实验保留为方法链路验证、代码基线和低维对照，不再作为论文主贡献。

## 投稿定位

目标定位为具备一区或二区投稿潜力的光学、激光、光电子或物理信息交叉方向论文。项目应优先满足以下要求：

- **问题有明确工程价值**：面向多路 CBC 相位误差快速估计，服务于高功率光纤激光相干合成。
- **方法有物理特色**：不是只比较通用 CNN，而是将傅里叶传播模型、相位周期性和下游补偿指标纳入方法体系。
- **实验链路完整**：包含数据生成、相位反演、物理约束、鲁棒性、补偿效果和消融分析。
- **结论可信且克制**：如实记录有效结果和负结果，不把未提升的模型包装成创新成果。
- **可复现性强**：保留训练脚本、评估脚本、随机种子、关键 CSV、结果图和论文草稿。

## 当前核心科学问题

1. 单帧远场强度图像中是否包含足够信息反演七光束外圈 6 路相对相位？
2. `sin/cos` 周期编码和周期误差统计能否稳定处理相位边界问题？
3. 傅里叶远场一致性损失是否能提升相位预测的物理可信度？
4. 残差结构、最佳验证 checkpoint 和物理约束组合是否优于普通 CNN baseline？
5. 相位 RMSE 的改善是否能转化为主瓣能量占比、Strehl 比、合成效率和残余相位 RMSE 的改善？
6. 物理约束模型在探测器噪声、振幅失配和位置偏移等非理想条件下是否具备更好泛化？
7. 与 Hou、Mills、Xie 等深度学习 CBC 文献相比，本项目的差距、优势和可发表创新点在哪里？

## 当前技术路线

### 1. 七光束相干合成仿真

- 构建中心参考光束 + 外圈 6 路光束的六边形阵列。
- 用高斯光束近似每一路近场复振幅。
- 使用 FFT 从近场复振幅生成远场强度图。
- 裁剪并归一化远场图像，作为网络输入。

### 2. 相位标签与评价指标

- 标签格式：

```text
[sin(phi_1), cos(phi_1), ..., sin(phi_6), cos(phi_6)]
```

- 相位误差使用周期 RMSE：

```text
error = atan2(sin(phi_pred - phi_true), cos(phi_pred - phi_true))
RMSE = sqrt(mean(error^2))
```

### 3. 物理约束训练

网络输出相位后，重新构建七光束近场复振幅，并通过 FFT 得到预测远场强度。总损失为：

```text
L_total = L_phase + lambda_phy * L_farfield
```

其中：

- `L_phase`：相位监督损失。
- `L_farfield`：预测相位重建远场与输入远场之间的 MSE。
- `lambda_phy`：物理约束权重。

### 4. 模型路线

当前主要候选路线：

- `SimplePhaseCNN`：普通 CNN baseline。
- `ResidualPhaseCNN`：残差结构候选。
- `ResidualPhaseCNN + FarFieldConsistencyLoss`：当前最优主线。

已验证但暂不作为主线：

- `CBCPhaseLiteCNN`：自研轻量网络，当前 50 epoch 结果未超过残差物理约束路线。
- `cyclic` / `cyclic_unit` 损失：在 `cbc_lite_cnn` 上未带来提升，后续应在当前最优残差物理约束路线中重新验证。

## 已完成关键结果

### 双光束低维验证

- 普通 CNN：测试 RMSE `0.003742 rad`。
- 物理约束 CNN：测试 RMSE `0.004291 rad`，远场一致性误差降低。
- 结论：双光束任务可验证代码链路，但相位维度过低，不足以支撑主论文贡献。

### 七光束 baseline 与物理约束

| 模型 | 物理约束 | RMSE(rad) | MAE(rad) | far-field MSE |
| --- | --- | ---: | ---: | ---: |
| 普通 CNN | 否 | 1.026976 | 0.819061 | `1.1935e-4` |
| 物理约束 CNN | 是，`lambda_phy=0.1` | 1.022686 | 0.816424 | `1.1501e-4` |

结论：首版物理约束带来小幅相位收益，并降低远场重建误差。

### 七光束补偿效果

| 状态 | 主瓣能量占比 | Strehl 比 | 合成效率 | 残余相位 RMSE(rad) |
| --- | ---: | ---: | ---: | ---: |
| 补偿前 | 0.359388 | 0.390687 | 0.532856 | 1.774909 |
| 普通 CNN 补偿后 | 0.519307 | 0.647172 | 0.786023 | 0.905907 |
| 物理约束 CNN 补偿后 | 0.521546 | 0.653564 | 0.789644 | 0.894276 |
| 理想相干 | 0.650631 | 1.000000 | 1.000000 | 0.000000 |

结论：模型预测相位能显著改善远场补偿效果，物理约束 CNN 相比普通 CNN 有小幅优势。

### 残差结构与最佳 checkpoint

| 模型 | 策略 | 测试 RMSE(rad) |
| --- | --- | ---: |
| `residual_cnn` final epoch | 最终 epoch | 1.269384 |
| `residual_cnn` best checkpoint | 最佳验证 checkpoint | 0.992071 |
| `residual_cnn + physics loss` | `lambda_phy=0.05`，最佳 checkpoint | 0.983128 |

结论：当前最优路线是 `ResidualPhaseCNN + FarFieldConsistencyLoss + best validation checkpoint`。

### 自研轻量网络与周期损失负结果

| 模型 | 损失 | 最佳 checkpoint 测试 RMSE(rad) |
| --- | --- | ---: |
| `cbc_lite_cnn` | `mse` | 1.219643 |
| `cbc_lite_cnn` | `cyclic` | 1.281704 |
| `cbc_lite_cnn` | `cyclic_unit` | 1.255836 |

结论：当前 `cbc_lite_cnn` 不作为论文主模型；周期损失需要在残差物理约束主线上重新验证。

## 无时间约束 Cycle 任务规划

从下一阶段开始，项目恢复 Cycle 管理，但 Cycle 只用于拆分任务和记录实验批次。每个 Cycle 都应包含：

- 研究目的：本 Cycle 要回答的论文问题。
- 主要任务：需要修改的代码、运行的实验或整理的文档。
- 产出文件：日志、CSV、图、模型或论文段落。
- 完成标准：能否支持论文中的一个明确结论。
- 后续判断：继续推进、调整路线或作为负结果保留。

### Cycle 27：补齐当前主模型补偿指标

状态：已完成。结果见 `result/logs/cycle27_residual_physics_compensation_2026-06-11.md`。

研究目的：验证 `residual_cnn + physics loss, lambda_phy=0.05` 的相位 RMSE 改善是否能转化为下游远场补偿质量提升。

主要任务：

- 找到当前最优 `residual_cnn + physics loss, lambda_phy=0.05` 的 best checkpoint 和对应配置。
- 复用或扩展 `train/evaluate_seven_beam_compensation_effect.py`。
- 计算主瓣能量占比、Strehl 比、合成效率、峰值旁瓣比和补偿后残余相位 RMSE。
- 与普通 CNN、首版物理约束 CNN、`residual_cnn_best` 做统一表格对比。

产出文件：

- `result/logs/cycle27_residual_physics_compensation_*.md`
- `result/metrics/cycle27_residual_physics_compensation_*.csv`
- `result/figures/cycle27_residual_physics_compensation_*.png`

完成标准：论文中可以明确说明“当前最优相位 RMSE 模型是否同步改善补偿物理指标”。

阶段判断：`residual_cnn_best` 在 256 样本补偿指标上优于 `residual_cnn + physics, lambda_phy=0.05`。后者虽然在 Cycle 25 的测试集相位 RMSE 上更低，但本次没有转化为更好的主瓣能量、Strehl 比或合成效率。因此后续主模型选择不能只看相位 RMSE，还必须联合补偿物理指标。

### Cycle 28：在最优主线上验证周期相位损失

研究目的：判断 Xie et al. 的周期相位损失思想是否能在本项目的残差物理约束路线中带来收益。

主要任务：

- 将 `--phase-loss cyclic` 和 `--phase-loss cyclic_unit` 接入 `ResidualPhaseCNN + FarFieldConsistencyLoss` 训练流程。
- 固定数据集、随机种子、epoch、batch size 和 `lambda_phy=0.05`。
- 与当前 `mse + physics` 结果进行公平对比。
- 记录最佳 checkpoint 与最终 epoch 的差异。

产出文件：

- `result/logs/cycle28_residual_physics_cyclic_*.md`
- `result/metrics/cycle28_residual_physics_cyclic_*.csv`
- `result/figures/cycle28_residual_physics_cyclic_*.png`

完成标准：明确判断 `cyclic` 是否进入论文主模型；若不提升，则作为高价值负结果写入消融分析。

### Cycle 29：七光束数据规模扩展实验

研究目的：判断当前 RMSE 偏高是否主要受限于 1024 样本数据规模。

主要任务：

- 生成或复查 `5000`、`10000` 样本七光束数据集。
- 使用同一训练脚本和同一模型路线比较不同数据规模。
- 记录训练耗时、验证 RMSE、测试 RMSE、远场 MSE 和补偿指标。

产出文件：

- `dataset/seven_beam/main_clean_5000/`
- `dataset/seven_beam/main_clean_10000/`
- `result/logs/cycle29_dataset_scale_*.md`
- `result/metrics/cycle29_dataset_scale_*.csv`

完成标准：得到“扩大数据是否显著改善七光束相位反演”的定量结论。

### Cycle 30：焦前/离焦图像数据路线

研究目的：验证非焦平面强度图是否比当前焦平面远场图包含更稳定的相位信息。

主要任务：

- 在七光束仿真中加入焦前/离焦传播距离参数。
- 生成同一组相位对应的焦平面、焦前和离焦图像。
- 训练同一网络结构，比较不同输入平面的 RMSE 和补偿指标。
- 与 Hou、Xie 等文献中的“非焦平面图像更适合相位识别”结论对齐讨论。

产出文件：

- `simulation/static/generate_seven_beam_defocus_dataset.py`
- `result/logs/cycle30_defocus_input_*.md`
- `result/metrics/cycle30_defocus_input_*.csv`
- `result/figures/cycle30_defocus_input_*.png`

完成标准：判断离焦输入是否应成为论文主线创新点。

### Cycle 31：噪声增强与稳健训练

研究目的：改善当前物理约束模型对探测器噪声不稳定的问题。

主要任务：

- 设计干净/噪声混合训练集。
- 比较训练时噪声增强、测试时噪声扰动和振幅/位置扰动的泛化表现。
- 尝试稳健远场一致性损失或噪声权重策略。
- 输出统一鲁棒性曲线。

产出文件：

- `result/logs/cycle31_noise_augmented_training_*.md`
- `result/metrics/cycle31_noise_augmented_training_*.csv`
- `result/figures/cycle31_noise_augmented_training_*.png`

完成标准：给出“物理约束 + 噪声增强是否能提高鲁棒性”的可发表结论。

### Cycle 32：六边形对称增强与通道均衡

研究目的：利用 7 光束六边形阵列的旋转/镜像对称性，减少网络对固定通道位置的偶然记忆，改善 6 路外圈相位预测的通道不平衡，并观察这种改进是否能转化为 Strehl 比、主瓣能量和合成效率提升。

主要任务：

- 实现七光束专用的标签感知数据增强：`60°` 倍数旋转、镜像翻转，以及外圈 6 路相位标签的同步循环重排/反向重排。
- 在 `DeepResidualPhaseCNN` 或当前 Cycle 30 主模型训练脚本中接入该增强，保持数据集、随机种子和训练轮次尽量可比。
- 对比无增强、普通图像增强、六边形对称增强三种设置。
- 同时记录相位 RMSE、逐通道 RMSE、通道不平衡、主瓣能量占比、Strehl 比和合成效率。

产出文件：

- `train/hexagonal_augmentation.py` 或等价的数据增强模块。
- `result/logs/cycle32_hex_symmetry_augmentation_*.md`
- `result/metrics/cycle32_hex_symmetry_augmentation_*.csv`
- `result/figures/cycle32_hex_symmetry_augmentation_*.png`

完成标准：判断六边形物理对称增强是否能降低通道不平衡，并给出其对补偿质量指标的定量影响。

### Cycle 33：补偿质量损失调度与单位圆约束

研究目的：解决相位 RMSE 与补偿质量不完全一致的问题，让训练目标更直接地服务于下游远场补偿效果，同时避免直接优化 Strehl/主瓣能量导致训练不稳定。

主要任务：

- 将 `CompensationQualityLoss` 从固定权重改为 warmup/调度策略：前期以相位监督和远场一致性为主，后期逐步加入补偿质量项。
- 增加 `sin^2 + cos^2 = 1` 的单位圆正则，约束网络输出更符合周期相位编码几何。
- 比较 `lambda_comp`、warmup epoch、`lambda_unit` 等关键超参数。
- 与 Cycle 30 主模型和 Cycle 32 最佳设置做统一补偿指标评估。

建议总损失：

```text
L_total = L_phase
        + lambda_phy * L_farfield
        + lambda_comp(t) * L_compensation
        + lambda_unit * L_unit_circle
```

产出文件：

- `train/train_seven_beam_compensation_loss.py` 的稳定版或新训练入口。
- `result/logs/cycle33_comp_loss_schedule_*.md`
- `result/metrics/cycle33_comp_loss_schedule_*.csv`
- `result/figures/cycle33_comp_loss_schedule_*.png`

完成标准：判断补偿质量损失调度和单位圆约束是否能在不显著牺牲相位 RMSE 的前提下提升 Strehl 比、主瓣能量占比和合成效率。

### Cycle 34：补偿损失 warmup 与单位圆约束稳定性扫描

研究目的：围绕 Cycle 33 的小正结果做参数稳定性验证，确定补偿质量损失加入训练的最佳节奏，以及单位圆约束权重的合理范围。

主要任务：

- 固定 `lambda_comp=0.5` 和当前主模型 `DeepResidualPhaseCNN`。
- 先扫描 `lambda_unit=0.003/0.01/0.03`，判断单位圆约束强弱对 RMSE 与补偿指标的影响。
- 在 `lambda_unit=0.01` 最佳折中点上，继续扫描 `comp_warmup_epochs=5/10/15`。
- 与 Cycle 30、Cycle 33 做统一补偿指标评估。

产出文件：

- `result/logs/cycle34_unit_weight_scan_*.md`
- `result/logs/cycle34_warmup_scan_*.md`
- `result/metrics/cycle34_*_summary.csv`
- `result/figures/cycle34_*_comparison.png`

完成标准：给出当前主线训练损失的默认推荐参数，并判断收益是否稳定。

### Cycle 35：焦平面/焦前输入的 attribution 解释性分析

研究目的：借鉴 Hou 和 Xie 对非焦平面图像的解释，比较单焦平面与焦前/多平面模型的 attribution/saliency map，判断焦前图像是否真的提供更局部、更可分的相位线索。

主要任务：

- 对 Cycle 30 单焦平面模型和 Cycle 31 多平面/焦前模型做 saliency 或 gradient attribution。
- 对每个外圈通道分别计算输入像素对该通道相位预测的敏感性。
- 比较 attribution 的局部性、对称性和光束相关区域分布。
- 结合 RMSE 与补偿指标解释为什么多平面在 10k 数据下收益有限。

产出文件：

- `train/analyze_phase_attribution.py`
- `result/logs/cycle35_attribution_analysis_*.md`
- `result/figures/cycle35_attribution_*.png`
- `result/metrics/cycle35_attribution_summary.csv`

完成标准：给出“焦前图像是否提供更强物理可解释线索”的阶段性判断；若解释性强但性能收益小，可作为补充实验和论文讨论依据。

### Cycle 41：未归一化 Strehl checkpoint 选择修复

研究目的：解决 Cycle 40 暴露的训练期 Strehl 指标失真问题，使训练过程中的 checkpoint 选择与最终补偿评估脚本使用的未归一化远场指标保持一致。

主要任务：

- 在 `SevenBeamFourierOptics` 或独立验证工具中增加未归一化远场输出接口，避免 `reconstruct_from_phase()` 的峰值归一化影响 Strehl 计算。
- 在 `train/train_multiplane.py` 中使用未归一化远场计算 `val_strehl_ratio`、`val_main_lobe_ratio` 和合成效率等验证指标。
- 保留归一化远场用于训练期 far-field MSE 的兼容路径，避免破坏既有物理一致性损失。
- 先运行 1-3 epoch smoke，确认 `val_strehl_ratio` 不再退化为接近 `1.0` 的恒定值。
- 将 smoke checkpoint 用最终补偿评估脚本复核，确认训练期指标排序与最终评估方向一致。

产出文件：

- `train/physics_loss.py` 的未归一化远场接口或等价工具函数。
- `train/train_multiplane.py` 的真实指标 checkpoint 选择逻辑。
- `result/logs/cycle41_unnormalized_strehl_checkpoint_*.md`
- `result/metrics/cycle41_unnormalized_strehl_smoke*_history.csv`
- `models/cycle41_*_best_strehl*.pth` 或 smoke 权重。

完成标准：训练日志中的 `val_strehl_ratio` 不再因峰值归一化退化；best-Strehl checkpoint 可被最终评估脚本正常读取，并且训练期选择方向与最终未归一化 Strehl/主瓣指标一致。

### Cycle 42：焦平面/焦前双分支特征融合（已完成）

研究目的：在 Cycle 35 多平面收益和 attribution 结果基础上，避免把焦平面与焦前图像简单堆叠为普通输入通道，转而显式学习两类观测的互补信息。该方向直接对齐 Hou 2019 与 Xie 2024 关于非焦平面/焦前图像包含更多相位线索的文献启发。

主要任务：

- 设计轻量双分支模型：焦平面图像和焦前图像分别进入 encoder，再通过 feature fusion、gating 或注意力权重融合。
- 与当前 `MultiPlanePhaseCNN` 的简单通道堆叠方式公平对比，保持数据集、训练轮次、随机种子和损失配置一致。
- 在验证阶段使用 Cycle 41 修复后的未归一化 Strehl/主瓣指标选择 checkpoint。
- 输出相位 RMSE、逐通道 RMSE、残余相位 RMSE、主瓣能量、Strehl 比、合成效率和 attribution 对比。
- 若双分支模型没有提升，应作为负结果保留，说明当前多平面收益主要来自额外传播约束或冗余观测，而非更强特征融合。

产出文件：

- `train/models.py` 中的双分支多平面模型，或独立模型模块。
- `train/train_multiplane_fusion.py` 或在 `train/train_multiplane.py` 中新增模型选择入口。
- `result/logs/cycle42_multiplane_fusion_*.md`
- `result/metrics/cycle42_multiplane_fusion_summary.csv`
- `result/figures/cycle42_multiplane_fusion_*.png`
- `models/cycle42_*_best_*.pth`

完成标准：已完成。Cycle42 `dual_plane_fusion_cnn` 以 `5.77M` 参数超过 Cycle41 简单双通道 `deep_residual_cnn` 的 `11.34M` 参数模型。paired 评估中，`cycle42_best_rmse` 的主瓣能量占比 `0.525304`、Strehl `0.682690`、合成效率 `0.795854`、残余相位 RMSE `0.892309 rad`，均优于 Cycle41 的 `0.524967`、`0.670898`、`0.795033`、`0.896828 rad`。因此“焦平面/焦前显式融合优于简单通道堆叠”阶段性成立。

### Cycle 43：双分支解释性与鲁棒性补强（已完成 ✓✓✓）

研究目的：在 Cycle42 正结果基础上，验证双分支门控融合是否真的利用了焦前图像的局部相位线索，并判断该收益在噪声扰动下是否稳定。该周期对齐 Xie 2024 的"误差统计 + attribution map + 噪声鲁棒性曲线"证据链。

主要任务：

- 对 Cycle41 简单双通道堆叠模型和 Cycle42 双分支融合模型做同样样本集合的 attribution/saliency 分析。
- 分别统计焦平面分支和焦前分支的梯度能量占比、top 10% 能量集中度和平均半径。
- 对 Cycle42 做探测器噪声或输入强度噪声扫描，输出主瓣能量、Strehl、合成效率和残余相位 RMSE 退化曲线。
- 若 attribution 显示焦前分支贡献更明确，且噪声下指标不明显劣化，则将 Cycle42 固定为论文主模型。

产出文件：

- `result/logs/cycle43_dual_plane_attribution_noise_2026-06-13.md`
- `result/metrics/cycle43_attribution_cycle41_64.csv`
- `result/metrics/cycle43_attribution_cycle42_64.csv`
- `result/metrics/cycle43_attribution_overview_64.csv`
- `result/metrics/cycle43_dual_plane_noise_robustness_summary.csv`
- `result/figures/cycle43_attribution_overview_64.png`
- `result/figures/cycle43_dual_plane_noise_robustness.png`

完成标准：已完成。Attribution 分析显示 Cycle42 双分支模型动态分配跨平面特征（焦平面 48.4%，焦前 51.6%，标准差 0.314），说明自适应使用两个输入平面。噪声鲁棒性在 σ≥0.005 时全面优于 Cycle41（σ=0.02 时 Strehl 0.481 vs 0.407）。**技术验证阶段完成，Cycle42 确认为补偿质量主模型**。

## 论文主线建议

当前最适合投稿的论文叙事不是“提出一个全新网络并大幅超过所有文献”，而是：

> 面向七光束 CBC 相位误差反演，建立一套包含周期相位编码、傅里叶物理一致性约束、最佳 checkpoint 选择和下游补偿物理指标评价的可复现实验框架；实验表明残差特征提取与傅里叶物理约束组合能够稳定优于普通 CNN，并在远场补偿质量上体现收益，同时揭示周期损失和轻量网络并非直接迁移即可有效。

该叙事的优势是可信、完整、能体现物理建模特色。要达到一区或二区投稿标准，还需要补强：

- 当前最优模型的补偿物理指标。
- 更大规模数据和离焦图像对比。
- 与已有深度学习 CBC 文献的公平讨论。
- 更充分的模型改进与消融实验，特别是六边形对称增强、补偿质量损失调度和补偿感知模型结构。
- 更规范的图表和论文写作。

## 文件管理说明

历史文件名中的 `cycleXX` 继续作为实验批次编号和结果索引保留。后续恢复使用 Cycle 管理任务，但每个 Cycle 只代表一个任务包，不绑定天数、日期或硬性截止时间；项目推进以论文质量和关键证据链是否完整为准。

## 下一步建议

1. 下一步执行 Cycle 43：对 Cycle42 与 Cycle41 做 attribution 对比，重点检查焦前分支是否提供更局部、更可分的相位线索。
2. 对 Cycle42 做噪声鲁棒性扫描，输出 Strehl、主瓣能量、合成效率和残余 RMSE 的退化曲线。
3. 后续优化方向继续保持为“正确物理指标 + 聪明信息融合 + 可解释证据”，暂不把更大模型、继续扩大 `lambda_comp` 网格或复跑同配置作为默认方向。
4. 若 Cycle43 支撑 Cycle42，则固定 Cycle42/Cycle37 双主模型进入论文图表和讨论整理。
