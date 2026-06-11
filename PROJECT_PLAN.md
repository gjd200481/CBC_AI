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

### Cycle 32：论文主图与表格定稿

研究目的：把已有实验转化为正式论文图表证据链。

主要任务：

- 统一图表风格、坐标轴、图例和单位。
- 汇总模型 RMSE、MAE、远场 MSE、主瓣能量、Strehl 比、合成效率等指标。
- 制作论文主图：
  1. 七光束阵列与仿真流程。
  2. 物理约束训练框架。
  3. 模型性能对比。
  4. 补偿前后远场图样。
  5. 鲁棒性曲线。
  6. 负结果消融。

产出文件：

- `paper/figures/`
- `paper/tables/`
- `result/logs/cycle32_paper_figures_*.md`

完成标准：论文初稿可以直接引用主图和主表。

### Cycle 33：论文初稿升级为投稿稿

研究目的：把当前中文阶段性初稿升级为接近期刊投稿格式的论文稿。

主要任务：

- 重写摘要、引言、方法、实验、讨论和结论。
- 增加英文标题、英文摘要和关键词。
- 加强与 Hou、Mills、Xie 等文献的差异化讨论。
- 将负结果写成合理消融，而不是简单失败记录。

产出文件：

- `paper/CBC_AI_manuscript_draft_*.md`
- `paper/references.bib` 或参考文献清单

完成标准：形成可以继续翻译、排版或投给目标期刊模板的论文稿。

### Cycle 34：投稿目标期刊筛选与补实验清单

研究目的：根据目标期刊要求反向检查论文缺口。

主要任务：

- 筛选 3 到 5 个一区/二区候选期刊。
- 比较栏目范围、图表要求、创新性要求和数据可用性要求。
- 根据目标期刊审稿标准整理必须补做的实验。

产出文件：

- `paper/journal_target_list_*.md`
- `paper/revision_checklist_*.md`

完成标准：确定优先投稿方向和下一轮补实验清单。

## 论文主线建议

当前最适合投稿的论文叙事不是“提出一个全新网络并大幅超过所有文献”，而是：

> 面向七光束 CBC 相位误差反演，建立一套包含周期相位编码、傅里叶物理一致性约束、最佳 checkpoint 选择和下游补偿物理指标评价的可复现实验框架；实验表明残差特征提取与傅里叶物理约束组合能够稳定优于普通 CNN，并在远场补偿质量上体现收益，同时揭示周期损失和轻量网络并非直接迁移即可有效。

该叙事的优势是可信、完整、能体现物理建模特色。要达到一区或二区投稿标准，还需要补强：

- 当前最优模型的补偿物理指标。
- 更大规模数据和离焦图像对比。
- 与已有深度学习 CBC 文献的公平讨论。
- 更充分的消融实验。
- 更规范的图表和论文写作。

## 文件管理说明

历史文件名中的 `cycleXX` 继续作为实验批次编号和结果索引保留。后续恢复使用 Cycle 管理任务，但每个 Cycle 只代表一个任务包，不绑定天数、日期或硬性截止时间；项目推进以论文质量和关键证据链是否完整为准。

## 下一步建议

1. 先执行 Cycle 27，补齐 `residual_cnn + physics loss, lambda_phy=0.05` 的补偿物理指标。
2. 再执行 Cycle 28，在当前最优残差物理约束路线中测试周期相位损失。
3. Cycle 29 与 Cycle 30 分别处理数据规模和离焦图像两个可能显著降低 RMSE 的方向。
4. Cycle 31 到 Cycle 34 面向鲁棒性、论文图表、投稿稿和目标期刊筛选。
