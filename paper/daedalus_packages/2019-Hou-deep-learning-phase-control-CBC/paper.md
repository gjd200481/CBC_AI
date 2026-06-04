---
title: "Deep-learning-based phase control method for tiled aperture coherent beam combining systems"
title_zh: "面向拼接孔径相干光束合成系统的深度学习相位控制方法"
authors:
  - "Tianyue Hou"
  - "Yi An"
  - "Qi Chang"
  - "Pengfei Ma"
  - "Jun Li"
  - "Dong Zhi"
  - "Liangjin Huang"
  - "Rongtao Su"
  - "Jian Wu"
  - "Yanxing Ma"
  - "Pu Zhou"
year: "2019"
venue: "High Power Laser Science and Engineering 7:e59"
doi: "10.1017/hpl.2019.46"
tags:
  - literature
  - method/deep-learning
  - optics/control
  - optics/coherent-beam-combining
  - task/phase-control
methods:
  - deep-learning
  - optical-control
material_system: "拼接孔径相干光束合成，CNN 相位控制"
task_type: "用深度学习估计相位误差并驱动 CBC 伺服补偿"
reading_status: "processed"
source_pdf: "source/2019-Hou-deep-learning-phase-control-CBC.pdf"
created: "2026-06-04"
---

# Deep-learning-based phase control method for tiled aperture coherent beam combining systems

## 0. 文献元信息

### 0.1 基础信息

| 项目 | 内容 |
|---|---|
| 英文题名 | Deep-learning-based phase control method for tiled aperture coherent beam combining systems |
| 中文题名 | 面向拼接孔径相干光束合成系统的深度学习相位控制方法 |
| 作者 | Tianyue Hou, Yi An, Qi Chang, Pengfei Ma, Jun Li, Dong Zhi, Liangjin Huang, Rongtao Su, Jian Wu, Yanxing Ma, Pu Zhou |
| 年份/来源 | 2019，High Power Laser Science and Engineering 7:e59 |
| DOI | 10.1017/hpl.2019.46 |
| 任务类型 | 用深度学习估计相位误差并驱动 CBC 伺服补偿 |
| 研究对象 | 拼接孔径相干光束合成，CNN 相位控制 |

### 0.2 Abstract 中英文对照

> 合规短摘：We incorporate deep learning (DL) into tiled aperture coherent beam combining (CBC) systems for the ﬁrst time, to the best of our knowledge. By using a well-trained convolutional n

**中文专业转述：**  
这篇论文围绕 拼接孔径相干光束合成，CNN 相位控制 中的核心控制难题展开。本文首次将深度学习引入拼接孔径相干光束合成相位控制，用非焦平面强度图训练 CNN 估计各子光束相位误差，并把估计结果直接用于伺服相位补偿。 方法上，作者构建 VGG 风格卷积神经网络，从非焦平面强度分布回归相对相位；同时比较焦平面和非焦平面输入，指出焦平面存在数据碰撞问题；再用 7 元和 19 元六角阵列仿真验证可扩展性。 结果层面，非焦平面 CNN 训练能稳定收敛，补偿后远场主瓣、Strehl ratio 和 power-in-bucket 接近理想相干合成；阵元数量增加时，CNN 推理复杂度没有像传统相位控制链路那样显著增加。

### 0.3 Conclusion 中英文对照

> 合规短摘：In this paper, we have shown that the DL-based phase control method could be implemented into CBC systems to directly compensate the phase error. Comprehensively considering simula

**中文专业转述：**  
作者最终强调：非焦平面 CNN 训练能稳定收敛，补偿后远场主瓣、Strehl ratio 和 power-in-bucket 接近理想相干合成；阵元数量增加时，CNN 推理复杂度没有像传统相位控制链路那样显著增加。 同时，论文也留下了明确边界：研究以数值仿真为主，真实光纤放大器噪声、执行器带宽、相机链路、热漂移和硬件闭环仍需要实验验证。

## 1. 快速总结表

| 模块 | 内容 |
|---|---|
| 文章信息 | 2019；High Power Laser Science and Engineering 7:e59；Tianyue Hou 等 |
| 研究背景 | 光学控制系统中，相位或波前状态随时间变化，而传感、计算和执行存在延迟或不可直接观测的问题。 |
| 研究目的 | 用深度学习估计相位误差并驱动 CBC 伺服补偿 |
| 核心方法 | 作者构建 VGG 风格卷积神经网络，从非焦平面强度分布回归相对相位；同时比较焦平面和非焦平面输入，指出焦平面存在数据碰撞问题；再用 7 元和 19 元六角阵列仿真验证可扩展性。 |
| 关键结果 | 非焦平面 CNN 训练能稳定收敛，补偿后远场主瓣、Strehl ratio 和 power-in-bucket 接近理想相干合成；阵元数量增加时，CNN 推理复杂度没有像传统相位控制链路那样显著增加。 |
| 主要结论 | 本文首次将深度学习引入拼接孔径相干光束合成相位控制，用非焦平面强度图训练 CNN 估计各子光束相位误差，并把估计结果直接用于伺服相位补偿。 |
| 文章好在哪里 | 它把深度学习放进具体物理控制链路，而不是只做通用图像识别；同时用指标、流程图和鲁棒性实验支持论证。 |
| 是否值得深读 | 值得，尤其适合做 CBC/AO 中“AI 预测 + 实时控制 + 物理约束”方向的文献基础。 |

## 2. 125 笔记整理表

| 类型 | 内容 | 对我研究/写作的用法 |
|---|---|---|
| 1 个思路 | 用神经网络从可观测图像或斜率序列中预测不可直接及时获得的控制变量。 | 可作为 CBC 相位控制、AO 延迟补偿、光场闭环控制的共同范式。 |
| 2 个图表 1 | 系统/数据流图展示从传感到模型再到控制的链路。 | 写方法章节时先画闭环，再讲模型。 |
| 2 个图表 2 | 性能对比图或表格展示预测方法相对非预测/传统方法的改善。 | 写结果章节时用“基线-模型-理想状态”三层对照。 |
| 5 个句式 1 | The observable intensity/slope pattern is treated as a proxy for the hidden control state. | 用于引出代理观测反演。 |
| 5 个句式 2 | The model reduces latency-induced error by predicting the future state before actuation. | 用于 AO 或实时控制论文。 |
| 5 个句式 3 | Performance must be evaluated at the downstream optical-control level, not only at prediction-error level. | 强调任务指标。 |
| 5 个句式 4 | Robustness is tested by varying noise, turbulence, delay, or array scale. | 写鲁棒性实验设计。 |
| 5 个句式 5 | The method is promising, but deployment depends on hardware timing and distribution shift. | 写局限与展望。 |

## 3. 图表证据链

### Figure 1. Figure 1. Experimental setup for implementing the DL-based phase control method for CBC. (SL: seed laser; PA: pre-ampliﬁer; FS: ﬁber splitter; FPM: ﬁber phase modulator; FA: ﬁber ampliﬁer; HRM: highly reﬂective mirror; FL: focus lens; BS: beam splitter.) be expressed as

![Figure 1](figures/Figure-1.png)

| 维度 | 解析 |
|---|---|
| 目的 | 展示本文证据链中的 Figure 1。批量抽取时保存为页面级截图，优先保证上下文不丢失。 |
| 描述 | Figure 1. Experimental setup for implementing the DL-based phase control method for CBC. (SL: seed laser; PA: pre-ampliﬁer; FS: ﬁber splitter; FPM: ﬁber phase modulator; FA: ﬁber ampliﬁer; HRM: highly reﬂective mirror; FL: focus lens; BS: beam splitter.) be expressed as |
| 结论 | 该图/表服务于论文主线：本文首次将深度学习引入拼接孔径相干光束合成相位控制，用非焦平面强度图训练 CNN 估计各子光束相位误差，并把估计结果直接用于伺服相位补偿。 |
| 支撑 | 它把方法、数据或结果中的一个环节可视化，帮助判断模型是否真正改善了相位或波前控制。 |
| 可复用性 | 可作为未来写作中“系统流程、模型结构、性能对比或鲁棒性验证”的图表组织参考。 |
### Figure 2. Figure 2. Illustration of the CNN for estimating the phase error in CBC systems. Softmax function after the last fully connected (FC) layer of the original VGG model is replaced by the Sigmoid function. The reason for this substitution is that estimating

![Figure 2](figures/Figure-2.png)

| 维度 | 解析 |
|---|---|
| 目的 | 展示本文证据链中的 Figure 2。批量抽取时保存为页面级截图，优先保证上下文不丢失。 |
| 描述 | Figure 2. Illustration of the CNN for estimating the phase error in CBC systems. Softmax function after the last fully connected (FC) layer of the original VGG model is replaced by the Sigmoid function. The reason for this substitution is that estimating |
| 结论 | 该图/表服务于论文主线：本文首次将深度学习引入拼接孔径相干光束合成相位控制，用非焦平面强度图训练 CNN 估计各子光束相位误差，并把估计结果直接用于伺服相位补偿。 |
| 支撑 | 它把方法、数据或结果中的一个环节可视化，帮助判断模型是否真正改善了相位或波前控制。 |
| 可复用性 | 可作为未来写作中“系统流程、模型结构、性能对比或鲁棒性验证”的图表组织参考。 |
### Figure 3. Figure 3. Intensity proﬁles of the beam arrays consisting of (a) 7 elements and (b) 19 elements. by multiplying 2π. Based on the relative phases estimated by the CNN, the phase error in the CBC system could be compensated directly.

![Figure 3](figures/Figure-3.png)

| 维度 | 解析 |
|---|---|
| 目的 | 展示本文证据链中的 Figure 3。批量抽取时保存为页面级截图，优先保证上下文不丢失。 |
| 描述 | Figure 3. Intensity proﬁles of the beam arrays consisting of (a) 7 elements and (b) 19 elements. by multiplying 2π. Based on the relative phases estimated by the CNN, the phase error in the CBC system could be compensated directly. |
| 结论 | 该图/表服务于论文主线：本文首次将深度学习引入拼接孔径相干光束合成相位控制，用非焦平面强度图训练 CNN 估计各子光束相位误差，并把估计结果直接用于伺服相位补偿。 |
| 支撑 | 它把方法、数据或结果中的一个环节可视化，帮助判断模型是否真正改善了相位或波前控制。 |
| 可复用性 | 可作为未来写作中“系统流程、模型结构、性能对比或鲁棒性验证”的图表组织参考。 |
### Figure 4. Figure 4. Average MSE of the CNN as a function of the number of training epochs. ﬁnal epoch, showing the high prediction accuracy of the convergent CNN. In contrast, the MSE of the focal plane case ﬂuctuates around 0.082 in the training process, indicating the

![Figure 4](figures/Figure-4.png)

| 维度 | 解析 |
|---|---|
| 目的 | 展示本文证据链中的 Figure 4。批量抽取时保存为页面级截图，优先保证上下文不丢失。 |
| 描述 | Figure 4. Average MSE of the CNN as a function of the number of training epochs. ﬁnal epoch, showing the high prediction accuracy of the convergent CNN. In contrast, the MSE of the focal plane case ﬂuctuates around 0.082 in the training process, indicating the |
| 结论 | 该图/表服务于论文主线：本文首次将深度学习引入拼接孔径相干光束合成相位控制，用非焦平面强度图训练 CNN 估计各子光束相位误差，并把估计结果直接用于伺服相位补偿。 |
| 支撑 | 它把方法、数据或结果中的一个环节可视化，帮助判断模型是否真正改善了相位或波前控制。 |
| 可复用性 | 可作为未来写作中“系统流程、模型结构、性能对比或鲁棒性验证”的图表组织参考。 |
### Figure 5. Figure 5. Performances of the trained CNN for phase control. Far-ﬁeld intensity proﬁles (a1)–(a5) without phase error compensation, and with phase error compensation using CNNs trained at (b1)–(b5) the focal plane and (c1)–(c5) the non-focal-plane. https://doi.org/10.1017/hpl.2019.46 Published online by Cambridge University Press

![Figure 5](figures/Figure-5.png)

| 维度 | 解析 |
|---|---|
| 目的 | 展示本文证据链中的 Figure 5。批量抽取时保存为页面级截图，优先保证上下文不丢失。 |
| 描述 | Figure 5. Performances of the trained CNN for phase control. Far-ﬁeld intensity proﬁles (a1)–(a5) without phase error compensation, and with phase error compensation using CNNs trained at (b1)–(b5) the focal plane and (c1)–(c5) the non-focal-plane. https://doi.org/10.1017/hpl.2019.46 Published online by Cambridge University Press |
| 结论 | 该图/表服务于论文主线：本文首次将深度学习引入拼接孔径相干光束合成相位控制，用非焦平面强度图训练 CNN 估计各子光束相位误差，并把估计结果直接用于伺服相位补偿。 |
| 支撑 | 它把方法、数据或结果中的一个环节可视化，帮助判断模型是否真正改善了相位或波前控制。 |
| 可复用性 | 可作为未来写作中“系统流程、模型结构、性能对比或鲁棒性验证”的图表组织参考。 |
### Figure 6. Figure 6. Far-ﬁeld intensity proﬁles of the (a) incoherently combined beam, (b) DL-based coherently combined beam and (c) ideal coherently combined beam, for the case of the 7-element hexagonal array. (d) Far-ﬁeld intensity proﬁles along the x axis for the ideal coherently combined beam (red), DL-based coherently combined beam (green) and incoherently combined beam (blue). (e) Power in the bucket (PIB) at the focal plane as a function of the bucket radius for the ideal coherently combined beam (

![Figure 6](figures/Figure-6.png)

| 维度 | 解析 |
|---|---|
| 目的 | 展示本文证据链中的 Figure 6。批量抽取时保存为页面级截图，优先保证上下文不丢失。 |
| 描述 | Figure 6. Far-ﬁeld intensity proﬁles of the (a) incoherently combined beam, (b) DL-based coherently combined beam and (c) ideal coherently combined beam, for the case of the 7-element hexagonal array. (d) Far-ﬁeld intensity proﬁles along the x axis for the ideal coherently combined beam (red), DL-based coherently combined beam (green) and incoherently combined beam (blue). (e) Power in the bucket (PIB) at the focal plane as a function of the bucket radius for the ideal coherently combined beam ( |
| 结论 | 该图/表服务于论文主线：本文首次将深度学习引入拼接孔径相干光束合成相位控制，用非焦平面强度图训练 CNN 估计各子光束相位误差，并把估计结果直接用于伺服相位补偿。 |
| 支撑 | 它把方法、数据或结果中的一个环节可视化，帮助判断模型是否真正改善了相位或波前控制。 |
| 可复用性 | 可作为未来写作中“系统流程、模型结构、性能对比或鲁棒性验证”的图表组织参考。 |
### Figure 7. Figure 7. Far-ﬁeld intensity proﬁles of the (a) incoherently combined beam, (b) DL-based coherently combined beam and (c) ideal coherently combined beam, for the case of the 19-element hexagonal array. (d) Far-ﬁeld intensity proﬁles along the x axis for the ideal coherently combined beam (red), DL-based coherently combined beam (green) and incoherently combined beam (blue). (e) Power in the bucket (PIB) at the focal plane as a function of the bucket radius for the ideal coherently combined beam 

![Figure 7](figures/Figure-7.png)

| 维度 | 解析 |
|---|---|
| 目的 | 展示本文证据链中的 Figure 7。批量抽取时保存为页面级截图，优先保证上下文不丢失。 |
| 描述 | Figure 7. Far-ﬁeld intensity proﬁles of the (a) incoherently combined beam, (b) DL-based coherently combined beam and (c) ideal coherently combined beam, for the case of the 19-element hexagonal array. (d) Far-ﬁeld intensity proﬁles along the x axis for the ideal coherently combined beam (red), DL-based coherently combined beam (green) and incoherently combined beam (blue). (e) Power in the bucket (PIB) at the focal plane as a function of the bucket radius for the ideal coherently combined beam  |
| 结论 | 该图/表服务于论文主线：本文首次将深度学习引入拼接孔径相干光束合成相位控制，用非焦平面强度图训练 CNN 估计各子光束相位误差，并把估计结果直接用于伺服相位补偿。 |
| 支撑 | 它把方法、数据或结果中的一个环节可视化，帮助判断模型是否真正改善了相位或波前控制。 |
| 可复用性 | 可作为未来写作中“系统流程、模型结构、性能对比或鲁棒性验证”的图表组织参考。 |


## 4. 方法与图表复用

这篇论文最值得复用的是“物理系统问题定义 → 可观测代理信号 → 神经网络预测 → 控制补偿 → 下游光学指标验证”的写作路径。对于 CBC 论文，重点看相位误差如何被预测并转化为补偿；对于 AO 论文，重点看延迟波前如何被预测并转化为残差下降。

```mermaid
flowchart LR
  A["物理系统状态"] --> B["相机图像或 WFS slope"]
  B --> C["深度学习预测器"]
  C --> D["相位/波前补偿量"]
  D --> E["光学系统输出"]
  E --> F["远场质量或残余误差评价"]
```

## 5. 中英陪读

| 关键词 | 中文解释 |
|---|---|
| coherent beam combining | 相干光束合成，通过控制多路光束相位使其叠加成高质量输出。 |
| adaptive optics | 自适应光学，通过实时测量和校正波前畸变改善成像或传输质量。 |
| wavefront prediction | 波前预测，用历史观测估计未来波前以补偿控制延迟。 |
| phase control | 相位控制，使不同通道保持目标相位关系。 |
| open-loop correction | 开环校正，控制量不直接依赖校正后的即时反馈，因而更依赖预测准确性。 |

## 6. Daedalus 深度剖析

### Act I: Opening Gambit - 核心价值命题

本文首次将深度学习引入拼接孔径相干光束合成相位控制，用非焦平面强度图训练 CNN 估计各子光束相位误差，并把估计结果直接用于伺服相位补偿。 这类论文的战略意义在于，它把光学系统中的“慢反馈”和“不可观测隐变量”问题，转化为机器学习可以处理的预测或反演问题。

### Act II: Narrative Unfolding - 从真实工程到科学核心

真实工程痛点是：光学系统并不会等控制器慢慢计算。CBC 中相位噪声会持续漂移；AO 中大气湍流会在传感、计算和变形镜响应之间继续演化。因此，系统需要提前知道或快速反推出下一步该如何补偿。

### Act III: Technical Heart - 方法核心

作者构建 VGG 风格卷积神经网络，从非焦平面强度分布回归相对相位；同时比较焦平面和非焦平面输入，指出焦平面存在数据碰撞问题；再用 7 元和 19 元六角阵列仿真验证可扩展性。 技术上最重要的是输入输出定义：输入不是抽象向量，而是带有物理含义的强度图、波前斜率或时间序列；输出也不是普通分类标签，而是相位、波前或补偿相关的连续控制量。

### Act IV: Chain Of Evidence - 结果证据链

本文图表从系统结构、模型结构、训练/测试设置、性能比较和鲁棒性分析几个层面组织证据。`## 3` 中列出的每个图表都对应证据链的一环：先证明系统和数据可信，再证明模型有效，最后证明方法在噪声、延迟、规模或目标变化下仍有边界内的稳定性。

### Act V: Intellectual Distillation - 页外洞察

关键洞察是：深度学习在这里不是替代物理，而是学习物理系统中难以显式建模或难以及时测量的映射。它的成功依赖训练分布覆盖、传感链路稳定、输出变量定义合理，以及评价指标真正对应光学任务。

### Act VI: Legacy And Outlook - 迁移与未来方向

未来可以沿三条线推进：第一，把模型部署到更真实的硬件闭环中，测量端到端延迟；第二，用物理约束或仿真-实验混合数据提升泛化；第三，面向更大阵列、更强湍流或更复杂目标光场做规模化验证。

## 7. 写作素材库

| 用途 | 可复用表达 |
|---|---|
| 引出问题 | 光学控制的关键瓶颈不是缺少测量，而是测量、推理和执行之间存在时间差与隐变量。 |
| 方法表述 | 将可观测光场图样或 WFS slope 序列作为隐藏相位/波前状态的代理信号。 |
| 结果表述 | 方法的价值应通过最终光学质量指标验证，而不仅是网络预测误差。 |
| 局限表述 | 部署性能取决于硬件延迟、训练分布、噪声模型和跨场景泛化能力。 |

## 8. 局限与复核清单

| 项目 | 状态 |
|---|---|
| PDF 文本抽取 | PyMuPDF 抽取成功，共 7 页。 |
| 图表抽取 | 已保存 7 张 Figure 页面截图、0 张 Table 页面截图。 |
| 抽取精度 | 批量模式采用页面级截图，可靠保留上下文，但不是所有图表都做了精确裁剪。 |
| Abstract/Conclusion | 因版权合规限制，仅放短摘和中文专业转述。 |
| 需人工复核 | 建议打开 Typora 检查页面截图是否需要后续手工裁剪成更精确的图。 |

## 本论文的通用知识迁移总结

- **核心思想**：把光学系统中的不可见或未来状态转化为可学习的预测/反演任务。
- **可复用模块**：代理观测、深度学习预测器、闭环补偿、鲁棒性扫描、下游光学指标验证。
- **关键避坑指南**：不要只报告模型误差；一定要说明硬件时延、噪声、训练分布和物理可达性边界。
