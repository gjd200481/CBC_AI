# 2026-06-15 工作总结：多平面数据物理问题发现与修正

---

## 🎯 核心成果

今天完成了项目中**最关键的物理问题修正**，这是在论文投稿前发现的重大问题。

### 发现的问题

当前所有"多平面"数据集的**焦平面和焦前平面完全相同**（差异仅 1e-19 ~ 1e-24），不是真正的多平面观测。

### 根本原因

旧代码对自由传播后的场再做 FFT，由于传播只改变相位、不改变幅值，导致不同平面的强度几乎完全退化。

### 修正方案

实现了正确的**透镜焦平面 + 离焦探测**物理链路：
```python
# 正确实现
field_with_lens = near_field * lens_phase(f)
field_at_detector = propagate(field_with_lens, f+Δz)
intensity = |field_at_detector|²  # 直接取强度，不再 FFT
```

### 验证结果

| 指标 | 修正前 | 修正后 | 改善倍数 |
|------|--------|--------|---------|
| Max diff | 2.17e-19 | **0.520** | **1.9×10¹⁸** |
| Mean diff | ~1e-20 | **0.0008** | **8×10¹⁶** |
| 退化样本 | 10/10 (100%) | 0/10 (0%) | - |

---

## ✅ 已完成工作清单

### 1. 问题验证
- [x] 创建验证脚本确认三个数据集（3cm/5cm/7cm）全部退化
- [x] 可视化对比修正前后的图像差异

### 2. 根本原因分析
- [x] 定位错误代码位置 (`propagation.py:89-96`)
- [x] 与 Hou 2019 / Xie 2024 文献对照物理模型
- [x] 理解为什么"传播后再FFT"会导致退化

### 3. 修正实现
- [x] 创建新模块 `propagation_corrected.py`
- [x] 实现 `lens_focus_multiplane_intensity()` 函数
- [x] 创建新数据生成脚本 `generate_corrected_multiplane.py`
- [x] 保留旧函数但添加 `DeprecationWarning`

### 4. 验证测试
- [x] 生成 32 样本 smoke 数据集
- [x] 验证两通道有显著差异（Max 0.42）
- [x] 生成 10k 完整修正数据集
- [x] 验证 10k 数据两通道差异（Max 0.52）

### 5. 文档记录
- [x] 创建改进路线图 (Priority 1-7)
- [x] 创建详细修正报告
- [x] 创建修正完成总结
- [x] 创建执行摘要
- [x] 更新项目总览文档

### 6. 启动验证训练
- [x] 准备快速验证训练脚本
- [x] 配置训练参数
- [x] 启动 15 epoch 快速验证
- [x] 创建训练状态文档
- 🔄 等待训练完成（~15分钟）

---

## 📊 影响范围评估

### ✅ 不受影响的结论（仍然有效）

这些结论依赖**架构设计和训练策略**，与输入数据的物理含义无关：

1. **Cycle 42**: 双分支融合优于简单通道堆叠（5.77M vs 11.34M）
2. **Cycle 44**: 动态噪声增强在 σ≥0.005 提升鲁棒性 +30%
3. **Cycle 43**: IG+Grad-CAM 验证模型学到物理一致特征
4. **Cycle 32**: 六边形对称增强无效（负结果）
5. **Cycle 26**: 周期损失无效（负结果）

### ⚠️ 需要重新验证的结论（依赖多平面物理）

这些结论直接依赖"焦前图像与焦平面不同"的假设：

1. **Cycle 35**: "焦前图像提供更局部的相位线索" - Attribution 分析需重做
2. **Cycle 31**: "多平面优于单平面 0.7-0.8%" - 需用修正数据重新对比
3. **Cycle 43**: "焦平面能量 48.4%，焦前 51.6%" - 能量分配需重新评估
4. **All Cycles**: RMSE 0.892, Strehl 0.683 上限 - 可能提升或下降

---

## 🎯 三种可能场景

当前训练完成后，结果会落入三种场景之一：

### 场景 A：乐观（真实多平面显著提升）

**预期**: RMSE < 0.85, Strehl > 0.70

**意义**: 
- 真实多平面数据提供更强相位信息
- 论文创新点"多平面融合"完全成立
- 可以强调物理正确建模的重要性

**下一步**:
- 配置扫描（离焦距离 × 光束间距）
- 数据规模扩展到 50k
- 重做 Attribution 验证焦前分支贡献

### 场景 B：现实（改善有限）

**预期**: RMSE 0.85-0.90, Strehl 0.65-0.70

**意义**:
- 多平面收益 < 5%，接近单焦平面上限
- 物理模型更正确，但性能提升有限
- 论文重点转向架构和鲁棒性

**下一步**:
- 混合物理优化（CNN + SPGD 微调）
- 周期约束和单位圆正则强化
- 诚实讨论物理正确性 vs 性能权衡

### 场景 C：保守（性能下降）

**预期**: RMSE > 0.90, Strehl < 0.65

**意义**:
- 伪多平面意外提供了正则效果
- 更正确的物理模型反而降低性能
- 仍是有价值的发现和负结果

**下一步**:
- 优先执行混合物理优化弥补性能
- 增强数据增强和正则化
- 论文讨论"错误模型有时也能工作"

**无论哪种场景，修正都是正确的选择**，因为当前数据与文献描述不符。

---

## 📁 新增文件（16个）

### 核心代码（3）
1. `simulation/common/propagation_corrected.py`
2. `simulation/static/generate_corrected_multiplane.py`
3. `train/quick_verify_corrected.py`

### 验证脚本（2）
4. `scripts/verify_multiplane_data_issue.py`
5. `scripts/visualize_corrected_vs_old.py`

### 数据集（2）
6. `dataset/seven_beam/multiplane_corrected_smoke/`
7. `dataset/seven_beam/multiplane_corrected_f1.0_d0.05/`

### 可视化（1）
8. `result/figures/multiplane_correction_comparison.png`

### 文档（8）
9. `PROJECT_IMPROVEMENT_ROADMAP.md`
10. `docs/MULTIPLANE_CORRECTION_REPORT.md`
11. `docs/CORRECTION_SUMMARY.md`
12. `docs/CORRECTION_EXECUTIVE_SUMMARY.md`
13. `docs/CORRECTED_TRAINING_STATUS.md`
14. 本文档
15. 更新 `PROJECT_OVERVIEW.md`
16. 更新 `PROJECT_STATUS.md` (待完成)

---

## ⏭️ 下一步行动（按优先级）

### Immediate（今天/明天）

1. 🔄 **等待验证训练完成**（~15分钟）
2. **评估结果**，判断场景 A/B/C
3. **更新文档**，记录修正后性能
4. **决定策略**，根据场景选择后续路线

### Short-term（本周）

根据场景执行不同策略：

**若场景 A（乐观）**:
- Priority 2: 配置扫描（离焦距离 × 光束间距）
- Priority 3: 数据规模扩展 50k
- Priority 7: 双分支架构重新验证

**若场景 B（现实）**:
- Priority 5: 混合物理优化（CNN + SPGD）
- Priority 4: 周期约束和单位圆正则强化
- Priority 6: 噪声增强策略优化

**若场景 C（保守）**:
- Priority 5: 混合物理优化（紧急）
- 增强数据增强和正则化
- 重新评估模型架构

### Medium-term（1-2周）

- 完整 30 epoch 训练最终模型
- 重新做完整 Attribution 分析
- 更新论文 Results 和 Discussion 章节
- 准备投稿材料

---

## 🎓 关键经验教训

### 1. 物理验证至关重要

即使代码运行正常、训练收敛、指标提升，也不代表物理模型正确。

**教训**: 对关键物理量必须做数值验证，不能只依赖"代码能跑"。

### 2. 文献对照要细致

不要只看方法概述，要对照实验配置细节、图注说明和参数表。

**发现**: 
- Hou Fig.3 是近场强度，不是间距消融
- Hou Fig.5 是补偿后结果，不是 CNN 输入
- 我们的物理参数与文献完全不同

### 3. 负结果也有价值

六边形增强、周期损失等负结果让我们能区分"架构有效性"和"数据正确性"。

**价值**: 完整的实验记录让问题定位更准确。

### 4. Cycle 管理的有效性

Cycle 1-44 的完整记录让我们能快速评估问题影响范围。

**改进**: 未来在关键物理假设处增加数值验证 checkpoint。

---

## 📊 工作量统计

- **代码行数**: ~600 行（新增+修改）
- **文档字数**: ~8000 字（6篇新文档）
- **数据生成**: 10032 样本（smoke + 10k）
- **验证脚本**: 2 个
- **可视化图**: 1 张
- **实际耗时**: ~3 小时（问题发现到数据生成完成）
- **预计总耗时**: ~4 小时（含验证训练）

---

## 🔗 快速访问

- **执行摘要**: [docs/CORRECTION_EXECUTIVE_SUMMARY.md](CORRECTION_EXECUTIVE_SUMMARY.md)
- **详细报告**: [docs/MULTIPLANE_CORRECTION_REPORT.md](MULTIPLANE_CORRECTION_REPORT.md)
- **改进路线**: [PROJECT_IMPROVEMENT_ROADMAP.md](../PROJECT_IMPROVEMENT_ROADMAP.md)
- **训练状态**: [docs/CORRECTED_TRAINING_STATUS.md](CORRECTED_TRAINING_STATUS.md)

---

**日期**: 2026-06-15  
**当前状态**: 🔄 Priority 1 完成，验证训练进行中  
**下一检查点**: 训练完成（预计 17:30）
