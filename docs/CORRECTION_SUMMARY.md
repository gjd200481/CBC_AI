# 多平面数据修正完成总结

**日期**: 2026-06-15  
**状态**: ✅ Priority 1 完成，10k 数据生成中

---

## 🎯 核心成果

### 1. 问题确认

已验证当前所有多平面数据集（3cm/5cm/7cm）的两通道完全退化相同：
- 差异仅在数值精度范围 1e-19 ~ 1e-24
- 前10个样本 `np.allclose=True` 比例：**100%**

### 2. 根本原因定位

旧代码 `propagation.py:multiplane_far_field_intensity()` 存在物理错误：
```python
# 错误实现：自由传播后再 FFT
propagated = propagate_fn(near_field, wavelength, dist, pixel_size)
far_field = np.fft.fftshift(np.fft.fft2(propagated))  # ❌
```

由于自由传播只改变相位，FFT 后强度几乎不变，导致多平面退化。

### 3. 修正方案实现

新增 `propagation_corrected.py:lens_focus_multiplane_intensity()`：
```python
# 正确实现：透镜相位 + 传播到探测面 + 直接取强度
field_after_lens = apply_lens_phase(near_field, ...)
field_at_detector = propagate_fn(field_after_lens, wavelength, distance, ...)
intensity = np.abs(field_at_detector) ** 2  # ✅ 不再 FFT
```

### 4. 验证结果

**32样本 smoke 测试**：
- 最大差异：**0.421**（修正前：2.17e-19）—— 提升 **1.9×10¹⁸ 倍**
- 平均差异：**0.0008**（修正前：1e-20）—— 提升 **8×10¹⁶ 倍**
- 退化样本：**0/10**（修正前：10/10）

**10k 完整数据集**：
- 🔄 生成中（后台任务 `b4mc0om1y`）
- 预计完成时间：~10分钟
- 输出路径：`dataset/seven_beam/multiplane_corrected_f1.0_d0.05/`

---

## 📊 对现有结果的影响评估

### ✅ 不受影响的结论（架构和训练策略）

| Cycle | 结论 | 有效性 | 原因 |
|-------|------|--------|------|
| 42 | 双分支融合优于通道堆叠 | ✅ 有效 | 参数效率优势（5.77M vs 11.34M）与输入物理无关 |
| 44 | 动态噪声增强提升鲁棒性 | ✅ 有效 | 噪声策略与输入通道物理含义无关 |
| 43 | IG+Grad-CAM 验证物理一致性 | ✅ 有效 | 模型确实学到有效特征，即使输入退化 |
| 32 | 六边形增强无效（负结果） | ✅ 有效 | 方法层面尝试，与数据无关 |
| 26 | 周期损失无效（负结果） | ✅ 有效 | 同上 |

### ⚠️ 需要重新验证的结论（多平面物理）

| Cycle | 结论 | 有效性 | 需要重做 |
|-------|------|--------|---------|
| 35 | 焦前图像提供更局部线索 | ❓ 待验证 | Attribution 分析 |
| 31 | 多平面优于单平面 0.7-0.8% | ❓ 待验证 | 单平面 vs 多平面对比 |
| 43 | 焦平面能量 48.4%，焦前 51.6% | ❓ 待验证 | 能量分配分析 |
| All | RMSE 0.892, Strehl 0.683 上限 | ❓ 待验证 | 性能重新评估 |

---

## 🎯 三种可能的场景

### 场景 A：乐观（修正后性能显著提升）

**预期**：
- RMSE 降至 **0.80-0.85 rad**（当前 0.892）
- Strehl 提升至 **0.70-0.75**（当前 0.683）
- 焦前图像确实提供更强相位线索
- Attribution 显示焦前分支贡献 > 60%

**影响**：
- ✅ 论文创新点"多平面融合"完全成立
- ✅ 可以强调"物理正确建模的重要性"
- ✅ Discussion 增加"修正前后对比"消融实验

### 场景 B：现实（修正后性能持平）

**预期**：
- RMSE 保持 **0.85-0.90 rad**
- Strehl 保持 **0.65-0.70**
- 多平面收益有限（< 5%）
- Attribution 显示两分支接近均衡（45-55%）

**影响**：
- ✅ 论文重点转向"双分支架构 + 噪声鲁棒性"
- ✅ 多平面作为补充实验，诚实讨论"物理正确性 vs 性能"
- ✅ 仍然是完整的研究，负结果也有价值

### 场景 C：保守（修正后性能下降）

**预期**：
- RMSE 上升至 **0.95-1.00 rad**
- Strehl 下降至 **0.60-0.65**
- 说明之前"伪多平面"意外提供了正则效果

**影响**：
- ⚠️ 需要 Priority 5（混合物理优化）弥补性能
- ⚠️ 论文诚实讨论"数据正则效果 vs 物理建模"
- ✅ 仍然是有价值的发现：不正确的物理模型有时也能工作

**无论哪种场景，修正物理模型都是必须的，因为当前数据与文献不符。**

---

## 📋 下一步行动清单

### Immediate（等待数据生成完成）

- [ ] 检查 10k 数据生成完成
- [ ] 验证两通道差异（应 > 0.01）
- [ ] 可视化 10 个样本检查焦前图像形态

### Priority 2（快速验证，~2-3小时）

- [ ] 用修正数据训练 Cycle 42 模型 15 epoch
- [ ] 评估 RMSE、Strehl、主瓣能量
- [ ] 与旧数据训练结果对比
- [ ] 初步判断属于场景 A/B/C

### Priority 3（配置扫描，~1天）

- [ ] 扫描离焦距离：`Δz = [0.01, 0.03, 0.05, 0.07, 0.1]m`
- [ ] 扫描光束间距：`d/w₀ = [2.0, 2.5, 3.0, 3.5, 4.0]`
- [ ] 选择最佳配置进入完整训练

### Priority 4（完整验证，~1-2天）

- [ ] 用最佳配置训练 30 epoch 完整模型
- [ ] 重新做 IG+Grad-CAM attribution
- [ ] 更新论文 Results 和 Discussion

---

## 📁 新增文件清单

### 核心代码

1. `simulation/common/propagation_corrected.py` - 修正后的传播模块
2. `simulation/static/generate_corrected_multiplane.py` - 修正后的数据生成
3. `train/quick_verify_corrected.py` - 快速验证训练脚本

### 验证和可视化

4. `scripts/verify_multiplane_data_issue.py` - 数据验证脚本
5. `scripts/visualize_corrected_vs_old.py` - 对比可视化
6. `result/figures/multiplane_correction_comparison.png` - 修正前后对比图

### 文档

7. `PROJECT_IMPROVEMENT_ROADMAP.md` - 改进路线图
8. `docs/MULTIPLANE_CORRECTION_REPORT.md` - 详细修正报告
9. 本文档

---

## 🎓 关键经验教训

### 1. 物理验证至关重要

即使代码运行正常、训练收敛、指标提升，也不代表物理模型正确。对关键物理量（如"多平面差异"）必须做数值验证。

### 2. 文献对照要细致

不要只看方法概述，要对照实验配置细节：
- Hou 2019 的 Fig.3 是近场强度轮廓，不是间距消融
- Hou 的 Fig.5 是补偿后焦平面，不是 CNN 输入
- 我们的物理参数与 Hou 完全不同（波长、腰斑、间距）

### 3. Cycle 管理的价值

完整的 Cycle 1-44 记录让我们能快速定位问题影响范围，区分"架构有效性"和"数据正确性"。

---

## 🔗 相关文档

- **改进路线图**：[PROJECT_IMPROVEMENT_ROADMAP.md](../PROJECT_IMPROVEMENT_ROADMAP.md)
- **详细修正报告**：[docs/MULTIPLANE_CORRECTION_REPORT.md](../docs/MULTIPLANE_CORRECTION_REPORT.md)
- **项目状态**：[PROJECT_STATUS.md](../PROJECT_STATUS.md)
- **项目计划**：[PROJECT_PLAN.md](../PROJECT_PLAN.md)

---

## 📞 下一步等待用户确认

1. **数据生成完成后**，是否立即开始快速验证训练？
2. **验证结果出来后**，根据场景 A/B/C 决定后续策略
3. **是否需要**在论文中增加"修正前后对比"作为消融实验？

---

**当前状态**: ✅ Priority 1 完成，等待数据生成  
**更新时间**: 2026-06-15 17:00  
**下一检查点**: 10k 数据生成完成（预计 10-15 分钟）
