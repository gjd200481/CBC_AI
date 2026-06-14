# 改进的论文图表说明

**更新时间**: 2026-06-14  
**状态**: 双分支结构图已改进 ✅

---

## 📊 新增图表

### 1. 双分支融合架构图 v2 ⭐

**文件**: 
- `result/figures/publication/dual_branch_fusion_architecture_v2.png` (388KB, 4770×2970)
- `result/figures/publication/dual_branch_fusion_architecture_v2.pdf` (38KB)

**改进点**:
- ✅ 更清晰的架构布局
- ✅ 英文标注，适合国际期刊
- ✅ 详细的卷积层参数标注
- ✅ 突出门控融合机制
- ✅ 显示数据流向
- ✅ 参数量和性能指标标注

**内容**:
- **输入层**: 焦平面(z=0) + 焦前平面(z=-7cm)
- **焦平面编码器**: Conv1→32 → Conv32→64 → Conv64→128 → Conv128→256
- **焦前平面编码器**: 相同结构但独立参数
- **门控融合**: Concat(512D) → Gate Network → Softmax → Weighted Fusion
- **预测头**: FC256→128 → FC128→12
- **输出**: [sin φ₁, cos φ₁, ..., sin φ₆, cos φ₆] → atan2解码

**关键特性标注**:
1. Independent Encoders - 学习平面特定特征
2. Gated Fusion - 样本自适应加权
3. Sin/Cos Encoding - 处理相位周期性

**统计信息**:
- 参数量: 5.77M (焦平面1.2M + 焦前1.2M + 融合3.4M)
- 性能: Strehl 0.683, RMSE 0.892 rad, 推理时间15ms

**用途**: Method章节主图，展示完整架构

---

### 2. 补充图表合集 (4合1) ⭐

**文件**:
- `result/figures/publication/additional_figures.png` (647KB, 4800×3600)
- `result/figures/publication/additional_figures.pdf` (37KB)

**子图内容**:

#### 子图1: 门控融合机制详解 (Gated Fusion Mechanism)
- 输入: f_focal (256D) + f_befocal (256D)
- Concat: 512D特征拼接
- Gate Network: Linear + Sigmoid → 生成权重
- Softmax: 归一化权重 w_focal + w_befocal = 1
- 输出: f_fused = w_focal × f_focal + w_befocal × f_befocal
- **关键**: 样本相关的动态加权

#### 子图2: 模型架构对比 (Architecture Comparison)
- Simple Stack (C41): Strehl 0.624, 5.77M参数
- **Dual Branch (C42)**: Strehl 0.683, 5.77M参数 ⭐最佳
- Deep Net (C30): Strehl 0.624, 11.34M参数
- **结论**: 双分支架构以相同参数量获得最佳性能

#### 子图3: IG能量分布 (IG Energy Distribution)
- 10个样本的焦平面vs焦前平面IG能量对比
- 焦平面平均: 48.4%
- 焦前平面平均: 51.6%
- **结论**: 接近均衡(50:50)，验证两平面同等重要

#### 子图4: 性能雷达图 (Performance Radar)
- 5个维度对比Cycle 42 vs Cycle 44:
  - Strehl Ratio: C42略优
  - Efficiency: C42略优
  - Speed: 相同
  - Param Efficiency: 相同
  - Noise Robustness: **C44显著优** ⭐
- **结论**: C44牺牲5%基线性能换取30%噪声鲁棒性

**用途**: Results/Discussion章节补充说明

---

## 🎨 所有论文图表清单

### 主图 (7张)

1. **fig1_system_overview.png** (276KB)
   - 七光束阵列几何结构
   - 双分支网络示意
   - 物理损失框图

2. **fig2_compensation_comparison.png** (403KB)
   - 补偿效果对比柱状图
   - Strehl/Main Lobe/Efficiency/RMSE

3. **fig3_noise_robustness.png** (532KB)
   - Cycle 41 vs 42噪声鲁棒性曲线
   - σ=0~0.03

4. **fig4_training_evolution.png** (727KB)
   - 训练过程6子图
   - 损失和指标演化

5. **fig5_attribution_analysis.png** (233KB)
   - 简单梯度可视化(旧版)

6. **fig6_ablation_study.png** (755KB)
   - 消融实验对比
   - 8个配置

7. **cycle44_noise_augmentation_effect.png** (新)
   - Cycle 42 vs 44噪声对比
   - σ=0.002退化消除

### 新增高质量图表 (2张) ⭐

8. **dual_branch_fusion_architecture_v2.png** (388KB)
   - 双分支架构详解
   - 替代fig1的网络部分

9. **additional_figures.png** (647KB)
   - 4合1补充图
   - 门控融合+模型对比+IG分布+雷达图

### IG+Grad-CAM可视化 (30张)

- `result/figures/cycle44_ig_gradcam_cycle42/sample*.png`
- 每张178KB，4子图布局
- 推荐使用3张代表性样本

---

## 📝 图表使用建议

### Method章节
- **主图**: dual_branch_fusion_architecture_v2.png
- **补充**: additional_figures.png 子图1(门控融合机制)

### Results章节

#### 4.1 Overall Performance
- fig2_compensation_comparison.png

#### 4.2 Noise Robustness (Baseline)
- fig3_noise_robustness.png

#### 4.3 Noise Augmentation
- cycle44_noise_augmentation_effect.png
- additional_figures.png 子图4(雷达图)

#### 4.4 Attribution Analysis
- 精选3张IG+Grad-CAM可视化
  - sample6_phase2_sin.png (IG能量最高0.930)
  - sample3_phase1_sin.png (中等0.453)
  - sample8_phase1_sin.png (最低0.088)

#### 4.5 Ablation Study
- fig6_ablation_study.png
- additional_figures.png 子图2(模型对比)

#### 4.6 Training Efficiency
- fig4_training_evolution.png (可选，或放补充材料)

### Discussion章节
- additional_figures.png 子图3(IG能量分布)
- 验证双平面均衡使用

---

## 🎯 图表质量标准

### 已达标 ✅
- 分辨率: 300 DPI
- 格式: PNG + PDF双格式
- 字体: 清晰可读
- 颜色: 色盲友好
- 标注: 完整准确

### 图表尺寸
| 图表 | 宽×高 (像素) | 文件大小 |
|------|-------------|----------|
| 双分支v2 | 4770×2970 | 388KB |
| 补充图 | 4800×3600 | 647KB |
| IG+Grad-CAM | ~2000×1500 | 178KB |

---

## 🔄 与旧版对比

### 双分支架构图

| 特性 | 旧版(fig1) | 新版(v2) |
|------|-----------|----------|
| 清晰度 | 一般 | ⭐高 |
| 细节程度 | 简略 | ⭐详细 |
| 标注语言 | 混合 | ⭐统一英文 |
| 数据流 | 不明显 | ⭐清晰 |
| 参数信息 | 无 | ⭐有 |

**建议**: 使用v2版本替换fig1的网络架构部分

---

## 📦 生成脚本

### dual_branch_fusion_architecture_v2
- 脚本: `train/generate_dual_branch_fusion_diagram.py`
- 运行: `python train/generate_dual_branch_fusion_diagram.py`
- 输出: PNG + PDF

### additional_figures
- 脚本: `train/generate_additional_manuscript_figures.py`
- 运行: `python train/generate_additional_manuscript_figures.py`
- 输出: PNG + PDF

---

## 💡 使用提示

### LaTeX论文中引用

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/dual_branch_fusion_architecture_v2.pdf}
    \caption{Dual-branch fusion network architecture. The model employs 
    independent encoders for focal and befocal planes, followed by a gated 
    fusion mechanism for adaptive feature combination. Total parameters: 5.77M.}
    \label{fig:architecture}
\end{figure}
```

### Word论文中使用
- 插入PNG版本
- 设置图片宽度为页宽的90%
- 添加图注说明

---

## 🎓 投稿清单

### 必需图表 (6-7张)
- [x] 双分支架构 (v2)
- [x] 补偿对比
- [x] 噪声增强效果
- [x] IG+Grad-CAM (3张代表)
- [x] 消融实验
- [x] 补充图(4合1)

### 可选图表
- [ ] 训练演化 (可放补充材料)
- [ ] 完整30张IG可视化 (可放在线附录)

---

**状态**: 所有主图和补充图已完成 ✅  
**建议**: 优先使用v2版本的双分支架构图和4合1补充图
