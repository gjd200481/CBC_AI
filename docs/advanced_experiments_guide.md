# 高级实验运行指南

本文档说明如何运行三个高级实验：
1. 噪声增强训练（应对σ=0.002局部退化）
2. 高级解释性分析（Integrated Gradients + Grad-CAM）
3. 50k大规模数据集验证

---

## 实验1: 噪声增强训练

### 目标
解决Cycle43发现的σ=0.002局部退化问题，通过训练时动态添加噪声提升模型在低噪声下的稳定性。

### 运行方法

#### 方案A: 动态噪声增强（推荐）
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

python train/train_noise_augmented.py \
  --noise-mode dynamic \
  --noise-sigma-min 0.0 \
  --noise-sigma-max 0.005 \
  --model-name dual_plane_fusion_cnn \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --lambda-phy 0.05 \
  --lambda-comp 0.5 \
  --output-prefix cycle44_noise_aug_dynamic \
  --device cuda
```

#### 方案B: 课程学习（渐进式噪声）
```bash
python train/train_noise_augmented.py \
  --noise-mode curriculum \
  --noise-sigma-min 0.0 \
  --noise-sigma-max 0.005 \
  --model-name dual_plane_fusion_cnn \
  --epochs 40 \
  --output-prefix cycle44_noise_aug_curriculum
```

#### 方案C: 固定噪声
```bash
python train/train_noise_augmented.py \
  --noise-mode fixed \
  --noise-sigma-max 0.003 \
  --output-prefix cycle44_noise_aug_fixed
```

### 验证噪声鲁棒性
训练完成后，使用现有的噪声鲁棒性评估工具：

```bash
python train/evaluate_multiplane_noise_robustness.py \
  --image-path dataset/seven_beam/multiplane_0_-0.07/images_multiplane_7cm.npy \
  --label-path dataset/seven_beam/multiplane_0_-0.07/labels_multiplane_7cm.npy \
  --model cycle44=models/cycle44_noise_aug_dynamic_best.pth \
  --model cycle42=models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth \
  --noise-levels 0 0.001 0.002 0.003 0.005 0.01 0.02 \
  --max-samples 256 \
  --summary-csv result/metrics/cycle44_noise_robustness_comparison.csv \
  --figure-path result/figures/cycle44_noise_robustness_comparison.png
```

### 预期结果
- σ=0.002时不再出现退化
- σ=0~0.005范围内性能更平滑
- 对σ≥0.005的噪声保持鲁棒性

---

## 实验2: 高级解释性分析

### 目标
使用Integrated Gradients和Grad-CAM替代简单梯度方法，提供更准确的特征归因。

### 运行方法

#### 分析Cycle42模型
```bash
python train/analyze_advanced_attribution.py \
  --image-path dataset/seven_beam/multiplane_0_-0.07/images_multiplane_7cm.npy \
  --model-path models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth \
  --num-samples 20 \
  --target-channels 0 1 2 3 4 5 \
  --use-ig \
  --use-gradcam \
  --ig-steps 50 \
  --output-prefix cycle44_ig_gradcam_cycle42 \
  --device cuda
```

#### 对比Cycle41和Cycle42
```bash
# 分析Cycle41
python train/analyze_advanced_attribution.py \
  --model-path models/cycle41_multiplane_7cm_unorm_best_strehl_30epoch.pth \
  --num-samples 20 \
  --output-prefix cycle44_ig_gradcam_cycle41

# 分析Cycle42
python train/analyze_advanced_attribution.py \
  --model-path models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth \
  --num-samples 20 \
  --output-prefix cycle44_ig_gradcam_cycle42
```

### 输出内容
- **Integrated Gradients可视化**: 
  - 每个样本每个相位通道的IG归因图
  - 相比简单梯度更准确
  
- **Grad-CAM热图**: 
  - 卷积特征激活区域
  - 叠加在原始图像上的热图
  - 定位模型关注的关键区域

- **统计数据**: 
  - IG能量分布
  - Grad-CAM峰值位置
  - 跨样本的一致性

### 预期发现
- **IG优势**: 消除梯度饱和伪影，更准确反映真实重要性
- **Grad-CAM优势**: 空间定位更清晰，显示模型关注区域
- **Cycle42特点**: 双分支融合导致更分散的注意力分配

---

## 实验3: 50k大规模数据集

### 目标
验证模型在5倍数据量下的性能上限，判断当前模型容量是否饱和。

### 步骤1: 生成50k数据集

```bash
# 需要较大内存和较长时间（约30-60分钟）
python simulation/static/generate_50k_dataset.py \
  --num-samples 50000 \
  --batch-size 1000 \
  --num-points 256 \
  --window-size 0.01 \
  --waist 0.0005 \
  --beam-distance 0.0015 \
  --crop-size 160 \
  --output-dir dataset/seven_beam/multiplane_50k \
  --prefix multiplane_50k
```

**注意事项**:
- 数据集大小约 **7-8 GB**
- 生成时间约 **30-60分钟**（取决于CPU性能）
- 需要至少 **10 GB** 可用硬盘空间
- 分批生成避免内存溢出

### 步骤2: 训练模型

#### 使用GPU（推荐）
```bash
# RTX 3060或更高
python train/train_50k_dataset.py \
  --image-path dataset/seven_beam/multiplane_50k/images_multiplane_50k.npy \
  --label-path dataset/seven_beam/multiplane_50k/labels_multiplane_50k.npy \
  --model-name dual_plane_fusion_cnn \
  --epochs 50 \
  --batch-size 64 \
  --accumulation-steps 2 \
  --learning-rate 1e-3 \
  --lambda-phy 0.05 \
  --lambda-comp 0.5 \
  --use-amp \
  --num-workers 4 \
  --pin-memory \
  --output-prefix cycle45_50k \
  --save-interval 10 \
  --device cuda
```

**训练参数说明**:
- `batch-size 64`: 单次前向传播的样本数
- `accumulation-steps 2`: 有效batch size = 64 × 2 = 128
- `use-amp`: 混合精度训练，节省显存约30-40%
- `save-interval 10`: 每10个epoch保存一次checkpoint
- 预计训练时间: **8-12小时**（RTX 3060）

#### 使用CPU（备选）
```bash
# 不推荐，训练时间极长（数天）
python train/train_50k_dataset.py \
  --epochs 30 \
  --batch-size 16 \
  --accumulation-steps 4 \
  --num-workers 2 \
  --device cpu
```

#### 使用预训练模型（迁移学习）
```bash
# 从10k模型开始，加速收敛
python train/train_50k_dataset.py \
  --pretrained-model models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth \
  --epochs 30 \
  --output-prefix cycle45_50k_finetune
```

### 步骤3: 评估结果

```bash
# 噪声鲁棒性
python train/evaluate_multiplane_noise_robustness.py \
  --model cycle45_50k=models/cycle45_50k_best.pth \
  --model cycle42_10k=models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth \
  --summary-csv result/metrics/cycle45_50k_vs_10k_noise.csv

# 补偿效果
python train/evaluate_seven_beam_compensation_effect.py \
  --model cycle45_50k=models/cycle45_50k_best.pth \
  --model cycle42_10k=models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth \
  --summary-csv result/metrics/cycle45_50k_vs_10k_compensation.csv
```

### 预期结果分析

#### 情况A: 显著提升（RMSE下降>5%）
→ 说明模型容量未饱和，数据规模是瓶颈  
→ 论文中强调"大数据下性能持续提升"  
→ 建议：继续扩展到100k

#### 情况B: 轻微提升（RMSE下降2-5%）
→ 说明模型容量接近饱和  
→ 论文中说明"当前模型已充分利用数据"  
→ 建议：优化模型结构而非数据量

#### 情况C: 无提升或退化（RMSE下降<2%）
→ 说明模型容量已饱和或过拟合  
→ 论文中讨论"模型容量与数据量的平衡"  
→ 建议：保持10k数据，增强正则化

---

## 实验对比表

| 实验 | 目的 | 预计时间 | 显存需求 | 预期收益 |
|------|------|----------|----------|----------|
| **噪声增强** | 修复σ=0.002退化 | 2-3小时 | 4-6 GB | 低噪声下稳定性+20% |
| **IG+Grad-CAM** | 更准确的解释性 | 0.5-1小时 | 2-4 GB | 论文解释性章节增强 |
| **50k数据集** | 验证性能上限 | 10-15小时 | 8-10 GB | RMSE可能降低3-8% |

---

## 论文撰写建议

### 如果三个实验都成功

**Method章节**:
- 增加"噪声增强训练策略"小节
- 增加"高级归因分析方法"小节

**Results章节**:
- 新增图表：噪声增强前后对比
- 新增图表：IG/Grad-CAM可视化
- 新增表格：10k vs 50k性能对比

**Discussion章节**:
- 讨论噪声增强的有效性
- 讨论IG相比简单梯度的优势
- 讨论数据规模与模型容量的关系

### 如果部分实验失败

**负结果也有价值**:
- 噪声增强无效 → 说明σ=0.002退化不是训练策略问题
- 50k无提升 → 说明当前模型容量已充分利用10k数据
- 论文Discussion中诚实报告负结果

---

## 故障排除

### 问题1: 内存不足
```bash
# 减小batch size
--batch-size 16

# 增加梯度累积
--accumulation-steps 4

# 禁用pin_memory
# 移除 --pin-memory 参数
```

### 问题2: 训练过慢
```bash
# 使用混合精度
--use-amp

# 增加num_workers
--num-workers 8

# 使用预训练模型
--pretrained-model models/cycle42_...pth
```

### 问题3: 生成数据集失败
```bash
# 减小batch_size
--batch-size 500

# 分多次生成然后合并
# 生成前25k
--num-samples 25000 --prefix multiplane_25k_part1

# 生成后25k
--num-samples 25000 --prefix multiplane_25k_part2

# 手动合并
python -c "
import numpy as np
part1_img = np.load('dataset/.../images_multiplane_25k_part1.npy')
part2_img = np.load('dataset/.../images_multiplane_25k_part2.npy')
merged = np.concatenate([part1_img, part2_img])
np.save('dataset/.../images_multiplane_50k.npy', merged)
"
```

---

## 时间规划

### 快速验证（1天）
1. 噪声增强训练（动态模式）: 2-3小时
2. IG/Grad-CAM分析: 1小时
3. 结果分析和图表生成: 2小时

### 完整验证（3-4天）
1. Day 1: 生成50k数据集 + 开始训练
2. Day 2-3: 继续训练 + 噪声增强实验
3. Day 4: IG/Grad-CAM分析 + 结果汇总

---

## 后续工作

完成这三个实验后：
1. ✅ 更新`PROJECT_STATUS.md`记录Cycle44-45
2. ✅ 生成新的主图（噪声增强对比、IG可视化）
3. ✅ 更新Results章节
4. ✅ 准备投稿材料

---

**现在你有了完整的工具链和运行指南，可以开始这三个高级实验了！建议先运行噪声增强（最快见效），然后是IG/Grad-CAM（增强论文），最后是50k（耗时最长但收益可能最大）。**
