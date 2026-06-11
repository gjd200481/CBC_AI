# Cycle 31: 多平面输入验证与消融实验

## 执行步骤

### Step 1: 生成多平面数据集 (Smoke测试)

生成1k双平面数据集（焦平面 + 焦前5cm）:

```powershell
python simulation\static\generate_seven_beam_multiplane_dataset.py `
  --num-samples 1024 `
  --distances "0,-0.05" `
  --wavelength 632.8e-9 `
  --noise-sigma 0 `
  --crop-size 160 `
  --seed 20260616 `
  --output-dir dataset\seven_beam\multiplane `
  --prefix multiplane_seven_beam
```

### Step 2: Smoke测试 (1-2天)

快速验证多平面是否有效:

```powershell
python train\train_multiplane_ablation.py `
  --mode smoke `
  --device cuda `
  --num-workers 2 `
  --output-dir result\cycle31_multiplane_smoke
```

**判断标准**: 
- 如果双平面 test_rmse 比单平面降低 **>3%**，继续Step 3
- 如果提升 <3%，则放弃多平面路线

### Step 3: 生成完整多平面数据集 (仅在Smoke通过后)

```powershell
# 焦前3cm
python simulation\static\generate_seven_beam_multiplane_dataset.py `
  --num-samples 10000 `
  --distances "0,-0.03" `
  --output-dir dataset\seven_beam\multiplane_0_-0.03 `
  --seed 20260616

# 焦前5cm (Xie推荐)
python simulation\static\generate_seven_beam_multiplane_dataset.py `
  --num-samples 10000 `
  --distances "0,-0.05" `
  --output-dir dataset\seven_beam\multiplane_0_-0.05 `
  --seed 20260616

# 焦前7cm
python simulation\static\generate_seven_beam_multiplane_dataset.py `
  --num-samples 10000 `
  --distances "0,-0.07" `
  --output-dir dataset\seven_beam\multiplane_0_-0.07 `
  --seed 20260616
```

### Step 4: 完整消融实验 (1周)

```powershell
python train\train_multiplane_ablation.py `
  --mode ablation `
  --device cuda `
  --num-workers 2 `
  --output-dir result\cycle31_multiplane_ablation
```

### Step 5: 评估补偿质量指标

```powershell
# 复用evaluate_seven_beam_compensation_effect.py
python train\evaluate_seven_beam_compensation_effect.py `
  --baseline-model models\cycle30_deep_comp_10k.pth `
  --baseline-name "Cycle30_single_focal" `
  --model "Multiplane_5cm=result\cycle31_multiplane_ablation\multiplane_befocal_5cm_10k_best.pth" `
  --num-samples 256 `
  --output-dir result\cycle31_multiplane_compensation
```

## 预期结果

### 保守估计 (基于Xie 2024)

| 指标 | Cycle 30 单焦平面 | Cycle 31 双平面 | 改善 |
|------|------------------|----------------|------|
| 测试 RMSE | 0.955 rad | **0.85-0.90 rad** | -6% to -11% |
| Strehl比 | 0.647 | **0.66-0.68** | +2% to +5% |
| 通道不平衡 | 0.073 rad | **0.05-0.06 rad** | -20% to -30% |

### 消融实验预期

| 配置 | 预期 test_rmse | 说明 |
|------|---------------|------|
| 单焦平面 (baseline) | 0.955 rad | 当前最优 |
| 双平面 (焦前3cm) | 0.88-0.92 rad | 距离偏小 |
| 双平面 (焦前5cm) | **0.85-0.90 rad** | Xie推荐 ✓ |
| 双平面 (焦前7cm) | 0.90-0.93 rad | 距离偏大 |

## 输出文件

### Smoke测试
```
result/cycle31_multiplane_smoke/
├── baseline_single_focal_best.pth
├── baseline_single_focal_history.csv
├── multiplane_dual_best.pth
├── multiplane_dual_history.csv
└── summary_smoke.csv
```

### 完整消融
```
result/cycle31_multiplane_ablation/
├── baseline_single_10k_best.pth
├── multiplane_befocal_3cm_10k_best.pth
├── multiplane_befocal_5cm_10k_best.pth  ← 预期最优
├── multiplane_befocal_7cm_10k_best.pth
└── summary_ablation.csv
```

## 论文价值

### 直接对标 Xie 2024

**他们的发现**: Camera A (焦前5cm) 误差 0.26 rad，Camera B (焦平面) 误差 0.41 rad

**我们的验证**: 
- 在七光束系统验证多平面策略
- 量化焦前距离对相位RMSE和补偿质量的影响
- 证明物理约束 + 多平面的协同效应

### 论文叙事

> 受Xie等人[ref]启发，我们验证了多平面输入策略在七光束CBC系统的有效性。实验表明，同时利用焦平面和焦前5cm图像可使相位RMSE降低X%，Strehl比提升Y%。物理解释为焦前图像保留了更多局部相位信息，与焦平面图像互补，使深度网络能够更均衡地提取多通道相位特征。

## 风险应对

### 风险1: Smoke测试提升<3%
**应对**: 放弃多平面，回到单焦平面主线，Cycle 31改为其他改进方向

### 风险2: 数据生成太慢
**应对**: 
- 减少num_points到128 (当前256)
- 使用Fresnel传播代替角谱传播
- 并行生成: 每个距离配置在不同终端运行

### 风险3: 显存不足
**应对**: 
- 减小batch_size到16
- 使用gradient checkpointing
- 只训练焦前5cm一个配置

## 时间估算

- Step 1 (1k数据生成): 5-10分钟
- Step 2 (Smoke测试): 30-60分钟
- Step 3 (10k数据生成 × 3): 1-2小时
- Step 4 (完整训练 × 4): 6-8小时
- Step 5 (补偿评估): 30分钟

**总计**: 如果Smoke通过，约1-2天完成全部实验
