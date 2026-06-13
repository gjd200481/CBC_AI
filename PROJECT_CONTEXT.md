# CBC_AI 项目对话历史

## 最新路线修订：指标修复 + 焦前/焦平面融合（2026-06-13）

用户认可下一阶段从“更大的模型”转向“更正确的物理指标 + 更聪明的焦前/焦平面信息融合”。结合 Hou 2019、Mills 2022、Xie 2024 与 Cycle35-42 结果，当前路线为：

1. **Cycle 41**：先修复训练期未归一化 Strehl / 主瓣指标，使 checkpoint 选择与最终补偿评估一致。
2. **Cycle 42**：已完成焦平面/焦前双分支特征融合，`cycle42_best_rmse` 在 paired 评估中取得主瓣能量 `0.525304`、Strehl `0.682690`、合成效率 `0.795854`、残余 RMSE `0.892309 rad`，优于 Cycle41。
3. **Cycle 43**：下一步做 attribution 和噪声鲁棒性补强，验证双分支收益是否具有物理解释和稳定性。

当前双主模型已更新：补偿质量主模型为 `models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth`，相位/残余 RMSE 主模型为 `models/cycle37_multiplane_7cm_lambda_comp0p3_30epoch.pth`。

## 最近会话：Cycle 31 多平面输入验证 (2026-06-12)

### 会话目标
用户要求继续追求更好的模型，阅读Xie 2024论文后决定实施多平面输入验证。

### 执行内容

#### 1. 文献调研
- 阅读 Xie et al. 2024 (Scientific Reports)
- 核心发现：焦前图像比焦平面图像相位预测误差降低36%
- 物理原理：焦前图像保留更多局部干涉条纹

#### 2. 技术实现
**新增模块**：
- `simulation/common/propagation.py` - 角谱/菲涅尔传播
- `simulation/static/generate_seven_beam_multiplane_dataset.py` - 多平面数据生成
- `train/models.MultiPlanePhaseCNN` - 多平面输入网络
- `train/train_multiplane_ablation.py` - 训练和消融脚本

**数据集生成**：
- 1k smoke数据（焦平面 + 焦前5cm）
- 10k完整数据 × 3个距离（3cm/5cm/7cm）

#### 3. 实验结果

**Smoke测试（1k数据）**：
- 单焦平面：RMSE = 1.4783 rad
- 双平面：RMSE = 1.2455 rad
- **改善：-15.7%** ✓ 通过3%阈值

**完整消融（10k数据）**：
- 单焦平面：RMSE = 0.9530 rad
- 双平面 3cm：RMSE = 0.9456 rad (-0.8%)
- 双平面 5cm：RMSE = 0.9462 rad (-0.7%)
- 双平面 7cm：RMSE = 0.9449 rad (-0.8%)

#### 4. 核心发现
- **多平面收益与数据规模负相关**：小数据(-15.7%)显著，大数据(-0.8%)有限
- **当前配置已接近单焦平面表示上限**：10k + 11.3M参数足够
- **焦前距离影响小**：3cm/5cm/7cm差异仅1.4%

#### 5. 项目决策
- **论文定位**：作为补充实验(supplementary)，不作主线创新
- **原因**：性能改善0.8%不如Cycle 30突破性（Strehl 0.647）
- **下一步**：暂不进入论文写作阶段，继续进行模型二次改进（Cycle 32-34）

### Git提交
```
Commit: 4c4c36f
Message: Cycle 31: 多平面输入验证与消融实验
Branch: cbc-lite-cyclic-phase
Files: 11 changed, +1334 lines
```

### 输出文件
- 周期报告：`result/logs/cycle31_multiplane_ablation_2026-06-12.md`
- 实验数据：`result/metrics/cycle31_multiplane_ablation_summary_2026-06-12.csv`
- 数据集：`dataset/seven_beam/multiplane_*` (31k样本)

---

## 项目当前状态（截至 Cycle 31）

### 核心技术路线
```
七光束远场图像 
→ DeepResidualPhaseCNN (11.3M参数)
→ 相位监督 + 物理一致性(λ=0.05) + 补偿质量(λ=0.5)
→ CosineAnnealingLR + 数据增强
→ Strehl比 0.647, 合成效率 0.787 ✓ 论文可接受水平
```

### 已完成的Cycle

| Cycle | 主题 | 核心成果 | 状态 |
|-------|------|---------|------|
| 1-10 | 双光束验证 | 建立完整训练流程 | ✓ |
| 11-20 | 七光束主线 | 物理约束+补偿指标 | ✓ |
| 21-27 | 模型优化 | 残差网络+最佳checkpoint | ✓ |
| 28 | 数据规模 | 10k样本，RMSE 0.936 | ✓ |
| 29 | 补偿损失 | 直接优化Strehl比 | ✓ |
| 30 | 深度网络 | 11.3M参数，Strehl 0.647 | ✓ 突破 |
| 31 | 多平面 | 验证边际收益递减 | ✓ 补充 |

### 项目阶段
- **技术验证阶段**：已完成 ✓
- **模型二次改进阶段**：进行中 →
  - Cycle 32: 六边形对称增强与通道均衡
  - Cycle 33: 补偿质量损失调度与单位圆约束
  - Cycle 34: 补偿感知模型头与物理 refinement
- **论文写作阶段**：后移，等待模型二次改进形成明确正结果或负结果后再集中推进

### 关键指标

**当前最优模型（Cycle 30）**：
- 测试RMSE: 0.955 rad
- Strehl比: 0.647 (论文可接受 ✓)
- 合成效率: 0.787 (论文可接受 ✓)
- 主瓣能量: 0.520
- 通道不平衡: 0.073 rad

**数据资产**：
- 双光束数据: 2k样本
- 七光束单平面: 11k样本
- 七光束多平面: 31k样本

**模型资产**：
- 训练好的模型: 20个
- 参数量范围: 0.3M - 11.3M

---

## 技术特色

### 1. 物理建模
- 傅里叶光学可微分前向模型
- 七光束近场到远场FFT传播
- 角谱传播支持多平面

### 2. 损失函数体系
- 相位监督：MSE on sin/cos编码
- 物理一致性：远场重建MSE
- 补偿质量：直接优化Strehl比和主瓣能量

### 3. 评价体系
- 相位指标：RMSE, MAE, 通道不平衡
- 补偿指标：Strehl比, 合成效率, 主瓣能量
- 鲁棒性：噪声/振幅/位置扰动

### 4. 项目管理
- Cycle管理：31个实验周期
- 版本控制：Git + 分支管理
- 文档完善：README, STATUS, PLAN, KEY_FILES

---

## 文献对标

### Xie et al. 2024 (Scientific Reports)
- **创新**：焦前图像 vs 焦平面，误差降低36%
- **我们的验证**：小数据(-15.7%)有效，大数据(-0.8%)有限
- **差异原因**：实验 vs 仿真，2M vs 11.3M网络

### Hou et al. 2019
- **方法**：深度学习相位控制
- **我们的扩展**：物理约束 + 补偿质量优化

### Mills et al. 2022
- **方法**：单步相位优化
- **我们的对比**：深度学习 vs 传统优化

---

## 代码结构

```
CBC_AI/
├── simulation/
│   ├── common/
│   │   ├── two_beam_core.py          # 双光束仿真
│   │   ├── multi_beam_core.py        # 七光束仿真
│   │   └── propagation.py            # 光场传播 (Cycle 31新增)
│   └── static/
│       ├── generate_two_beam_dataset.py
│       ├── generate_seven_beam_dataset.py
│       └── generate_seven_beam_multiplane_dataset.py  (Cycle 31新增)
├── train/
│   ├── data_utils.py                 # 数据加载（支持多平面）
│   ├── models.py                     # 网络架构（含MultiPlanePhaseCNN）
│   ├── phase_metrics.py              # 相位评估
│   ├── physics_loss.py               # 物理损失
│   ├── train_seven_beam_baseline.py
│   ├── train_seven_beam_physics_constrained_cnn.py
│   ├── train_deep_residual_final.py  # Cycle 30最优
│   ├── train_multiplane_ablation.py  # Cycle 31
│   └── train_multiplane_quick.py     # Cycle 31
├── result/
│   ├── logs/cycle*.md                # 31个实验报告
│   ├── metrics/cycle*.csv            # 实验数据
│   └── figures/cycle*.png            # 结果图
└── paper/
    ├── journals/chinese/             # 中文文献
    └── daedalus_packages/            # 精读文献包

```

---

## 下次对话建议

### 如果继续技术探索
1. **方案A**：小网络+多平面（验证参数效率）
2. **方案B**：实验数据验证（搭建SLM系统）
3. **方案C**：其他改进方向（注意力机制、transformer）

### 如果继续修改模型（当前推荐）
1. **Cycle 32**：实现七光束六边形对称增强，降低通道不平衡
2. **Cycle 33**：稳定化补偿质量损失，加入 warmup 调度和单位圆约束
3. **Cycle 34**：探索补偿感知模型头或少步物理 refinement

### 快速启动命令
```bash
# 查看项目状态
cat README.md
cat PROJECT_STATUS.md

# 查看最新实验
cat result/logs/cycle31_multiplane_ablation_2026-06-12.md

# 查看下一步计划
cat PROJECT_PLAN.md

# 继续训练（如果需要）
python train/train_deep_residual_final.py --device cuda
```

---

## 关键文件索引

- **项目概览**：`README.md`
- **当前进度**：`PROJECT_STATUS.md`
- **研究计划**：`PROJECT_PLAN.md`
- **文件说明**：`KEY_FILES.md`
- **最新实验**：`result/logs/cycle31_multiplane_ablation_2026-06-12.md`
- **最优模型配置**：`train/train_deep_residual_final.py`

---

## 联系上下文的关键点

1. **当前阶段**：技术验证完成，进入模型二次改进
2. **核心成果**：Cycle 30达到论文可接受水平（Strehl 0.647）
3. **Cycle 31角色**：补充实验，证明当前配置已接近上限
4. **下一步**：Cycle 32实现六边形对称增强与通道均衡实验
5. **Git分支**：cbc-lite-cyclic-phase
6. **最新commit**：4c4c36f

---

*最后更新：2026-06-12*
*总实验周期：31个*
*总代码行数：~15000行*
*总实验时间：~200小时*
