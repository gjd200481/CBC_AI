# CBC_AI

本项目面向相干光束合成（Coherent Beam Combining, CBC）中的相位误差反演问题，当前主线是 **7 光束多路相干合成相位误差智能估计**。项目目标已调整为形成一篇具备一区或二区期刊投稿潜力的研究论文；项目恢复 `Cycle` 管理方式，但 `Cycle` 只用于任务分割和实验批次记录，不绑定日期或硬性截止时间。

核心思路：

```text
7 光束远场光强图像
↓
CNN 相位反演
↓
6 路相对相位 sin/cos 编码
↓
傅里叶光学远场重建
↓
物理一致性损失约束
```

中心光束 `beam_0` 作为参考，相位固定为 0；外圈 6 路光束按六边形阵列排列，网络预测 `phi_1 ... phi_6`。标签格式为：

```text
[sin(phi_1), cos(phi_1), ..., sin(phi_6), cos(phi_6)]
```

## 当前进度

已完成：

- 双光束低维验证基线。
- 7 光束六边形阵列仿真模块。
- 7 光束静态数据集生成脚本。
- 7 光束普通 CNN baseline。
- 7 光束傅里叶光学物理一致性损失。
- 7 光束物理约束 CNN。
- 7 光束 `lambda_phy` 权重消融。
- 7 光束探测器噪声鲁棒性实验。
- 7 光束振幅失配与位置偏移鲁棒性实验。
- 7 光束主瓣能量占比与相位补偿效果评估。
- 7 光束 Strehl 比评估。
- 7 光束相位补偿综合效果实验。
- 双光束/7 光束系统规模对比。
- 7 光束网络结构快速消融。
- RTX 3060 长轮次训练准备。
- **Cycle 28: 数据规模扩展至10k样本，RMSE降至0.936 rad**。
- **Cycle 29: 补偿质量损失函数，直接优化Strehl比和主瓣能量**。
- **Cycle 30: 深度残差网络(11M参数) + 组合改进，Strehl比0.647，合成效率0.787，达到论文可接受水平** ✓。
- **Cycle 31: 多平面输入验证，小数据(-15.7%)收益显著，大数据(-0.8%)收益有限，作为补充实验**。
- **Cycle 32-40: 六边形对称增强、补偿损失调度、checkpoint选择策略、未归一化Strehl修复**。
- **Cycle 41: 修复训练期Strehl指标，使checkpoint选择与最终评估一致**。
- **Cycle 42: 焦平面/焦前双分支融合，以5.77M参数超过Cycle 41的11.34M模型** ✓✓。
- **Cycle 43: Attribution解释性与噪声鲁棒性验证，技术验证阶段完成** ✓✓✓。

当前双主模型配置：

```text
补偿质量主模型 (Cycle 42, 焦平面/焦前双分支融合):
DualPlaneFusionPhaseCNN (5.77M参数)
+ 焦平面/焦前显式门控融合
+ CompensationQualityLoss (lambda_comp=0.5)
+ 未归一化Strehl checkpoint选择

补偿后Strehl比: 0.683 (论文可接受水平 ✓✓)
补偿后合成效率: 0.796 (论文可接受水平 ✓✓)
补偿后主瓣能量: 0.525
残余相位RMSE: 0.892 rad

噪声鲁棒性 (σ=0.02): Strehl 0.481 > Cycle41 0.407

相位精度主模型 (Cycle 37):
MultiPlanePhaseCNN, lambda_comp=0.3
测试相位RMSE: 0.932 rad
残余相位RMSE: 0.866 rad (当前最低)
```

## 目录结构

```text
CBC_AI/
├── README.md
├── NAMING_CONVENTIONS.md
├── PROJECT_PLAN.md
├── PROJECT_STATUS.md
├── KEY_FILES.md
├── MULTI_BEAM_TRANSITION.md
├── examples/
│   ├── demo_evaluate_two_beam_model.py
│   └── demo_two_beam_inference.py
├── simulation/
│   ├── common/
│   │   ├── two_beam_core.py
│   │   └── multi_beam_core.py
│   ├── static/
│   │   ├── generate_two_beam_dataset.py
│   │   ├── generate_seven_beam_dataset.py
│   │   └── legacy/
│   └── dynamic/
│       └── generate_two_beam_sequence_dataset.py
├── train/
│   ├── data_utils.py
│   ├── models.py
│   ├── phase_metrics.py
│   ├── physics_loss.py
│   ├── train_seven_beam_baseline.py
│   ├── train_seven_beam_physics_constrained_cnn.py
│   └── sweep_seven_beam_lambda.py
├── result/
│   ├── logs/
│   ├── metrics/
│   └── figures/
├── dataset/
├── models/
└── paper/
```

说明：

- `dataset/` 保存本地生成数据集，不提交 Git。
- `models/` 保存本地模型权重，不提交 Git。
- `result/` 默认忽略，但关键日志、CSV 和图会按重要实验结果强制提交。
- `examples/` 保存演示和快速推理脚本。
- `simulation/static/legacy/` 保存早期验证脚本，不作为主训练入口。

## 快速复现实验

### 1. 生成 7 光束静态数据集

```powershell
python simulation\static\generate_seven_beam_dataset.py --num-samples 1024 --noise-sigma 0 --num-points 256 --window-size 0.01 --waist 0.0005 --beam-distance 0.0015 --crop-size 160 --seed 20260612 --output-dir dataset\seven_beam\main_static --prefix main_clean_seven_beam
```

### 2. 训练 7 光束普通 CNN baseline

```powershell
python train\train_seven_beam_baseline.py --epochs 30 --batch-size 32 --learning-rate 0.001 --seed 20260612 --no-plot
```

### 3. 训练 7 光束物理约束 CNN

```powershell
python train\train_seven_beam_physics_constrained_cnn.py --lambda-phy 0.1 --epochs 30 --batch-size 32 --learning-rate 0.001 --seed 20260612 --no-plot
```

### 4. 进行物理损失权重消融

```powershell
python train\sweep_seven_beam_lambda.py --epochs 12 --batch-size 32 --learning-rate 0.001 --seed 20260612 --no-plot
```

## 关键结果

当前 7 光束主数据集测试结果：

| 模型 | lambda_phy | RMSE(rad) | MAE(rad) | far-field MSE |
| --- | --- | --- | --- | --- |
| 普通 CNN | 0 | `1.02698` | `0.81906` | `1.1935e-4` |
| 物理约束 CNN | 0.1 | `1.02269` | `0.81642` | `1.1501e-4` |
| 物理约束 CNN | 0.5 | `1.05027` | `0.82944` | `1.2103e-4` |

噪声鲁棒性阶段性结论：

- 干净数据上 `lambda_phy=0.1` 物理约束 CNN 略优。
- 当 `noise>=0.03` 时，当前干净训练的物理约束模型比普通 CNN 更敏感。
- 后续鲁棒性提升应考虑噪声增强训练或去噪物理一致性目标。

复杂扰动阶段性结论：

- 在振幅失配 `0~0.3` 下，物理约束 CNN 的 RMSE 相比普通 CNN 降低约 `1.25%~2.39%`。
- 在位置偏移 `0~100um` 下，物理约束 CNN 的 RMSE 相比普通 CNN 降低约 `0.99%~2.36%`。
- 当前物理约束对光束状态扰动更友好，对探测器噪声不自动鲁棒。

主瓣能量占比阶段性结论：

- 补偿前主瓣能量占比约 `0.35939`。
- 普通 CNN 补偿后约 `0.51931`。
- 物理约束 CNN 补偿后约 `0.52155`。
- 理想相干约 `0.65063`。

Strehl 比阶段性结论：

- 补偿前 Strehl 均值约 `0.39069`。
- 普通 CNN 补偿后约 `0.64717`。
- 物理约束 CNN 补偿后约 `0.65356`。
- **Cycle 42 双分支融合补偿后约 `0.68269`** ✓✓。
- 理想相干为 `1.00000`。

相位补偿综合效果阶段性结论：

- 补偿前合成效率约 `0.53286`。
- 普通 CNN 补偿后合成效率约 `0.78602`。
- 物理约束 CNN 补偿后合成效率约 `0.78964`。
- **Cycle 42 双分支融合补偿后合成效率约 `0.79585`** ✓✓。
- Cycle 42 模型在主瓣能量占比、Strehl 比、合成效率和噪声鲁棒性上全面优于早期模型。

系统规模对比阶段性结论：

- 7 光束待预测相位数量和网络输出维度均为双光束的 `6` 倍。
- 双光束普通 CNN RMSE 为 `0.003742 rad`，7 光束普通 CNN RMSE 为 `1.026976 rad`。
- 双光束定位为方法验证和低维基线，7 光束定位为论文主实验对象。

网络结构快速消融阶段性结论：

- 已新增 `wide_cnn` 和 `residual_cnn` 两类候选结构。
- 96 样本、2 epoch 快速筛选中，`residual_cnn` 的测试 RMSE 为 `1.709031 rad`，优于同设置下的 `simple_cnn` 和 `wide_cnn`。
- 该结果只用于候选筛选，后续需要完整数据长训练验证。

技术验证历程总结：

- Cycle 1-20：建立双光束到七光束完整训练流程，验证物理约束有效性。
- Cycle 21-27：网络结构消融，确认残差网络优于简单CNN，发现相位RMSE与补偿质量不完全一致。
- Cycle 28-30：数据规模扩展(10k)，补偿质量损失重构，达到Strehl 0.647论文可接受水平。
- Cycle 31：多平面输入验证，发现大数据集下收益有限(0.8%)，作为补充实验保留。
- Cycle 32-34：六边形对称增强、补偿损失调度(warmup)、单位圆约束参数扫描。
- Cycle 35-40：Attribution解释性、多平面7cm训练、lambda_comp扫描、checkpoint选择策略验证。
- Cycle 41：修复未归一化Strehl指标，使训练期选择与最终评估一致。
- **Cycle 42：焦平面/焦前双分支门控融合，以更小参数量(5.77M)超过简单堆叠(11.34M)**。
- **Cycle 43：Attribution显示动态跨平面特征分配，噪声鲁棒性验证在σ≥0.005全面优于Cycle 41**。

关键负结果记录：

- 周期相位损失(cyclic)在cbc_lite_cnn上未超过MSE。
- 轻量网络cbc_lite_cnn未超过残差物理约束路线。
- 六边形对称增强未转化为补偿质量收益。
- 多平面在10k数据下收益仅0.8%，但双分支显式融合有效。

## 项目文档

- `PROJECT_PLAN.md`：面向一区/二区投稿目标的研究路线图和无时间约束 Cycle 任务规划。
- `PROJECT_STATUS.md`：当前进度、阶段性结论和下一步。
- `KEY_FILES.md`：关键文件地址和作用说明。
- `NAMING_CONVENTIONS.md`：目录和文件命名规范。
- `MULTI_BEAM_TRANSITION.md`：从双光束升级到 7 光束的路线说明。

## Git 提交规则

提交到 Git：

- 源码。
- README、计划、进度、关键文件说明。
- 关键实验日志。
- 指标 CSV。
- 重要结果图。

不提交到 Git：

- `.npy` 数据集。
- `.pth`、`.pt`、`.ckpt` 模型权重。
- 本地缓存和 `__pycache__`。
- IDE 临时文件。
