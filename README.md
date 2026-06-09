# CBC_AI

本项目面向相干光束合成（Coherent Beam Combining, CBC）中的相位误差反演问题，当前主线是 **7 光束多路相干合成相位误差智能估计**。

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

当前暂定 7 光束主实验物理损失权重为：

```text
lambda_phy = 0.1
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
- `result/` 默认忽略，但关键日志、CSV 和图会按周期强制提交。
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
- 理想相干为 `1.00000`。

相位补偿综合效果阶段性结论：

- 补偿前合成效率约 `0.53286`。
- 普通 CNN 补偿后合成效率约 `0.78602`。
- 物理约束 CNN 补偿后合成效率约 `0.78964`。
- 物理约束 CNN 在主瓣能量占比、Strehl 比、合成效率和残余相位 RMSE 上均略优于普通 CNN。

## 项目文档

- `PROJECT_PLAN.md`：两天一个周期的项目计划。
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
