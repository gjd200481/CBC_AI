# 当前模型结构说明

本文档说明当前补偿质量主模型 `cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth` 的网络结构、卷积与池化配置、焦平面/焦前融合门控、参数量和训练参数。

## 1. 模型定位

当前采用“双主模型”策略：

| 角色 | 模型权重 | 说明 |
| --- | --- | --- |
| 补偿质量主模型 | `models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth` | 当前主瓣能量、Strehl、合成效率综合表现最好 |
| 相位/残余 RMSE 主模型 | `models/cycle37_multiplane_7cm_lambda_comp0p3_30epoch.pth` | 当前残余相位 RMSE 更低 |

本文档重点描述补偿质量主模型：

```text
model_name = dual_plane_fusion_cnn
model class = DualPlaneFusionPhaseCNN
code = train/models.py
```

## 2. 输入与输出

| 项目 | 设置 |
| --- | --- |
| 数据集 | `dataset/seven_beam/multiplane_0_-0.07/` |
| 图像文件 | `images_multiplane_7cm.npy` |
| 标签文件 | `labels_multiplane_7cm.npy` |
| 输入张量 | `[B, 2, 160, 160]` |
| 通道 0 | 焦平面远场强度图 |
| 通道 1 | 焦前/离焦 7cm 强度图 |
| 输出张量 | `[B, 12]` |
| 输出含义 | 6 路外圈相对相位的 `[sin(phi_i), cos(phi_i)]` |
| 中心光束 | 参考相位固定为 0 |

相位解码方式：

```text
phi_i = atan2(sin(phi_i), cos(phi_i))
```

## 3. 总体结构

Cycle42 不再把焦平面和焦前图像简单堆叠后交给一个共享 CNN，而是采用双分支编码：

```text
focal plane image   -> focal_encoder   -> focal feature
befocal image       -> befocal_encoder -> befocal feature
focal + befocal     -> fusion_gate     -> fused feature
fused feature       -> channel_attention
attended feature    -> phase regression head -> 12-dim output
```

两个 encoder 结构相同但参数不共享。每个分支输出 `[B, 256, 5, 5]` 特征图。

## 4. 单分支卷积结构

单个分支使用：

```text
PlaneFeatureEncoder(input_channels=1, base_channels=32)
```

尺寸变化如下：

| 阶段 | 操作 | 输出尺寸 |
| --- | --- | --- |
| input | 单通道图像 | `[B, 1, 160, 160]` |
| stem conv | `7x7 Conv, stride=2, padding=3, 1 -> 32` | `[B, 32, 80, 80]` |
| stem pool | `3x3 MaxPool, stride=2, padding=1` | `[B, 32, 40, 40]` |
| layer1 | 2 个 `BasicResBlock`, `32 -> 32` | `[B, 32, 40, 40]` |
| layer2 | 2 个 `BasicResBlock`, 首块 stride=2, `32 -> 64` | `[B, 64, 20, 20]` |
| layer3 | 2 个 `BasicResBlock`, 首块 stride=2, `64 -> 128` | `[B, 128, 10, 10]` |
| layer4 | 2 个 `BasicResBlock`, 首块 stride=2, `128 -> 256` | `[B, 256, 5, 5]` |

`BasicResBlock` 结构：

```text
3x3 Conv -> BatchNorm -> ReLU
3x3 Conv -> BatchNorm
shortcut add
ReLU
```

当通道数或 stride 不匹配时，shortcut 使用：

```text
1x1 Conv(stride=stride) -> BatchNorm
```

## 5. 池化设计

模型中有三类池化：

| 位置 | 池化 | 作用 |
| --- | --- | --- |
| 每个 encoder stem 后 | `MaxPool2d(kernel=3, stride=2, padding=1)` | 从 `80x80` 降到 `40x40` |
| `fusion_gate` 内 | `AdaptiveAvgPool2d(1)` | 将 `[B, 512, 5, 5]` 压缩为通道描述向量 |
| `channel_attention` 与输出头内 | `AdaptiveAvgPool2d(1)` | 生成通道注意力和全连接回归输入 |

输出头不是直接 flatten `5x5` 全特征图，而是先做全局平均池化，这减少参数量并降低对空间位置的偶然记忆。

## 6. 融合门控

两个分支先得到：

```text
focal   = focal_encoder(x[:, 0:1])    # [B, 256, 5, 5]
befocal = befocal_encoder(x[:, 1:2])  # [B, 256, 5, 5]
```

然后拼接：

```text
concat = cat([focal, befocal], dim=1) # [B, 512, 5, 5]
```

`fusion_gate` 结构：

```text
AdaptiveAvgPool2d(1)
Flatten
Linear(512 -> 128)
ReLU
Linear(128 -> 256)
Sigmoid
```

融合公式：

```text
fused = gate * focal + (1 - gate) * befocal
```

`gate` 是 `[B, 256]` 的样本级、通道级权重。`gate` 接近 1 时该特征通道更依赖焦平面，接近 0 时更依赖焦前图像。

## 7. 通道注意力

融合后再做一次轻量通道注意力：

```text
AdaptiveAvgPool2d(1)
Flatten
Linear(256 -> 16)
ReLU
Linear(16 -> 256)
Sigmoid
```

该模块不再区分特征来自焦平面还是焦前，而是对融合后的 256 个通道重新加权：

```text
fused = fused * attention
```

## 8. 相位回归头

相位回归头：

```text
AdaptiveAvgPool2d(1)
Flatten
Linear(256 -> 256)
ReLU
Dropout(p=0.25)
Linear(256 -> 12)
```

输出顺序固定为：

```text
[sin(phi_1), cos(phi_1), ..., sin(phi_6), cos(phi_6)]
```

## 9. 参数量

| 模块 | 参数量 |
| --- | ---: |
| `focal_encoder` | 2,795,744 |
| `befocal_encoder` | 2,795,744 |
| `fusion_gate` | 98,688 |
| `channel_attention` | 8,464 |
| `fc` 相位回归头 | 68,876 |
| **总计** | **5,767,516** |

对照模型：

| 模型 | 输入方式 | 参数量 |
| --- | --- | ---: |
| Cycle41 `deep_residual_cnn` | 双通道直接堆叠 | 11,341,100 |
| Cycle42 `dual_plane_fusion_cnn` | 双分支门控融合 | 5,767,516 |

Cycle42 的收益不是来自更大的参数量，而是来自更合适的焦平面/焦前信息融合方式。

## 10. 训练参数

Cycle42 正式训练配置：

| 参数 | 值 |
| --- | --- |
| 样本数 | 10,000 |
| train / val / test | 7,000 / 1,500 / 1,500 |
| 输入尺寸 | `2 x 160 x 160` |
| epoch | 30 |
| batch size | 32 |
| optimizer | Adam |
| 初始学习率 | `1e-3` |
| scheduler | CosineAnnealingLR |
| 最小学习率 | `1e-6` |
| weight decay | `1e-5` |
| seed | `20260616` |
| `lambda_phy` | `0.05` |
| `lambda_comp` | `0.5` |
| `comp_warmup_epochs` | `0` |
| `lambda_unit` | `0.0` |
| `focal_plane_index` | `0` |

训练总损失：

```text
L_total = L_phase
        + lambda_phy * L_farfield
        + lambda_comp * L_compensation
        + lambda_unit * L_unit_circle
```

其中：

| 损失项 | 含义 |
| --- | --- |
| `L_phase` | 预测 sin/cos 标签与真实 sin/cos 标签的 MSE |
| `L_farfield` | 用预测相位重建焦平面远场，与输入焦平面图像保持一致 |
| `L_compensation` | 基于未归一化远场的 Strehl / 主瓣补偿质量损失 |
| `L_unit_circle` | `sin^2 + cos^2 = 1` 单位圆约束，本模型未启用 |

## 11. Checkpoint 选择

训练脚本同时保存四类 checkpoint：

```text
best_rmse
best_comp
best_strehl
best_main_lobe
```

当前补偿质量主模型选择：

```text
models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth
```

关键训练结果：

| checkpoint | epoch | 测试 RMSE(rad) |
| --- | ---: | ---: |
| best RMSE | 28 | 0.974026 |
| best comp | 29 | 0.973058 |
| best Strehl | 29 | 0.973058 |
| best main-lobe | 27 | 0.974182 |

统一 paired 评估中，`best_rmse` 同时给出最好的主瓣能量、Strehl、合成效率和较好的残余 RMSE，因此被选为当前补偿质量主模型。

## 12. 当前性能摘要

256 样本 paired 评估：

| 模型 | 主瓣能量占比 | Strehl | 合成效率 | 残余 RMSE(rad) |
| --- | ---: | ---: | ---: | ---: |
| Cycle37 相位/RMSE 主模型 | 0.520248 | 0.652884 | 0.787546 | 0.865573 |
| Cycle41 简单双通道模型 | 0.524967 | 0.670898 | 0.795033 | 0.896828 |
| Cycle42 双分支融合模型 | 0.525304 | 0.682690 | 0.795854 | 0.892309 |

Cycle43 噪声扫描显示，Cycle42 在 `sigma >= 0.005` 的输入噪声下也优于 Cycle41；但 `sigma=0.002` 有一次局部退化，后续如继续优化可考虑噪声增强训练或门控正则。
