# Cycle42 双分支焦前/焦平面融合模型说明

本文档说明当前补偿质量主模型 `cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth` 的结构、参数设置、卷积与池化设计、融合方式和训练配置。

## 1. 当前主模型

当前补偿质量主模型：

```text
models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth
```

模型类：

```text
DualPlaneFusionPhaseCNN
model_name = dual_plane_fusion_cnn
```

代码位置：

```text
train/models.py
```

当前任务输入输出：

| 项目 | 设置 |
| --- | --- |
| 输入图像 | 7cm 双平面远场强度图 |
| 输入张量 | `[B, 2, 160, 160]` |
| 第 1 个通道 | 焦平面图像 |
| 第 2 个通道 | 焦前/离焦 7cm 图像 |
| 输出张量 | `[B, 12]` |
| 输出含义 | 6 路外圈相对相位的 `[sin(phi_i), cos(phi_i)]` |
| 中心通道相位 | 固定为参考相位 0 |
| 参数量 | `5,767,516` |

## 2. 设计动机

Cycle41 使用的是简单双通道堆叠方式：把焦平面和焦前图像作为普通输入通道送入同一个 `DeepResidualPhaseCNN`。这种方式能利用多平面信息，但没有显式区分两类观测的物理含义。

Cycle42 改为双分支结构：

```text
focal plane image   -> focal encoder
befocal image       -> befocal encoder
two feature maps    -> gated fusion
fused feature       -> channel attention
phase head          -> 12-dim sin/cos phase output
```

这个设计对应 Hou 2019 和 Xie 2024 的文献启发：非焦平面/焦前图像并不是焦平面图像的普通冗余通道，而是经过不同传播距离后的物理观测，可能携带更局部、更可分的相位线索。

## 3. 总体结构

模型由 5 个主要部分组成：

| 模块 | 参数量 | 作用 |
| --- | ---: | --- |
| `focal_encoder` | `2,795,744` | 编码焦平面图像 |
| `befocal_encoder` | `2,795,744` | 编码焦前 7cm 图像 |
| `fusion_gate` | `98,688` | 学习每个特征通道更偏向焦平面还是焦前 |
| `channel_attention` | `8,464` | 对融合后的通道重要性再加权 |
| `fc` | `68,876` | 全局池化后回归 12 维相位编码 |
| **总计** | **`5,767,516`** | 当前 Cycle42 主模型 |

作为对照，Cycle41 的简单双通道 `deep_residual_cnn` 参数量为 `11,341,100`。因此 Cycle42 的提升不是来自更大的模型容量，而是来自更合适的焦平面/焦前信息融合结构。

## 4. 单分支 Encoder 结构

两个分支使用相同结构，但参数不共享：

```text
PlaneFeatureEncoder(input_channels=1, base_channels=32)
```

每个 encoder 的输入是 `[B, 1, 160, 160]`，输出是 `[B, 256, 5, 5]`。

特征图尺寸变化：

| 阶段 | 操作 | 输出尺寸 |
| --- | --- | --- |
| input | 单通道图像 | `[B, 1, 160, 160]` |
| `conv1` | `7x7 conv, stride=2, padding=3, 1 -> 32` | `[B, 32, 80, 80]` |
| `pool` | `3x3 max pool, stride=2, padding=1` | `[B, 32, 40, 40]` |
| `layer1` | 2 个残差块，通道 `32 -> 32` | `[B, 32, 40, 40]` |
| `layer2` | 2 个残差块，首块 stride=2，通道 `32 -> 64` | `[B, 64, 20, 20]` |
| `layer3` | 2 个残差块，首块 stride=2，通道 `64 -> 128` | `[B, 128, 10, 10]` |
| `layer4` | 2 个残差块，首块 stride=2，通道 `128 -> 256` | `[B, 256, 5, 5]` |

## 5. 残差块结构

每个 `BasicResBlock` 包含：

```text
3x3 Conv -> BatchNorm -> ReLU
3x3 Conv -> BatchNorm
shortcut add
ReLU
```

当输入输出通道数不同，或 stride 不等于 1 时，shortcut 分支使用：

```text
1x1 Conv(stride=stride) -> BatchNorm
```

因此降采样发生在 `layer2/layer3/layer4` 的第一个残差块中。主分支用 `3x3 stride=2` 卷积降采样，shortcut 用 `1x1 stride=2` 保持尺寸一致。

## 6. 池化设计

模型中有三类池化：

1. 初始下采样池化：

```text
MaxPool2d(kernel_size=3, stride=2, padding=1)
```

它位于每个 encoder 的 stem 后，把 `80x80` 特征图降为 `40x40`。

2. 融合门控池化：

```text
AdaptiveAvgPool2d(1)
```

它把 `[B, 512, 5, 5]` 的焦平面/焦前拼接特征压缩为 `[B, 512]`，用于生成通道级融合权重。

3. 输出头池化：

```text
AdaptiveAvgPool2d(1)
```

它把融合后的 `[B, 256, 5, 5]` 特征压缩为 `[B, 256]`，再进入全连接相位回归头。

## 7. 门控融合

两个分支先得到：

```text
focal   = focal_encoder(x[:, 0:1])    # [B, 256, 5, 5]
befocal = befocal_encoder(x[:, 1:2])  # [B, 256, 5, 5]
```

然后拼接：

```text
concat = cat([focal, befocal], dim=1) # [B, 512, 5, 5]
```

`fusion_gate` 的结构：

```text
AdaptiveAvgPool2d(1)
Flatten
Linear(512 -> 128)
ReLU
Linear(128 -> 256)
Sigmoid
```

得到的 `gate` 是 `[B, 256]`，扩展到 `[B, 256, 1, 1]` 后执行：

```text
fused = gate * focal + (1 - gate) * befocal
```

物理含义：每个特征通道都可以自行选择更依赖焦平面特征，还是更依赖焦前特征。`gate` 接近 1 表示偏向焦平面，接近 0 表示偏向焦前。

## 8. 通道注意力

融合后再经过一个轻量通道注意力：

```text
AdaptiveAvgPool2d(1)
Flatten
Linear(256 -> 16)
ReLU
Linear(16 -> 256)
Sigmoid
```

然后：

```text
fused = fused * attention
```

该模块不区分焦平面/焦前来源，而是在融合后的表示上重新选择对相位反演最重要的通道。

## 9. 相位回归头

输出头结构：

```text
AdaptiveAvgPool2d(1)
Flatten
Linear(256 -> 256)
ReLU
Dropout(p=0.25)
Linear(256 -> 12)
```

输出 12 个数，对应 6 路相对相位：

```text
[sin(phi_1), cos(phi_1), ..., sin(phi_6), cos(phi_6)]
```

后续评估时通过：

```text
phi_i = atan2(sin(phi_i), cos(phi_i))
```

恢复相位角，并计算补偿后的残余相位：

```text
residual = wrap(true_phase - predicted_phase)
```

## 10. 训练配置

Cycle42 正式训练配置：

| 参数 | 值 |
| --- | --- |
| 数据集 | `dataset/seven_beam/multiplane_0_-0.07/` |
| 图像文件 | `images_multiplane_7cm.npy` |
| 标签文件 | `labels_multiplane_7cm.npy` |
| 输入尺寸 | `2 x 160 x 160` |
| 样本数 | `10000` |
| train/val/test | `7000 / 1500 / 1500` |
| epoch | `30` |
| batch size | `32` |
| optimizer | Adam |
| 初始学习率 | `1e-3` |
| scheduler | CosineAnnealingLR |
| 最小学习率 | `1e-6` |
| weight decay | `1e-5` |
| `lambda_phy` | `0.05` |
| `lambda_comp` | `0.5` |
| `lambda_unit` | `0.0` |
| `comp_warmup_epochs` | `0` |
| checkpoint 选择 | best RMSE / best comp / best Strehl / best main-lobe 同时保存 |

最佳 checkpoint：

```text
best_rmse epoch = 28
selection_metric = val_rmse_rad
selection_value = 0.989105
test_rmse_rad = 0.974026
```

逐通道测试 RMSE：

| 通道 | RMSE(rad) |
| --- | ---: |
| 1 | 0.975283 |
| 2 | 0.985191 |
| 3 | 0.974490 |
| 4 | 0.975554 |
| 5 | 0.958664 |
| 6 | 0.974786 |

## 11. 损失函数

训练总损失：

```text
loss = phase_loss + lambda_phy * farfield_loss + lambda_comp * comp_loss + lambda_unit * unit_loss
```

当前 Cycle42 设置：

```text
lambda_phy = 0.05
lambda_comp = 0.5
lambda_unit = 0.0
```

各项含义：

| 损失项 | 含义 |
| --- | --- |
| `phase_loss` | 预测 sin/cos 标签与真实 sin/cos 标签的 MSE |
| `farfield_loss` | 用预测相位重建远场，与输入焦平面图像保持一致 |
| `comp_loss` | 基于未归一化远场的 Strehl / 主瓣补偿质量损失 |
| `unit_loss` | sin/cos 单位圆约束，本周期未启用 |

## 12. Cycle42 最终补偿指标

统一 256 样本 paired 评估：

| 模型 | 主瓣能量占比 | Strehl | 合成效率 | 残余相位 RMSE |
| --- | ---: | ---: | ---: | ---: |
| Cycle37 相位/RMSE 主模型 | 0.520248 | 0.652884 | 0.787546 | 0.865573 |
| Cycle41 简单双通道主模型 | 0.524967 | 0.670898 | 0.795033 | 0.896828 |
| Cycle42 双分支融合主模型 | 0.525304 | 0.682690 | 0.795854 | 0.892309 |

结论：

1. Cycle42 是当前补偿质量主模型。
2. 它以更小参数量超过 Cycle41 的简单双通道堆叠。
3. 它提升了主瓣能量、Strehl 和合成效率，并略微改善 Cycle41 的残余相位 RMSE。
4. 但相位/残余 RMSE 仍不如 Cycle37，因此当前仍保留双主模型策略。

## 13. 当前双主模型

补偿质量主模型：

```text
models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth
```

相位/残余 RMSE 主模型：

```text
models/cycle37_multiplane_7cm_lambda_comp0p3_30epoch.pth
```

后续 Cycle43 需要用 attribution 和噪声鲁棒性验证 Cycle42 的正结果是否具有物理解释和稳定性。
