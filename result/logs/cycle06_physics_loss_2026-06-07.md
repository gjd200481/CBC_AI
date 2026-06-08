# Cycle 06 傅里叶光学物理一致性损失记录

## 任务目标

实现可反向传播的傅里叶光学物理一致性模块，使后续物理约束 CNN 能够在训练时同时优化：

```text
L_total = L_phase + lambda_phy * L_farfield
```

其中 `L_phase` 是相位 `sin/cos` 监督损失，`L_farfield` 是由预测相位重建远场后与输入远场之间的误差。

## 新增文件

### `train/physics_loss.py`

主要内容：

- `crop_center_torch`
  - torch 版本中心裁剪函数。
  - 支持 `[B, H, W]` 和 `[B, C, H, W]`。

- `normalize_intensity`
  - 按每张图最大值归一化远场光强。

- `TwoBeamFourierOptics`
  - 双光束可微分傅里叶光学模型。
  - 根据网络预测的 `[sin(phi), cos(phi)]` 解码相位。
  - 重建双光束近场复振幅。
  - 使用 `torch.fft.fft2` 和 `torch.fft.fftshift` 得到远场光强。
  - 归一化并裁剪中心区域，输出 `[B, crop_size, crop_size]`。

- `FarFieldConsistencyLoss`
  - 计算重建远场与输入远场之间的 MSE 或 L1 损失。
  - 当前默认使用 MSE。

## 参数对齐

torch 版物理模块默认参数与 `simulation/common/two_beam_core.py` 保持一致：

- `num_points = 256`
- `window_size = 10e-3`
- `waist = 0.5e-3`
- `beam_distance = 1.5e-3`
- `crop_size = 160`

这保证了 Cycle 03 生成的数据集可以直接用于物理一致性验证。

## 验证 1：真实标签重建远场

验证命令读取 `main_clean_two_beam` 数据集前 16 个样本，用真实 `labels` 重建远场，并与原始图像比较。

结果：

```text
recon_shape = (16, 160, 160)
mse_true_label = 1.0808640024e-16
max_abs_true_label = 4.7683715820e-07
```

结论：真实标签代入 torch 版傅里叶模型后，几乎可以精确重建 NumPy 仿真生成的远场图像，说明两套传播模型口径一致。

## 验证 2：物理损失可反向传播

使用真实标签计算损失：

```text
backward_loss = 1.0808640024e-16
grad_norm = 1.2469546229e-11
```

由于真实标签几乎完美重建远场，损失和梯度都接近 0，符合预期。

## 验证 3：扰动预测下梯度有效

给 `labels` 加入小扰动，模拟网络预测不准确时的情况。

结果：

```text
perturbed_loss = 9.2805896657e-07
perturbed_grad_norm = 6.9834600254e-06
grad_is_finite = True
```

结论：当预测相位存在偏差时，远场物理一致性损失非零，并且能够产生有限梯度，可用于后续物理约束 CNN 训练。

## 备注

在 Windows 环境中运行验证脚本时，PyTorch 与 NumPy 可能触发 OpenMP 运行库重复初始化提示。验证时临时设置：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

该设置只用于本地验证，没有写入项目源码。

## 结论

Cycle 06 的核心目标已完成：傅里叶光学物理一致性损失已经实现，并通过了数值一致性与反向传播检查。下一步 Cycle 07 可以在普通 CNN baseline 的训练流程中加入：

```text
loss = phase_loss + lambda_phy * farfield_loss
```

并测试不同 `lambda_phy` 下的训练稳定性和相位 RMSE。
