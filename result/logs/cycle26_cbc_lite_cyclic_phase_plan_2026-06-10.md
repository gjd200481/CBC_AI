# Cycle 26：CBC 自研轻量网络与周期相位损失计划

日期：2026-06-10

## 背景

在对 Xie et al., Scientific Reports 2024 进行对比后，可以采纳的关键思想包括：

- 相位变量具有周期性，普通 MSE 会在 `-pi/pi` 边界附近放大物理上等价的误差。
- 单张远场/离焦强度图可以作为隐藏相位状态的代理观测。
- 训练过程应保存最佳验证 checkpoint，而不是只使用最终 epoch。

但本项目不能直接移用该文的 MobileNetV3-Small 结构。当前路线调整为：保留周期相位损失思想，在本项目已有 7 光束仿真、sin/cos 标签、傅里叶物理约束和补偿评估链条上，迭代一个面向 CBC 远场条纹图像的自研候选模型。

## 本周期代码调整

新增或扩展内容：

- `train/phase_metrics.py`
  - 新增 `cyclic_phase_loss_from_sin_cos`。
  - 新增 `CyclicPhaseLoss`。
  - 新增 `build_phase_loss()`，支持 `mse`、`cyclic`、`cyclic_unit`。

- `train/models.py`
  - 新增 `CBCPhaseLiteCNN`。
  - 新增 `SpatialChannelGate`，用于空间/通道门控。
  - 新增 `SeparableResidualBlock`，用于轻量深度可分离残差特征提取。
  - 新增 `MultiScalePhaseHead`，用于汇聚不同尺度的条纹特征。
  - `build_phase_model()` 新增模型名 `cbc_lite_cnn`。

- 训练入口
  - `train/train_seven_beam_baseline.py` 支持 `--model-name`、`--phase-loss`、`--unit-loss-weight`。
  - `train/sweep_seven_beam_architecture.py` 支持 `--phase-loss`、`--unit-loss-weight`。
  - `train/train_seven_beam_physics_constrained_cnn.py` 支持 `--phase-loss`、`--unit-loss-weight`。

- 运行脚本
  - 新增 `scripts/run_cycle26_gpu_cbc_lite.ps1`。

## 与 Xie et al. 的关系

采纳：

- 周期相位损失思想。
- 7 光束相对相位回归任务的评价视角。
- 最佳验证 checkpoint 与闭环补偿指标需要分开记录。

不照搬：

- 不使用 MobileNetV3-Small 作为项目主结构。
- 不把文献中的实验采集配置直接当作本项目数据流程。
- 不仅追求相位 RMSE，还要验证主瓣能量、Strehl 比、合成效率和补偿后残余相位。

本项目创新点定位：

- 面向 CBC 远场条纹图像的轻量残差分离卷积结构。
- 空间/通道门控用于强调干涉条纹和能量分布区域。
- 多尺度池化头同时保留全局能量分布和局部条纹信息。
- 与傅里叶远场物理一致性损失、补偿指标评估链条结合。

## 烟测结果

本地 CPU 小样本验证命令：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python train\sweep_seven_beam_architecture.py --models cbc_lite_cnn --epochs 1 --batch-size 8 --max-samples 16 --device cpu --phase-loss cyclic --no-save-model --experiment-tag cycle26_cbc_lite_smoke --history-dir result\metrics\cycle26_cbc_lite_smoke --summary-csv result\metrics\cycle26_cbc_lite_smoke_2026-06-10.csv --figure-path result\figures\cycle26_cbc_lite_smoke_2026-06-10.png
```

输出摘要：

```text
model: cbc_lite_cnn
phase_loss: cyclic
train samples: 11
val samples: 2
test samples: 3
epoch: 1
train loss: 1.848251
val RMSE: 1.870283 rad
test RMSE: 1.741463 rad
```

该结果只用于验证代码路径可运行，不用于论文结论。

## GPU 下一步

推荐运行：

```powershell
.\scripts\run_cycle26_gpu_cbc_lite.ps1 -Epochs 50 -BatchSize 64 -LearningRate 0.001 -NumWorkers 2 -Seed 20260612 -PhaseLoss cyclic
```

如周期损失训练不稳定，补跑：

```powershell
.\scripts\run_cycle26_gpu_cbc_lite.ps1 -Epochs 50 -BatchSize 64 -LearningRate 0.001 -NumWorkers 2 -Seed 20260612 -PhaseLoss cyclic_unit
```

## 判断标准

优先比较：

- `best_checkpoint_test_rmse_rad`
- 逐通道 RMSE 是否均衡
- 是否优于 `residual_cnn_best` 的 `0.992071 rad`
- 是否优于 `residual_cnn + physics loss, lambda_phy=0.05` 的 `0.983128 rad`

如果相位 RMSE 有优势，再进入统一补偿评估：

- 主瓣能量占比
- Strehl 比
- 合成效率
- 峰值旁瓣比
- 补偿后残余相位 RMSE

## 当前结论

Cycle 26 的重点不再是复刻文献网络，而是把文献中的周期相位建模思想转化为本项目自己的模型创新。`cbc_lite_cnn + cyclic phase loss` 是下一轮需要在 RTX 3060 上验证的候选路线。
