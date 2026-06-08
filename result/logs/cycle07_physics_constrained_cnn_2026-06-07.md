# Cycle 07 第一版物理约束 CNN 实验记录

## 任务目标

训练第一版物理约束 CNN，将普通相位监督损失和傅里叶光学远场一致性损失组合：

```text
L_total = L_phase + lambda_phy * L_farfield
```

本周期目标是验证物理约束训练流程可以完整跑通，并获得第一版 `physics_constrained_cnn` 结果。完整的 `lambda_phy` 权重消融留到 Cycle 08。

## 新增脚本

### `train/train_physics_constrained_cnn.py`

主要功能：

- 读取 Cycle 03 主静态双光束数据集。
- 使用 `SimplePhaseCNN` 输出 `[sin(phi), cos(phi)]`。
- 使用 `MSELoss` 计算相位监督损失。
- 使用 `FarFieldConsistencyLoss` 计算远场物理一致性损失。
- 记录每个 epoch 的：
  - 总损失。
  - 相位损失。
  - 远场重建损失。
  - 验证集相位 RMSE 和 MAE。
- 保存模型、训练曲线、测试集 summary 和结果图。

## 数据集

- 数据集名称：`main_clean_two_beam`
- 图像文件：`dataset/two_beam/main_static/images_main_clean_two_beam.npy`
- 标签文件：`dataset/two_beam/main_static/labels_main_clean_two_beam.npy`
- 样本数：2000
- 划分：训练集 1400，验证集 300，测试集 300
- 噪声强度：0

## 训练参数

- 模型：`SimplePhaseCNN`
- 物理模块：`TwoBeamFourierOptics`
- `lambda_phy`：0.1
- epoch：10
- batch size：32
- learning rate：0.001
- seed：20260608
- device：CPU

## 训练命令

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -m train.train_physics_constrained_cnn `
  --epochs 10 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --lambda-phy 0.1 `
  --seed 20260608 `
  --model-path models\physics_cnn_lambda_0.1_main_clean.pth `
  --metrics-path result\metrics\physics_cnn_lambda_0.1_main_clean_2026-06-07.csv `
  --summary-path result\metrics\physics_cnn_lambda_0.1_main_clean_summary_2026-06-07.csv `
  --figure-path result\figures\physics_cnn_lambda_0.1_main_clean_2026-06-07.png `
  --no-plot
```

说明：`KMP_DUPLICATE_LIB_OK` 只用于本地 Windows 环境绕过 OpenMP 重复初始化提示，没有写入源码。

## 输出文件

- 模型权重：`models/physics_cnn_lambda_0.1_main_clean.pth`
- 训练指标：`result/metrics/physics_cnn_lambda_0.1_main_clean_2026-06-07.csv`
- 测试摘要：`result/metrics/physics_cnn_lambda_0.1_main_clean_summary_2026-06-07.csv`
- 结果图：`result/figures/physics_cnn_lambda_0.1_main_clean_2026-06-07.png`

## 训练过程摘要

| epoch | train_total | train_phase | train_farfield | val_rmse(rad) | val_farfield |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4905078 | 0.4904673 | 4.053762e-04 | 1.1290103 | 2.131611e-04 |
| 2 | 0.0834052 | 0.0834015 | 3.704529e-05 | 0.0648358 | 1.010920e-06 |
| 3 | 0.0014341 | 0.0014340 | 3.713569e-07 | 0.0134371 | 5.290328e-08 |
| 4 | 0.0001994 | 0.0001993 | 3.192641e-08 | 0.0092855 | 2.391731e-08 |
| 8 | 0.0000462 | 0.0000462 | 9.869678e-09 | 0.0058031 | 1.011197e-08 |
| 10 | 0.0000445 | 0.0000445 | 8.840490e-09 | 0.0057887 | 9.710685e-09 |

## 测试集结果

| 指标 | 数值 |
|---|---:|
| RMSE(rad) | 0.0057821637 |
| RMSE(deg) | 0.3312935766 |
| MAE(rad) | 0.0043467321 |
| MAE(deg) | 0.2490494011 |
| Mean error(rad) | 0.0006691354 |
| Mean error(deg) | 0.0383386322 |
| Phase loss | 3.9130254554e-05 |
| Far-field loss | 9.3537719910e-09 |
| Total loss | 3.9131189624e-05 |

## 与普通 CNN baseline 的初步对比

| 方法 | epoch | lambda_phy | RMSE(rad) | RMSE(deg) | 说明 |
|---|---:|---:|---:|---:|---|
| 普通 CNN baseline | 20 | 0 | 0.0037421337 | 0.2144084693 | 只使用相位监督损失 |
| 物理约束 CNN 初版 | 10 | 0.1 | 0.0057821637 | 0.3312935766 | 使用相位监督 + 远场一致性损失 |

## 结论

第一版物理约束 CNN 已经完整跑通，远场一致性损失能随训练快速下降，最终测试集远场重建 MSE 约为 `9.35e-9`。相位 RMSE 目前略高于普通 CNN baseline，但这次物理约束模型只训练了 10 个 epoch，而普通 CNN baseline 训练了 20 个 epoch。

当前结论应谨慎表述为：

- 物理约束训练流程可用。
- `lambda_phy=0.1` 下模型能同时降低相位误差和远场重建误差。
- 干净双光束数据上，普通 CNN 已经很强，物理约束优势需要在后续噪声、振幅失配、位置偏移或权重消融中进一步体现。

下一步 Cycle 08 应完成 `lambda_phy` 消融实验，例如 `0, 0.01, 0.05, 0.1, 0.5, 1.0`，并统一训练轮数后比较相位 RMSE、远场重建误差和训练稳定性。
