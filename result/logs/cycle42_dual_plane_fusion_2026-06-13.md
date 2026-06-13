# Cycle 42：焦平面/焦前双分支特征融合

日期：2026-06-13

## 研究目的

Cycle 42 延续 2026-06-12 的路线修订：优化方向从“更大的模型”转向“更正确的物理指标 + 更聪明的焦前/焦平面信息融合”。Cycle 41 已修复未归一化 Strehl / 主瓣指标，使训练期 checkpoint 选择与最终补偿评估一致；本周期在此基础上验证焦平面/焦前双分支融合是否优于简单多平面通道堆叠。

文献依据：

1. Hou 2019 指出非焦平面强度图比焦平面图更适合相位控制，并且结果应以 Strehl、PIB、远场主瓣等下游光学质量评价。
2. Xie 2024 在七束 CBC 实验中显示焦前 Camera A 的相位预测误差约 `0.26 rad`，优于焦平面 Camera B 的约 `0.41 rad`；attribution map 也显示焦前图像相位线索更局部、更可分。
3. Mills 2022 强调相位预测必须落到 power-in-bucket、物理可达性和噪声韧性等任务指标，而不能只报告网络误差。

## 代码修改

修改/新增文件：

```text
train/models.py
train/train_multiplane.py
train/plot_cycle42_literature_figure.py
```

关键修改：

1. 新增 `PlaneFeatureEncoder`：用于单个观测平面的轻量残差编码。
2. 新增 `DualPlaneFusionPhaseCNN`：焦平面和焦前图像分别进入两个 encoder，再通过门控融合得到共享特征，最后回归 6 路外圈相位的 sin/cos 表示。
3. `build_phase_model()` 注册 `dual_plane_fusion_cnn`。
4. `train_multiplane.py` 新增 `--model-name` 参数，并在 checkpoint 中保存真实 `model_name`，保证后续评估脚本能自动重建双分支模型。
5. 新增 Cycle42 文献风格出图脚本，将方法结构、训练轨迹、下游补偿指标和典型远场图样组织成一张综合证据图。

模型规模：

```text
deep_residual_cnn(2-channel): 11,341,100 parameters
dual_plane_fusion_cnn(2-channel): 5,767,516 parameters
```

因此 Cycle 42 不是靠更大模型取胜，而是在更小模型容量下验证更合理的焦平面/焦前信息融合结构。

## 验证

### 1. 编译与前向自检

已通过：

```text
python -m py_compile train\models.py train\train_multiplane.py train\plot_cycle42_literature_figure.py
```

前向自检：

```text
deep_residual_cnn params 11341100 out (2, 12)
dual_plane_fusion_cnn params 5767516 out (2, 12)
```

### 2. 1 epoch smoke

命令使用 7cm 双平面数据、`dual_plane_fusion_cnn`、`lambda_phy=0.05`、`lambda_comp=0.5`、`lambda_unit=0.0`。

输出：

```text
Epoch 001 | val_rmse=1.404678 | strehl=0.488591 | main=0.456351 | eff=0.684041
Best-RMSE model saved: models\cycle42_dual_plane_fusion_7cm_smoke1_rmse.pth
Best-comp model saved: models\cycle42_dual_plane_fusion_7cm_smoke1_comp.pth
Best-Strehl model saved: models\cycle42_dual_plane_fusion_7cm_smoke1_strehl.pth
Best-main-lobe model saved: models\cycle42_dual_plane_fusion_7cm_smoke1_main_lobe.pth
```

结论：训练入口、四类 checkpoint 保存、GPU 路径和模型加载链路正常。

## 正式 30 epoch 实验

训练命令使用：

```text
python train\train_multiplane.py
  --model-name dual_plane_fusion_cnn
  --image-path dataset\seven_beam\multiplane_0_-0.07\images_multiplane_7cm.npy
  --label-path dataset\seven_beam\multiplane_0_-0.07\labels_multiplane_7cm.npy
  --epochs 30
  --batch-size 32
  --lambda-phy 0.05
  --lambda-comp 0.5
  --lambda-unit 0.0
```

关键输出：

```text
result/logs/cycle42_dual_plane_fusion_7cm_30epoch_console_2026-06-13.log
result/metrics/cycle42_dual_plane_fusion_7cm_30epoch_history.csv
models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth
models/cycle42_dual_plane_fusion_7cm_best_comp_30epoch.pth
models/cycle42_dual_plane_fusion_7cm_best_strehl_30epoch.pth
models/cycle42_dual_plane_fusion_7cm_best_main_lobe_30epoch.pth
```

训练期关键点：

```text
best RMSE: epoch 28, test RMSE=0.974026 rad
best comp: epoch 29, test RMSE=0.973058 rad
best Strehl: epoch 29, test RMSE=0.973058 rad
best main-lobe: epoch 27, test RMSE=0.974182 rad
```

验证集峰值：

```text
val_rmse = 0.989105 rad at epoch 28
val_strehl = 0.662112 at epoch 29
val_main_lobe = 0.518959 at epoch 27
val_synthesis_efficiency = 0.785579 at epoch 27
```

## Paired 最终评估

统一 256 样本 paired 评估输出：

```text
result/metrics/cycle42_dual_plane_fusion_paired_summary.csv
result/metrics/cycle42_dual_plane_fusion_paired_detail.csv
result/figures/cycle42_dual_plane_fusion_paired.png
```

核心结果：

| 模型 | 主瓣能量占比 | Strehl | 合成效率 | 残余相位 RMSE |
| --- | ---: | ---: | ---: | ---: |
| before | 0.361631 | 0.384810 | 0.536195 | 1.803903 |
| comp0p3_best_rmse / Cycle37 | 0.520248 | 0.652884 | 0.787546 | 0.865573 |
| cycle41_best_strehl | 0.524967 | 0.670898 | 0.795033 | 0.896828 |
| cycle42_best_rmse | 0.525304 | 0.682690 | 0.795854 | 0.892309 |
| cycle42_best_comp / best_strehl | 0.524495 | 0.681633 | 0.794593 | 0.894897 |
| cycle42_best_main_lobe | 0.524713 | 0.680548 | 0.794941 | 0.894337 |
| ideal | 0.650631 | 1.000000 | 1.000000 | 0.000000 |

相对 Cycle 41 主模型：

```text
主瓣能量占比：0.524967 -> 0.525304  （+0.000337）
Strehl：      0.670898 -> 0.682690  （+0.011793）
合成效率：    0.795033 -> 0.795854  （+0.000821）
残余 RMSE：   0.896828 -> 0.892309  （-0.004519 rad）
```

结论：Cycle42 `best_rmse` 是正结果。它在更小参数量下同时提升了主瓣能量、Strehl、合成效率，并略微降低残余相位 RMSE。当前补偿质量主模型应从 Cycle41 更新为 Cycle42 `best_rmse`。

## 文献风格图与解释

新增综合证据图：

```text
train/plot_cycle42_literature_figure.py
result/figures/cycle42_literature_style_fusion_evidence.png
```

该图仿照 Hou 2019、Mills 2022、Xie 2024 的证据链组织方式，按“结构假设 -> 训练轨迹 -> 下游补偿指标 -> 典型远场图样”的顺序解释 Cycle42。

### 图 (a)：焦平面/焦前双分支融合结构

图 (a) 展示 Cycle42 的核心方法变化：焦平面图像和焦前图像不再作为普通输入通道直接堆叠，而是分别进入两个 encoder，再通过 gated fusion 融合。这个设计对应 Hou/Xie 的物理启发：焦平面和非焦平面图像不是同质通道，而是包含不同传播状态下的相位线索。

### 图 (b)：训练期指标轨迹

图 (b) 展示 30 epoch 中验证 RMSE、Strehl、主瓣能量占比和合成效率的变化。RMSE 在前 5-10 epoch 快速下降，随后缓慢收敛；Strehl 和合成效率同步上升，说明双分支模型的训练没有出现补偿指标失真。

### 图 (c)：下游补偿质量对比

图 (c) 对比补偿前、Cycle37 相位精度模型、Cycle41 补偿质量模型、Cycle42 双分支融合模型和理想相干状态。Cycle42 的 Strehl 达到 `0.682690`，高于 Cycle41 的 `0.670898`；主瓣能量占比和合成效率也略高于 Cycle41。

### 图 (d)：残余相位 RMSE 折中

图 (d) 显示 Cycle42 没有完全追上 Cycle37 的残余 RMSE，但相较 Cycle41 已略有改善：`0.896828 rad -> 0.892309 rad`。这说明双分支融合不仅提升了下游补偿质量，也没有进一步恶化相位残差。

### 图 (e)-(h)：典型样本远场图样

图 (e)-(h) 选取代表样本 `sample_index=228`，展示补偿前、Cycle41、Cycle42 和理想相干状态的远场图样。补偿前能量更分散；Cycle41 与 Cycle42 均明显增强中心区域；Cycle42 的中心强度和旁瓣压制更接近高质量补偿状态；理想相干图样作为理论上限参考。

## 更新后的主模型判断

补偿质量主模型更新为：

```text
models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth
```

相位/残余 RMSE 主模型暂时保持：

```text
models/cycle37_multiplane_7cm_lambda_comp0p3_30epoch.pth
```

原因：Cycle42 的补偿质量指标已经超过 Cycle41，但残余相位 RMSE 仍不如 Cycle37。因此当前仍采用“双主模型”策略：Cycle42 代表补偿质量最优，Cycle37 代表相位/残余 RMSE 最优。

## 下一步建议

下一周期建议进入 Cycle43：围绕 Cycle42 做解释性和稳健性补强，而不是立刻继续扩模型。

优先任务：

1. 对 Cycle41 与 Cycle42 做 attribution 对比，检查双分支 gated fusion 是否真的更依赖焦前局部线索。
2. 对 Cycle42 做噪声鲁棒性评估，模仿 Xie 2024 的相位噪声/power-in-bucket 证据逻辑，判断正结果是否稳定。
3. 若解释性和鲁棒性成立，则把 Cycle42 作为论文主模型；若不成立，则保留为结构正结果但降低主线权重。
