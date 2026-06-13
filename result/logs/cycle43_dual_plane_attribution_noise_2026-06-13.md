# Cycle 43: 双分支解释性与噪声鲁棒性补强

日期：2026-06-13

## 研究目的

Cycle 43 延续 Cycle 42 的正结果，验证焦平面/焦前双分支融合模型的收益是否具有可解释性和噪声稳定性。重点问题是：

1. `dual_plane_fusion_cnn` 是否真的在利用焦前图像，而不是仅靠模型容量或训练随机性取得收益。
2. Cycle42 相比 Cycle41 简单双通道堆叠的补偿质量收益，在输入强度噪声下是否稳定。
3. 若结果成立，是否可以把 Cycle42 固定为论文中的补偿质量主模型。

## 对比模型与数据

对比模型：

```text
Cycle41 simple stack:
models/cycle41_multiplane_7cm_unorm_best_strehl_30epoch.pth

Cycle42 dual-plane fusion:
models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth
```

数据：

```text
dataset/seven_beam/multiplane_0_-0.07/images_multiplane_7cm.npy
dataset/seven_beam/multiplane_0_-0.07/labels_multiplane_7cm.npy
```

输入为 7cm 双平面图像，形状为 `[N, 2, 160, 160]`，第 1 个通道为焦平面图像，第 2 个通道为焦前/离焦 7cm 图像。

## Attribution 分析

已完成 64 个样本、6 个相位通道的梯度 attribution 分析。每个模型共有 `64 x 6 = 384` 条统计记录。

输出文件：

```text
result/metrics/cycle43_attribution_cycle41_64.csv
result/metrics/cycle43_attribution_cycle42_64.csv
result/metrics/cycle43_attribution_overview_64.csv
result/figures/cycle43_attribution_cycle41_64/
result/figures/cycle43_attribution_cycle42_64/
result/figures/cycle43_attribution_overview_64.png
```

核心统计：

| 模型 | 记录数 | 平均敏感半径(px) | top 10% 能量占比 | 焦平面能量占比 | 焦前能量占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cycle41 简单双通道堆叠 | 384 | 23.956 | 0.791 | 0.526 | 0.474 |
| Cycle42 双分支融合 | 384 | 34.861 | 0.648 | 0.484 | 0.516 |

阶段判断：

- Cycle41 的梯度能量更集中，两个输入平面的平均贡献接近固定均衡，标准差很小。
- Cycle42 的平均焦前能量占比略高于焦平面，但两个平面能量占比的标准差达到约 `0.314`，说明模型不是固定偏向某一个平面，而是在不同样本和不同相位通道之间动态切换信息来源。
- 当前梯度 attribution 指标没有支持“Cycle42 一定比 Cycle41 更局部”的强结论；相反，Cycle42 的敏感区域更分散。这更适合解释为：双分支门控提供了更灵活的跨平面特征选择，而不是简单的局部化增强。

## 噪声鲁棒性扫描

运行命令：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONWARNINGS='ignore::FutureWarning'
python train\evaluate_multiplane_noise_robustness.py `
  --image-path dataset\seven_beam\multiplane_0_-0.07\images_multiplane_7cm.npy `
  --label-path dataset\seven_beam\multiplane_0_-0.07\labels_multiplane_7cm.npy `
  --model cycle41_stack=models\cycle41_multiplane_7cm_unorm_best_strehl_30epoch.pth `
  --model cycle42_fusion=models\cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth `
  --noise-levels 0 0.002 0.005 0.01 0.02 0.03 `
  --max-samples 256 `
  --batch-size 64 `
  --summary-csv result\metrics\cycle43_dual_plane_noise_robustness_summary.csv `
  --figure-path result\figures\cycle43_dual_plane_noise_robustness.png `
  --device auto
```

说明：Windows 环境下首次运行遇到 OpenMP runtime 重复初始化问题，使用 `KMP_DUPLICATE_LIB_OK=TRUE` 与单线程 MKL/OMP 后正常完成。

输出文件：

```text
result/logs/cycle43_dual_plane_noise_robustness_console_2026-06-13.log
result/metrics/cycle43_dual_plane_noise_robustness_summary.csv
result/figures/cycle43_dual_plane_noise_robustness.png
```

核心结果：

| 噪声 sigma | 模型 | 主瓣能量 | Strehl | 合成效率 | 残余 RMSE(rad) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.000 | Cycle41 | 0.524967 | 0.670898 | 0.795033 | 0.896828 |
| 0.000 | Cycle42 | 0.525304 | 0.682690 | 0.795854 | 0.892309 |
| 0.002 | Cycle41 | 0.524937 | 0.670306 | 0.794975 | 0.898973 |
| 0.002 | Cycle42 | 0.513535 | 0.624339 | 0.776075 | 0.989943 |
| 0.005 | Cycle41 | 0.444108 | 0.487559 | 0.665232 | 1.349128 |
| 0.005 | Cycle42 | 0.456474 | 0.503555 | 0.684991 | 1.272879 |
| 0.010 | Cycle41 | 0.390757 | 0.420989 | 0.581440 | 1.630248 |
| 0.010 | Cycle42 | 0.436481 | 0.471450 | 0.653156 | 1.367090 |
| 0.020 | Cycle41 | 0.373066 | 0.406717 | 0.554006 | 1.718218 |
| 0.020 | Cycle42 | 0.440190 | 0.481045 | 0.658991 | 1.363921 |
| 0.030 | Cycle41 | 0.372739 | 0.406891 | 0.553510 | 1.719810 |
| 0.030 | Cycle42 | 0.432440 | 0.470657 | 0.646836 | 1.385301 |

阶段判断：

- 无噪声时，Cycle42 延续 Cycle42 paired 评估结论，在主瓣能量、Strehl、合成效率和残余 RMSE 上均优于 Cycle41。
- 极低噪声 `sigma=0.002` 是例外：Cycle42 的补偿指标出现明显回落，而 Cycle41 基本保持不变。该点提示双分支门控可能对轻微输入扰动存在局部敏感区间。
- 从 `sigma=0.005` 到 `0.03`，Cycle42 在所有四个指标上均优于 Cycle41。以 `sigma=0.02` 为例，Cycle42 的 Strehl 为 `0.481045`，高于 Cycle41 的 `0.406717`；残余 RMSE 为 `1.363921 rad`，低于 Cycle41 的 `1.718218 rad`。
- 因此，Cycle42 的噪声稳定性结论是“中高噪声区间优于简单双通道堆叠”，但不应表述为全噪声范围单调占优。

## Cycle43 结论

1. Cycle42 可以继续作为当前补偿质量主模型。其干净输入下的补偿质量优于 Cycle41，并且在 `sigma >= 0.005` 的输入噪声下保持更好的主瓣能量、Strehl、合成效率和残余 RMSE。
2. Attribution 结果支持“双分支模型确实在自适应使用两个输入平面”，但当前梯度指标没有支持“焦前分支带来更局部 saliency”的强断言。
3. 论文中建议采用克制表述：显式焦平面/焦前融合提升了补偿质量，并在中高输入噪声下表现出更强稳定性；其可解释性证据主要体现为跨平面贡献的动态分配，而不是单一平面的绝对主导。
4. 后续若要强化解释性，可以增加门控权重统计、Integrated Gradients 或 Grad-CAM 风格分析，避免只依赖输入梯度幅值。

## 后续建议

- 将 Cycle42 固定为补偿质量主模型，Cycle37 继续作为相位/残余 RMSE 主模型。
- 论文图表可使用 Cycle42 paired 评估图、Cycle43 噪声退化曲线，以及 attribution overview 作为补充证据。
- 若继续做模型改进，优先考虑噪声增强训练或门控正则，而不是继续扩大模型参数量。
