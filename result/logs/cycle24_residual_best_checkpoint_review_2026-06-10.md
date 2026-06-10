# Cycle 24：residual_cnn 最佳 checkpoint 结果复核与下一步建议

## 结果来源

本周期合入并复核 RTX 3060 分支：

```text
origin/cycle23-residual-best-50epoch-results-20260610
```

主要结果文件：

```text
result/logs/cycle23_residual_best_50epoch_gpu_run_2026-06-10.md
result/metrics/cycle23_residual_best_50epoch_2026-06-10.csv
result/metrics/cycle23_residual_best_50epoch/residual_cnn_history.csv
result/metrics/cycle23_residual_best_50epoch/residual_cnn_summary.csv
result/figures/cycle23_residual_best_50epoch_2026-06-10.png
```

## 关键结论

`residual_cnn` 最终 epoch 与最佳 checkpoint 差异明显：

| 模型状态 | epoch | 测试 RMSE(rad) | 测试 MAE(rad) |
| --- | ---: | ---: | ---: |
| 最终 checkpoint | 50 | `1.269384` | `0.983511` |
| 最佳验证 checkpoint | 17 | `0.992071` | `0.812456` |

当前基线：

| 模型 | RMSE(rad) | MAE(rad) |
| --- | ---: | ---: |
| 普通 CNN | `1.02698` | `0.81906` |
| 物理约束 CNN | `1.02269` | `0.81642` |
| `residual_cnn_best` | `0.992071` | `0.812456` |

因此，`residual_cnn_best` 在相位 RMSE 上已经优于现有两条主线：

- 相比普通 CNN，RMSE 相对降低约 `3.40%`。
- 相比物理约束 CNN，RMSE 相对降低约 `2.99%`。

## 代码修正

已修改：

```text
train/evaluate_seven_beam_noise_robustness.py
train/evaluate_seven_beam_compensation_effect.py
train/evaluate_seven_beam_strehl.py
```

其中 `load_seven_beam_model()` 现在会读取 checkpoint 中的 `model_name` 字段，并自动构建对应网络：

```text
simple_cnn
wide_cnn
residual_cnn
```

旧模型 checkpoint 没有 `model_name` 时，默认按 `simple_cnn` 加载，保持兼容。

`evaluate_seven_beam_compensation_effect.py` 和 `evaluate_seven_beam_strehl.py` 已新增：

```text
--candidate-model
--candidate-name
```

后续将 `residual_cnn_best` 权重文件带回后，可直接作为第三个候选模型加入主瓣能量、Strehl、合成效率和补偿后残余相位 RMSE 对比。

## 下一步建议

下一步不建议继续只比较相位 RMSE，而应验证相位误差改善是否能带来远场补偿收益。建议将 3060 电脑上的最佳权重文件带回当前电脑：

```text
models/cycle23_residual_best_50epoch_residual_cnn_seven_beam_best.pth
```

然后进行以下评估：

- `residual_cnn_best` 的主瓣能量占比。
- `residual_cnn_best` 的 Strehl 比。
- `residual_cnn_best` 的合成效率。
- `residual_cnn_best` 的补偿后残余相位 RMSE。
- 与普通 CNN、物理约束 CNN、理想相干状态同表比较。

## 阶段判断

`residual_cnn_best` 可以升级为候选主模型，但还不能直接作为最终论文主模型。是否替换当前 `simple_cnn + physics loss` 主线，取决于下一步物理补偿指标是否同样领先。
