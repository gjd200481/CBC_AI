# CBC_AI 文件命名规范

本文档用于统一项目文件夹和文件命名，避免后续实验越做越乱。

## 1. 总体原则

- Python 文件使用小写字母和下划线：`seven_beam_dataset.py`。
- 实验结果文件包含周期、任务、日期：`cycle14_seven_beam_lambda_sweep_2026-06-08.csv`。
- 数据集前缀体现任务、扰动和用途：`main_clean_seven_beam`。
- 模型权重只放在 `models/`，不提交 Git。
- 演示脚本统一放在 `examples/`。
- 早期验证脚本统一放在 `simulation/static/legacy/`。

## 2. 目录命名

| 目录 | 用途 |
| --- | --- |
| `simulation/common/` | 公共光学仿真模块 |
| `simulation/static/` | 静态远场数据集生成脚本 |
| `simulation/static/legacy/` | 早期验证和演示脚本 |
| `simulation/dynamic/` | 动态序列数据脚本，当前为拓展备用 |
| `train/` | 数据读取、模型、损失、训练和评估脚本 |
| `examples/` | 推理和快速评估示例 |
| `dataset/` | 本地数据集，不提交 Git |
| `models/` | 本地模型权重，不提交 Git |
| `result/logs/` | 实验记录 Markdown |
| `result/metrics/` | 指标 CSV |
| `result/figures/` | 结果图 |
| `paper/` | 论文 PDF、中文期刊和文献阅读结果 |

## 3. 仿真脚本命名

主数据生成脚本使用：

```text
generate_<beam_config>_<dataset_type>.py
```

示例：

```text
generate_two_beam_dataset.py
generate_seven_beam_dataset.py
generate_two_beam_noise_robustness_dataset.py
generate_two_beam_amplitude_mismatch_dataset.py
generate_two_beam_sequence_dataset.py
```

公共仿真模块使用：

```text
<beam_config>_core.py
```

示例：

```text
two_beam_core.py
multi_beam_core.py
```

## 4. 训练脚本命名

训练脚本使用：

```text
train_<beam_config>_<model_type>.py
```

示例：

```text
train_seven_beam_baseline.py
train_seven_beam_physics_constrained_cnn.py
train_physics_constrained_cnn.py
```

评估脚本使用：

```text
evaluate_<experiment_type>.py
```

示例：

```text
evaluate_noise_robustness.py
evaluate_amplitude_mismatch.py
evaluate_two_beam.py
```

消融脚本使用：

```text
sweep_<beam_config>_<factor>.py
```

示例：

```text
sweep_seven_beam_lambda.py
```

## 5. 数据集命名

数据集目录建议使用：

```text
dataset/<beam_config>/<experiment_name>/
```

示例：

```text
dataset/seven_beam/main_static/
dataset/seven_beam/cycle11_smoke/
dataset/two_beam/noise_robustness/
dataset/two_beam/amplitude_mismatch/
```

`.npy` 文件命名：

```text
images_<prefix>.npy
labels_<prefix>.npy
phases_<prefix>.npy
config_<prefix>.json
```

示例：

```text
images_main_clean_seven_beam.npy
labels_main_clean_seven_beam.npy
phases_main_clean_seven_beam.npy
config_main_clean_seven_beam.json
```

## 6. 结果文件命名

周期结果使用：

```text
cycle<cycle_id>_<experiment_name>_<YYYY-MM-DD>.<ext>
```

示例：

```text
cycle13_seven_beam_physics_cnn_2026-06-08.md
cycle14_seven_beam_lambda_sweep_2026-06-08.csv
cycle14_seven_beam_lambda_sweep_2026-06-08.png
```

模型训练结果使用：

```text
<model_name>_<dataset_name>_<YYYY-MM-DD>.csv
<model_name>_<dataset_name>_summary_<YYYY-MM-DD>.csv
<model_name>_<dataset_name>_<YYYY-MM-DD>.png
```

示例：

```text
baseline_cnn_main_clean_seven_beam_2026-06-08.csv
physics_cnn_lambda_0.1_main_clean_seven_beam_summary_2026-06-08.csv
```

## 7. 模型权重命名

模型权重只保存在本地 `models/`：

```text
<model_name>_<dataset_name>_<YYYY-MM-DD>.pth
```

示例：

```text
baseline_cnn_main_clean_seven_beam_2026-06-08.pth
physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.pth
```

## 8. 后续新增文件检查清单

新增文件前先确认：

- 是否属于源码、数据、权重、结果、文献或演示。
- 文件名是否能看出任务、模型、扰动和日期。
- 是否应该提交 Git。
- 是否需要同步更新 `README.md`、`KEY_FILES.md` 或 `PROJECT_STATUS.md`。
