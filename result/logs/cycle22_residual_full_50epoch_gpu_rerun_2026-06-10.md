# Cycle 22：RTX 3060 上 residual_cnn 完整数据 50 epoch 复跑

## 任务背景

上一阶段已经完成 RTX 3060 长轮次训练准备，并将 Cycle 21 中快速筛选出的 `residual_cnn` 作为优先候选结构。本次任务是在带 RTX 3060 的本地机器上，按照 `GPU_TRAINING_3060.md` 的说明，对 `residual_cnn` 进行完整 7 光束数据集 50 epoch 复跑，并记录正式指标。

本次复跑的目的不是新增结构，而是验证 `residual_cnn` 在完整数据和较长训练轮次下，是否能够优于当前 7 光束普通 CNN baseline。

## 运行环境

| 项目 | 数值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| `nvidia-smi` CUDA | 12.3 |
| Python | 3.11.7，Anaconda |
| PyTorch | `2.5.1+cu121` |
| PyTorch CUDA 可用 | True |

本机初始 Python 环境缺少 PyTorch，因此先安装了 CUDA 版 PyTorch：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

安装后检查结果：

```text
torch: 2.5.1+cu121
cuda available: True
device: NVIDIA GeForce RTX 3060 Laptop GPU
cuda version: 12.1
```

## 数据集准备

训练前重新生成 7 光束干净静态数据集：

```powershell
python simulation\static\generate_seven_beam_dataset.py --num-samples 1024 --noise-sigma 0 --num-points 256 --window-size 0.01 --waist 0.0005 --beam-distance 0.0015 --crop-size 160 --seed 20260612 --output-dir dataset\seven_beam\main_static --prefix main_clean_seven_beam
```

生成结果：

| 文件 | 形状 |
| --- | --- |
| `dataset/seven_beam/main_static/images_main_clean_seven_beam.npy` | `(1024, 160, 160)` |
| `dataset/seven_beam/main_static/labels_main_clean_seven_beam.npy` | `(1024, 12)` |
| `dataset/seven_beam/main_static/phases_main_clean_seven_beam.npy` | `(1024, 6)` |
| `dataset/seven_beam/main_static/config_main_clean_seven_beam.json` | 配置文件 |

数据集文件属于本地生成产物，不提交 Git。

## 运行命令

本次使用项目提供的 PowerShell 启动脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_cycle22_gpu_residual.ps1 -Epochs 50 -BatchSize 64 -LearningRate 0.001 -NumWorkers 2
```

等价训练配置：

| 参数 | 数值 |
| --- | --- |
| 模型 | `residual_cnn` |
| 数据范围 | 完整 7 光束主数据集 |
| epoch | 50 |
| batch size | 64 |
| learning rate | 0.001 |
| device | `cuda` |
| num workers | 2 |
| pin memory | True |
| experiment tag | `cycle22_residual_full_50epoch` |

训练结束时 Anaconda 输出了 OpenMP 运行时冲突提示：

```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

该提示出现在结果文件和模型权重已经写出之后。随后使用已有 history CSV 补生成训练曲线图，并临时设置 `KMP_DUPLICATE_LIB_OK=TRUE` 避免绘图阶段再次触发同类冲突。

## 复跑结果

| 项目 | 数值 |
| --- | ---: |
| 参数量 | `1,008,492` |
| 训练样本 | `716` |
| 验证样本 | `153` |
| 测试样本 | `155` |
| 训练耗时 | `683.04 s` |
| 测试 RMSE | `1.319034 rad` |
| 测试 MAE | `1.030842 rad` |
| 测试 loss | `0.554049` |
| 最优验证 RMSE | `0.973325 rad` |
| 最终验证 RMSE | `1.219996 rad` |

逐通道测试 RMSE：

| 通道 | RMSE(rad) |
| --- | ---: |
| channel 1 | `1.280143` |
| channel 2 | `1.209993` |
| channel 3 | `1.326953` |
| channel 4 | `1.396413` |
| channel 5 | `1.443398` |
| channel 6 | `1.241810` |

## 与当前基线对比

当前 README 中记录的 7 光束主数据集测试结果为：

| 模型 | RMSE(rad) | MAE(rad) | far-field MSE |
| --- | ---: | ---: | ---: |
| 普通 CNN | `1.02698` | `0.81906` | `1.1935e-4` |
| 物理约束 CNN，`lambda_phy=0.1` | `1.02269` | `0.81642` | `1.1501e-4` |

本次 `residual_cnn` 50 epoch 完整数据复跑的测试 RMSE 为 `1.319034 rad`，未优于当前普通 CNN baseline，也未优于当前物理约束 CNN。

从训练曲线看，验证 RMSE 在中途达到较低值，但最终验证 RMSE 回升到 `1.219996 rad`。这说明当前设置下后期可能存在过拟合，或者该结构在当前数据划分、学习率和训练轮次下不够稳定。

## 输出文件

```text
result/metrics/cycle22_residual_full_50epoch_2026-06-09.csv
result/metrics/cycle22_residual_full_50epoch/residual_cnn_history.csv
result/metrics/cycle22_residual_full_50epoch/residual_cnn_summary.csv
result/figures/cycle22_residual_full_50epoch_2026-06-10.png
models/cycle22_residual_full_50epoch_residual_cnn_seven_beam.pth
```

其中 `models/*.pth` 为本地模型权重，不建议提交 Git。`dataset/` 也为本地生成数据集，不提交 Git。

## 阶段结论

本次复跑完成了 RTX 3060 上 `residual_cnn` 的完整数据 50 epoch 验证。结果表明，`residual_cnn` 虽然在 Cycle 21 的 96 样本快速筛选中表现最好，但在完整数据长训练下并没有直接转化为更优测试指标。

因此当前不建议将 `residual_cnn` 直接替换为论文主模型。后续若继续探索该结构，建议优先做以下方向：

- 固定相同数据划分后，与 `simple_cnn` 做 50 epoch 公平对比。
- 尝试更小学习率或学习率调度，观察后期验证 RMSE 回升是否缓解。
- 加入早停策略，记录最优验证 epoch 对应的测试表现。
- 若显存允许，可补跑 80 epoch，但需要重点关注过拟合，而不是只看最终 epoch。

## 注意事项

本次训练结果文件当前被 `.gitignore` 的 `result/` 规则忽略。如果需要提交本日志、CSV 或图，需要使用 `git add -f` 强制加入。模型权重和数据集仍按项目规则保留在本地，不提交。
