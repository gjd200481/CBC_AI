# RTX 3060 长轮次训练说明

本文档用于在有 RTX 3060 的电脑上运行 7 光束候选网络结构长训练。当前建议优先训练 `residual_cnn`，因为 Cycle 21 的快速结构筛选中它的早期 RMSE 最低。

## 1. 环境检查

进入项目目录后先检查 PyTorch 是否能识别 CUDA：

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

如果 `cuda available` 为 `False`，需要安装带 CUDA 的 PyTorch。

## 2. 数据准备

训练脚本默认读取：

```text
dataset/seven_beam/main_static/images_main_clean_seven_beam.npy
dataset/seven_beam/main_static/labels_main_clean_seven_beam.npy
```

如果 GPU 电脑没有数据集，先在该电脑上运行：

```powershell
python simulation\static\generate_seven_beam_dataset.py --num-samples 1024 --noise-sigma 0 --num-points 256 --window-size 0.01 --waist 0.0005 --beam-distance 0.0015 --crop-size 160 --seed 20260612 --output-dir dataset\seven_beam\main_static --prefix main_clean_seven_beam
```

## 3. 推荐长训练命令

优先跑残差 CNN：

```powershell
python train\sweep_seven_beam_architecture.py --models residual_cnn --full-dataset --epochs 50 --batch-size 64 --learning-rate 0.001 --seed 20260612 --device cuda --num-workers 2 --pin-memory --experiment-tag cycle23_residual_best_50epoch --history-dir result\metrics\cycle23_residual_best_50epoch --summary-csv result\metrics\cycle23_residual_best_50epoch_2026-06-10.csv --figure-path result\figures\cycle23_residual_best_50epoch_2026-06-10.png
```

如果显存充足，可以尝试：

```powershell
python train\sweep_seven_beam_architecture.py --models residual_cnn --full-dataset --epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 20260612 --device cuda --num-workers 2 --pin-memory --experiment-tag cycle23_residual_best_80epoch_lr3e4 --history-dir result\metrics\cycle23_residual_best_80epoch_lr3e4 --summary-csv result\metrics\cycle23_residual_best_80epoch_lr3e4_2026-06-10.csv --figure-path result\figures\cycle23_residual_best_80epoch_lr3e4_2026-06-10.png
```

## 4. 公平对比命令

如果时间允许，建议同样训练 `simple_cnn`，用于和 Cycle 12 的普通 CNN baseline 做结构公平对比：

```powershell
python train\sweep_seven_beam_architecture.py --models simple_cnn residual_cnn --full-dataset --epochs 50 --batch-size 64 --learning-rate 0.001 --seed 20260612 --device cuda --num-workers 2 --pin-memory --experiment-tag cycle23_arch_fair_50epoch --history-dir result\metrics\cycle23_arch_fair_50epoch --summary-csv result\metrics\cycle23_arch_fair_50epoch_2026-06-10.csv --figure-path result\figures\cycle23_arch_fair_50epoch_2026-06-10.png
```

## 5. 结果带回本项目

长训练结束后，重点保留以下文件：

```text
result/metrics/cycle23_residual_best_50epoch_2026-06-10.csv
result/metrics/cycle23_residual_best_50epoch/residual_cnn_history.csv
result/metrics/cycle23_residual_best_50epoch/residual_cnn_summary.csv
result/figures/cycle23_residual_best_50epoch_2026-06-10.png
models/cycle23_residual_best_50epoch_residual_cnn_seven_beam.pth
models/cycle23_residual_best_50epoch_residual_cnn_seven_beam_best.pth
```

其中 `models/*.pth` 不提交 Git，但需要保留在本地用于后续补偿效果评估。

## 6. 判断标准

优先看：

- `rmse_rad` 是否低于当前 7 光束普通 CNN baseline：`1.02698 rad`。
- `best_checkpoint_test_rmse_rad` 是否低于当前 7 光束普通 CNN baseline：`1.02698 rad`。
- 逐通道 RMSE 是否比 `simple_cnn` 更均衡。
- 训练曲线是否稳定下降，验证 RMSE 是否出现明显过拟合。

如果 `residual_cnn` 完整数据长训练后 RMSE 明显低于 `simple_cnn`，后续可以将它作为论文主模型候选。
