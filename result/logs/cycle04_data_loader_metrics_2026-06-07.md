# Cycle 04 数据读取与相位误差模块记录

## 任务目标

本周期目标是整理 PyTorch 数据读取、固定数据集划分、相位解码和周期相位误差计算函数，为后续普通 CNN baseline 和物理约束 CNN 共用同一套训练评估基础。

## 新增模块

### `train/data_utils.py`

主要内容：

- `FarFieldPhaseDataset`
  - 读取远场光强图像 `.npy` 和相位标签 `.npy`。
  - 检查图像形状是否为 `[N, H, W]`。
  - 检查标签形状是否为 `[N, 2 * num_phases]`。
  - 在 `__getitem__` 中将图像转换为 `[1, H, W]`，适配 `Conv2d` 输入。

- `split_dataset`
  - 使用固定随机种子划分训练集、验证集和测试集。
  - 默认比例为 `0.7 / 0.15 / 0.15`。

- `build_dataloaders`
  - 一次性构建 `train`、`val`、`test` 三个 DataLoader。
  - 返回数据集元信息和划分样本数。

### `train/phase_metrics.py`

主要内容：

- `decode_sin_cos`
  - 将 `[sin(phi), cos(phi)]` 解码为相位角。

- `wrap_phase_error`
  - 将相位误差折回 `[-pi, pi]`。

- `phase_rmse_from_angles`
  - 根据预测相位和真实相位计算周期 RMSE。

- `phase_rmse_from_sin_cos`
  - 直接根据 sin/cos 编码计算周期 RMSE。

- `phase_metrics_from_sin_cos`
  - 输出 RMSE、MAE 和平均误差，单位同时包含 rad 和 deg。

### `train/models.py`

主要内容：

- `SimplePhaseCNN`
  - 当前双光束 CNN baseline 模型。
  - 输入为单通道远场光强图。
  - 默认输出 `[sin(phi), cos(phi)]`。
  - 后续多光束扩展时可通过 `output_dim` 调整输出维度。

## 已改造脚本

### `train/evaluate_two_beam.py`

已从路径写死的训练脚本改为可传参的 baseline 训练入口。

支持参数：

- `--image-path`
- `--label-path`
- `--model-path`
- `--metrics-path`
- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--train-ratio`
- `--val-ratio`
- `--seed`
- `--image-size`
- `--no-plot`

默认数据集指向 Cycle 03 生成的主静态数据集：

```text
dataset/two_beam/main_static/images_main_clean_two_beam.npy
dataset/two_beam/main_static/labels_main_clean_two_beam.npy
```

### `model/demo_evaluate_two_beam_model.py`

已改为复用：

- `train.data_utils.FarFieldPhaseDataset`
- `train.models.SimplePhaseCNN`
- `train.phase_metrics`

这样训练和评估使用同一套数据格式检查、模型结构和相位误差计算逻辑。

## 验证命令

### 1. 检查训练脚本参数

```powershell
python -m train.evaluate_two_beam --help
```

结果：命令行参数正常显示。

### 2. 检查主数据集读取与划分

```powershell
python - <<'PY'
from train.data_utils import build_dataloaders
from train.phase_metrics import phase_metrics_from_sin_cos

loaders = build_dataloaders(
    'dataset/two_beam/main_static/images_main_clean_two_beam.npy',
    'dataset/two_beam/main_static/labels_main_clean_two_beam.npy',
    batch_size=64,
)
print(loaders['dataset'].image_size, loaders['dataset'].num_phases, loaders['splits'])
labels = loaders['dataset'].labels[:8]
print(phase_metrics_from_sin_cos(labels, labels))
PY
```

结果：

```text
image_size = (160, 160)
num_phases = 1
splits = {'train': 1400, 'val': 300, 'test': 300}
self-check rmse = 0.0
```

### 3. 冒烟训练

```powershell
python -m train.evaluate_two_beam `
  --epochs 1 `
  --batch-size 64 `
  --model-path models\smoke_cycle04_cnn.pth `
  --metrics-path result\metrics\smoke_cycle04_train.csv `
  --no-plot
```

结果：

```text
Splits: {'train': 1400, 'val': 300, 'test': 300}
Epoch 001 | train_loss=0.537980 | val_loss=0.497469 | val_rmse=1.707471 rad
Test RMSE(rad): 1.7587032318
```

说明：该结果只用于验证训练链路是否跑通。由于只训练 1 个 epoch，误差较大是正常现象。

### 4. 冒烟评估

```powershell
python model\demo_evaluate_two_beam_model.py `
  --model-path models\smoke_cycle04_cnn.pth `
  --image-path dataset\two_beam\main_static\images_main_clean_two_beam.npy `
  --label-path dataset\two_beam\main_static\labels_main_clean_two_beam.npy `
  --batch-size 128 `
  --no-plot
```

结果：

```text
Samples: 2000
RMSE(rad): 1.7541974783
RMSE(deg): 100.5081119388
```

## 结论

Cycle 04 的核心目标已完成：数据读取、固定划分、相位解码、周期误差计算、CNN 模型定义、训练脚本和评估脚本已经统一。下一步可以进入 Cycle 05，使用 `main_clean_two_beam` 数据集训练正式普通 CNN baseline，并记录完整训练曲线、测试 RMSE 和典型远场预测图。
