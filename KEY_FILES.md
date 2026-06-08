# CBC_AI 关键文件说明

本文档用于说明当前项目中关键文件和文件夹的地址、作用与使用顺序。项目当前主线已从双光束验证升级为 7 光束多路相干合成相位反演：

```text
7 光束远场光强图像 -> CNN 相位反演 -> 6 路相对相位 sin/cos 编码 -> FFT 物理一致性损失
```

双光束相关文件仍然保留，用作低维验证基线、代码原型和论文对照实验。

## 根目录文件

### `PROJECT_PLAN.md`

- 地址：`D:\CBC_AI\PROJECT_PLAN.md`
- 作用：项目总计划文件。
- 当前内容：
  - 明确 2026 年 7 月底前的项目目标。
  - 将主线确定为“7 光束多路相干合成下基于傅里叶光学约束的 CNN 相位误差反演”。
  - 按两天一个周期安排数据生成、模型训练、物理约束、鲁棒性实验、论文写作。
  - 将此前的 `CNN + LSTM` 远场序列预测路线暂时降级为后续拓展方向。
  - 将双光束实验定位为低维验证基线，后续主实验转向 7 光束。
- 使用建议：
  - 每完成一个周期后，在对应 Cycle 下补充“状态”。
  - 每次修改研究路线或摘要后，应优先同步修改这个文件。

### `KEY_FILES.md`

- 地址：`D:\CBC_AI\KEY_FILES.md`
- 作用：当前文件，用于快速理解项目结构和关键文件用途。
- 使用建议：
  - 新增重要脚本、数据集、实验结果或论文材料后，应同步补充说明。

### `.gitignore`

- 地址：`D:\CBC_AI\.gitignore`
- 作用：控制哪些文件不提交到 Git。
- 当前重点规则：
  - `dataset/`：本地数据集较大，不提交。
  - `models/`：模型权重较大，不提交。
  - `result/`：结果目录默认忽略，但关键实验日志和指标表可在需要时强制提交。
  - `*.npy`、`*.pth`、`*.pt`、`*.ckpt`：数组数据和模型权重默认不提交。
- 使用建议：
  - 数据集配置、实验日志、指标 CSV 可以选择性强制提交。
  - 原始大数据和训练权重一般不提交，避免仓库膨胀。

## 仿真代码

### `simulation/common/two_beam_core.py`

- 地址：`D:\CBC_AI\simulation\common\two_beam_core.py`
- 作用：双光束相干合成仿真的核心公共模块，是当前最重要的物理建模文件。
- 主要功能：
  - `create_grid()`：建立二维近场坐标网格。
  - `gaussian_beam()`：生成单束高斯光近场复振幅。
  - `two_beam_near_field()`：生成双光束近场复振幅，第一束为参考相位，第二束带相位误差。
  - `far_field_intensity()`：通过 FFT 由近场复振幅计算远场光强。
  - `add_gaussian_noise()`：给归一化远场图像加入探测器高斯噪声。
  - `crop_center()`：裁剪远场中心区域。
  - `phase_to_sin_cos()`：将相位转换为 `[sin(phi), cos(phi)]` 标签。
  - `wrap_phase()`：将相位折回 `[-pi, pi]`。
  - `generate_two_beam_dataset()`：生成静态双光束远场图像、sin/cos 标签和原始相位。
  - `dataset_config()`：生成静态数据集 JSON 配置。
  - `generate_two_beam_sequence_dataset()`：生成动态序列数据，当前作为后续拓展备用。
- 当前主线用途：
  - 提供双光束低维验证基线。
  - 为后续 7 光束公共仿真模块提供接口风格参考。
  - 后续 7 光束物理一致性损失应复用这里的裁剪、归一化、相位编码等设计思路。
- 注意事项：
  - 该文件只覆盖双光束，第一束相位固定为 0，网络估计第二束相对相位。
  - 7 光束主线应新增独立公共模块，避免把双光束文件改得过于臃肿。
  - `generate_two_beam_dataset()` 已支持 `phase_min` 和 `phase_max`，默认完整覆盖 `[-pi, pi]`。

### `simulation/common/multi_beam_core.py`

- 地址：`D:\CBC_AI\simulation\common\multi_beam_core.py`
- 作用：作为 7 光束主系统的核心物理仿真文件。
- 主要功能：
  - 生成中心 + 外圈六边形 7 光束坐标。
  - 固定中心参考相位为 0。
  - 随机生成外圈 6 路相对相位。
  - 构造 7 光束近场复振幅。
  - 通过 FFT 得到远场光强。
  - 输出 12 维 `sin/cos` 标签。
  - 支持后续振幅失配、位置偏移和噪声扰动。
- 当前用途：
  - 作为后续 7 光束 CNN baseline、物理约束 CNN 和扰动鲁棒性实验的公共仿真底座。
  - 统一 7 光束相位定义，避免后续训练、评估和物理损失之间标签含义不一致。
  - 提供 `save_dataset()` 与配置字典，保证生成数据可追溯。

### `simulation/static/generate_seven_beam_dataset.py`

- 地址：`D:\CBC_AI\simulation\static\generate_seven_beam_dataset.py`
- 作用：生成 7 光束静态远场数据集，是下一阶段 7 光束普通 CNN baseline 的数据入口。
- 输入参数：
  - `--num-samples`：样本数。
  - `--noise-sigma`：探测器高斯噪声标准差。
  - `--num-points`：近场计算网格采样点数。
  - `--window-size`：近场窗口物理尺寸，单位 m。
  - `--waist`：高斯光束腰斑半径，单位 m。
  - `--beam-distance`：中心光束到外圈光束的距离，单位 m。
  - `--crop-size`：远场中心裁剪尺寸。
  - `--phase-min`、`--phase-max`：外圈 6 路相位采样范围。
  - `--seed`：随机种子。
  - `--output-dir`：输出目录。
  - `--prefix`：输出文件名前缀。
- 输出文件：
  - `images_<prefix>.npy`：7 光束远场光强图像。
  - `labels_<prefix>.npy`：6 路相对相位的 12 维 `sin/cos` 标签。
  - `phases_<prefix>.npy`：6 路原始相位，单位 rad。
  - `config_<prefix>.json`：数据集参数配置。
- 当前 smoke 记录：
  - `D:\CBC_AI\result\logs\cycle11_seven_beam_smoke_2026-06-08.md`

### `simulation/static/generate_two_beam_dataset.py`

- 地址：`D:\CBC_AI\simulation\static\generate_two_beam_dataset.py`
- 作用：生成可复现的静态双光束远场数据集。
- 输入参数：
  - `--num-samples`：样本数。
  - `--noise-sigma`：探测器高斯噪声标准差。
  - `--num-points`：近场计算网格采样点数。
  - `--window-size`：近场窗口物理尺寸，单位 m。
  - `--waist`：高斯光束腰斑半径，单位 m。
  - `--beam-distance`：两束光中心间距，单位 m。
  - `--crop-size`：远场中心裁剪尺寸。
  - `--phase-min`、`--phase-max`：相位采样范围。
  - `--seed`：随机种子。
  - `--output-dir`：输出目录。
  - `--prefix`：输出文件名前缀。
  - `--save-phases`：是否保存原始相位数组。
- 输出文件：
  - `images_<prefix>.npy`
  - `labels_<prefix>.npy`
  - `phases_<prefix>.npy`
  - `config_<prefix>.json`
- 当前主数据集生成命令见：
  - `D:\CBC_AI\result\logs\cycle03_static_dataset_2026-06-07.md`

### `simulation/static/two_beam_diff_noise.py`

- 地址：`D:\CBC_AI\simulation\static\two_beam_diff_noise.py`
- 作用：生成指定噪声强度的双光束数据集。
- 当前用途：
  - 用于噪声鲁棒性实验。
  - 可生成 `noise_0.01`、`noise_0.03`、`noise_0.05` 等数据集。
- 与 `generate_two_beam_dataset.py` 的关系：
  - 两者底层都调用 `simulation/common/two_beam_core.py`。
  - `generate_two_beam_dataset.py` 更通用，建议作为以后主数据生成入口。
  - `two_beam_diff_noise.py` 保留用于兼容旧训练脚本和快速生成噪声数据。

### `simulation/dynamic/generate_two_beam_sequence_dataset.py`

- 地址：`D:\CBC_AI\simulation\dynamic\generate_two_beam_sequence_dataset.py`
- 作用：生成远场序列数据。
- 当前状态：
  - 原本服务于 `CNN + LSTM` 未来相位预测路线。
  - 新摘要下暂不作为 7 月底主线任务。
- 后续用途：
  - 如果论文后续需要扩展“动态扰动预测”或“闭环控制”，可以重新启用。

### `simulation/static/*.py` 其他脚本

- 地址：`D:\CBC_AI\simulation\static\`
- 作用：早期仿真、验证和演示脚本。
- 文件包括：
  - `gaussian_fft.py`：高斯光束 FFT 传播早期验证。
  - `two_beam_interference.py`：双光束干涉图样演示。
  - `twobeam_gaussian.py`：双高斯光束近场/远场演示。
  - `day2_5_diffraction.py`：早期衍射仿真实验脚本。
- 当前优先级：
  - 这些脚本主要用于理解和回溯，不作为主训练入口。
  - 新实验应优先调用 `simulation/common/two_beam_core.py`。

## 训练与评估代码

### `train/evaluate_two_beam.py`

- 地址：`D:\CBC_AI\train\evaluate_two_beam.py`
- 作用：当前双光束 CNN baseline 训练入口。
- 当前功能：
  - 支持命令行传入图像、标签、模型保存路径、指标保存路径和训练超参数。
  - 默认读取 Cycle 03 主静态数据集 `main_clean_two_beam`。
  - 使用 `train.data_utils` 构建训练/验证/测试 DataLoader。
  - 使用 `train.models.SimplePhaseCNN` 作为普通 CNN baseline。
  - 使用 `train.phase_metrics` 计算周期相位 RMSE、MAE 和平均误差。
- 当前默认数据：
  - `dataset/two_beam/main_static/images_main_clean_two_beam.npy`
  - `dataset/two_beam/main_static/labels_main_clean_two_beam.npy`
- 后续用途：
  - 已用于 Cycle 05 训练正式普通 CNN baseline。
  - 后续物理约束 CNN 可复用其中的数据读取、训练循环和指标记录思路。

### `train/train_seven_beam_baseline.py`

- 地址：`D:\CBC_AI\train\train_seven_beam_baseline.py`
- 作用：7 光束普通 CNN baseline 训练入口。
- 当前功能：
  - 默认读取 `dataset/seven_beam/main_static/images_main_clean_seven_beam.npy` 和 `labels_main_clean_seven_beam.npy`。
  - 使用 `train.data_utils.build_dataloaders()` 构建训练、验证和测试集。
  - 使用 `train.models.SimplePhaseCNN(image_size=160, output_dim=12)` 输出 6 路相对相位的 12 维 `sin/cos` 编码。
  - 使用普通 `MSELoss` 作为监督损失，不包含物理一致性项。
  - 输出整体 RMSE、MAE、平均误差，以及 6 个外圈通道各自的 RMSE。
- 当前结果：
  - 已用于 Cycle 12 训练首版 7 光束 baseline。
  - 测试集整体 RMSE 为 `1.02698 rad`，MAE 为 `0.81906 rad`。
  - 第 4 通道 RMSE 最高，约为 `1.14974 rad`，说明多通道反演中存在通道偏差。
- 后续用途：
  - 作为 7 光束物理约束 CNN 的直接对照。
  - 后续如果引入更深 CNN、残差网络或动态卷积，可先与该脚本结果比较。

### `train/data_utils.py`

- 地址：`D:\CBC_AI\train\data_utils.py`
- 作用：统一管理远场数据集读取和 DataLoader 构建。
- 主要功能：
  - `FarFieldPhaseDataset`：读取远场图像和 `[sin(phi), cos(phi)]` 标签。
  - `split_dataset`：用固定随机种子划分训练集、验证集和测试集。
  - `build_dataloaders`：返回训练、验证、测试三个 DataLoader 和划分信息。
- 当前默认格式：
  - 输入图像：`[N, H, W]`。
  - 网络输入：`[batch, 1, H, W]`。
  - 标签：`[N, 2 * num_phases]`。

### `train/phase_metrics.py`

- 地址：`D:\CBC_AI\train\phase_metrics.py`
- 作用：统一管理相位解码和周期误差指标。
- 主要功能：
  - `decode_sin_cos`：将 `[sin(phi), cos(phi)]` 解码为相位。
  - `wrap_phase_error`：将相位误差折回 `[-pi, pi]`。
  - `phase_rmse_from_sin_cos`：根据 sin/cos 编码计算周期 RMSE。
  - `phase_metrics_from_sin_cos`：输出 RMSE、MAE 和平均误差。
- 当前用途：
  - 普通 CNN 训练评估。
  - 后续物理约束 CNN 训练评估。

### `train/models.py`

- 地址：`D:\CBC_AI\train\models.py`
- 作用：统一保存训练阶段使用的神经网络结构。
- 当前模型：
  - `SimplePhaseCNN`：三层卷积 + 全连接回归头。
  - 输入：单通道远场光强图。
  - 双光束输出：`[sin(phi), cos(phi)]`。
  - 7 光束输出：设置 `output_dim=12`，对应 6 路相对相位的 `sin/cos` 编码。

### `train/physics_loss.py`

- 地址：`D:\CBC_AI\train\physics_loss.py`
- 作用：实现傅里叶光学物理一致性损失，是后续物理约束 CNN 的核心模块。
- 主要功能：
  - `TwoBeamFourierOptics`：torch 版双光束近场重建与 FFT 远场传播模型。
  - `FarFieldConsistencyLoss`：计算预测相位重建远场与输入远场之间的 MSE 或 L1 损失。
  - `crop_center_torch`：torch 版中心裁剪函数。
  - `normalize_intensity`：按单张图最大值归一化远场光强。
- 当前验证结果：
  - 真实标签重建远场 MSE 约 `1.08e-16`。
  - 最大像素误差约 `4.77e-7`。
  - 扰动预测下物理损失可反向传播，梯度有限。
- 后续用途：
  - Cycle 07 中与相位监督损失组合：

```text
L_total = L_phase + lambda_phy * L_farfield
```

### `train/train_physics_constrained_cnn.py`

- 地址：`D:\CBC_AI\train\train_physics_constrained_cnn.py`
- 作用：训练第一版物理约束 CNN。
- 当前功能：
  - 读取远场图像和 `sin/cos` 相位标签。
  - 使用 `SimplePhaseCNN` 预测相位编码。
  - 使用 `MSELoss` 计算相位监督损失。
  - 使用 `FarFieldConsistencyLoss` 计算远场物理一致性损失。
  - 按 `L_total = L_phase + lambda_phy * L_farfield` 训练模型。
  - 保存训练指标、测试摘要、模型权重和结果图。
- 当前实验：
  - `lambda_phy = 0.1`
  - 10 epoch
  - 测试集 RMSE 为 `0.005782 rad`，约 `0.331 deg`
  - 远场重建 MSE 为 `9.35e-9`

### `train/evaluate_noise_robustness.py`

- 地址：`D:\CBC_AI\train\evaluate_noise_robustness.py`
- 作用：评估普通 CNN 和物理约束 CNN 在不同探测器噪声强度下的相位反演性能。
- 当前功能：
  - 加载 `baseline_cnn_main_clean.pth`。
  - 加载 `sweep_lambda_0.01_main_clean.pth` 作为物理约束 CNN 候选。
  - 对多个 `noise_sigma` 数据集计算 RMSE、MAE、平均误差和远场 MSE。
  - 输出噪声强度-误差曲线。
- 当前结论：
  - 在 `noise=0.01, 0.03, 0.05` 下，物理约束 CNN 相比普通 CNN 有更低 RMSE。

### `simulation/static/generate_two_beam_noise_robustness_dataset.py`

- 地址：`D:\CBC_AI\simulation\static\generate_two_beam_noise_robustness_dataset.py`
- 作用：生成噪声鲁棒性实验专用数据集。
- 设计特点：
  - 多个噪声强度共用同一组相位标签。
  - 只改变探测器噪声强度，保证噪声曲线对比公平。
- 当前输出目录：
  - `dataset/two_beam/noise_robustness/`

### `simulation/static/generate_two_beam_amplitude_mismatch_dataset.py`

- 地址：`D:\CBC_AI\simulation\static\generate_two_beam_amplitude_mismatch_dataset.py`
- 作用：生成振幅失配鲁棒性实验专用数据集。
- 设计特点：
  - 第一束振幅固定为 `1.0`。
  - 第二束振幅从 `[1-r, 1+r]` 随机采样。
  - 多个失配等级共用同一组相位标签。
- 当前输出目录：
  - `dataset/two_beam/amplitude_mismatch/`

### `train/evaluate_amplitude_mismatch.py`

- 地址：`D:\CBC_AI\train\evaluate_amplitude_mismatch.py`
- 作用：评估普通 CNN 和物理约束 CNN 在不同振幅失配范围下的相位反演性能。
- 当前结论：
  - 振幅失配范围到 `0.3` 时，两类模型都较稳定。
  - 当前设置下普通 CNN 的 RMSE 低于物理约束 CNN。

### `model/demo_evaluate_two_beam_model.py`

- 地址：`D:\CBC_AI\model\demo_evaluate_two_beam_model.py`
- 作用：加载已训练 CNN 模型并评估相位 RMSE。
- 当前功能：
  - 从 `.pth` 模型文件读取网络权重。
  - 读取评估数据集。
  - 将网络输出 `[sin(phi), cos(phi)]` 解码为相位。
  - 计算周期相位误差和 RMSE。
- 当前实现：
  - 已复用 `train.data_utils.FarFieldPhaseDataset`。
  - 已复用 `train.models.SimplePhaseCNN`。
  - 已复用 `train.phase_metrics`。
- 使用场景：
  - 训练完成后快速复查模型性能。
  - 对比不同噪声数据集上的泛化误差。

### `model/demo_two_beam_inference.py`

- 地址：`D:\CBC_AI\model\demo_two_beam_inference.py`
- 作用：单样本推理演示脚本。
- 使用场景：
  - 展示模型如何从一张远场图像预测相位。
  - 后续可用于生成论文中的典型样例图。

## 数据目录

### `dataset/`

- 地址：`D:\CBC_AI\dataset\`
- 作用：保存本地生成的数据集。
- Git 状态：
  - 被 `.gitignore` 忽略，不提交到仓库。
- 当前重要数据集：
  - `dataset/two_beam/main_static/`
    - 当前主路线第一版干净静态双光束数据集。
    - 样本数：2000。
    - 图像尺寸：`160 x 160`。
    - 相位范围：`[-pi, pi]`。
    - 噪声强度：0。
  - `dataset/two_beam/`
    - 旧版噪声数据集，如 `noise_0.05`。
  - `dataset/two_beam_sequence/`
    - 动态序列数据，当前为后续拓展备用。
- 当前主数据集文件：
  - `D:\CBC_AI\dataset\two_beam\main_static\images_main_clean_two_beam.npy`
  - `D:\CBC_AI\dataset\two_beam\main_static\labels_main_clean_two_beam.npy`
  - `D:\CBC_AI\dataset\two_beam\main_static\phases_main_clean_two_beam.npy`
  - `D:\CBC_AI\dataset\two_beam\main_static\config_main_clean_two_beam.json`

## 结果目录

### `result/`

- 地址：`D:\CBC_AI\result\`
- 作用：保存实验日志、指标和图表。
- 子目录：
  - `result/logs/`：实验记录，写清任务目标、命令、参数、结果和结论。
  - `result/metrics/`：CSV 指标表，便于后续画图和汇总。
  - `result/figures/`：训练曲线、远场图、误差图等。
- Git 状态：
  - 默认被 `.gitignore` 忽略。
  - 关键日志和指标表可强制提交。

### `result/logs/cycle03_static_dataset_2026-06-07.md`

- 地址：`D:\CBC_AI\result\logs\cycle03_static_dataset_2026-06-07.md`
- 作用：记录 Cycle 03 主静态数据集生成过程。
- 内容：
  - 数据集路径。
  - 生成命令。
  - 数据形状。
  - 数值检查。
  - Cycle 03 结论。

### `result/metrics/cycle03_static_dataset_2026-06-07.csv`

- 地址：`D:\CBC_AI\result\metrics\cycle03_static_dataset_2026-06-07.csv`
- 作用：以 CSV 形式保存 Cycle 03 数据集检查指标。
- 当前指标：
  - 样本数。
  - 图像形状。
  - 标签形状。
  - 噪声强度。
  - 相位范围。
  - 图像均值和标准差。
  - 标签与真实相位 sin/cos 的最大误差。

### `result/logs/cycle05_baseline_cnn_2026-06-07.md`

- 地址：`D:\CBC_AI\result\logs\cycle05_baseline_cnn_2026-06-07.md`
- 作用：记录普通 CNN baseline 的正式训练过程和测试结果。
- 当前结论：
  - 在无噪声双光束主数据集上，测试集 RMSE 为 `0.003742 rad`，约 `0.214 deg`。
  - 后续物理约束 CNN 的重点应放在鲁棒性、远场重建一致性和物理指标上。

### `result/logs/cycle06_physics_loss_2026-06-07.md`

- 地址：`D:\CBC_AI\result\logs\cycle06_physics_loss_2026-06-07.md`
- 作用：记录傅里叶光学物理一致性损失模块的实现和验证过程。
- 当前结论：
  - torch 版 FFT 传播与 NumPy 数据生成口径一致。
  - 物理一致性损失可以反向传播。
  - 可进入 Cycle 07 的物理约束 CNN 训练。

### `result/logs/cycle07_physics_constrained_cnn_2026-06-07.md`

- 地址：`D:\CBC_AI\result\logs\cycle07_physics_constrained_cnn_2026-06-07.md`
- 作用：记录第一版物理约束 CNN 的训练过程和测试结果。
- 当前结论：
  - 物理约束训练流程已经完整跑通。
  - `lambda_phy=0.1` 下，测试集 RMSE 为 `0.005782 rad`，约 `0.331 deg`。
  - 远场重建 MSE 为 `9.35e-9`。
  - 后续需要在 Cycle 08 做 `lambda_phy` 权重消融。

### `result/logs/cycle08_lambda_sweep_2026-06-07.md`

- 地址：`D:\CBC_AI\result\logs\cycle08_lambda_sweep_2026-06-07.md`
- 作用：记录物理损失权重 `lambda_phy` 的消融实验。
- 测试权重：
  - `0`
  - `0.01`
  - `0.05`
  - `0.1`
  - `0.5`
  - `1.0`
- 当前结论：
  - 在干净双光束数据集、8 epoch 设置下，`lambda_phy=0.01` 最优。
  - 其测试集 RMSE 为 `0.004291 rad`，约 `0.24585 deg`。
  - 远场重建 MSE 为 `4.82e-9`。
  - 下一阶段噪声鲁棒性实验建议优先使用 `lambda_phy=0.01`。

### `result/metrics/cycle08_lambda_sweep_2026-06-07.csv`

- 地址：`D:\CBC_AI\result\metrics\cycle08_lambda_sweep_2026-06-07.csv`
- 作用：保存 `lambda_phy` 消融实验汇总指标。

### `result/figures/cycle08_lambda_sweep_2026-06-07.png`

- 地址：`D:\CBC_AI\result\figures\cycle08_lambda_sweep_2026-06-07.png`
- 作用：保存 `lambda_phy` 与相位 RMSE、远场 MSE、相位监督损失之间的关系图。

### `result/logs/cycle09_noise_robustness_2026-06-08.md`

- 地址：`D:\CBC_AI\result\logs\cycle09_noise_robustness_2026-06-08.md`
- 作用：记录探测器噪声鲁棒性实验。
- 当前结论：
  - 在中等噪声 `0.01, 0.03, 0.05` 下，`lambda_phy=0.01` 物理约束 CNN 相比普通 CNN 的 RMSE 降低约 `10.60%` 到 `15.99%`。
  - 在高噪声 `0.08` 下，物理约束 CNN 略差，说明当前方法存在噪声适用边界。

### `PROJECT_STATUS.md`

- 地址：`D:\CBC_AI\PROJECT_STATUS.md`
- 作用：详细记录项目任务目标、研究路线、已完成工作、当前进度和下一步计划。

### `result/logs/cycle10_amplitude_mismatch_2026-06-08.md`

- 地址：`D:\CBC_AI\result\logs\cycle10_amplitude_mismatch_2026-06-08.md`
- 作用：记录振幅失配鲁棒性实验。
- 当前结论：
  - 当前双光束设置下，振幅失配对相位反演影响较小。
  - 普通 CNN 在该扰动下仍优于物理约束 CNN。
  - 后续需要继续测试位置偏移和混合扰动。

### `result/metrics/baseline_cnn_main_clean_2026-06-07.csv`

- 地址：`D:\CBC_AI\result\metrics\baseline_cnn_main_clean_2026-06-07.csv`
- 作用：保存普通 CNN baseline 每个 epoch 的训练损失、验证损失和验证相位误差。

### `result/metrics/baseline_cnn_main_clean_summary_2026-06-07.csv`

- 地址：`D:\CBC_AI\result\metrics\baseline_cnn_main_clean_summary_2026-06-07.csv`
- 作用：保存普通 CNN baseline 最终测试集指标。

### `result/figures/baseline_cnn_main_clean_2026-06-07.png`

- 地址：`D:\CBC_AI\result\figures\baseline_cnn_main_clean_2026-06-07.png`
- 作用：保存普通 CNN baseline 的训练曲线、预测-真实相位散点图和误差分布图。

## 文献目录

### `paper/`

- 地址：`D:\CBC_AI\paper\`
- 作用：保存论文阅读材料、期刊 PDF、学位论文和 Daedalus 解析结果。
- 子目录：
  - `paper/journals/`：英文和中文期刊论文 PDF。
  - `paper/journals/chinese/`：中文期刊论文 PDF 和清单。
  - `paper/theses/`：学位论文。
  - `paper/daedalus_packages/`：用 valey-literature-daedalus 生成的论文陪读包。

### `paper/journals/chinese/README.md`

- 地址：`D:\CBC_AI\paper\journals\chinese\README.md`
- 作用：中文期刊论文清单。
- 内容：
  - 每篇中文论文的文件名。
  - 期刊和年份。
  - 主题。
  - 对本项目的用途。
  - 来源链接。
  - 推荐阅读顺序。
- 当前用途：
  - 支撑论文引言和相关工作。
  - 支撑传统 SPGD、主动相位控制、机器学习自适应光学等背景论述。

### `paper/daedalus_packages/`

- 地址：`D:\CBC_AI\paper\daedalus_packages\`
- 作用：保存文献精读结果。
- 典型内容：
  - `paper.md`：论文中英文陪读式解析。
  - `figures/`：论文图表提取结果。
  - `source/`：原始 PDF、抽取文本和解析日志。
- 当前用途：
  - 为论文写作提供方法、实验指标和对比文献参考。

## 模型权重目录

### `models/`

- 地址：`D:\CBC_AI\models\`
- 作用：保存训练得到的模型权重。
- Git 状态：
  - 被 `.gitignore` 忽略，不提交到仓库。
- 当前用途：
  - 保存 CNN baseline 权重。
  - 后续保存物理约束 CNN 权重。
- 注意事项：
  - 提交代码时不要把 `.pth` 文件加入 Git。
  - 需要复现实验时，应优先通过训练脚本重新生成模型权重。

## 当前优先使用顺序

1. 查看计划：`PROJECT_PLAN.md`
2. 生成或复查数据：`simulation/static/generate_two_beam_dataset.py`
3. 检查物理仿真逻辑：`simulation/common/two_beam_core.py`
4. 查看主数据集记录：`result/logs/cycle03_static_dataset_2026-06-07.md`
5. 进入下一步训练前，整理：`train/evaluate_two_beam.py`
6. 评估模型时使用：`model/demo_evaluate_two_beam_model.py`
7. 写论文背景时查：`paper/journals/chinese/README.md` 和 `paper/daedalus_packages/`

## 下一步建议

按照新计划，下一步是 Cycle 04：

- 将 `Dataset`、`DataLoader`、相位解码、周期相位误差 RMSE 独立成可复用模块。
- 让训练脚本支持命令行传入数据路径、模型路径、epoch、batch size 和随机种子。
- 为后续普通 CNN baseline 和物理约束 CNN 共用同一套数据读取与评估函数。
