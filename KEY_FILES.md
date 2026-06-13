# CBC_AI 关键文件说明

本文档用于说明当前项目中关键文件和文件夹的地址、作用与使用顺序。项目当前主线已从双光束验证升级为 7 光束多路相干合成相位反演：

```text
7 光束远场光强图像 -> CNN 相位反演 -> 6 路相对相位 sin/cos 编码 -> FFT 物理一致性损失
```

双光束相关文件仍然保留，用作低维验证基线、代码原型和论文对照实验。

## 根目录文件

### `PROJECT_PLAN.md`

- 地址：`D:\CBC_AI\PROJECT_PLAN.md`
- 作用：面向一区/二区期刊投稿目标的研究路线图和无时间约束 Cycle 任务规划。
- 当前内容：
  - 明确项目目标已调整为形成具备一区或二区投稿潜力的论文。
  - 将主线确定为“7 光束多路相干合成下基于傅里叶光学约束的 CNN 相位误差反演”。
  - 恢复 `Cycle` 管理方式，但 Cycle 只作为任务分割和实验批次记录，不绑定日期或硬性截止时间。
  - 从 Cycle 27 到 Cycle 43 规划主模型补偿指标、周期损失验证、大规模数据、多平面输入、六边形对称增强、补偿质量损失调度、checkpoint 选择、未归一化 Strehl 修复、焦平面/焦前双分支融合，以及后续解释性与鲁棒性补强。
- 使用建议：
  - 每次修改研究路线、目标期刊定位或论文主结论后，应优先同步修改这个文件。
  - 新 Cycle 是否继续推进，以能否增强一区/二区论文证据链为判断标准。

### `KEY_FILES.md`

- 地址：`D:\CBC_AI\KEY_FILES.md`
- 作用：当前文件，用于快速理解项目结构和关键文件用途。
- 使用建议：
  - 新增重要脚本、数据集、实验结果或论文材料后，应同步补充说明。

### `README.md`

- 地址：`D:\CBC_AI\README.md`
- 作用：项目总入口，说明研究目标、目录结构、快速复现实验命令和当前关键结果。
- 使用建议：
  - 新成员或重新打开项目时优先阅读。
  - 当主实验路线、关键结果或目录结构发生变化时同步更新。

### `NAMING_CONVENTIONS.md`

- 地址：`D:\CBC_AI\NAMING_CONVENTIONS.md`
- 作用：统一项目目录、源码、数据集、结果文件和模型权重命名规范。
- 使用建议：
  - 新增脚本、结果或数据集前先检查命名是否符合该规范。

### `GPU_TRAINING_3060.md`

- 地址：`D:\CBC_AI\GPU_TRAINING_3060.md`
- 作用：RTX 3060 电脑上的完整数据长轮次训练说明。
- 当前内容：
  - CUDA 可用性检查命令。
  - 7 光束主数据集生成命令。
  - `residual_cnn` 50 epoch 和 80 epoch 推荐训练命令。
  - `simple_cnn` 与 `residual_cnn` 公平对比命令。
  - 长训练结果带回当前项目后的判断标准。
- 使用建议：
  - 在 GPU 电脑上训练前先阅读。
  - 长训练完成后，把 CSV、图和本地模型权重带回当前项目继续评估。

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

### `simulation/static/generate_seven_beam_noise_robustness_dataset.py`

- 地址：`D:\CBC_AI\simulation\static\generate_seven_beam_noise_robustness_dataset.py`
- 作用：生成 7 光束探测器噪声鲁棒性实验数据集。
- 当前功能：
  - 多个噪声等级共用同一组 6 路相位样本。
  - 输出 7 光束远场图像、12 维 sin/cos 标签、原始相位和配置文件。
  - 默认噪声等级为 `0, 0.01, 0.03, 0.05, 0.08`。
- 当前输出目录：
  - `D:\CBC_AI\dataset\seven_beam\noise_robustness\`

### `simulation/static/generate_seven_beam_complex_robustness_dataset.py`

- 地址：`D:\CBC_AI\simulation\static\generate_seven_beam_complex_robustness_dataset.py`
- 作用：生成 7 光束振幅失配和位置偏移鲁棒性实验数据集。
- 当前功能：
  - 多个扰动等级共用同一组 6 路相位样本。
  - 振幅失配：中心参考光束振幅固定为 1，外圈 6 路随机变化。
  - 位置偏移：7 路光束中心在指定范围内随机偏移。
  - 保存远场图像、标签、相位、振幅数组、偏移数组和配置文件。
- 当前输出目录：
  - `D:\CBC_AI\dataset\seven_beam\complex_robustness\`

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
  - 当前投稿主线暂不使用该动态序列路线。
- 后续用途：
  - 如果论文后续需要扩展“动态扰动预测”或“闭环控制”，可以重新启用。

### `simulation/static/legacy/*.py` 早期脚本

- 地址：`D:\CBC_AI\simulation\static\legacy\`
- 作用：早期仿真、验证和演示脚本。
- 文件包括：
  - `gaussian_fft.py`：高斯光束 FFT 传播早期验证。
  - `two_beam_interference.py`：双光束干涉图样演示。
  - `two_beam_gaussian.py`：双高斯光束近场/远场演示。
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
- 当前增强能力：
  - `augment_mode="noise"`：训练集随机探测器噪声增强。
  - `augment_mode="hex"`：训练集七光束六边形对称增强。
  - 增强只作用于训练集，验证集和测试集保持干净。
- 当前默认格式：
  - 输入图像：`[N, H, W]`。
  - 网络输入：`[batch, 1, H, W]`。
  - 标签：`[N, 2 * num_phases]`。

### `train/hexagonal_augmentation.py`

- 地址：`D:\CBC_AI\train\hexagonal_augmentation.py`
- 作用：Cycle 32 新增的七光束六边形物理对称增强模块。
- 当前功能：
  - 对远场图像做 `60°` 倍数旋转，并同步循环重排外圈 6 路 sin/cos 标签。
  - 对远场图像做左右镜像，并同步执行外圈通道反向映射。
  - 支持单平面 `[H, W]` 和多平面 `[P, H, W]` 图像。
- 当前用途：
  - 配合 `train/train_deep_residual_final.py --augment-mode hex` 做 Cycle 32 模型改进实验。
  - 重点观察逐通道 RMSE、通道不平衡和补偿物理指标是否改善。

### `train/phase_metrics.py`

- 地址：`D:\CBC_AI\train\phase_metrics.py`
- 作用：统一管理相位解码和周期误差指标。
- 主要功能：
  - `decode_sin_cos`：将 `[sin(phi), cos(phi)]` 解码为相位。
  - `wrap_phase_error`：将相位误差折回 `[-pi, pi]`。
  - `phase_rmse_from_sin_cos`：根据 sin/cos 编码计算周期 RMSE。
  - `phase_metrics_from_sin_cos`：输出 RMSE、MAE 和平均误差。
  - `cyclic_phase_loss_from_sin_cos`：实现 `2 - 2*cos(delta)` 周期相位损失，适配现有 `sin/cos` 标签。
  - `build_phase_loss()`：按名称构建 `mse`、`cyclic` 或 `cyclic_unit` 损失函数。
- 当前用途：
  - 普通 CNN 训练评估。
  - 后续物理约束 CNN 训练评估。

### `train/models.py`

- 地址：`D:\CBC_AI\train\models.py`
- 作用：统一保存训练阶段使用的神经网络结构。
- 当前模型：
  - `SimplePhaseCNN`：三层卷积 + 全连接回归头。
  - `WidePhaseCNN`：更宽的三层卷积结构，并使用自适应池化降低全连接层参数量。
  - `ResidualPhaseCNN`：残差连接 + 自适应池化的候选结构。
  - `CBCPhaseLiteCNN`：面向 CBC 远场条纹图像的自研轻量模型，包含深度可分离残差块、空间/通道门控和多尺度池化回归头。
  - `DualPlaneFusionPhaseCNN`：Cycle42 新增焦平面/焦前双分支门控融合模型，当前补偿质量主模型使用该结构。
  - 输入：单通道远场光强图。
  - 双光束输出：`[sin(phi), cos(phi)]`。
  - 7 光束输出：设置 `output_dim=12`，对应 6 路相对相位的 `sin/cos` 编码。
- 辅助函数：
  - `build_phase_model()`：按模型名称构建网络，支持结构消融。
  - `count_parameters()`：统计可训练参数量。

### `train/physics_loss.py`

- 地址：`D:\CBC_AI\train\physics_loss.py`
- 作用：实现傅里叶光学物理一致性损失，是后续物理约束 CNN 的核心模块。
- 主要功能：
  - `TwoBeamFourierOptics`：torch 版双光束近场重建与 FFT 远场传播模型。
  - `SevenBeamFourierOptics`：torch 版 7 光束近场重建与 FFT 远场传播模型。
  - `FarFieldConsistencyLoss`：计算预测相位重建远场与输入远场之间的 MSE 或 L1 损失。
  - `crop_center_torch`：torch 版中心裁剪函数。
  - `normalize_intensity`：按单张图最大值归一化远场光强。
- 当前验证结果：
  - 双光束真实标签重建远场 MSE 约 `1.08e-16`。
  - 7 光束真实标签重建远场 MSE 约 `1.20e-16`。
  - 7 光束真实标签重建远场最大像素误差约 `1.01e-6`。
  - 物理一致性损失可以反向传播。
- 后续用途：
  - Cycle 07 中与双光束相位监督损失组合。
  - Cycle 13 中与 7 光束相位监督损失组合。

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

### `train/train_seven_beam_physics_constrained_cnn.py`

- 地址：`D:\CBC_AI\train\train_seven_beam_physics_constrained_cnn.py`
- 作用：7 光束物理约束 CNN 训练入口。
- 当前功能：
  - 默认读取 7 光束主静态数据集 `main_clean_seven_beam`。
  - 支持 `--model-name simple_cnn` 或 `--model-name residual_cnn` 输出 6 路相对相位的 sin/cos 编码。
  - 使用 `SevenBeamFourierOptics` 根据预测相位重建 7 光束远场。
  - 总损失为 `L_total = L_phase + lambda_phy * L_farfield`。
  - 输出整体 RMSE、MAE、远场重建 MSE 和逐通道 RMSE。
  - 保存最终 checkpoint 和最佳验证 RMSE checkpoint。
- 当前结果：
  - 已用于 Cycle 13 训练 `lambda_phy=0.1` 的 7 光束物理约束模型。
  - 测试集 RMSE 为 `1.02269 rad`，略低于普通 CNN 的 `1.02698 rad`。
  - 远场重建 MSE 为 `1.1501e-4`，低于普通 CNN 的 `1.1935e-4`。
- 后续用途：
  - Cycle 14 将基于该脚本进行 `lambda_phy` 权重消融。
  - 后续噪声、振幅失配和位置偏移实验也会调用该模型作为物理约束对照。

### `train/sweep_seven_beam_lambda.py`

- 地址：`D:\CBC_AI\train\sweep_seven_beam_lambda.py`
- 作用：7 光束物理损失权重消融脚本。
- 当前功能：
  - 固定同一数据集、同一划分和同一初始化种子。
  - 批量训练多个 `lambda_phy` 设置。
  - 输出每个权重的训练历史、汇总指标、模型权重和结果图。
  - 汇总整体 RMSE、MAE、远场重建 MSE 和逐通道 RMSE。
- 当前结果：
  - Cycle 14 已比较 `0, 0.01, 0.05, 0.1, 0.5, 1.0`。
  - 12 epoch 快速消融中 `lambda_phy=0.1` 相位 RMSE 最低。
  - 30 epoch 候选复训后，`lambda_phy=0.1` 仍优于 `lambda_phy=0.5`。
- 后续用途：
  - 若需要论文主实验更稳，可围绕 `0.05, 0.1, 0.2` 做更细长训练搜索。

### `train/sweep_seven_beam_architecture.py`

- 地址：`D:\CBC_AI\train\sweep_seven_beam_architecture.py`
- 作用：第 21 周期新增的 7 光束网络结构快速消融脚本。
- 当前功能：
  - 对比 `simple_cnn`、`wide_cnn` 和 `residual_cnn`。
  - 记录每个模型的参数量、训练耗时、验证 RMSE、测试 RMSE、MAE 和逐通道 RMSE。
  - 支持 `--max-samples` 限制样本数，便于 CPU 环境下快速筛选结构。
  - 支持 `--full-dataset`、`--device cuda`、`--num-workers` 和 `--pin-memory`，便于 RTX 3060 上完整数据长训练。
  - 支持 `--phase-loss mse|cyclic|cyclic_unit`，便于比较普通 MSE 与周期相位损失。
  - 支持 `--experiment-tag`，避免长训练结果覆盖快速筛选结果。
  - 自动保存最终 epoch checkpoint 和最佳验证 RMSE checkpoint，并在 CSV 中记录二者测试表现。
  - 输出结构消融汇总 CSV、每个模型的训练历史 CSV 和对比图。
- 当前结论：
  - 96 样本、2 epoch 快速筛选中，`residual_cnn` 测试 RMSE 为 `1.709031 rad`，是三者中最低。
  - 该结果仅用于选择候选结构，后续需要完整数据长训练验证。

### `scripts/run_cycle22_gpu_residual.ps1`

- 地址：`D:\CBC_AI\scripts\run_cycle22_gpu_residual.ps1`
- 作用：第 22 周期新增的 RTX 3060 长训练启动脚本。
- 当前功能：
  - 默认运行 `residual_cnn` 完整 7 光束数据训练。
  - 默认参数为 `50 epoch`、`batch size=64`、`learning rate=0.001`、`seed=20260612`、`num_workers=2`。
  - 自动设置结果文件名为 `cycle23_residual_best_<epoch>epoch`。
  - 训练脚本会同时保存最终 epoch 权重和最佳验证 RMSE 权重。
- 使用示例：

```powershell
.\scripts\run_cycle22_gpu_residual.ps1 -Epochs 50 -BatchSize 64 -LearningRate 0.001 -NumWorkers 2
```

### `scripts/run_cycle25_gpu_residual_physics.ps1`

- 地址：`D:\CBC_AI\scripts\run_cycle25_gpu_residual_physics.ps1`
- 作用：在 RTX 3060 上训练 `residual_cnn + physics loss`。
- 当前功能：
  - 使用 `ResidualPhaseCNN`。
  - 使用 `SevenBeamFourierOptics` 远场物理一致性损失。
  - 默认 `lambda_phy=0.1`、`50 epoch`、`batch size=32`。
  - 输出最终 checkpoint 与最佳验证 checkpoint。
- 使用示例：

```powershell
.\scripts\run_cycle25_gpu_residual_physics.ps1 -Epochs 50 -BatchSize 32 -LearningRate 0.001 -LambdaPhy 0.1 -NumWorkers 2 -Seed 20260612
```

### `scripts/run_cycle26_gpu_cbc_lite.ps1`

- 地址：`D:\CBC_AI\scripts\run_cycle26_gpu_cbc_lite.ps1`
- 作用：在 RTX 3060 上训练项目自研 `cbc_lite_cnn + cyclic phase loss`。
- 当前功能：
  - 使用 `CBCPhaseLiteCNN`，不直接复用 MobileNetV3-Small。
  - 默认运行完整 7 光束数据集。
  - 默认参数为 `50 epoch`、`batch size=64`、`learning rate=0.001`、`seed=20260612`、`phase_loss=cyclic`。
  - 输出最终 checkpoint 与最佳验证 checkpoint。
  - 用于 Cycle 26 的文献启发创新模型验证。
- 使用示例：

```powershell
.\scripts\run_cycle26_gpu_cbc_lite.ps1 -Epochs 50 -BatchSize 64 -LearningRate 0.001 -NumWorkers 2 -Seed 20260612 -PhaseLoss cyclic
```

### `train/evaluate_seven_beam_noise_robustness.py`

- 地址：`D:\CBC_AI\train\evaluate_seven_beam_noise_robustness.py`
- 作用：评估 7 光束普通 CNN 和物理约束 CNN 在探测器噪声下的鲁棒性。
- 当前功能：
  - 加载 `baseline_cnn_main_clean_seven_beam_2026-06-08.pth`。
  - 加载 `physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.pth`。
  - 可根据 checkpoint 中的 `model_name` 自动加载 `simple_cnn`、`wide_cnn` 或 `residual_cnn`。
  - 计算整体 RMSE、MAE、逐通道 RMSE 和远场重建 MSE。
  - 输出噪声强度-误差曲线。
- 当前结论：
  - 干净数据上物理约束 CNN 略优。
  - `noise>=0.03` 时物理约束 CNN 的相位 RMSE 明显高于普通 CNN。
  - 后续需要噪声增强训练或更合理的去噪物理一致性目标。

### `train/evaluate_seven_beam_complex_robustness.py`

- 地址：`D:\CBC_AI\train\evaluate_seven_beam_complex_robustness.py`
- 作用：评估 7 光束普通 CNN 和物理约束 CNN 在振幅失配、位置偏移下的鲁棒性。
- 当前功能：
  - 加载 7 光束普通 CNN 和 `lambda_phy=0.1` 物理约束 CNN。
  - 评估振幅失配等级 `0, 0.05, 0.1, 0.2, 0.3`。
  - 评估位置偏移等级 `0, 10um, 20um, 50um, 100um`。
  - 输出整体 RMSE、MAE、逐通道 RMSE、远场重建 MSE 和对比曲线。
- 当前结论：
  - 物理约束 CNN 在振幅失配和位置偏移下均保持小幅 RMSE 优势。
  - 该结果与探测器噪声实验形成对照，说明物理约束更适合光束状态扰动泛化。

### `train/evaluate_seven_beam_compensation_metrics.py`

- 地址：`D:\CBC_AI\train\evaluate_seven_beam_compensation_metrics.py`
- 作用：评估 7 光束相位补偿后的主瓣能量占比。
- 当前功能：
  - 加载 7 光束普通 CNN 和 `lambda_phy=0.1` 物理约束 CNN。
  - 根据预测相位计算补偿后残余相位。
  - 重建补偿前、普通 CNN 补偿后、物理约束 CNN 补偿后和理想相干远场。
  - 计算中心圆形主瓣区域能量占比。
  - 输出统计 CSV 和典型远场图。
- 当前结论：
  - 补偿前主瓣能量占比约 `0.35939`。
  - 普通 CNN 补偿后约 `0.51931`。
  - 物理约束 CNN 补偿后约 `0.52155`。
  - 理想相干约 `0.65063`。

### `train/evaluate_seven_beam_strehl.py`

- 地址：`D:\CBC_AI\train\evaluate_seven_beam_strehl.py`
- 作用：评估 7 光束相位补偿后的 Strehl 比。
- 当前功能：
  - 以理想相干远场峰值强度为基准。
  - 计算补偿前、普通 CNN 补偿后、物理约束 CNN 补偿后和理想相干状态的 Strehl 比。
  - 支持 `--candidate-model` 和 `--candidate-name`，可将 `residual_cnn_best` 纳入 Strehl 对比。
  - 同时记录残余相位 RMSE。
  - 输出明细 CSV、汇总 CSV 和 Strehl 对比图。
- 当前结论：
  - 补偿前 Strehl 均值约 `0.39069`。
  - 普通 CNN 补偿后约 `0.64717`。
  - 物理约束 CNN 补偿后约 `0.65356`。
  - 理想相干为 `1.00000`。

### `train/evaluate_seven_beam_compensation_effect.py`

- 地址：`D:\CBC_AI\train\evaluate_seven_beam_compensation_effect.py`
- 作用：第 19 周期新增的 7 光束相位补偿综合效果评估脚本。
- 当前功能：
  - 加载 7 光束普通 CNN 和 `lambda_phy=0.1` 物理约束 CNN。
  - 使用预测相位作为补偿量，计算补偿后残余相位。
  - 统一评估补偿前、普通 CNN 补偿后、物理约束 CNN 补偿后和理想相干四种状态。
  - 支持 `--candidate-model` 和 `--candidate-name`，可加入 `residual_cnn_best` 等候选模型。
  - 支持重复传入 `--model state_name=checkpoint_path`，用于一次评估多个任意命名 checkpoint。
  - 输出主瓣能量占比、旁瓣能量占比、Strehl 比、合成效率、峰值旁瓣比和残余相位 RMSE。
  - 保存明细 CSV、汇总 CSV 和综合对比图。
- 当前结论：
  - 补偿前合成效率约 `0.53286`。
  - 普通 CNN 补偿后合成效率约 `0.78602`。
  - 物理约束 CNN 补偿后合成效率约 `0.78964`。
  - Cycle 27 显示 `residual_cnn_best` 的补偿指标优于普通 CNN、首版物理约束 CNN 和 `residual_cnn + physics, lambda_phy=0.05`。

### `train/compare_system_scale.py`

- 地址：`D:\CBC_AI\train\compare_system_scale.py`
- 作用：第 20 周期新增的双光束/7 光束系统规模对比脚本。
- 当前功能：
  - 读取已有双光束和 7 光束实验 CSV，不重新训练模型。
  - 汇总光束数量、待预测相位数量、网络输出维度、样本数、训练轮数、RMSE、MAE 和远场一致性误差。
  - 计算 7 光束相对双光束的规模放大倍数。
  - 输出系统规模对比 CSV、倍数对比 CSV 和结果图。
- 当前结论：
  - 7 光束待预测相位数量和网络输出维度均为双光束的 `6` 倍。
  - 双光束适合作为方法验证和低维基线。
  - 7 光束更适合作为论文主实验对象，用于体现多通道 CBC 相位反演难度。

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

### `examples/demo_evaluate_two_beam_model.py`

- 地址：`D:\CBC_AI\examples\demo_evaluate_two_beam_model.py`
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

### `examples/demo_two_beam_inference.py`

- 地址：`D:\CBC_AI\examples\demo_two_beam_inference.py`
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
2. 查看当前论文初稿：`paper/CBC_AI_paper_draft_2026-06-10.md`
3. 复查七光束主数据生成：`simulation/static/generate_seven_beam_dataset.py`
4. 检查七光束物理约束：`train/physics_loss.py`
5. 训练当前主线候选：`train/train_seven_beam_physics_constrained_cnn.py`
6. 复查关键结果：`result/logs/`、`result/metrics/`、`result/figures/`
7. 写论文背景时查：`paper/journals/chinese/README.md` 和 `paper/daedalus_packages/`

## 下一步建议

围绕一区或二区投稿目标，下一步按无时间约束 Cycle 推进：

- Cycle 27：已补齐 `residual_cnn + physics loss, lambda_phy=0.05` 的补偿物理指标，结论见 `result/logs/cycle27_residual_physics_compensation_2026-06-11.md`。
- Cycle 28：在残差主线上测试周期相位损失，并同时观察相位 RMSE 与补偿物理指标。
- Cycle 29：设计并运行更大规模七光束数据集实验。
- Cycle 30：已生成焦前/离焦图像数据集并比较相位反演效果。
- Cycle 32 到 Cycle 40：已完成六边形对称增强、补偿质量损失调度、多平面 7cm 主模型、lambda_comp 扫描、双 checkpoint 和显式指标 checkpoint 工具验证。
- Cycle 41：优先修复未归一化 torch 远场/Strehl 验证函数，使训练期 checkpoint 选择与最终补偿评估一致。
- Cycle 42：已完成焦平面/焦前双分支特征融合，`cycle42_best_rmse` 超过 Cycle41，当前补偿质量主模型更新为 `models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth`。
- Cycle 43：下一步做 Cycle42 attribution 解释性分析和噪声鲁棒性验证。
- 当前默认不继续盲目扩大 `lambda_comp` 网格，也不把更大模型作为下一阶段主方向。
