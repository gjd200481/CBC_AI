import json
from pathlib import Path

import numpy as np


def create_grid(num_points=256, window_size=10e-3):
    """创建二维近场坐标网格。

    参数含义：
    - num_points：每个方向的采样点数，最终得到 num_points x num_points 网格。
    - window_size：近场计算窗口的物理尺寸，单位为 m。

    返回值：
    - x_grid, y_grid：两个二维坐标矩阵，分别表示每个采样点的 x/y 坐标。
    """
    # 坐标范围取 [-window_size/2, window_size/2]，让光束阵列自然位于窗口中心附近。
    x = np.linspace(-window_size / 2, window_size / 2, num_points)
    return np.meshgrid(x, x)


def gaussian_beam(x_grid, y_grid, center_x, center_y, waist, amplitude=1.0, phase=0.0):
    """生成单束高斯光的近场复振幅。

    这里的复振幅同时包含振幅和相位：
    E(x, y) = A * exp(-r^2 / w0^2) * exp(j * phase)

    参数含义：
    - center_x, center_y：光束中心位置。
    - waist：高斯光束腰斑半径 w0。
    - amplitude：光束振幅系数，后续扩展振幅失配时会用到。
    - phase：该光束携带的相位。
    """
    # envelope 是实数高斯包络，只描述空间振幅分布。
    envelope = np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / waist**2)
    # exp(1j * phase) 给整束光叠加统一相位，得到复数形式的电场。
    return amplitude * envelope * np.exp(1j * phase)


def two_beam_near_field(x_grid, y_grid, waist, beam_distance, phase):
    """生成双光束相干合成的近场复振幅。

    第一束光作为相位参考，相位固定为 0；第二束光相对第一束光具有 phase。
    当前任务要学习的正是这个相位差 phase。
    """
    # 第一束光放在 x = -d/2，作为参考光束。
    beam_1 = gaussian_beam(
        x_grid,
        y_grid,
        center_x=-beam_distance / 2,
        center_y=0.0,
        waist=waist,
        phase=0.0,
    )
    # 第二束光放在 x = +d/2，并叠加待估计的相位误差。
    beam_2 = gaussian_beam(
        x_grid,
        y_grid,
        center_x=beam_distance / 2,
        center_y=0.0,
        waist=waist,
        phase=phase,
    )
    # CBC 的近场合成是复电场相加，不是光强相加。
    return beam_1 + beam_2


def far_field_intensity(near_field, normalize=True):
    """由近场复振幅计算远场归一化光强。

    在夫琅禾费近似下，远场复振幅可由近场复振幅的傅里叶变换得到。
    探测器测到的是光强，因此取复振幅模长平方。
    """
    # fft2 计算二维傅里叶变换，fftshift 将零频分量移动到图像中心。
    far_field = np.fft.fftshift(np.fft.fft2(near_field))
    intensity = np.abs(far_field) ** 2
    if normalize:
        # 每张图归一化到最大值 1，降低总能量尺度变化对网络训练的影响。
        max_value = np.max(intensity)
        if max_value > 0:
            intensity = intensity / max_value
    return intensity


def add_gaussian_noise(intensity, noise_sigma, rng):
    """向归一化远场光强图中加入高斯探测噪声。

    参数 noise_sigma 是噪声标准差。由于光强图已归一化到 [0, 1]，
    加噪后需要 clip 回 [0, 1]，避免出现负光强或超过归一化上限。
    """
    if noise_sigma <= 0:
        return intensity
    noisy = intensity + rng.normal(0, noise_sigma, intensity.shape)
    return np.clip(noisy, 0, 1)


def crop_center(image, crop_size):
    """从远场图像中心裁剪正方形区域。

    中心区域通常包含主瓣和主要干涉条纹，是相位估计最关键的信息。
    裁剪后图像尺寸更小，训练速度和显存占用都会更友好。
    """
    if crop_size > min(image.shape):
        raise ValueError(f"crop_size={crop_size} is larger than image shape {image.shape}")

    center_y = image.shape[0] // 2
    center_x = image.shape[1] // 2
    half = crop_size // 2

    if crop_size % 2 == 0:
        # 偶数尺寸如 160：左右各取 80 个像素。
        return image[
            center_y - half:center_y + half,
            center_x - half:center_x + half,
        ]

    # 奇数尺寸时需要包含中心像素，因此右边界多取 1。
    return image[
        center_y - half:center_y + half + 1,
        center_x - half:center_x + half + 1,
    ]


def phase_to_sin_cos(phase):
    """将相位编码为 [sin(phi), cos(phi)]。

    相位具有 2pi 周期性。直接回归 phi 时，-pi 和 pi 附近会出现数值不连续；
    使用 sin/cos 后，网络学习的是连续的圆周表示，训练更稳定。
    """
    return [np.sin(phase), np.cos(phase)]


def wrap_phase(phase):
    """把任意相位折回 [-pi, pi] 区间。

    动态扰动中，相位可能因为漂移或随机游走逐渐超过 [-pi, pi]。
    但物理上相位是周期变量，因此需要统一折回主值区间，便于训练和评估。
    """
    return np.arctan2(np.sin(phase), np.cos(phase))


def generate_phase_trajectory(
    sequence_length,
    mode,
    rng,
    step_sigma=0.08,
    sine_amplitude=1.0,
    sine_frequency=0.04,
    drift_velocity=0.03,
    step_probability=0.08,
    step_scale=0.6,
):
    """生成一条随时间变化的相位扰动轨迹。

    支持的动态模式：
    - random_walk：随机游走，模拟低频相位漂移和累积扰动。
    - sine：正弦扰动，模拟周期性机械振动或热扰动。
    - step：阶跃扰动，模拟突然冲击或控制器跳变。
    - drift：线性漂移叠加小噪声，模拟缓慢热漂移。

    返回值是长度为 sequence_length 的相位序列，单位为 rad。
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    # 起始相位随机采样，避免所有序列都从同一相位状态开始。
    initial_phase = rng.uniform(-np.pi, np.pi)
    time_index = np.arange(sequence_length, dtype=np.float32)

    if mode == "random_walk":
        # 每一帧在上一帧基础上加入一个小相位增量。
        increments = rng.normal(0, step_sigma, sequence_length)
        trajectory = initial_phase + np.cumsum(increments)
    elif mode == "sine":
        # 频率以“每帧周期数”表示，phase_offset 让不同序列具有不同振动初相。
        phase_offset = rng.uniform(-np.pi, np.pi)
        trajectory = initial_phase + sine_amplitude * np.sin(
            2 * np.pi * sine_frequency * time_index + phase_offset
        )
    elif mode == "step":
        # 阶跃模式先保持随机游走背景，再以一定概率出现相位跳变。
        trajectory = np.full(sequence_length, initial_phase, dtype=np.float32)
        current_phase = initial_phase
        for i in range(sequence_length):
            current_phase += rng.normal(0, step_sigma)
            if rng.random() < step_probability:
                current_phase += rng.normal(0, step_scale)
            trajectory[i] = current_phase
    elif mode == "drift":
        # 线性漂移用于模拟慢变扰动，小噪声用于避免轨迹过于理想。
        noise = rng.normal(0, step_sigma, sequence_length)
        trajectory = initial_phase + drift_velocity * time_index + noise
    else:
        raise ValueError(
            f"Unknown phase trajectory mode: {mode}. "
            "Expected random_walk, sine, step, or drift."
        )

    return wrap_phase(trajectory).astype(np.float32)


def generate_sequence_from_phases(
    phases,
    noise_sigma,
    x_grid,
    y_grid,
    waist,
    beam_distance,
    crop_size,
    rng,
):
    """根据一条相位轨迹生成对应的远场图像序列。

    输入 phases 是一维相位数组，每个相位对应一个时刻。
    输出 images 的形状为 [time, crop_size, crop_size]。
    """
    frames = []

    for phase in phases:
        near_field = two_beam_near_field(
            x_grid=x_grid,
            y_grid=y_grid,
            waist=waist,
            beam_distance=beam_distance,
            phase=phase,
        )
        intensity = far_field_intensity(near_field)
        intensity = add_gaussian_noise(intensity, noise_sigma=noise_sigma, rng=rng)
        frames.append(crop_center(intensity, crop_size=crop_size).astype(np.float32))

    return np.array(frames, dtype=np.float32)


def generate_two_beam_dataset(
    num_samples=1000,
    noise_sigma=0.0,
    num_points=256,
    window_size=10e-3,
    waist=0.5e-3,
    beam_distance=1.5e-3,
    crop_size=160,
    seed=None,
):
    """生成双光束相位估计数据集。

    每个样本流程：
    1. 随机采样相位差 phase。
    2. 构造双光束近场复振幅。
    3. 通过 FFT 得到远场光强。
    4. 可选加入探测器噪声。
    5. 裁剪中心区域。
    6. 保存图像、sin/cos 标签和原始相位。
    """
    # default_rng 支持显式 seed，便于复现实验。
    rng = np.random.default_rng(seed)
    x_grid, y_grid = create_grid(num_points=num_points, window_size=window_size)

    images = []
    labels = []
    phases = []

    for _ in range(num_samples):
        # 相位差均匀分布在 [-pi, pi]，覆盖完整相位周期。
        phase = rng.uniform(-np.pi, np.pi)
        near_field = two_beam_near_field(
            x_grid=x_grid,
            y_grid=y_grid,
            waist=waist,
            beam_distance=beam_distance,
            phase=phase,
        )
        intensity = far_field_intensity(near_field)
        intensity = add_gaussian_noise(intensity, noise_sigma=noise_sigma, rng=rng)
        crop = crop_center(intensity, crop_size=crop_size)

        # 图像保存为 float32，匹配 PyTorch 默认训练精度并节省磁盘空间。
        images.append(crop.astype(np.float32))
        labels.append(phase_to_sin_cos(phase))
        # phases 不是训练必需项，但方便后续调试、可视化和物理误差分析。
        phases.append(phase)

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(phases, dtype=np.float32),
    )


def generate_two_beam_sequence_dataset(
    num_sequences=1000,
    input_length=8,
    predict_steps=1,
    phase_mode="random_walk",
    noise_sigma=0.0,
    num_points=256,
    window_size=10e-3,
    waist=0.5e-3,
    beam_distance=1.5e-3,
    crop_size=160,
    seed=None,
    step_sigma=0.08,
    sine_amplitude=1.0,
    sine_frequency=0.04,
    drift_velocity=0.03,
    step_probability=0.08,
    step_scale=0.6,
):
    """生成用于 CNN+LSTM 的双光束远场序列数据集。

    每个样本包含：
    - 输入远场序列：前 input_length 帧，形状为 [T, H, W]。
    - 标签：未来第 predict_steps 帧的相位 [sin(phi), cos(phi)]。
    - input_phases：输入序列对应的真实相位，便于调试和画图。
    - target_phases：未来目标相位，便于计算预测误差。
    - all_phases：完整相位轨迹，长度为 input_length + predict_steps。

    例如 input_length=8, predict_steps=1 时，模型读取 t0 到 t7 的远场图，
    目标是预测 t8 的相位误差。
    """
    if input_length <= 0:
        raise ValueError("input_length must be positive")
    if predict_steps <= 0:
        raise ValueError("predict_steps must be positive")

    rng = np.random.default_rng(seed)
    x_grid, y_grid = create_grid(num_points=num_points, window_size=window_size)

    total_length = input_length + predict_steps
    images = []
    labels = []
    input_phases = []
    target_phases = []
    all_phases = []
    modes = []

    available_modes = ["random_walk", "sine", "step", "drift"]

    for _ in range(num_sequences):
        # mixed 模式会为每条序列随机选择一种扰动类型，用于提高训练集多样性。
        current_mode = rng.choice(available_modes) if phase_mode == "mixed" else phase_mode
        phases = generate_phase_trajectory(
            sequence_length=total_length,
            mode=current_mode,
            rng=rng,
            step_sigma=step_sigma,
            sine_amplitude=sine_amplitude,
            sine_frequency=sine_frequency,
            drift_velocity=drift_velocity,
            step_probability=step_probability,
            step_scale=step_scale,
        )
        frames = generate_sequence_from_phases(
            phases=phases,
            noise_sigma=noise_sigma,
            x_grid=x_grid,
            y_grid=y_grid,
            waist=waist,
            beam_distance=beam_distance,
            crop_size=crop_size,
            rng=rng,
        )

        target_phase = phases[-1]

        images.append(frames[:input_length])
        labels.append(phase_to_sin_cos(target_phase))
        input_phases.append(phases[:input_length])
        target_phases.append(target_phase)
        all_phases.append(phases)
        modes.append(current_mode)

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(input_phases, dtype=np.float32),
        np.array(target_phases, dtype=np.float32),
        np.array(all_phases, dtype=np.float32),
        np.array(modes),
    )


def dataset_config(
    num_samples,
    noise_sigma,
    num_points,
    window_size,
    waist,
    beam_distance,
    crop_size,
    seed,
    image_path,
    label_path,
    phase_path=None,
):
    """构造可写入 JSON 的数据集配置。

    该配置记录数据生成所需的关键参数。以后复现实验时，只要保留这个配置，
    就能知道当前 .npy 数据是怎样生成的。
    """
    return {
        "task": "two_beam_phase_estimation",
        "num_samples": num_samples,
        "noise_sigma": noise_sigma,
        "num_points": num_points,
        "window_size_m": window_size,
        "waist_m": waist,
        "beam_distance_m": beam_distance,
        "crop_size": crop_size,
        "seed": seed,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "phase_path": None if phase_path is None else str(phase_path),
        "label_format": "[sin(phi), cos(phi)]",
    }


def sequence_dataset_config(
    num_sequences,
    input_length,
    predict_steps,
    phase_mode,
    noise_sigma,
    num_points,
    window_size,
    waist,
    beam_distance,
    crop_size,
    seed,
    image_path,
    label_path,
    input_phase_path,
    target_phase_path,
    all_phase_path,
    mode_path,
    step_sigma,
    sine_amplitude,
    sine_frequency,
    drift_velocity,
    step_probability,
    step_scale,
):
    """构造远场序列数据集的 JSON 配置。

    与静态数据集相比，序列数据需要额外记录输入帧数、预测步长和相位动态模式。
    这些参数直接决定 CNN+LSTM 的训练任务定义。
    """
    return {
        "task": "two_beam_future_phase_prediction",
        "model_target": "CNN+LSTM",
        "num_sequences": num_sequences,
        "input_length": input_length,
        "predict_steps": predict_steps,
        "total_sequence_length": input_length + predict_steps,
        "phase_mode": phase_mode,
        "noise_sigma": noise_sigma,
        "num_points": num_points,
        "window_size_m": window_size,
        "waist_m": waist,
        "beam_distance_m": beam_distance,
        "crop_size": crop_size,
        "seed": seed,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "input_phase_path": str(input_phase_path),
        "target_phase_path": str(target_phase_path),
        "all_phase_path": str(all_phase_path),
        "mode_path": str(mode_path),
        "label_format": "[sin(phi_future), cos(phi_future)]",
        "image_shape": "[num_sequences, input_length, crop_size, crop_size]",
        "step_sigma": step_sigma,
        "sine_amplitude": sine_amplitude,
        "sine_frequency": sine_frequency,
        "drift_velocity": drift_velocity,
        "step_probability": step_probability,
        "step_scale": step_scale,
    }


def save_dataset(
    images,
    labels,
    output_dir,
    image_name,
    label_name,
    config_name,
    config,
    phases=None,
    phase_name=None,
):
    """保存数据集数组和配置文件。

    保存内容：
    - images：远场光强图像。
    - labels：网络训练标签 [sin(phi), cos(phi)]。
    - phases：可选，原始相位值，便于调试和分析。
    - config：JSON 格式参数记录。
    """
    output_dir = Path(output_dir)
    # parents=True 允许一次性创建多级目录，例如 dataset/two_beam/cycle02_smoke。
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / image_name
    label_path = output_dir / label_name
    config_path = output_dir / config_name

    np.save(image_path, images)
    np.save(label_path, labels)

    # 原始 phase 不直接作为训练标签，但在画图和核对 sin/cos 编码时很有用。
    if phases is not None and phase_name is not None:
        np.save(output_dir / phase_name, phases)

    # ensure_ascii=False 让中文说明保持可读，而不是写成 unicode 转义。
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return image_path, label_path, config_path


def save_sequence_dataset(
    images,
    labels,
    input_phases,
    target_phases,
    all_phases,
    modes,
    output_dir,
    prefix,
    config,
):
    """保存远场序列数据集。

    保存文件：
    - images_<prefix>.npy：输入远场序列。
    - labels_<prefix>.npy：未来相位 sin/cos 标签。
    - input_phases_<prefix>.npy：输入帧真实相位。
    - target_phases_<prefix>.npy：未来目标相位。
    - all_phases_<prefix>.npy：完整相位轨迹。
    - modes_<prefix>.npy：每条序列使用的扰动模式。
    - config_<prefix>.json：完整参数配置。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / f"images_{prefix}.npy"
    label_path = output_dir / f"labels_{prefix}.npy"
    input_phase_path = output_dir / f"input_phases_{prefix}.npy"
    target_phase_path = output_dir / f"target_phases_{prefix}.npy"
    all_phase_path = output_dir / f"all_phases_{prefix}.npy"
    mode_path = output_dir / f"modes_{prefix}.npy"
    config_path = output_dir / f"config_{prefix}.json"

    np.save(image_path, images)
    np.save(label_path, labels)
    np.save(input_phase_path, input_phases)
    np.save(target_phase_path, target_phases)
    np.save(all_phase_path, all_phases)
    np.save(mode_path, modes)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return {
        "image_path": image_path,
        "label_path": label_path,
        "input_phase_path": input_phase_path,
        "target_phase_path": target_phase_path,
        "all_phase_path": all_phase_path,
        "mode_path": mode_path,
        "config_path": config_path,
    }
