import json
from pathlib import Path

import numpy as np


def create_grid(num_points=256, window_size=10e-3):
    """创建二维近场坐标网格。"""
    x = np.linspace(-window_size / 2, window_size / 2, num_points)
    return np.meshgrid(x, x)


def gaussian_beam(x_grid, y_grid, center_x, center_y, waist, amplitude=1.0, phase=0.0):
    """生成单束高斯光近场复振幅。"""
    envelope = np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / waist**2)
    return amplitude * envelope * np.exp(1j * phase)


def seven_beam_centers(beam_distance=1.5e-3):
    """生成中心 + 外圈六边形 7 光束阵列坐标。

    返回顺序：
    - beam_0：中心参考光束。
    - beam_1 ... beam_6：外圈六边形光束，按角度 0, 60, ..., 300 度排列。
    """
    centers = [(0.0, 0.0)]
    for index in range(6):
        angle = index * np.pi / 3
        centers.append(
            (
                beam_distance * np.cos(angle),
                beam_distance * np.sin(angle),
            )
        )
    return np.array(centers, dtype=np.float32)


def phase_vector_to_sin_cos(phases):
    """将多路相位向量编码为 [sin(phi_i), cos(phi_i)] 串联形式。"""
    phases = np.asarray(phases, dtype=np.float32)
    encoded = np.empty(phases.size * 2, dtype=np.float32)
    encoded[0::2] = np.sin(phases)
    encoded[1::2] = np.cos(phases)
    return encoded


def sin_cos_to_phase_vector(labels):
    """将多路 sin/cos 标签解码为相位向量。"""
    labels = np.asarray(labels, dtype=np.float32)
    if labels.shape[-1] % 2 != 0:
        raise ValueError(f"Expected an even label dimension, got {labels.shape[-1]}")
    return np.arctan2(labels[..., 0::2], labels[..., 1::2]).astype(np.float32)


def crop_center(image, crop_size):
    """从远场图像中心裁剪正方形区域。"""
    if crop_size > min(image.shape):
        raise ValueError(f"crop_size={crop_size} is larger than image shape {image.shape}")

    center_y = image.shape[0] // 2
    center_x = image.shape[1] // 2
    half = crop_size // 2

    if crop_size % 2 == 0:
        return image[
            center_y - half:center_y + half,
            center_x - half:center_x + half,
        ]

    return image[
        center_y - half:center_y + half + 1,
        center_x - half:center_x + half + 1,
    ]


def far_field_intensity(near_field, normalize=True):
    """通过 FFT 由近场复振幅计算远场光强。"""
    far_field = np.fft.fftshift(np.fft.fft2(near_field))
    intensity = np.abs(far_field) ** 2
    if normalize:
        max_value = np.max(intensity)
        if max_value > 0:
            intensity = intensity / max_value
    return intensity


def add_gaussian_noise(intensity, noise_sigma, rng):
    """给归一化远场光强加入高斯探测噪声。"""
    if noise_sigma <= 0:
        return intensity
    noisy = intensity + rng.normal(0, noise_sigma, intensity.shape)
    return np.clip(noisy, 0, 1)


def seven_beam_near_field(
    x_grid,
    y_grid,
    waist,
    beam_distance,
    phases,
    amplitudes=None,
    position_offsets=None,
):
    """生成 7 光束相干合成近场复振幅。

    相位定义：
    - 中心 beam_0 为参考光束，相位固定为 0。
    - phases 长度为 6，对应外圈 beam_1 ... beam_6 的相对相位。

    amplitudes 可选，长度为 7；position_offsets 可选，形状为 [7, 2]。
    """
    phases = np.asarray(phases, dtype=np.float32)
    if phases.shape != (6,):
        raise ValueError(f"Expected phases with shape (6,), got {phases.shape}")

    centers = seven_beam_centers(beam_distance=beam_distance)
    if amplitudes is None:
        amplitudes = np.ones(7, dtype=np.float32)
    else:
        amplitudes = np.asarray(amplitudes, dtype=np.float32)
        if amplitudes.shape != (7,):
            raise ValueError(f"Expected amplitudes with shape (7,), got {amplitudes.shape}")

    if position_offsets is None:
        position_offsets = np.zeros((7, 2), dtype=np.float32)
    else:
        position_offsets = np.asarray(position_offsets, dtype=np.float32)
        if position_offsets.shape != (7, 2):
            raise ValueError(
                f"Expected position_offsets with shape (7, 2), got {position_offsets.shape}"
            )

    all_phases = np.concatenate([[0.0], phases]).astype(np.float32)
    near_field = np.zeros_like(x_grid, dtype=np.complex64)

    for index, ((center_x, center_y), amplitude, phase) in enumerate(
        zip(centers + position_offsets, amplitudes, all_phases)
    ):
        near_field += gaussian_beam(
            x_grid=x_grid,
            y_grid=y_grid,
            center_x=float(center_x),
            center_y=float(center_y),
            waist=waist,
            amplitude=float(amplitude),
            phase=float(phase),
        ).astype(np.complex64)

    return near_field


def generate_seven_beam_dataset(
    num_samples=1000,
    noise_sigma=0.0,
    num_points=256,
    window_size=10e-3,
    waist=0.5e-3,
    beam_distance=1.5e-3,
    crop_size=160,
    phase_min=-np.pi,
    phase_max=np.pi,
    seed=None,
):
    """生成 7 光束相位反演数据集。

    输出：
    - images: [N, crop_size, crop_size]
    - labels: [N, 12]，即 6 路相位的 sin/cos 编码。
    - phases: [N, 6]，外圈 6 路真实相对相位。
    """
    if phase_min >= phase_max:
        raise ValueError("phase_min must be smaller than phase_max")

    rng = np.random.default_rng(seed)
    x_grid, y_grid = create_grid(num_points=num_points, window_size=window_size)

    images = []
    labels = []
    phases_all = []

    for _ in range(num_samples):
        phases = rng.uniform(phase_min, phase_max, size=6).astype(np.float32)
        near_field = seven_beam_near_field(
            x_grid=x_grid,
            y_grid=y_grid,
            waist=waist,
            beam_distance=beam_distance,
            phases=phases,
        )
        intensity = far_field_intensity(near_field)
        intensity = add_gaussian_noise(intensity, noise_sigma=noise_sigma, rng=rng)
        crop = crop_center(intensity, crop_size=crop_size)

        images.append(crop.astype(np.float32))
        labels.append(phase_vector_to_sin_cos(phases))
        phases_all.append(phases)

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(phases_all, dtype=np.float32),
    )


def seven_beam_dataset_config(
    num_samples,
    noise_sigma,
    num_points,
    window_size,
    waist,
    beam_distance,
    crop_size,
    phase_min,
    phase_max,
    seed,
    image_path,
    label_path,
    phase_path=None,
):
    """构造 7 光束数据集 JSON 配置。"""
    return {
        "task": "seven_beam_phase_estimation",
        "num_beams": 7,
        "num_reference_beams": 1,
        "num_predicted_phases": 6,
        "array_geometry": "center_plus_hexagonal_ring",
        "num_samples": num_samples,
        "noise_sigma": noise_sigma,
        "num_points": num_points,
        "window_size_m": window_size,
        "waist_m": waist,
        "beam_distance_m": beam_distance,
        "crop_size": crop_size,
        "phase_min_rad": phase_min,
        "phase_max_rad": phase_max,
        "seed": seed,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "phase_path": None if phase_path is None else str(phase_path),
        "label_format": "[sin(phi_1), cos(phi_1), ..., sin(phi_6), cos(phi_6)]",
        "image_shape": "[num_samples, crop_size, crop_size]",
        "label_shape": "[num_samples, 12]",
        "phase_shape": "[num_samples, 6]",
        "reference_phase": "beam_0 fixed to 0 rad",
        "beam_centers": seven_beam_centers(beam_distance=beam_distance).tolist(),
    }


def save_dataset(
    images,
    labels,
    phases,
    output_dir,
    image_name,
    label_name,
    phase_name,
    config_name,
    config,
):
    """保存 7 光束数据集和配置文件。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / image_name
    label_path = output_dir / label_name
    phase_path = output_dir / phase_name
    config_path = output_dir / config_name

    np.save(image_path, images)
    np.save(label_path, labels)
    np.save(phase_path, phases)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return image_path, label_path, phase_path, config_path
