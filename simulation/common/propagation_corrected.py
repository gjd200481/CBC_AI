"""修正后的光场传播模块：支持透镜焦平面 + 离焦探测

关键修正：
- 不再是"自由传播 + FFT"（会导致两平面退化相同）
- 改为"透镜相位 + 传播到探测面 + 直接取|U|²"
- 参考 Hou 2019: f=20m, 焦前 0.4-0.6m
"""
import numpy as np


def fresnel_propagate(near_field, wavelength, distance, pixel_size):
    """菲涅尔衍射传播

    Args:
        near_field: 近场复振幅 [H, W]
        wavelength: 波长 (m)
        distance: 传播距离 (m)，正值为沿光轴传播
        pixel_size: 像素物理尺寸 (m)

    Returns:
        propagated_field: 传播后复振幅 [H, W]
    """
    H, W = near_field.shape
    k = 2 * np.pi / wavelength

    # 频域坐标
    fx = np.fft.fftfreq(W, pixel_size)
    fy = np.fft.fftfreq(H, pixel_size)
    FX, FY = np.meshgrid(fx, fy)

    # 传递函数
    H_fresnel = np.exp(1j * k * distance) * np.exp(-1j * np.pi * wavelength * distance * (FX**2 + FY**2))

    # 传播
    field_fft = np.fft.fft2(near_field)
    propagated_fft = field_fft * H_fresnel
    propagated_field = np.fft.ifft2(propagated_fft)

    return propagated_field


def angular_spectrum_propagate(near_field, wavelength, distance, pixel_size):
    """角谱传播（更精确）

    Args:
        near_field: 近场复振幅 [H, W]
        wavelength: 波长 (m)
        distance: 传播距离 (m)
        pixel_size: 像素物理尺寸 (m)

    Returns:
        propagated_field: 传播后复振幅 [H, W]
    """
    H, W = near_field.shape
    k = 2 * np.pi / wavelength

    # 频域坐标
    fx = np.fft.fftfreq(W, pixel_size)
    fy = np.fft.fftfreq(H, pixel_size)
    FX, FY = np.meshgrid(fx, fy)

    # 传递函数（角谱）
    kz_sq = k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2
    kz_sq = np.maximum(kz_sq, 0)  # 消逝波处理
    kz = np.sqrt(kz_sq)
    H_angular = np.exp(1j * kz * distance)

    # 传播
    field_fft = np.fft.fft2(near_field)
    propagated_fft = field_fft * H_angular
    propagated_field = np.fft.ifft2(propagated_fft)

    return propagated_field


def apply_lens_phase(near_field, wavelength, focal_length, x_grid, y_grid):
    """施加透镜相位调制

    Args:
        near_field: 近场复振幅 [H, W]
        wavelength: 波长 (m)
        focal_length: 透镜焦距 (m)
        x_grid, y_grid: 空间坐标网格 (m)

    Returns:
        field_with_lens: 经过透镜后的复振幅 [H, W]
    """
    k = 2 * np.pi / wavelength

    # 透镜相位: exp(-j * k * (x² + y²) / (2f))
    lens_phase = np.exp(-1j * k * (x_grid**2 + y_grid**2) / (2 * focal_length))

    return near_field * lens_phase


def lens_focus_multiplane_intensity(
    near_field,
    wavelength,
    focal_length,
    defocus_distances,
    x_grid,
    y_grid,
    pixel_size,
    crop_size=None,
    normalize=True,
    method='angular'
):
    """生成透镜焦平面 + 离焦平面的多平面强度

    正确的物理链路：
    1. 近场经过透镜（施加透镜相位）
    2. 传播到焦平面 z=f，或离焦平面 z=f+Δz
    3. 直接取 |U(x,y,z)|² 作为探测强度
    4. 不再对传播后的场做 FFT

    Args:
        near_field: 近场复振幅 [H, W]
        wavelength: 波长 (m)
        focal_length: 透镜焦距 (m)，例如 0.5, 1.0, 2.0
        defocus_distances: 离焦距离列表 (m)，例如 [0, -0.05, 0.05]
            - 0 表示焦平面 z=f
            - -0.05 表示焦前 z=f-0.05
            - +0.05 表示焦后 z=f+0.05
        x_grid, y_grid: 近场空间坐标网格 (m)
        pixel_size: 近场像素物理尺寸 (m)
        crop_size: 裁剪尺寸，None则不裁剪
        normalize: 是否按最大值归一化
        method: 'angular' 或 'fresnel'

    Returns:
        intensities: [num_planes, H_crop, W_crop]
    """
    propagate_fn = angular_spectrum_propagate if method == 'angular' else fresnel_propagate

    # 施加透镜相位
    field_after_lens = apply_lens_phase(near_field, wavelength, focal_length, x_grid, y_grid)

    intensities = []
    for defocus in defocus_distances:
        # 传播到探测面
        distance = focal_length + defocus

        if distance <= 0:
            raise ValueError(f"传播距离必须为正值，当前 f={focal_length}, defocus={defocus}, total={distance}")

        field_at_detector = propagate_fn(field_after_lens, wavelength, distance, pixel_size)

        # 直接取强度（不再做 FFT）
        intensity = np.abs(field_at_detector) ** 2

        # 裁剪
        if crop_size is not None:
            intensity = _crop_center(intensity, crop_size)

        # 归一化
        if normalize:
            max_val = np.max(intensity)
            if max_val > 0:
                intensity = intensity / max_val

        intensities.append(intensity)

    return np.stack(intensities, axis=0)


def _crop_center(image, crop_size):
    """中心裁剪"""
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    half = crop_size // 2

    if crop_size % 2 == 0:
        return image[center_y - half:center_y + half, center_x - half:center_x + half]
    else:
        return image[center_y - half:center_y + half + 1, center_x - half:center_x + half + 1]


# 向后兼容：保留旧函数名，但添加警告
def multiplane_far_field_intensity(near_field, wavelength, distances, pixel_size,
                                   crop_size=None, normalize=True, method='angular'):
    """【已弃用】旧版多平面远场强度生成

    ⚠️ 警告：此函数生成的"多平面"数据存在物理问题：
    - 对传播后的场再做 FFT 会导致不同平面强度退化相同
    - 请改用 lens_focus_multiplane_intensity()

    保留此函数仅为了兼容旧代码。
    """
    import warnings
    warnings.warn(
        "multiplane_far_field_intensity() 已弃用，会导致多平面退化相同。"
        "请改用 lens_focus_multiplane_intensity()。",
        DeprecationWarning,
        stacklevel=2
    )

    propagate_fn = angular_spectrum_propagate if method == 'angular' else fresnel_propagate

    intensities = []
    for dist in distances:
        if dist == 0:
            far_field = np.fft.fftshift(np.fft.fft2(near_field))
        else:
            propagated = propagate_fn(near_field, wavelength, dist, pixel_size)
            far_field = np.fft.fftshift(np.fft.fft2(propagated))

        intensity = np.abs(far_field) ** 2

        if crop_size is not None:
            intensity = _crop_center(intensity, crop_size)

        if normalize:
            max_val = np.max(intensity)
            if max_val > 0:
                intensity = intensity / max_val

        intensities.append(intensity)

    return np.stack(intensities, axis=0)
