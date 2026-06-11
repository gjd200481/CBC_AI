"""光场传播模块：支持菲涅尔传播和角谱传播"""
import numpy as np


def fresnel_propagate(near_field, wavelength, distance, pixel_size):
    """菲涅尔衍射传播
    
    Args:
        near_field: 近场复振幅 [H, W]
        wavelength: 波长 (m)
        distance: 传播距离 (m)，正值为沿光轴传播，负值为反向
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


def multiplane_far_field_intensity(near_field, wavelength, distances, pixel_size, 
                                   crop_size=None, normalize=True, method='angular'):
    """生成多平面远场强度
    
    Args:
        near_field: 近场复振幅 [H, W]
        wavelength: 波长 (m)
        distances: 传播距离列表 (m)，例如 [0, -0.05, 0.05]
        pixel_size: 近场像素物理尺寸 (m)
        crop_size: 裁剪尺寸，None则不裁剪
        normalize: 是否按最大值归一化
        method: 'angular' 或 'fresnel'
    
    Returns:
        intensities: [num_planes, H_crop, W_crop]
    """
    propagate_fn = angular_spectrum_propagate if method == 'angular' else fresnel_propagate
    
    intensities = []
    for dist in distances:
        if dist == 0:
            # 焦平面直接FFT
            far_field = np.fft.fftshift(np.fft.fft2(near_field))
        else:
            # 先传播再FFT
            propagated = propagate_fn(near_field, wavelength, dist, pixel_size)
            far_field = np.fft.fftshift(np.fft.fft2(propagated))
        
        intensity = np.abs(far_field) ** 2
        
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
