"""生成多平面七光束数据集：焦前+焦平面+离焦三通道输入。

参考Xie 2024的发现：焦前图像相位信息更集中，RMSE可达0.26 rad。
本脚本生成三个传播距离的远场图像，作为3通道输入。
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.multi_beam_core import (
    create_grid,
    gaussian_beam,
    seven_beam_centers,
    phase_vector_to_sin_cos,
    crop_center,
)


def fresnel_propagate(near_field, z, wavelength=632.8e-9, dx=10e-3/256):
    """菲涅尔传播到距离z的平面。
    
    Args:
        near_field: 近场复振幅 [N, N]
        z: 传播距离 (m)，正值为离焦（远离），负值为焦前（靠近）
        wavelength: 波长 (m)
        dx: 近场采样间隔 (m)
    """
    N = near_field.shape[0]
    k = 2 * np.pi / wavelength
    
    # 频域坐标
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    
    # 传递函数
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    
    # 傅里叶传播
    field_ft = np.fft.fft2(near_field)
    propagated_ft = field_ft * H
    propagated = np.fft.ifft2(propagated_ft)
    
    return propagated


def generate_multiplane_seven_beam_dataset(
    num_samples=1000,
    z_planes=(-5e-3, 0, 5e-3),  # 焦前5mm, 焦平面, 离焦5mm
    num_points=256,
    window_size=10e-3,
    waist=0.5e-3,
    beam_distance=1.5e-3,
    crop_size=160,
    phase_min=-np.pi,
    phase_max=np.pi,
    noise_sigma=0.0,
    seed=20260616,
):
    """生成多平面七光束数据集。
    
    Returns:
        images: [num_samples, 3, crop_size, crop_size] 三通道图像
        labels: [num_samples, 12] sin/cos标签
        phases: [num_samples, 6] 原始相位
    """
    np.random.seed(seed)
    
    x_grid, y_grid = create_grid(num_points, window_size)
    centers = seven_beam_centers(beam_distance)
    
    images = np.zeros((num_samples, 3, crop_size, crop_size), dtype=np.float32)
    labels = np.zeros((num_samples, 12), dtype=np.float32)
    phases = np.zeros((num_samples, 6), dtype=np.float32)
    
    for i in range(num_samples):
        # 随机生成6路相位
        phase_outer = np.random.uniform(phase_min, phase_max, 6).astype(np.float32)
        phases[i] = phase_outer
        labels[i] = phase_vector_to_sin_cos(phase_outer)
        
        # 构建近场
        near_field = np.zeros((num_points, num_points), dtype=np.complex64)
        near_field += gaussian_beam(x_grid, y_grid, centers[0, 0], centers[0, 1], waist, phase=0.0)
        for j in range(6):
            near_field += gaussian_beam(x_grid, y_grid, centers[j+1, 0], centers[j+1, 1], waist, phase=phase_outer[j])
        
        # 传播到三个平面
        for plane_idx, z in enumerate(z_planes):
            if z == 0:
                # 焦平面：直接FFT
                far_field = np.fft.fftshift(np.fft.fft2(near_field))
            else:
                # 焦前/离焦：菲涅尔传播
                propagated = fresnel_propagate(near_field, z)
                far_field = np.fft.fftshift(np.fft.fft2(propagated))
            
            intensity = np.abs(far_field) ** 2
            intensity = intensity / (intensity.max() + 1e-12)
            
            if noise_sigma > 0:
                intensity += np.random.randn(*intensity.shape) * noise_sigma
                intensity = np.clip(intensity, 0, None)
            
            images[i, plane_idx] = crop_center(intensity, crop_size)
        
        if (i + 1) % 500 == 0:
            print(f"Generated {i+1}/{num_samples} samples")
    
    return images, labels, phases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--z-defocus", type=float, default=5e-3, help="Defocus distance in meters")
    p.add_argument("--crop-size", type=int, default=160)
    p.add_argument("--noise-sigma", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20260616)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dataset/seven_beam/multiplane_10k")
    p.add_argument("--prefix", default="multiplane_seven_beam_10k")
    args = p.parse_args()
    
    z_planes = (-args.z_defocus, 0, args.z_defocus)
    
    print(f"Generating multiplane dataset: z={z_planes}")
    images, labels, phases = generate_multiplane_seven_beam_dataset(
        num_samples=args.num_samples,
        z_planes=z_planes,
        crop_size=args.crop_size,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(args.output_dir / f"images_{args.prefix}.npy", images)
    np.save(args.output_dir / f"labels_{args.prefix}.npy", labels)
    np.save(args.output_dir / f"phases_{args.prefix}.npy", phases)
    
    config = {
        "num_samples": args.num_samples,
        "z_planes_m": list(z_planes),
        "crop_size": args.crop_size,
        "noise_sigma": args.noise_sigma,
        "seed": args.seed,
    }
    
    with open(args.output_dir / f"config_{args.prefix}.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nMultiplane dataset generated:")
    print(f"Images: {images.shape} {images.dtype}")
    print(f"Labels: {labels.shape} {labels.dtype}")
    print(f"Phases: {phases.shape} {phases.dtype}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
