"""生成七光束多平面数据集"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.multi_beam_core import (
    create_grid, seven_beam_near_field, phase_vector_to_sin_cos, add_gaussian_noise
)
from simulation.common.propagation import multiplane_far_field_intensity


def generate_multiplane_dataset(
    num_samples=1024,
    distances=(0, -0.05),  # 焦平面和焦前5cm
    wavelength=632.8e-9,
    noise_sigma=0.0,
    num_points=256,
    window_size=0.01,
    waist=0.0005,
    beam_distance=0.0015,
    crop_size=160,
    phase_min=-np.pi,
    phase_max=np.pi,
    seed=20260612,
    output_dir=None,
    prefix="multiplane_seven_beam",
):
    """生成多平面七光束数据集
    
    Args:
        distances: 传播距离元组，单位m，例如 (0, -0.05) 表示焦平面和焦前5cm
    
    Returns:
        images: [N, num_planes, H, W]
        labels: [N, 12] sin/cos编码
        phases: [N, 6] 原始相位
    """
    rng = np.random.RandomState(seed)
    num_planes = len(distances)
    
    # 计算像素物理尺寸
    pixel_size = window_size / num_points
    
    images_list = []
    labels_list = []
    phases_list = []
    
    x_grid, y_grid = create_grid(num_points, window_size)
    
    print(f"Generating {num_samples} samples with {num_planes} planes...")
    start_time = time.time()
    
    for i in range(num_samples):
        # 随机相位
        phases = rng.uniform(phase_min, phase_max, size=6).astype(np.float32)
        
        # 生成近场
        near_field = seven_beam_near_field(x_grid, y_grid, waist, beam_distance, phases)
        
        # 多平面传播
        multiplane_intensity = multiplane_far_field_intensity(
            near_field, wavelength, distances, pixel_size, 
            crop_size=crop_size, normalize=True, method='angular'
        )
        
        # 添加噪声
        if noise_sigma > 0:
            for j in range(num_planes):
                multiplane_intensity[j] = add_gaussian_noise(multiplane_intensity[j], noise_sigma, rng)
        
        # 编码标签
        labels = phase_vector_to_sin_cos(phases)
        
        images_list.append(multiplane_intensity)
        labels_list.append(labels)
        phases_list.append(phases)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  {i+1}/{num_samples} samples, elapsed {elapsed:.1f}s")
    
    images = np.stack(images_list, axis=0).astype(np.float32)
    labels = np.stack(labels_list, axis=0).astype(np.float32)
    phases = np.stack(phases_list, axis=0).astype(np.float32)
    
    print(f"\nDataset generation complete!")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Phases shape: {phases.shape}")
    
    # 保存
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / f"images_{prefix}.npy", images)
        np.save(output_dir / f"labels_{prefix}.npy", labels)
        np.save(output_dir / f"phases_{prefix}.npy", phases)
        
        config = {
            "num_samples": num_samples,
            "num_planes": num_planes,
            "distances": list(distances),
            "wavelength": wavelength,
            "noise_sigma": noise_sigma,
            "num_points": num_points,
            "window_size": window_size,
            "waist": waist,
            "beam_distance": beam_distance,
            "crop_size": crop_size,
            "phase_min": float(phase_min),
            "phase_max": float(phase_max),
            "seed": seed,
        }
        with open(output_dir / f"config_{prefix}.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"\nSaved to {output_dir}/")
    
    return images, labels, phases


def main():
    parser = argparse.ArgumentParser(description="生成七光束多平面数据集")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--distances", type=str, default="0,-0.05", 
                       help="传播距离，逗号分隔，单位m，例如 '0,-0.05,0.05'")
    parser.add_argument("--wavelength", type=float, default=632.8e-9, help="波长(m)")
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=0.01)
    parser.add_argument("--waist", type=float, default=0.0005)
    parser.add_argument("--beam-distance", type=float, default=0.0015)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="multiplane_seven_beam")
    args = parser.parse_args()
    
    distances = tuple(float(d) for d in args.distances.split(","))
    
    generate_multiplane_dataset(
        num_samples=args.num_samples,
        distances=distances,
        wavelength=args.wavelength,
        noise_sigma=args.noise_sigma,
        num_points=args.num_points,
        window_size=args.window_size,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.crop_size,
        seed=args.seed,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
