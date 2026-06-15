"""生成修正后的七光束多平面数据集

修正内容：
- 使用 lens_focus_multiplane_intensity() 替代旧版 multiplane_far_field_intensity()
- 实现真正的"透镜焦平面 + 离焦探测"物理链路
- 新增焦距参数 focal_length
"""
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
from simulation.common.propagation_corrected import lens_focus_multiplane_intensity


def generate_corrected_multiplane_dataset(
    num_samples=1024,
    focal_length=1.0,  # 新增：透镜焦距(m)
    defocus_distances=(0, -0.05),  # 焦平面和焦前5cm
    wavelength=632.8e-9,
    noise_sigma=0.0,
    num_points=256,
    window_size=0.01,
    waist=0.0005,
    beam_distance=0.0015,
    crop_size=160,
    phase_min=-np.pi,
    phase_max=np.pi,
    seed=20260615,
    output_dir=None,
    prefix="multiplane_corrected",
):
    """生成修正后的多平面七光束数据集

    Args:
        focal_length: 透镜焦距(m)，建议 0.5-2.0
        defocus_distances: 离焦距离元组(m)，例如 (0, -0.05) 表示焦平面和焦前5cm
            - 0: 焦平面 z=f
            - -0.05: 焦前 z=f-0.05
            - +0.05: 焦后 z=f+0.05

    Returns:
        images: [N, num_planes, H, W]
        labels: [N, 12] sin/cos编码
        phases: [N, 6] 原始相位
    """
    rng = np.random.RandomState(seed)
    num_planes = len(defocus_distances)

    # 计算像素物理尺寸
    pixel_size = window_size / num_points

    images_list = []
    labels_list = []
    phases_list = []

    x_grid, y_grid = create_grid(num_points, window_size)

    print(f"Generating CORRECTED multiplane dataset:")
    print(f"  Samples: {num_samples}")
    print(f"  Focal length: {focal_length} m")
    print(f"  Defocus distances: {defocus_distances} m")
    print(f"  Num planes: {num_planes}")
    print(f"  Wavelength: {wavelength*1e9:.1f} nm")
    print(f"  Waist: {waist*1e3:.2f} mm")
    print(f"  Beam distance: {beam_distance*1e3:.2f} mm")

    start_time = time.time()

    for i in range(num_samples):
        # 随机相位
        phases = rng.uniform(phase_min, phase_max, size=6).astype(np.float32)

        # 生成近场
        near_field = seven_beam_near_field(x_grid, y_grid, waist, beam_distance, phases)

        # 多平面传播（修正版）
        multiplane_intensity = lens_focus_multiplane_intensity(
            near_field,
            wavelength,
            focal_length,
            defocus_distances,
            x_grid,
            y_grid,
            pixel_size,
            crop_size=crop_size,
            normalize=True,
            method='angular'
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
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            print(f"  {i+1}/{num_samples} samples | {rate:.1f} samples/s | ETA {eta:.1f}s")

    images = np.stack(images_list, axis=0).astype(np.float32)
    labels = np.stack(labels_list, axis=0).astype(np.float32)
    phases = np.stack(phases_list, axis=0).astype(np.float32)

    elapsed = time.time() - start_time
    print(f"\n[OK] Dataset generation complete in {elapsed:.1f}s!")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Phases shape: {phases.shape}")

    # 验证两通道差异
    if num_planes >= 2:
        plane0 = images[:, 0]
        plane1 = images[:, 1]
        max_diff = np.max(np.abs(plane0 - plane1))
        mean_diff = np.mean(np.abs(plane0 - plane1))
        identical_count = sum(np.allclose(plane0[i], plane1[i], atol=1e-10) for i in range(min(10, num_samples)))

        print(f"\n[OK] Channel difference verification:")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  Identical in first 10: {identical_count}/10")

        if identical_count >= 5:
            print(f"  [WARNING] Channels may still be degenerate!")
        else:
            print(f"  [OK] Channels have significant differences")

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
            "focal_length": focal_length,
            "defocus_distances": list(defocus_distances),
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
            "generation_method": "lens_focus_corrected",
        }
        with open(output_dir / f"config_{prefix}.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n[OK] Saved to {output_dir}/")

    return images, labels, phases


def main():
    parser = argparse.ArgumentParser(description="生成修正后的七光束多平面数据集")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--focal-length", type=float, default=1.0, help="透镜焦距(m)，建议0.5-2.0")
    parser.add_argument("--defocus-distances", type=str, default="0,-0.05",
                       help="离焦距离，逗号分隔(m)，例如 '0,-0.05' 表示焦平面和焦前5cm")
    parser.add_argument("--wavelength", type=float, default=632.8e-9, help="波长(m)")
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=0.01)
    parser.add_argument("--waist", type=float, default=0.0005)
    parser.add_argument("--beam-distance", type=float, default=0.0015)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="multiplane_corrected")
    args = parser.parse_args()

    defocus_distances = tuple(float(d) for d in args.defocus_distances.split(","))

    generate_corrected_multiplane_dataset(
        num_samples=args.num_samples,
        focal_length=args.focal_length,
        defocus_distances=defocus_distances,
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
