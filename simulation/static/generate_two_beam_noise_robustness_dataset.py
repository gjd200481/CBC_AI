import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.two_beam_core import (
    add_gaussian_noise,
    create_grid,
    crop_center,
    dataset_config,
    far_field_intensity,
    phase_to_sin_cos,
    save_dataset,
    two_beam_near_field,
)


def format_noise_tag(noise_sigma):
    if noise_sigma == 0:
        return "0"
    return str(noise_sigma).rstrip("0").rstrip(".")


def generate_images_from_fixed_phases(
    phases,
    noise_sigma,
    rng,
    num_points=256,
    window_size=10e-3,
    waist=0.5e-3,
    beam_distance=1.5e-3,
    crop_size=160,
):
    """用固定相位数组生成指定噪声强度下的远场图像。"""
    x_grid, y_grid = create_grid(num_points=num_points, window_size=window_size)
    images = []

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
        images.append(crop_center(intensity, crop_size=crop_size).astype(np.float32))

    return np.array(images, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Generate noise robustness datasets with shared phase samples."
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0, 0.01, 0.03, 0.05, 0.08])
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--phase-min", type=float, default=-np.pi)
    parser.add_argument("--phase-max", type=float, default=np.pi)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/two_beam/noise_robustness"))
    args = parser.parse_args()

    if args.phase_min >= args.phase_max:
        raise ValueError("phase_min must be smaller than phase_max")

    phase_rng = np.random.default_rng(args.seed)
    phases = phase_rng.uniform(args.phase_min, args.phase_max, size=args.num_samples).astype(np.float32)
    labels = np.array([phase_to_sin_cos(phase) for phase in phases], dtype=np.float32)

    for noise_sigma in args.noise_levels:
        noise_tag = format_noise_tag(noise_sigma)
        prefix = f"noise_{noise_tag}"
        noise_seed = args.seed + int(round(noise_sigma * 10000)) + 17
        noise_rng = np.random.default_rng(noise_seed)

        images = generate_images_from_fixed_phases(
            phases=phases,
            noise_sigma=noise_sigma,
            rng=noise_rng,
            num_points=args.num_points,
            window_size=args.window_size,
            waist=args.waist,
            beam_distance=args.beam_distance,
            crop_size=args.crop_size,
        )

        image_name = f"images_{prefix}.npy"
        label_name = f"labels_{prefix}.npy"
        phase_name = f"phases_{prefix}.npy"
        config_name = f"config_{prefix}.json"

        image_path = args.output_dir / image_name
        label_path = args.output_dir / label_name
        phase_path = args.output_dir / phase_name
        config = dataset_config(
            num_samples=args.num_samples,
            noise_sigma=noise_sigma,
            num_points=args.num_points,
            window_size=args.window_size,
            waist=args.waist,
            beam_distance=args.beam_distance,
            crop_size=args.crop_size,
            phase_min=args.phase_min,
            phase_max=args.phase_max,
            seed=args.seed,
            image_path=image_path,
            label_path=label_path,
            phase_path=phase_path,
        )
        config["dataset_family"] = "noise_robustness_shared_phases"
        config["noise_seed"] = noise_seed

        save_dataset(
            images=images,
            labels=labels,
            output_dir=args.output_dir,
            image_name=image_name,
            label_name=label_name,
            config_name=config_name,
            config=config,
            phases=phases,
            phase_name=phase_name,
        )
        print(
            f"Generated {prefix}: images={images.shape}, labels={labels.shape}, "
            f"noise_sigma={noise_sigma}, noise_seed={noise_seed}"
        )


if __name__ == "__main__":
    main()
