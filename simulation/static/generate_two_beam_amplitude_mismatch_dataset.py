import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.two_beam_core import (
    crop_center,
    create_grid,
    dataset_config,
    far_field_intensity,
    gaussian_beam,
    phase_to_sin_cos,
    save_dataset,
)


def format_mismatch_tag(mismatch):
    if mismatch == 0:
        return "0"
    return str(mismatch).rstrip("0").rstrip(".")


def two_beam_near_field_with_amplitude(
    x_grid,
    y_grid,
    waist,
    beam_distance,
    phase,
    amplitude_1=1.0,
    amplitude_2=1.0,
):
    """生成带振幅失配的双光束近场复振幅。"""
    beam_1 = gaussian_beam(
        x_grid=x_grid,
        y_grid=y_grid,
        center_x=-beam_distance / 2,
        center_y=0.0,
        waist=waist,
        amplitude=amplitude_1,
        phase=0.0,
    )
    beam_2 = gaussian_beam(
        x_grid=x_grid,
        y_grid=y_grid,
        center_x=beam_distance / 2,
        center_y=0.0,
        waist=waist,
        amplitude=amplitude_2,
        phase=phase,
    )
    return beam_1 + beam_2


def generate_images_from_fixed_phases(
    phases,
    mismatch_range,
    rng,
    num_points=256,
    window_size=10e-3,
    waist=0.5e-3,
    beam_distance=1.5e-3,
    crop_size=160,
):
    """用固定相位数组生成指定振幅失配范围下的远场图像。

    amplitude_1 固定为 1.0，amplitude_2 从 [1-r, 1+r] 均匀采样。
    """
    x_grid, y_grid = create_grid(num_points=num_points, window_size=window_size)
    images = []
    amplitude_2_values = rng.uniform(
        1.0 - mismatch_range,
        1.0 + mismatch_range,
        size=len(phases),
    ).astype(np.float32)

    for phase, amplitude_2 in zip(phases, amplitude_2_values):
        near_field = two_beam_near_field_with_amplitude(
            x_grid=x_grid,
            y_grid=y_grid,
            waist=waist,
            beam_distance=beam_distance,
            phase=phase,
            amplitude_1=1.0,
            amplitude_2=float(amplitude_2),
        )
        intensity = far_field_intensity(near_field)
        images.append(crop_center(intensity, crop_size=crop_size).astype(np.float32))

    return np.array(images, dtype=np.float32), amplitude_2_values


def main():
    parser = argparse.ArgumentParser(
        description="Generate amplitude mismatch datasets with shared phase samples."
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--mismatch-levels", nargs="+", type=float, default=[0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--phase-min", type=float, default=-np.pi)
    parser.add_argument("--phase-max", type=float, default=np.pi)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/two_beam/amplitude_mismatch"))
    args = parser.parse_args()

    if args.phase_min >= args.phase_max:
        raise ValueError("phase_min must be smaller than phase_max")

    phase_rng = np.random.default_rng(args.seed)
    phases = phase_rng.uniform(args.phase_min, args.phase_max, size=args.num_samples).astype(np.float32)
    labels = np.array([phase_to_sin_cos(phase) for phase in phases], dtype=np.float32)

    for mismatch in args.mismatch_levels:
        if mismatch < 0 or mismatch >= 1:
            raise ValueError("mismatch levels must be in [0, 1)")

        mismatch_tag = format_mismatch_tag(mismatch)
        prefix = f"amp_{mismatch_tag}"
        mismatch_seed = args.seed + int(round(mismatch * 10000)) + 31
        mismatch_rng = np.random.default_rng(mismatch_seed)

        images, amplitude_2_values = generate_images_from_fixed_phases(
            phases=phases,
            mismatch_range=mismatch,
            rng=mismatch_rng,
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
        amplitude_name = f"amplitude2_{prefix}.npy"

        image_path = args.output_dir / image_name
        label_path = args.output_dir / label_name
        phase_path = args.output_dir / phase_name
        config = dataset_config(
            num_samples=args.num_samples,
            noise_sigma=0.0,
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
        config["dataset_family"] = "amplitude_mismatch_shared_phases"
        config["mismatch_range"] = mismatch
        config["amplitude_1"] = 1.0
        config["amplitude_2_range"] = [1.0 - mismatch, 1.0 + mismatch]
        config["mismatch_seed"] = mismatch_seed
        config["amplitude2_path"] = str(args.output_dir / amplitude_name)

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
        np.save(args.output_dir / amplitude_name, amplitude_2_values)
        print(
            f"Generated {prefix}: images={images.shape}, labels={labels.shape}, "
            f"amplitude_2=[{amplitude_2_values.min():.3f}, {amplitude_2_values.max():.3f}]"
        )


if __name__ == "__main__":
    main()
