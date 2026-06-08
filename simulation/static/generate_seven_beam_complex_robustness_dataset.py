import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.multi_beam_core import (
    crop_center,
    create_grid,
    far_field_intensity,
    phase_vector_to_sin_cos,
    save_dataset,
    seven_beam_dataset_config,
    seven_beam_near_field,
)


def format_float_tag(value):
    """将浮点数转换成适合文件名的短标签。"""
    if value == 0:
        return "0"
    return str(value).rstrip("0").rstrip(".")


def format_offset_tag(offset_m):
    """将位置偏移范围转换为微米标签。"""
    return f"{int(round(offset_m * 1e6))}um"


def generate_amplitude_images(
    phases,
    mismatch_range,
    rng,
    num_points=256,
    window_size=10e-3,
    waist=0.5e-3,
    beam_distance=1.5e-3,
    crop_size=160,
):
    """生成外圈 6 路振幅失配的 7 光束远场图像。"""
    x_grid, y_grid = create_grid(num_points=num_points, window_size=window_size)
    images = []
    amplitudes_all = []

    for phase_vector in phases:
        amplitudes = np.ones(7, dtype=np.float32)
        if mismatch_range > 0:
            amplitudes[1:] = rng.uniform(
                1.0 - mismatch_range,
                1.0 + mismatch_range,
                size=6,
            ).astype(np.float32)

        near_field = seven_beam_near_field(
            x_grid=x_grid,
            y_grid=y_grid,
            waist=waist,
            beam_distance=beam_distance,
            phases=phase_vector,
            amplitudes=amplitudes,
        )
        intensity = far_field_intensity(near_field)
        images.append(crop_center(intensity, crop_size=crop_size).astype(np.float32))
        amplitudes_all.append(amplitudes)

    return np.array(images, dtype=np.float32), np.array(amplitudes_all, dtype=np.float32)


def generate_position_images(
    phases,
    offset_range_m,
    rng,
    num_points=256,
    window_size=10e-3,
    waist=0.5e-3,
    beam_distance=1.5e-3,
    crop_size=160,
):
    """生成 7 路光束中心位置随机偏移的远场图像。"""
    x_grid, y_grid = create_grid(num_points=num_points, window_size=window_size)
    images = []
    offsets_all = []

    for phase_vector in phases:
        offsets = np.zeros((7, 2), dtype=np.float32)
        if offset_range_m > 0:
            offsets = rng.uniform(
                -offset_range_m,
                offset_range_m,
                size=(7, 2),
            ).astype(np.float32)

        near_field = seven_beam_near_field(
            x_grid=x_grid,
            y_grid=y_grid,
            waist=waist,
            beam_distance=beam_distance,
            phases=phase_vector,
            position_offsets=offsets,
        )
        intensity = far_field_intensity(near_field)
        images.append(crop_center(intensity, crop_size=crop_size).astype(np.float32))
        offsets_all.append(offsets)

    return np.array(images, dtype=np.float32), np.array(offsets_all, dtype=np.float32)


def save_extra_array(output_dir, file_name, array):
    """保存振幅或位置偏移等扰动参数数组。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / file_name
    np.save(path, array)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Generate seven-beam amplitude mismatch and position offset robustness datasets."
    )
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--amplitude-levels", nargs="+", type=float, default=[0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument(
        "--position-levels",
        nargs="+",
        type=float,
        default=[0, 1e-5, 2e-5, 5e-5, 1e-4],
        help="Uniform position offset range in meters.",
    )
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--phase-min", type=float, default=-np.pi)
    parser.add_argument("--phase-max", type=float, default=np.pi)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/seven_beam/complex_robustness"))
    args = parser.parse_args()

    if args.phase_min >= args.phase_max:
        raise ValueError("phase_min must be smaller than phase_max")

    phase_rng = np.random.default_rng(args.seed)
    phases = phase_rng.uniform(
        args.phase_min,
        args.phase_max,
        size=(args.num_samples, 6),
    ).astype(np.float32)
    labels = np.array([phase_vector_to_sin_cos(item) for item in phases], dtype=np.float32)

    for mismatch_range in args.amplitude_levels:
        tag = format_float_tag(mismatch_range)
        prefix = f"amplitude_{tag}"
        rng = np.random.default_rng(args.seed + int(round(mismatch_range * 10000)) + 131)
        images, amplitudes = generate_amplitude_images(
            phases=phases,
            mismatch_range=mismatch_range,
            rng=rng,
            num_points=args.num_points,
            window_size=args.window_size,
            waist=args.waist,
            beam_distance=args.beam_distance,
            crop_size=args.crop_size,
        )

        config = seven_beam_dataset_config(
            num_samples=args.num_samples,
            noise_sigma=0,
            num_points=args.num_points,
            window_size=args.window_size,
            waist=args.waist,
            beam_distance=args.beam_distance,
            crop_size=args.crop_size,
            phase_min=args.phase_min,
            phase_max=args.phase_max,
            seed=args.seed,
            image_path=args.output_dir / f"images_{prefix}.npy",
            label_path=args.output_dir / f"labels_{prefix}.npy",
            phase_path=args.output_dir / f"phases_{prefix}.npy",
        )
        config["dataset_family"] = "seven_beam_amplitude_mismatch_shared_phases"
        config["amplitude_mismatch_range"] = mismatch_range
        config["amplitude_file"] = str(args.output_dir / f"amplitudes_{prefix}.npy")

        save_dataset(
            images=images,
            labels=labels,
            phases=phases,
            output_dir=args.output_dir,
            image_name=f"images_{prefix}.npy",
            label_name=f"labels_{prefix}.npy",
            phase_name=f"phases_{prefix}.npy",
            config_name=f"config_{prefix}.json",
            config=config,
        )
        save_extra_array(args.output_dir, f"amplitudes_{prefix}.npy", amplitudes)
        print(f"Generated {prefix}: images={images.shape}, labels={labels.shape}")

    for offset_range_m in args.position_levels:
        tag = format_offset_tag(offset_range_m)
        prefix = f"position_{tag}"
        rng = np.random.default_rng(args.seed + int(round(offset_range_m * 1e9)) + 257)
        images, offsets = generate_position_images(
            phases=phases,
            offset_range_m=offset_range_m,
            rng=rng,
            num_points=args.num_points,
            window_size=args.window_size,
            waist=args.waist,
            beam_distance=args.beam_distance,
            crop_size=args.crop_size,
        )

        config = seven_beam_dataset_config(
            num_samples=args.num_samples,
            noise_sigma=0,
            num_points=args.num_points,
            window_size=args.window_size,
            waist=args.waist,
            beam_distance=args.beam_distance,
            crop_size=args.crop_size,
            phase_min=args.phase_min,
            phase_max=args.phase_max,
            seed=args.seed,
            image_path=args.output_dir / f"images_{prefix}.npy",
            label_path=args.output_dir / f"labels_{prefix}.npy",
            phase_path=args.output_dir / f"phases_{prefix}.npy",
        )
        config["dataset_family"] = "seven_beam_position_offset_shared_phases"
        config["position_offset_range_m"] = offset_range_m
        config["offset_file"] = str(args.output_dir / f"offsets_{prefix}.npy")

        save_dataset(
            images=images,
            labels=labels,
            phases=phases,
            output_dir=args.output_dir,
            image_name=f"images_{prefix}.npy",
            label_name=f"labels_{prefix}.npy",
            phase_name=f"phases_{prefix}.npy",
            config_name=f"config_{prefix}.json",
            config=config,
        )
        save_extra_array(args.output_dir, f"offsets_{prefix}.npy", offsets)
        print(f"Generated {prefix}: images={images.shape}, labels={labels.shape}")


if __name__ == "__main__":
    main()
