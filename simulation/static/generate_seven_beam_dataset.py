import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.multi_beam_core import (
    generate_seven_beam_dataset,
    save_dataset,
    seven_beam_dataset_config,
)


def format_noise_tag(noise_sigma):
    if noise_sigma == 0:
        return "0"
    return str(noise_sigma).rstrip("0").rstrip(".")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a reproducible seven-beam far-field phase dataset."
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--phase-min", type=float, default=-3.141592653589793)
    parser.add_argument("--phase-max", type=float, default=3.141592653589793)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/seven_beam"))
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    prefix = args.prefix
    if prefix is None:
        prefix = f"noise_{format_noise_tag(args.noise_sigma)}"

    image_name = f"images_{prefix}.npy"
    label_name = f"labels_{prefix}.npy"
    phase_name = f"phases_{prefix}.npy"
    config_name = f"config_{prefix}.json"

    images, labels, phases = generate_seven_beam_dataset(
        num_samples=args.num_samples,
        noise_sigma=args.noise_sigma,
        num_points=args.num_points,
        window_size=args.window_size,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.crop_size,
        phase_min=args.phase_min,
        phase_max=args.phase_max,
        seed=args.seed,
    )

    image_path = args.output_dir / image_name
    label_path = args.output_dir / label_name
    phase_path = args.output_dir / phase_name
    config = seven_beam_dataset_config(
        num_samples=args.num_samples,
        noise_sigma=args.noise_sigma,
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

    image_path, label_path, phase_path, config_path = save_dataset(
        images=images,
        labels=labels,
        phases=phases,
        output_dir=args.output_dir,
        image_name=image_name,
        label_name=label_name,
        phase_name=phase_name,
        config_name=config_name,
        config=config,
    )

    print("Seven-beam dataset generated successfully!")
    print("Images:", image_path, images.shape, images.dtype)
    print("Labels:", label_path, labels.shape, labels.dtype)
    print("Phases:", phase_path, phases.shape, phases.dtype)
    print("Config:", config_path)

    if args.show:
        plt.imshow(images[0], cmap="jet")
        plt.title("Seven-beam far field")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()
