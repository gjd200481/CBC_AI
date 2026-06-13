import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.multi_beam_core import create_grid
from train.evaluate_seven_beam_compensation_effect import farfield_metrics, phase_rmse_from_vector
from train.evaluate_seven_beam_compensation_metrics import (
    farfield_crop_from_phases,
    make_main_lobe_mask,
    predict_labels,
)
from train.evaluate_seven_beam_noise_robustness import load_seven_beam_model
from train.phase_metrics import decode_sin_cos, wrap_phase_error


METRIC_NAMES = [
    "main_lobe_ratio",
    "strehl_ratio",
    "synthesis_efficiency",
    "phase_rmse_rad",
]


def parse_model_specs(model_specs):
    parsed = []
    for spec in model_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --model value {spec!r}. Expected name=checkpoint_path.")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = Path(path.strip())
        if not name:
            raise ValueError(f"Invalid --model value {spec!r}: empty model name.")
        parsed.append((name, path))
    return parsed


def save_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model_on_images(
    model,
    images,
    labels,
    batch_size,
    device,
    x_grid,
    y_grid,
    main_lobe_mask,
    ideal_peak,
    ideal_main_lobe_energy,
    args,
):
    pred = predict_labels(model=model, images=images, batch_size=batch_size, device=device)
    true_phases = decode_sin_cos(labels)
    pred_phases = decode_sin_cos(pred)
    residual_phases = wrap_phase_error(true_phases, pred_phases)

    detail = []
    for sample_index, phases in enumerate(residual_phases):
        farfield = farfield_crop_from_phases(
            phases=phases,
            x_grid=x_grid,
            y_grid=y_grid,
            waist=args.waist,
            beam_distance=args.beam_distance,
            crop_size=args.crop_size,
        )
        metrics = farfield_metrics(
            farfield=farfield,
            main_lobe_mask=main_lobe_mask,
            ideal_peak=ideal_peak,
            ideal_main_lobe_energy=ideal_main_lobe_energy,
        )
        metrics["phase_rmse_rad"] = phase_rmse_from_vector(phases)
        detail.append(metrics)
    return detail


def summarize_detail(model_name, noise_sigma, detail):
    row = {
        "model": model_name,
        "noise_sigma": noise_sigma,
        "samples": len(detail),
    }
    for metric_name in METRIC_NAMES:
        values = np.array([item[metric_name] for item in detail], dtype=np.float64)
        row[f"{metric_name}_mean"] = float(values.mean())
        row[f"{metric_name}_std"] = float(values.std())
    return row


def plot_noise_summary(rows, figure_path):
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])

    panels = [
        ("main_lobe_ratio_mean", "Main-lobe ratio"),
        ("strehl_ratio_mean", "Strehl"),
        ("synthesis_efficiency_mean", "Synthesis efficiency"),
        ("phase_rmse_rad_mean", "Residual phase RMSE (rad)"),
    ]
    plt.figure(figsize=(12, 8))
    for index, (metric_name, title) in enumerate(panels, start=1):
        plt.subplot(2, 2, index)
        for model_name in models:
            model_rows = [row for row in rows if row["model"] == model_name]
            xs = [row["noise_sigma"] for row in model_rows]
            ys = [row[metric_name] for row in model_rows]
            plt.plot(xs, ys, marker="o", linewidth=2, label=model_name)
        plt.xlabel("Input noise sigma")
        plt.title(title)
        plt.grid(alpha=0.25)
        if index == 1:
            plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()
    print("Figure saved to:", figure_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate noise robustness for multiplane seven-beam models.")
    parser.add_argument("--image-path", type=Path, default=REPO_ROOT / "dataset/seven_beam/multiplane_0_-0.07/images_multiplane_7cm.npy")
    parser.add_argument("--label-path", type=Path, default=REPO_ROOT / "dataset/seven_beam/multiplane_0_-0.07/labels_multiplane_7cm.npy")
    parser.add_argument("--model", action="append", required=True, help="Model spec in name=checkpoint_path format.")
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.002, 0.005, 0.01, 0.02])
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--main-lobe-radius", type=int, default=3)
    parser.add_argument("--summary-csv", type=Path, default=REPO_ROOT / "result/metrics/cycle43_dual_plane_noise_robustness_summary.csv")
    parser.add_argument("--figure-path", type=Path, default=REPO_ROOT / "result/figures/cycle43_dual_plane_noise_robustness.png")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    images = np.load(args.image_path)
    labels = np.load(args.label_path)
    if args.max_samples is not None:
        images = images[: args.max_samples]
        labels = labels[: args.max_samples]

    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model_specs = parse_model_specs(args.model)
    models = []
    for model_name, model_path in model_specs:
        models.append((model_name, load_seven_beam_model(model_path, device)))

    x_grid, y_grid = create_grid(num_points=args.num_points, window_size=args.window_size)
    main_lobe_mask = make_main_lobe_mask(image_size=args.crop_size, radius=args.main_lobe_radius)
    ideal_farfield = farfield_crop_from_phases(
        phases=np.zeros(6, dtype=np.float32),
        x_grid=x_grid,
        y_grid=y_grid,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.crop_size,
    )
    ideal_peak = float(np.max(ideal_farfield))
    ideal_main_lobe_energy = float(np.sum(ideal_farfield[main_lobe_mask]))

    rows = []
    for level_index, noise_sigma in enumerate(args.noise_levels):
        rng = np.random.default_rng(args.seed + level_index)
        if noise_sigma == 0:
            noisy_images = images.astype(np.float32, copy=True)
        else:
            noise = rng.normal(0.0, noise_sigma, size=images.shape).astype(np.float32)
            noisy_images = np.clip(images + noise, 0.0, 1.0).astype(np.float32)
        for model_name, model in models:
            detail = evaluate_model_on_images(
                model=model,
                images=noisy_images,
                labels=labels,
                batch_size=args.batch_size,
                device=device,
                x_grid=x_grid,
                y_grid=y_grid,
                main_lobe_mask=main_lobe_mask,
                ideal_peak=ideal_peak,
                ideal_main_lobe_energy=ideal_main_lobe_energy,
                args=args,
            )
            row = summarize_detail(model_name, noise_sigma, detail)
            rows.append(row)
            print(
                f"{model_name} | noise={noise_sigma:g} | "
                f"main={row['main_lobe_ratio_mean']:.6f} | "
                f"strehl={row['strehl_ratio_mean']:.6f} | "
                f"eff={row['synthesis_efficiency_mean']:.6f} | "
                f"rmse={row['phase_rmse_rad_mean']:.6f}"
            )

    save_csv(rows, args.summary_csv)
    plot_noise_summary(rows, args.figure_path)
    print("Summary saved to:", args.summary_csv)


if __name__ == "__main__":
    main()
