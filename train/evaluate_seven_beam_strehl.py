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
from train.evaluate_seven_beam_compensation_metrics import (
    farfield_crop_from_phases,
    predict_labels,
)
from train.evaluate_seven_beam_noise_robustness import load_seven_beam_model
from train.phase_metrics import decode_sin_cos, wrap_phase_error


def phase_rmse_from_phase_vector(phases):
    """计算单个样本 6 路残余相位的周期 RMSE。"""
    wrapped = wrap_phase_error(phases, np.zeros_like(phases))
    return float(np.sqrt(np.mean(wrapped**2)))


def summarize_state(rows, state_name):
    """汇总某个状态的 Strehl 比和残余相位 RMSE。"""
    state_rows = [row for row in rows if row["state"] == state_name]
    strehl_values = np.array([row["strehl_ratio"] for row in state_rows], dtype=np.float64)
    phase_values = np.array([row["phase_rmse_rad"] for row in state_rows], dtype=np.float64)
    peak_values = np.array([row["peak_intensity"] for row in state_rows], dtype=np.float64)

    return {
        "state": state_name,
        "samples": len(state_rows),
        "strehl_mean": float(np.mean(strehl_values)),
        "strehl_std": float(np.std(strehl_values)),
        "strehl_min": float(np.min(strehl_values)),
        "strehl_max": float(np.max(strehl_values)),
        "phase_rmse_mean_rad": float(np.mean(phase_values)),
        "phase_rmse_std_rad": float(np.std(phase_values)),
        "peak_intensity_mean": float(np.mean(peak_values)),
    }


def save_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(detail_rows, summary_rows, example_images, figure_path):
    """绘制 Strehl 统计、相位 RMSE 关系和典型远场。"""
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    states = [row["state"] for row in summary_rows]
    strehl_means = [row["strehl_mean"] for row in summary_rows]
    strehl_stds = [row["strehl_std"] for row in summary_rows]

    plt.figure(figsize=(14, 9))

    plt.subplot(2, 3, 1)
    plt.bar(states, strehl_means, yerr=strehl_stds, capsize=4)
    plt.xticks(rotation=20)
    plt.ylabel("Strehl ratio")
    plt.title("Mean Strehl ratio")

    plt.subplot(2, 3, 2)
    for state in ["before", "baseline_compensated", "physics_compensated"]:
        state_rows = [row for row in detail_rows if row["state"] == state]
        plt.scatter(
            [row["phase_rmse_rad"] for row in state_rows],
            [row["strehl_ratio"] for row in state_rows],
            s=8,
            alpha=0.6,
            label=state,
        )
    plt.xlabel("Residual phase RMSE(rad)")
    plt.ylabel("Strehl ratio")
    plt.title("Phase RMSE vs Strehl")
    plt.legend()

    for index, state in enumerate(["before", "baseline_compensated", "physics_compensated", "ideal"], start=3):
        plt.subplot(2, 3, index)
        image = example_images[state]
        display = np.log10(image / max(float(np.max(image)), 1e-12) + 1e-6)
        plt.imshow(display, cmap="inferno")
        plt.axis("off")
        plt.title(state)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print("Figure saved to:", figure_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate seven-beam Strehl ratio before and after phase compensation."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=REPO_ROOT
        / "dataset"
        / "seven_beam"
        / "main_static"
        / "images_main_clean_seven_beam.npy",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=REPO_ROOT
        / "dataset"
        / "seven_beam"
        / "main_static"
        / "labels_main_clean_seven_beam.npy",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=REPO_ROOT / "models" / "baseline_cnn_main_clean_seven_beam_2026-06-08.pth",
    )
    parser.add_argument(
        "--physics-model",
        type=Path,
        default=REPO_ROOT / "models" / "physics_cnn_lambda_0.1_main_clean_seven_beam_2026-06-08.pth",
    )
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "cycle18_seven_beam_strehl_detail_2026-06-09.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "cycle18_seven_beam_strehl_summary_2026-06-09.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "figures"
        / "cycle18_seven_beam_strehl_2026-06-09.png",
    )
    args = parser.parse_args()

    images = np.load(args.image_path)
    labels = np.load(args.label_path)
    if args.max_samples is not None:
        images = images[:args.max_samples]
        labels = labels[:args.max_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = load_seven_beam_model(args.baseline_model, device)
    physics_model = load_seven_beam_model(args.physics_model, device)

    baseline_pred = predict_labels(
        model=baseline_model,
        images=images,
        batch_size=args.batch_size,
        device=device,
    )
    physics_pred = predict_labels(
        model=physics_model,
        images=images,
        batch_size=args.batch_size,
        device=device,
    )

    true_phases = decode_sin_cos(labels)
    baseline_phases = decode_sin_cos(baseline_pred)
    physics_phases = decode_sin_cos(physics_pred)

    baseline_residual = wrap_phase_error(true_phases, baseline_phases)
    physics_residual = wrap_phase_error(true_phases, physics_phases)
    ideal_phases = np.zeros(6, dtype=np.float32)

    x_grid, y_grid = create_grid(num_points=args.num_points, window_size=args.window_size)
    ideal_farfield = farfield_crop_from_phases(
        phases=ideal_phases,
        x_grid=x_grid,
        y_grid=y_grid,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.crop_size,
    )
    ideal_peak = float(np.max(ideal_farfield))

    detail_rows = []
    example_images = {}

    for index in range(len(images)):
        phase_sets = {
            "before": true_phases[index],
            "baseline_compensated": baseline_residual[index],
            "physics_compensated": physics_residual[index],
            "ideal": ideal_phases,
        }
        for state, phases in phase_sets.items():
            farfield = farfield_crop_from_phases(
                phases=phases,
                x_grid=x_grid,
                y_grid=y_grid,
                waist=args.waist,
                beam_distance=args.beam_distance,
                crop_size=args.crop_size,
            )
            peak = float(np.max(farfield))
            phase_rmse = phase_rmse_from_phase_vector(phases)
            detail_rows.append(
                {
                    "sample_index": index,
                    "state": state,
                    "strehl_ratio": peak / ideal_peak,
                    "peak_intensity": peak,
                    "ideal_peak_intensity": ideal_peak,
                    "phase_rmse_rad": phase_rmse,
                }
            )
            if index == args.example_index:
                example_images[state] = farfield

    summary_rows = [
        summarize_state(detail_rows, "before"),
        summarize_state(detail_rows, "baseline_compensated"),
        summarize_state(detail_rows, "physics_compensated"),
        summarize_state(detail_rows, "ideal"),
    ]

    before_mean = summary_rows[0]["strehl_mean"]
    for row in summary_rows:
        if row["state"] == "before":
            row["relative_gain_vs_before_percent"] = 0.0
        else:
            row["relative_gain_vs_before_percent"] = 100 * (
                row["strehl_mean"] - before_mean
            ) / before_mean

    save_csv(detail_rows, args.detail_csv)
    save_csv(summary_rows, args.summary_csv)
    plot_results(detail_rows, summary_rows, example_images, args.figure_path)

    print("Using device:", device)
    print("Samples:", len(images))
    print("Ideal peak intensity:", ideal_peak)
    for row in summary_rows:
        print(
            f"{row['state']} | strehl={row['strehl_mean']:.6f} | "
            f"phase_rmse={row['phase_rmse_mean_rad']:.6f} rad | "
            f"gain={row['relative_gain_vs_before_percent']:.2f}%"
        )
    print("Detail saved to:", args.detail_csv)
    print("Summary saved to:", args.summary_csv)


if __name__ == "__main__":
    main()
