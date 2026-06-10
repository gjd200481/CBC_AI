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
    make_main_lobe_mask,
    predict_labels,
)
from train.evaluate_seven_beam_noise_robustness import load_seven_beam_model
from train.phase_metrics import decode_sin_cos, wrap_phase_error


METRIC_NAMES = [
    "main_lobe_ratio",
    "side_lobe_ratio",
    "strehl_ratio",
    "synthesis_efficiency",
    "peak_to_sidelobe_ratio",
    "phase_rmse_rad",
]


def phase_rmse_from_vector(phases):
    """计算单个样本 6 路残余相位的周期 RMSE。"""
    wrapped = wrap_phase_error(phases, np.zeros_like(phases))
    return float(np.sqrt(np.mean(wrapped**2)))


def farfield_metrics(farfield, main_lobe_mask, ideal_peak, ideal_main_lobe_energy, eps=1e-12):
    """从未归一化远场光强中计算补偿效果指标。"""
    total_energy = float(np.sum(farfield))
    main_lobe_energy = float(np.sum(farfield[main_lobe_mask]))
    side_lobe_energy = max(total_energy - main_lobe_energy, 0.0)
    peak_intensity = float(np.max(farfield))
    max_side_lobe = float(np.max(farfield[~main_lobe_mask]))

    return {
        "main_lobe_energy": main_lobe_energy,
        "side_lobe_energy": side_lobe_energy,
        "total_energy": total_energy,
        "main_lobe_ratio": main_lobe_energy / max(total_energy, eps),
        "side_lobe_ratio": side_lobe_energy / max(total_energy, eps),
        "peak_intensity": peak_intensity,
        "max_side_lobe_intensity": max_side_lobe,
        "strehl_ratio": peak_intensity / max(ideal_peak, eps),
        "synthesis_efficiency": main_lobe_energy / max(ideal_main_lobe_energy, eps),
        "peak_to_sidelobe_ratio": peak_intensity / max(max_side_lobe, eps),
    }


def summarize_state(rows, state_name):
    """汇总某个状态下所有补偿指标的均值、标准差和范围。"""
    state_rows = [row for row in rows if row["state"] == state_name]
    summary = {
        "state": state_name,
        "samples": len(state_rows),
    }

    for metric_name in METRIC_NAMES:
        values = np.array([row[metric_name] for row in state_rows], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(np.mean(values))
        summary[f"{metric_name}_std"] = float(np.std(values))
        summary[f"{metric_name}_min"] = float(np.min(values))
        summary[f"{metric_name}_max"] = float(np.max(values))

    return summary


def add_relative_gain(summary_rows, baseline_state="before"):
    """补充相对补偿前的百分比变化，便于论文表格直接引用。"""
    baseline = next(row for row in summary_rows if row["state"] == baseline_state)

    higher_is_better = [
        "main_lobe_ratio",
        "strehl_ratio",
        "synthesis_efficiency",
        "peak_to_sidelobe_ratio",
    ]
    lower_is_better = [
        "side_lobe_ratio",
        "phase_rmse_rad",
    ]

    for row in summary_rows:
        for metric_name in higher_is_better + lower_is_better:
            base_value = baseline[f"{metric_name}_mean"]
            value = row[f"{metric_name}_mean"]
            if abs(base_value) < 1e-12:
                gain = 0.0
            elif metric_name in higher_is_better:
                gain = 100 * (value - base_value) / base_value
            else:
                gain = 100 * (base_value - value) / base_value
            row[f"{metric_name}_gain_vs_before_percent"] = float(gain)


def save_csv(rows, output_path):
    """保存字典列表为 CSV 文件。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_compensation_effect(summary_rows, example_images, figure_path):
    """绘制综合补偿指标和典型远场图。"""
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    states = [row["state"] for row in summary_rows]
    base_colors = ["#4B5563", "#2563EB", "#DC2626", "#7C3AED", "#059669"]
    colors = [base_colors[index % len(base_colors)] for index in range(len(states))]

    metric_panels = [
        ("main_lobe_ratio", "Main-lobe energy ratio"),
        ("strehl_ratio", "Strehl ratio"),
        ("synthesis_efficiency", "Synthesis efficiency"),
        ("side_lobe_ratio", "Side-lobe energy ratio"),
        ("phase_rmse_rad", "Residual phase RMSE(rad)"),
        ("peak_to_sidelobe_ratio", "Peak-to-sidelobe ratio"),
    ]

    plt.figure(figsize=(18, 12))

    for panel_index, (metric_name, title) in enumerate(metric_panels, start=1):
        plt.subplot(3, 4, panel_index)
        means = [row[f"{metric_name}_mean"] for row in summary_rows]
        stds = [row[f"{metric_name}_std"] for row in summary_rows]
        plt.bar(states, means, yerr=stds, capsize=4, color=colors)
        plt.xticks(rotation=18, ha="right")
        plt.title(title)

    for panel_index, state in enumerate(states, start=7):
        plt.subplot(3, 4, panel_index)
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
        description="Evaluate seven-beam phase compensation with main-lobe, side-lobe, Strehl and efficiency metrics."
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
    parser.add_argument("--candidate-model", type=Path, default=None)
    parser.add_argument("--candidate-name", default="candidate_compensated")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--main-lobe-radius", type=int, default=3)
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "cycle19_seven_beam_compensation_effect_detail_2026-06-09.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "cycle19_seven_beam_compensation_effect_summary_2026-06-09.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "figures"
        / "cycle19_seven_beam_compensation_effect_2026-06-09.png",
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
    candidate_model = None
    if args.candidate_model is not None:
        candidate_model = load_seven_beam_model(args.candidate_model, device)

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
    candidate_pred = None
    if candidate_model is not None:
        candidate_pred = predict_labels(
            model=candidate_model,
            images=images,
            batch_size=args.batch_size,
            device=device,
        )

    true_phases = decode_sin_cos(labels)
    baseline_phases = decode_sin_cos(baseline_pred)
    physics_phases = decode_sin_cos(physics_pred)
    candidate_phases = None
    if candidate_pred is not None:
        candidate_phases = decode_sin_cos(candidate_pred)

    phase_sets_by_state = {
        "before": true_phases,
        "baseline_compensated": wrap_phase_error(true_phases, baseline_phases),
        "physics_compensated": wrap_phase_error(true_phases, physics_phases),
        "ideal": np.zeros_like(true_phases),
    }
    if candidate_phases is not None:
        phase_sets_by_state[args.candidate_name] = wrap_phase_error(true_phases, candidate_phases)

    x_grid, y_grid = create_grid(num_points=args.num_points, window_size=args.window_size)
    main_lobe_mask = make_main_lobe_mask(
        image_size=args.crop_size,
        radius=args.main_lobe_radius,
    )
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

    detail_rows = []
    example_images = {}
    states = ["before", "baseline_compensated", "physics_compensated"]
    if candidate_phases is not None:
        states.append(args.candidate_name)
    states.append("ideal")

    for sample_index in range(len(images)):
        for state in states:
            phases = phase_sets_by_state[state][sample_index]
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

            detail_rows.append(
                {
                    "sample_index": sample_index,
                    "state": state,
                    **metrics,
                }
            )
            if sample_index == args.example_index:
                example_images[state] = farfield

    summary_rows = [summarize_state(detail_rows, state) for state in states]
    add_relative_gain(summary_rows)

    save_csv(detail_rows, args.detail_csv)
    save_csv(summary_rows, args.summary_csv)
    plot_compensation_effect(summary_rows, example_images, args.figure_path)

    print("Using device:", device)
    print("Samples:", len(images))
    print("Main-lobe radius(px):", args.main_lobe_radius)
    print("Ideal peak intensity:", ideal_peak)
    print("Ideal main-lobe energy:", ideal_main_lobe_energy)
    for row in summary_rows:
        print(
            f"{row['state']} | main={row['main_lobe_ratio_mean']:.6f} | "
            f"side={row['side_lobe_ratio_mean']:.6f} | "
            f"strehl={row['strehl_ratio_mean']:.6f} | "
            f"eff={row['synthesis_efficiency_mean']:.6f} | "
            f"rmse={row['phase_rmse_rad_mean']:.6f}"
        )
    print("Detail saved to:", args.detail_csv)
    print("Summary saved to:", args.summary_csv)


if __name__ == "__main__":
    main()
