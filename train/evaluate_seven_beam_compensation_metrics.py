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

from simulation.common.multi_beam_core import (
    crop_center,
    create_grid,
    far_field_intensity,
    seven_beam_near_field,
)
from train.evaluate_seven_beam_noise_robustness import load_seven_beam_model
from train.phase_metrics import decode_sin_cos, wrap_phase_error


def make_main_lobe_mask(image_size, radius):
    """生成中心圆形主瓣掩膜。"""
    yy, xx = np.ogrid[:image_size, :image_size]
    center = image_size // 2
    return (yy - center) ** 2 + (xx - center) ** 2 <= radius**2


def main_lobe_energy_ratio(intensity, mask, eps=1e-12):
    """计算主瓣区域能量占比。"""
    total = float(np.sum(intensity))
    if total <= eps:
        return 0.0
    return float(np.sum(intensity[mask]) / total)


def farfield_crop_from_phases(
    phases,
    x_grid,
    y_grid,
    waist,
    beam_distance,
    crop_size,
):
    """根据 6 路相对相位生成未按峰值归一化的裁剪远场光强。"""
    near_field = seven_beam_near_field(
        x_grid=x_grid,
        y_grid=y_grid,
        waist=waist,
        beam_distance=beam_distance,
        phases=phases,
    )
    intensity = far_field_intensity(near_field, normalize=False)
    return crop_center(intensity, crop_size=crop_size).astype(np.float64)


def predict_labels(model, images, batch_size, device):
    """批量预测 sin/cos 相位标签。"""
    preds = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            batch_tensor = torch.as_tensor(batch, dtype=torch.float32)
            if batch_tensor.ndim == 3:
                batch_tensor = batch_tensor.unsqueeze(1)
            elif batch_tensor.ndim != 4:
                raise ValueError(f"Expected batch [B,H,W] or [B,C,H,W], got {tuple(batch_tensor.shape)}")
            expected_in_channels = getattr(model, "expected_in_channels", batch_tensor.shape[1])
            if batch_tensor.shape[1] != expected_in_channels:
                if expected_in_channels == 1:
                    batch_tensor = batch_tensor[:, :1]
                else:
                    raise ValueError(
                        f"Model expects {expected_in_channels} channels, "
                        f"but batch has {batch_tensor.shape[1]} channels."
                    )
            batch_tensor = batch_tensor.to(device)
            preds.append(model(batch_tensor).cpu().numpy())

    return np.concatenate(preds, axis=0)


def summarize_state(rows, state_name):
    """汇总某个补偿状态的主瓣能量占比。"""
    values = np.array(
        [row["main_lobe_ratio"] for row in rows if row["state"] == state_name],
        dtype=np.float64,
    )
    return {
        "state": state_name,
        "samples": len(values),
        "main_lobe_ratio_mean": float(np.mean(values)),
        "main_lobe_ratio_std": float(np.std(values)),
        "main_lobe_ratio_min": float(np.min(values)),
        "main_lobe_ratio_max": float(np.max(values)),
    }


def save_detail_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(detail_rows, summary_rows, example_images, figure_path):
    """绘制主瓣能量占比统计和典型远场图。"""
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    states = [row["state"] for row in summary_rows]
    means = [row["main_lobe_ratio_mean"] for row in summary_rows]
    stds = [row["main_lobe_ratio_std"] for row in summary_rows]

    plt.figure(figsize=(14, 9))

    plt.subplot(2, 3, 1)
    plt.bar(states, means, yerr=stds, capsize=4)
    plt.xticks(rotation=20)
    plt.ylabel("Main-lobe energy ratio")
    plt.title("Mean main-lobe ratio")

    plt.subplot(2, 3, 2)
    data = [
        [row["main_lobe_ratio"] for row in detail_rows if row["state"] == state]
        for state in states
    ]
    plt.boxplot(data, tick_labels=states)
    plt.xticks(rotation=20)
    plt.ylabel("Main-lobe energy ratio")
    plt.title("Distribution")

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
        description="Evaluate seven-beam phase compensation using main-lobe energy ratio."
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
    parser.add_argument("--main-lobe-radius", type=int, default=8)
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "cycle17_seven_beam_main_lobe_detail_2026-06-09.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "cycle17_seven_beam_main_lobe_summary_2026-06-09.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "figures"
        / "cycle17_seven_beam_main_lobe_2026-06-09.png",
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
    mask = make_main_lobe_mask(image_size=args.crop_size, radius=args.main_lobe_radius)

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
            ratio = main_lobe_energy_ratio(farfield, mask)
            detail_rows.append(
                {
                    "sample_index": index,
                    "state": state,
                    "main_lobe_ratio": ratio,
                    "total_energy": float(np.sum(farfield)),
                    "main_lobe_energy": float(np.sum(farfield[mask])),
                    "peak_intensity": float(np.max(farfield)),
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

    before_mean = summary_rows[0]["main_lobe_ratio_mean"]
    for row in summary_rows:
        if row["state"] == "before":
            row["relative_gain_vs_before_percent"] = 0.0
        else:
            row["relative_gain_vs_before_percent"] = 100 * (
                row["main_lobe_ratio_mean"] - before_mean
            ) / before_mean

    save_detail_csv(detail_rows, args.detail_csv)
    save_summary_csv(summary_rows, args.summary_csv)
    plot_results(detail_rows, summary_rows, example_images, args.figure_path)

    print("Using device:", device)
    print("Samples:", len(images))
    print("Main-lobe radius(px):", args.main_lobe_radius)
    for row in summary_rows:
        print(
            f"{row['state']} | mean={row['main_lobe_ratio_mean']:.6f} | "
            f"std={row['main_lobe_ratio_std']:.6f} | "
            f"gain={row['relative_gain_vs_before_percent']:.2f}%"
        )
    print("Detail saved to:", args.detail_csv)
    print("Summary saved to:", args.summary_csv)


if __name__ == "__main__":
    main()
