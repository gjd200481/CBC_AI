import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.multi_beam_core import create_grid
from train.evaluate_seven_beam_compensation_metrics import farfield_crop_from_phases, predict_labels
from train.evaluate_seven_beam_noise_robustness import load_seven_beam_model
from train.phase_metrics import decode_sin_cos, wrap_phase_error


def draw_box(ax, xy, text, facecolor, edgecolor="#334155"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        0.25,
        0.18,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + 0.125, y + 0.09, text, ha="center", va="center", fontsize=9)


def plot_method_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.98, "(a) Metric correction", fontsize=12, fontweight="bold", va="top")
    draw_box(ax, (0.02, 0.58), "Predicted\nphase", "#DBEAFE")
    draw_box(ax, (0.37, 0.58), "Residual\nphase", "#E0F2FE")
    draw_box(ax, (0.72, 0.58), "Far-field\ncompensation", "#DCFCE7")
    ax.annotate("", xy=(0.36, 0.67), xytext=(0.28, 0.67), arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.annotate("", xy=(0.71, 0.67), xytext=(0.63, 0.67), arrowprops=dict(arrowstyle="->", lw=1.3))

    draw_box(ax, (0.08, 0.22), "Old validation\npeak-normalized", "#FEE2E2", edgecolor="#B91C1C")
    draw_box(ax, (0.42, 0.22), "Cycle 41\nunnormalized", "#DCFCE7", edgecolor="#15803D")
    draw_box(ax, (0.72, 0.22), "Final eval\nsame metric", "#FEF3C7", edgecolor="#B45309")
    ax.annotate("", xy=(0.42, 0.31), xytext=(0.33, 0.31), arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.annotate("", xy=(0.72, 0.31), xytext=(0.67, 0.31), arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.text(0.08, 0.08, "Goal: select checkpoints by downstream optical quality,\nnot by a normalized proxy.", fontsize=9)


def choose_example(detail_df, state="cycle41_best_strehl"):
    rows = detail_df[detail_df["state"] == state].copy()
    mean_value = rows["strehl_ratio"].mean()
    rows["distance_to_mean"] = (rows["strehl_ratio"] - mean_value).abs()
    return int(rows.sort_values("distance_to_mean").iloc[0]["sample_index"])


def compute_example_farfields(args, example_index):
    images = np.load(args.image_path)[: args.max_samples]
    labels = np.load(args.label_path)[: args.max_samples]
    true_phases = decode_sin_cos(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_specs = {
        "Cycle 35": args.cycle35_model,
        "Cycle 41": args.cycle41_model,
    }
    phase_sets = {
        "Before": true_phases,
        "Ideal": np.zeros_like(true_phases),
    }
    for label, path in model_specs.items():
        model = load_seven_beam_model(path, device)
        pred = predict_labels(model, images, batch_size=args.batch_size, device=device)
        pred_phases = decode_sin_cos(pred)
        phase_sets[label] = wrap_phase_error(true_phases, pred_phases)

    x_grid, y_grid = create_grid(num_points=args.num_points, window_size=args.window_size)
    farfields = {}
    for label in ["Before", "Cycle 35", "Cycle 41", "Ideal"]:
        farfields[label] = farfield_crop_from_phases(
            phases=phase_sets[label][example_index],
            x_grid=x_grid,
            y_grid=y_grid,
            waist=args.waist,
            beam_distance=args.beam_distance,
            crop_size=args.crop_size,
        )
    return farfields


def main():
    parser = argparse.ArgumentParser(description="Build a literature-style Cycle 41 evidence figure.")
    parser.add_argument("--history-csv", type=Path, default=REPO_ROOT / "result/metrics/cycle41_multiplane_7cm_unorm_30epoch_history.csv")
    parser.add_argument("--summary-csv", type=Path, default=REPO_ROOT / "result/metrics/cycle41_unnormalized_strehl_paired_summary.csv")
    parser.add_argument("--detail-csv", type=Path, default=REPO_ROOT / "result/metrics/cycle41_unnormalized_strehl_paired_detail.csv")
    parser.add_argument("--image-path", type=Path, default=REPO_ROOT / "dataset/seven_beam/multiplane_0_-0.07/images_multiplane_7cm.npy")
    parser.add_argument("--label-path", type=Path, default=REPO_ROOT / "dataset/seven_beam/multiplane_0_-0.07/labels_multiplane_7cm.npy")
    parser.add_argument("--cycle35-model", type=Path, default=REPO_ROOT / "models/cycle35_multiplane_7cm_10k_30epoch.pth")
    parser.add_argument("--cycle41-model", type=Path, default=REPO_ROOT / "models/cycle41_multiplane_7cm_unorm_best_strehl_30epoch.pth")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "result/figures/cycle41_literature_style_evidence.png")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--crop-size", type=int, default=160)
    args = parser.parse_args()

    history = pd.read_csv(args.history_csv)
    summary = pd.read_csv(args.summary_csv).set_index("state")
    detail = pd.read_csv(args.detail_csv)
    example_index = choose_example(detail)
    farfields = compute_example_farfields(args, example_index)

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.0, 1.1])

    ax_method = fig.add_subplot(gs[0, :2])
    plot_method_panel(ax_method)

    ax_curve = fig.add_subplot(gs[0, 2:])
    ax_curve.set_title("(b) Validation trajectory", loc="left", fontweight="bold")
    ax_curve.plot(history["epoch"], history["val_rmse_rad"], color="#7C2D12", lw=2, label="RMSE")
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Phase RMSE (rad)", color="#7C2D12")
    ax_curve.tick_params(axis="y", labelcolor="#7C2D12")
    ax2 = ax_curve.twinx()
    ax2.plot(history["epoch"], history["val_strehl_ratio"], color="#2563EB", lw=2, label="Strehl")
    ax2.plot(history["epoch"], history["val_main_lobe_ratio"], color="#059669", lw=2, label="Main lobe")
    ax2.plot(history["epoch"], history["val_synthesis_efficiency"], color="#7C3AED", lw=2, label="Efficiency")
    ax2.set_ylabel("Optical quality")
    lines = ax_curve.get_lines() + ax2.get_lines()
    ax_curve.legend(lines, [line.get_label() for line in lines], frameon=False, ncol=2, loc="lower right")

    states = ["before", "comp0p3_best_rmse", "comp0p5_cycle35", "cycle41_best_strehl", "ideal"]
    labels = ["Before", "C37 phase", "C35 quality", "C41 quality", "Ideal"]
    x = np.arange(len(states))
    width = 0.25
    ax_metrics = fig.add_subplot(gs[1, :2])
    ax_metrics.set_title("(c) Downstream optical quality", loc="left", fontweight="bold")
    for offset, metric, color, name in [
        (-width, "main_lobe_ratio_mean", "#059669", "Main lobe"),
        (0.0, "strehl_ratio_mean", "#2563EB", "Strehl"),
        (width, "synthesis_efficiency_mean", "#7C3AED", "Efficiency"),
    ]:
        ax_metrics.bar(x + offset, [summary.loc[state, metric] for state in states], width=width, color=color, label=name)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(labels, rotation=15, ha="right")
    ax_metrics.set_ylim(0.3, 1.05)
    ax_metrics.legend(frameon=False, ncol=3, loc="upper left")
    ax_metrics.axhline(summary.loc["comp0p5_cycle35", "strehl_ratio_mean"], color="#94A3B8", ls="--", lw=1)

    ax_rmse = fig.add_subplot(gs[1, 2:])
    ax_rmse.set_title("(d) Residual phase trade-off", loc="left", fontweight="bold")
    rmse_values = [summary.loc[state, "phase_rmse_rad_mean"] for state in states[:-1]]
    rmse_labels = labels[:-1]
    ax_rmse.bar(rmse_labels, rmse_values, color=["#64748B", "#0F766E", "#EA580C", "#2563EB"])
    ax_rmse.set_ylabel("Residual phase RMSE (rad)")
    ax_rmse.tick_params(axis="x", rotation=15)
    ax_rmse.axhline(summary.loc["comp0p3_best_rmse", "phase_rmse_rad_mean"], color="#0F766E", ls="--", lw=1)

    for idx, label in enumerate(["Before", "Cycle 35", "Cycle 41", "Ideal"]):
        ax = fig.add_subplot(gs[2, idx])
        image = farfields[label]
        display = np.log10(image / max(float(np.max(image)), 1e-12) + 1e-6)
        ax.imshow(display, cmap="inferno")
        ax.set_title(f"({chr(ord('e') + idx)}) {label}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Cycle 41 evidence chain: unnormalized Strehl selection for seven-beam CBC", fontsize=14, fontweight="bold")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {args.output}")
    print(f"Representative example index: {example_index}")


if __name__ == "__main__":
    main()
