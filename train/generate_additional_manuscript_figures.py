# -*- coding: utf-8 -*-
"""Generate additional manuscript figures for the seven-beam CBC paper.

The existing publication figures cover the main comparison curves.  This script
adds supporting figures that are useful for a Chinese journal manuscript:
data examples, detailed architecture, phase-prediction diagnostics, fusion-gate
statistics, compensated far-field case studies, and metric relationships.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.multi_beam_core import (  # noqa: E402
    create_grid,
    crop_center,
    far_field_intensity,
    seven_beam_near_field,
)
from train.evaluate_seven_beam_compensation_metrics import (  # noqa: E402
    farfield_crop_from_phases,
    predict_labels,
)
from train.evaluate_seven_beam_noise_robustness import load_seven_beam_model  # noqa: E402
from train.phase_metrics import decode_sin_cos, wrap_phase_error  # noqa: E402


plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#16a34a",
    "orange": "#ea580c",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "gray": "#64748b",
    "dark": "#0f172a",
}


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")


def log_display(image: np.ndarray, floor: float = 1e-5) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    image = image / max(float(np.max(image)), floor)
    return np.log10(np.clip(image, floor, None))


def select_examples(images: np.ndarray, count: int = 4) -> list[int]:
    """Pick visually diverse examples using focal-plane peak and plane distance."""
    focal = images[:, 0]
    befocal = images[:, 1]
    peak = focal.reshape(len(images), -1).max(axis=1)
    delta = np.mean(np.abs(focal - befocal), axis=(1, 2))
    score = 0.55 * (peak - peak.min()) / (peak.ptp() + 1e-8)
    score += 0.45 * (delta - delta.min()) / (delta.ptp() + 1e-8)
    quantiles = np.linspace(0.1, 0.9, count)
    order = np.argsort(score)
    return [int(order[min(len(order) - 1, round(q * (len(order) - 1)))]) for q in quantiles]


def fig_dataset_plane_examples(images: np.ndarray, output_dir: Path) -> None:
    indices = select_examples(images, count=4)
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 7.2), constrained_layout=True)
    row_labels = ["焦平面", "焦前平面", "平面差异"]

    for col, sample_index in enumerate(indices):
        focal = np.asarray(images[sample_index, 0])
        befocal = np.asarray(images[sample_index, 1])
        panels = [
            log_display(focal),
            log_display(befocal),
            np.abs(focal - befocal),
        ]
        cmaps = ["inferno", "inferno", "magma"]
        for row, panel in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(panel, cmap=cmaps[row])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"样本 {sample_index}", fontweight="bold")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontweight="bold")
            if row == 2:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("多平面远场强度样例：焦平面与焦前平面的互补信息", fontweight="bold")
    save_figure(fig, output_dir, "add_fig1_multiplane_data_examples")


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    fontsize: int = 8,
    linewidth: float = 1.2,
) -> None:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.04",
        facecolor=facecolor,
        edgecolor=COLORS["dark"],
        linewidth=linewidth,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=COLORS["dark"], lw=1.4, shrinkA=3, shrinkB=3),
    )


def fig_model_structure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    y_top, y_bottom = 5.2, 2.2
    x_positions = [0.3, 1.8, 3.35, 4.9, 6.45]
    labels = [
        "输入\n1 x 160 x 160",
        "7x7 Conv s=2\n32 x 80 x 80",
        "3x3 MaxPool s=2\n32 x 40 x 40",
        "ResBlock x2\n32 x 40 x 40",
        "ResBlock x2 s=2\n64 x 20 x 20",
    ]
    labels_2 = [
        "ResBlock x2 s=2\n128 x 10 x 10",
        "ResBlock x2 s=2\n256 x 5 x 5",
    ]

    for row_y, row_title, color in [
        (y_top, "焦平面分支", "#dbeafe"),
        (y_bottom, "焦前分支", "#e0f2fe"),
    ]:
        ax.text(0.2, row_y + 0.78, row_title, fontsize=11, fontweight="bold", color=COLORS["dark"])
        for i, text in enumerate(labels):
            draw_box(ax, (x_positions[i], row_y), 1.25, 0.72, text, color)
            if i > 0:
                draw_arrow(ax, (x_positions[i - 1] + 1.25, row_y + 0.36), (x_positions[i], row_y + 0.36))
        draw_box(ax, (8.0, row_y), 1.25, 0.72, labels_2[0], color)
        draw_arrow(ax, (x_positions[-1] + 1.25, row_y + 0.36), (8.0, row_y + 0.36))
        draw_box(ax, (9.6, row_y), 1.25, 0.72, labels_2[1], color)
        draw_arrow(ax, (9.25, row_y + 0.36), (9.6, row_y + 0.36))

    draw_box(ax, (11.35, 4.15), 1.25, 1.1, "拼接\n512 x 5 x 5", "#fef3c7")
    draw_arrow(ax, (10.85, y_top + 0.36), (11.35, 4.95))
    draw_arrow(ax, (10.85, y_bottom + 0.36), (11.35, 4.42))

    draw_box(ax, (11.35, 2.75), 1.25, 0.95, "融合门控\nGAP + MLP\n512->128->256", "#dcfce7")
    draw_arrow(ax, (11.98, 4.15), (11.98, 3.7))

    draw_box(ax, (8.75, 0.85), 1.55, 0.95, "门控融合\nw·F_focal\n+(1-w)·F_befocal", "#bbf7d0")
    draw_arrow(ax, (11.35, 3.1), (10.3, 1.45))
    draw_arrow(ax, (10.25, y_top), (9.85, 1.8))
    draw_arrow(ax, (10.25, y_bottom), (9.85, 1.8))

    draw_box(ax, (6.55, 0.85), 1.55, 0.95, "通道注意力\nGAP + MLP\n256->16->256", "#fde68a")
    draw_arrow(ax, (8.75, 1.32), (8.1, 1.32))

    draw_box(ax, (4.35, 0.85), 1.55, 0.95, "回归头\nGAP + FC\n256->256->12", "#fecaca")
    draw_arrow(ax, (6.55, 1.32), (5.9, 1.32))

    draw_box(ax, (2.35, 0.85), 1.35, 0.95, "输出\n6路相位\nsin/cos编码", "#ddd6fe")
    draw_arrow(ax, (4.35, 1.32), (3.7, 1.32))

    ax.text(
        0.3,
        0.25,
        "Cycle42 dual_plane_fusion_cnn：两个分支结构相同但参数不共享；总可训练参数量 5,767,516。",
        fontsize=10,
        color=COLORS["dark"],
    )
    fig.suptitle("双分支门控融合网络结构细图", fontweight="bold")
    save_figure(fig, output_dir, "add_fig2_model_structure_detail")


def run_inference(
    images: np.ndarray,
    labels: np.ndarray,
    checkpoint: Path,
    batch_size: int,
    device_name: str,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.device]:
    if device_name == "cuda":
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_seven_beam_model(checkpoint, device)
    pred_labels = predict_labels(model=model, images=images, batch_size=batch_size, device=device)
    true_phases = decode_sin_cos(labels)
    pred_phases = decode_sin_cos(pred_labels)
    errors = wrap_phase_error(pred_phases, true_phases)
    residual = wrap_phase_error(true_phases, pred_phases)
    return model, pred_labels, true_phases, pred_phases, errors, residual, device


def fig_phase_prediction(true_phases: np.ndarray, pred_phases: np.ndarray, errors: np.ndarray, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.ravel()
    aligned_pred = true_phases + errors
    lim = (-np.pi - 0.55, np.pi + 0.55)

    for beam_idx, ax in enumerate(axes):
        rmse = float(np.sqrt(np.mean(errors[:, beam_idx] ** 2)))
        mae = float(np.mean(np.abs(errors[:, beam_idx])))
        ax.scatter(
            true_phases[:, beam_idx],
            aligned_pred[:, beam_idx],
            s=13,
            alpha=0.58,
            color=COLORS["blue"],
            linewidths=0,
        )
        ax.plot(lim, lim, "--", color=COLORS["red"], lw=1.0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_title(f"Beam {beam_idx + 1}: RMSE={rmse:.3f} rad, MAE={mae:.3f} rad")
        if beam_idx in {3, 4, 5}:
            ax.set_xlabel("真实相位 / rad")
        if beam_idx in {0, 3}:
            ax.set_ylabel("周期对齐后的预测相位 / rad")

    fig.suptitle("六路相对相位预测一致性", fontweight="bold")
    save_figure(fig, output_dir, "add_fig3_phase_prediction_scatter")


def fig_phase_error_distribution(errors: np.ndarray, output_dir: Path) -> None:
    sample_rmse = np.sqrt(np.mean(errors**2, axis=1))
    fig = plt.figure(figsize=(12.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[:, :2])
    data = [errors[:, idx] for idx in range(errors.shape[1])]
    violin = ax1.violinplot(data, showmeans=True, showextrema=True, widths=0.82)
    for body in violin["bodies"]:
        body.set_facecolor(COLORS["cyan"])
        body.set_edgecolor(COLORS["dark"])
        body.set_alpha(0.55)
    for key in ["cbars", "cmins", "cmaxes", "cmeans"]:
        violin[key].set_color(COLORS["dark"])
        violin[key].set_linewidth(1.0)
    ax1.axhline(0, color=COLORS["red"], linestyle="--", lw=1.0)
    ax1.set_xticks(np.arange(1, errors.shape[1] + 1))
    ax1.set_xticklabels([f"Beam {i}" for i in range(1, errors.shape[1] + 1)])
    ax1.set_ylabel("周期相位误差 / rad")
    ax1.set_title("(a) 各相位通道误差分布", fontweight="bold")
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(sample_rmse, bins=24, color=COLORS["orange"], alpha=0.78, edgecolor="white")
    ax2.axvline(float(np.mean(sample_rmse)), color=COLORS["dark"], lw=1.2, label=f"均值 {np.mean(sample_rmse):.3f}")
    ax2.set_xlabel("单样本相位 RMSE / rad")
    ax2.set_ylabel("样本数")
    ax2.set_title("(b) 单样本 RMSE 直方图", fontweight="bold")
    ax2.legend(frameon=False)

    ax3 = fig.add_subplot(gs[1, 2])
    sorted_rmse = np.sort(sample_rmse)
    cdf = np.arange(1, len(sorted_rmse) + 1) / len(sorted_rmse)
    ax3.plot(sorted_rmse, cdf, color=COLORS["green"], lw=2.0)
    ax3.axvline(float(np.median(sample_rmse)), color=COLORS["dark"], linestyle="--", lw=1.1)
    ax3.set_xlabel("单样本相位 RMSE / rad")
    ax3.set_ylabel("累计概率")
    ax3.set_ylim(0, 1.02)
    ax3.set_title("(c) 误差累计分布", fontweight="bold")
    ax3.grid(alpha=0.25, linestyle=":")

    fig.suptitle("相位误差统计诊断", fontweight="bold")
    save_figure(fig, output_dir, "add_fig4_phase_error_distribution")


def compute_gate_and_attention(
    model: torch.nn.Module,
    images: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(model, "focal_encoder") or not hasattr(model, "fusion_gate"):
        raise TypeError("The loaded model does not expose focal_encoder/fusion_gate.")

    gates: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.as_tensor(images[start : start + batch_size], dtype=torch.float32, device=device)
            focal = model.focal_encoder(batch[:, 0:1])
            befocal = model.befocal_encoder(batch[:, 1:2])
            gate = model.fusion_gate(torch.cat([focal, befocal], dim=1))
            fused = gate.unsqueeze(-1).unsqueeze(-1) * focal + (1.0 - gate.unsqueeze(-1).unsqueeze(-1)) * befocal
            attention = model.channel_attention(fused)
            gates.append(gate.cpu().numpy())
            attentions.append(attention.cpu().numpy())
    return np.concatenate(gates, axis=0), np.concatenate(attentions, axis=0)


def fig_gate_statistics(gates: np.ndarray, attentions: np.ndarray, sample_rmse: np.ndarray, output_dir: Path) -> None:
    gate_mean_per_sample = gates.mean(axis=1)
    gate_std_per_sample = gates.std(axis=1)
    gate_mean_per_channel = gates.mean(axis=0)
    att_mean_per_channel = attentions.mean(axis=0)

    fig = plt.figure(figsize=(12.2, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(gate_mean_per_sample, bins=22, color=COLORS["blue"], alpha=0.78, edgecolor="white")
    ax1.axvline(0.5, color=COLORS["red"], linestyle="--", lw=1.0, label="均衡融合")
    ax1.axvline(float(np.mean(gate_mean_per_sample)), color=COLORS["dark"], lw=1.1, label=f"均值 {np.mean(gate_mean_per_sample):.3f}")
    ax1.set_xlabel("样本平均门控权重")
    ax1.set_ylabel("样本数")
    ax1.set_title("(a) 样本级焦平面权重", fontweight="bold")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(gate_mean_per_channel, bins=24, color=COLORS["cyan"], alpha=0.78, edgecolor="white")
    ax2.axvline(0.5, color=COLORS["red"], linestyle="--", lw=1.0)
    ax2.set_xlabel("通道平均门控权重")
    ax2.set_ylabel("通道数")
    ax2.set_title("(b) 通道级焦平面权重", fontweight="bold")

    ax3 = fig.add_subplot(gs[0, 2])
    sc = ax3.scatter(
        gate_mean_per_sample,
        sample_rmse,
        c=gate_std_per_sample,
        cmap="viridis",
        s=22,
        alpha=0.78,
        linewidths=0,
    )
    ax3.set_xlabel("样本平均门控权重")
    ax3.set_ylabel("单样本相位 RMSE / rad")
    ax3.set_title("(c) 融合权重与误差", fontweight="bold")
    fig.colorbar(sc, ax=ax3, label="门控通道标准差")

    ax4 = fig.add_subplot(gs[1, :2])
    channel_index = np.arange(gates.shape[1])
    ax4.plot(channel_index, gate_mean_per_channel, color=COLORS["blue"], lw=1.6, label="融合门控")
    ax4.plot(channel_index, att_mean_per_channel, color=COLORS["orange"], lw=1.2, alpha=0.88, label="通道注意力")
    ax4.axhline(0.5, color=COLORS["gray"], linestyle=":", lw=1.0)
    ax4.set_xlabel("特征通道索引")
    ax4.set_ylabel("平均权重")
    ax4.set_title("(d) 256 个融合通道的权重轮廓", fontweight="bold")
    ax4.legend(frameon=False)
    ax4.grid(alpha=0.22, linestyle=":")

    ax5 = fig.add_subplot(gs[1, 2])
    labels = ["<0.4\n焦前主导", "0.4-0.6\n均衡", ">0.6\n焦平面主导"]
    values = [
        float(np.mean(gates < 0.4)),
        float(np.mean((gates >= 0.4) & (gates <= 0.6))),
        float(np.mean(gates > 0.6)),
    ]
    ax5.bar(labels, values, color=[COLORS["cyan"], COLORS["green"], COLORS["blue"]], alpha=0.82)
    ax5.set_ylim(0, max(values) * 1.25 + 0.02)
    ax5.set_ylabel("门控元素占比")
    ax5.set_title("(e) 融合偏向比例", fontweight="bold")
    for idx, value in enumerate(values):
        ax5.text(idx, value, f"{value * 100:.1f}%", ha="center", va="bottom")

    fig.suptitle("双平面融合门控与通道注意力统计", fontweight="bold")
    save_figure(fig, output_dir, "add_fig5_gate_attention_statistics")


def make_farfield(phases: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    return farfield_crop_from_phases(
        phases=phases,
        x_grid=x_grid,
        y_grid=y_grid,
        waist=0.5e-3,
        beam_distance=1.5e-3,
        crop_size=160,
    )


def fig_farfield_case_gallery(
    true_phases: np.ndarray,
    residual: np.ndarray,
    errors: np.ndarray,
    output_dir: Path,
) -> None:
    sample_rmse = np.sqrt(np.mean(errors**2, axis=1))
    order = np.argsort(sample_rmse)
    indices = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    case_labels = ["低误差样本", "中位误差样本", "高误差样本"]

    x_grid, y_grid = create_grid(num_points=256, window_size=10e-3)
    ideal = make_farfield(np.zeros(6, dtype=np.float32), x_grid, y_grid)

    fig, axes = plt.subplots(3, 3, figsize=(9.6, 9.4), constrained_layout=True)
    col_labels = ["补偿前", "Cycle42 补偿后", "理想同相"]

    for row, (sample_index, case_label) in enumerate(zip(indices, case_labels)):
        panels = [
            make_farfield(true_phases[sample_index], x_grid, y_grid),
            make_farfield(residual[sample_index], x_grid, y_grid),
            ideal,
        ]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(log_display(panel, floor=1e-7), cmap="inferno", vmin=-6, vmax=0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_labels[col], fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{case_label}\n样本 {sample_index}\nRMSE={sample_rmse[sample_index]:.3f} rad", fontweight="bold")

    fig.suptitle("典型样本远场光斑补偿前后对比（log10 归一化强度）", fontweight="bold")
    save_figure(fig, output_dir, "add_fig6_farfield_case_gallery")


def fig_metric_relationships(detail_csv: Path, output_dir: Path) -> None:
    detail = pd.read_csv(detail_csv)
    detail = detail[detail["state"].isin(["before", "cycle42_best_rmse"])].copy()
    state_names = {
        "before": "补偿前",
        "cycle42_best_rmse": "Cycle42 补偿后",
    }
    detail["state_cn"] = detail["state"].map(state_names)

    fig = plt.figure(figsize=(12.2, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[:, 0])
    box_data = [detail.loc[detail["state"] == state, "strehl_ratio"].to_numpy() for state in ["before", "cycle42_best_rmse"]]
    box = ax1.boxplot(box_data, patch_artist=True, labels=["补偿前", "Cycle42\n补偿后"], widths=0.55)
    for patch, color in zip(box["boxes"], [COLORS["gray"], COLORS["blue"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax1.set_ylabel("Strehl 比")
    ax1.set_title("(a) Strehl 分布提升", fontweight="bold")
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    ax2 = fig.add_subplot(gs[0, 1:])
    for state, color in [("before", COLORS["gray"]), ("cycle42_best_rmse", COLORS["blue"])]:
        subset = detail[detail["state"] == state]
        ax2.scatter(
            subset["phase_rmse_rad"],
            subset["strehl_ratio"],
            s=18,
            alpha=0.55,
            color=color,
            label=state_names[state],
            linewidths=0,
        )
    ax2.set_xlabel("残余相位 RMSE / rad")
    ax2.set_ylabel("Strehl 比")
    ax2.set_title("(b) 相位误差与峰值质量关系", fontweight="bold")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25, linestyle=":")

    ax3 = fig.add_subplot(gs[1, 1])
    states = ["before", "cycle42_best_rmse"]
    means = [detail.loc[detail["state"] == state, "main_lobe_ratio"].mean() for state in states]
    stds = [detail.loc[detail["state"] == state, "main_lobe_ratio"].std() for state in states]
    ax3.bar(["补偿前", "Cycle42"], means, yerr=stds, capsize=4, color=[COLORS["gray"], COLORS["green"]], alpha=0.78)
    ax3.set_ylabel("主瓣能量占比")
    ax3.set_title("(c) 主瓣能量统计", fontweight="bold")
    ax3.grid(axis="y", alpha=0.25, linestyle=":")

    ax4 = fig.add_subplot(gs[1, 2])
    before = detail[detail["state"] == "before"].sort_values("sample_index")
    after = detail[detail["state"] == "cycle42_best_rmse"].sort_values("sample_index")
    gain = after["strehl_ratio"].to_numpy() - before["strehl_ratio"].to_numpy()
    ax4.hist(gain, bins=24, color=COLORS["purple"], alpha=0.76, edgecolor="white")
    ax4.axvline(float(np.mean(gain)), color=COLORS["dark"], lw=1.2, label=f"均值 {np.mean(gain):.3f}")
    ax4.set_xlabel("Strehl 绝对提升")
    ax4.set_ylabel("样本数")
    ax4.set_title("(d) 单样本提升分布", fontweight="bold")
    ax4.legend(frameon=False)

    fig.suptitle("相位误差与补偿后光束质量之间的统计关系", fontweight="bold")
    save_figure(fig, output_dir, "add_fig7_metric_relationships")


def fig_noise_augmentation_gain(noise_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(noise_csv)
    base = df[df["model"] == "cycle42_baseline"].set_index("noise_sigma")
    aug = df[df["model"] == "cycle44_noise_aug"].set_index("noise_sigma")
    sigmas = sorted(set(base.index).intersection(set(aug.index)))

    strehl_gain = np.array([(aug.loc[s, "strehl_ratio_mean"] - base.loc[s, "strehl_ratio_mean"]) / base.loc[s, "strehl_ratio_mean"] * 100 for s in sigmas])
    rmse_reduction = np.array([(base.loc[s, "phase_rmse_rad_mean"] - aug.loc[s, "phase_rmse_rad_mean"]) / base.loc[s, "phase_rmse_rad_mean"] * 100 for s in sigmas])
    retention_base = np.array([base.loc[s, "strehl_ratio_mean"] / base.loc[0.0, "strehl_ratio_mean"] * 100 for s in sigmas])
    retention_aug = np.array([aug.loc[s, "strehl_ratio_mean"] / aug.loc[0.0, "strehl_ratio_mean"] * 100 for s in sigmas])

    fig = plt.figure(figsize=(12.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sigmas, retention_base, marker="o", lw=2.0, color=COLORS["gray"], label="Cycle42 基线")
    ax1.plot(sigmas, retention_aug, marker="s", lw=2.0, color=COLORS["green"], label="Cycle44 噪声增强")
    ax1.set_xlabel("输入噪声标准差")
    ax1.set_ylabel("Strehl 保持率 / %")
    ax1.set_title("(a) 相对干净数据的质量保持率", fontweight="bold")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25, linestyle=":")

    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.0018
    ax2.bar(np.array(sigmas) - width / 2, strehl_gain, width=width, color=COLORS["blue"], alpha=0.78, label="Strehl 相对提升")
    ax2.bar(np.array(sigmas) + width / 2, rmse_reduction, width=width, color=COLORS["orange"], alpha=0.78, label="RMSE 相对下降")
    ax2.axhline(0, color=COLORS["dark"], lw=0.9)
    ax2.set_xlabel("输入噪声标准差")
    ax2.set_ylabel("相对变化 / %")
    ax2.set_title("(b) Cycle44 相对 Cycle42 的收益", fontweight="bold")
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.25, linestyle=":")

    ax3 = fig.add_subplot(gs[1, :])
    metrics = [
        ("main_lobe_ratio_mean", "主瓣能量"),
        ("strehl_ratio_mean", "Strehl"),
        ("synthesis_efficiency_mean", "合成效率"),
        ("phase_rmse_rad_mean", "相位RMSE"),
    ]
    x = np.arange(len(metrics))
    sigma = 0.02
    base_vals = [float(base.loc[sigma, metric]) for metric, _ in metrics]
    aug_vals = [float(aug.loc[sigma, metric]) for metric, _ in metrics]
    bar_width = 0.36
    ax3.bar(x - bar_width / 2, base_vals, width=bar_width, color=COLORS["gray"], alpha=0.72, label="Cycle42 基线")
    ax3.bar(x + bar_width / 2, aug_vals, width=bar_width, color=COLORS["green"], alpha=0.78, label="Cycle44 噪声增强")
    ax3.set_xticks(x)
    ax3.set_xticklabels([label for _, label in metrics])
    ax3.set_title("(c) 噪声标准差 0.02 时的核心指标对比", fontweight="bold")
    ax3.legend(frameon=False)
    ax3.grid(axis="y", alpha=0.25, linestyle=":")
    for idx, (b, a) in enumerate(zip(base_vals, aug_vals)):
        ax3.text(idx - bar_width / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
        ax3.text(idx + bar_width / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("动态噪声增强的鲁棒性收益拆解", fontweight="bold")
    save_figure(fig, output_dir, "add_fig8_noise_augmentation_gain")


def _read_state_metric(summary_csv: Path, state: str, metric: str) -> float:
    df = pd.read_csv(summary_csv)
    row = df[df["state"] == state]
    if row.empty:
        raise ValueError(f"State {state!r} not found in {summary_csv}")
    return float(row.iloc[0][metric])


def fig_model_evolution(output_dir: Path, noise_csv: Path) -> None:
    """Show the main performance trajectory from early baseline to Cycle44."""
    cycle30 = REPO_ROOT / "result/metrics/cycle30_final_comparison_summary.csv"
    cycle41 = REPO_ROOT / "result/metrics/cycle41_unnormalized_strehl_paired_summary.csv"
    cycle42 = REPO_ROOT / "result/metrics/cycle42_dual_plane_fusion_paired_summary.csv"
    noise_df = pd.read_csv(noise_csv)
    clean_aug = noise_df[
        (noise_df["model"] == "cycle44_noise_aug") & (noise_df["noise_sigma"] == 0.0)
    ].iloc[0]

    rows = [
        {
            "label": "补偿前",
            "params": 0.0,
            "strehl": _read_state_metric(cycle42, "before", "strehl_ratio_mean"),
            "eff": _read_state_metric(cycle42, "before", "synthesis_efficiency_mean"),
            "rmse": _read_state_metric(cycle42, "before", "phase_rmse_rad_mean"),
            "color": COLORS["gray"],
        },
        {
            "label": "Cycle30\n深残差",
            "params": 11.34,
            "strehl": _read_state_metric(cycle30, "cycle30_deep_final", "strehl_ratio_mean"),
            "eff": _read_state_metric(cycle30, "cycle30_deep_final", "synthesis_efficiency_mean"),
            "rmse": _read_state_metric(cycle30, "cycle30_deep_final", "phase_rmse_rad_mean"),
            "color": COLORS["orange"],
        },
        {
            "label": "Cycle41\n指标修复",
            "params": 11.34,
            "strehl": _read_state_metric(cycle41, "cycle41_best_rmse", "strehl_ratio_mean"),
            "eff": _read_state_metric(cycle41, "cycle41_best_rmse", "synthesis_efficiency_mean"),
            "rmse": _read_state_metric(cycle41, "cycle41_best_rmse", "phase_rmse_rad_mean"),
            "color": COLORS["cyan"],
        },
        {
            "label": "Cycle42\n双分支融合",
            "params": 5.77,
            "strehl": _read_state_metric(cycle42, "cycle42_best_rmse", "strehl_ratio_mean"),
            "eff": _read_state_metric(cycle42, "cycle42_best_rmse", "synthesis_efficiency_mean"),
            "rmse": _read_state_metric(cycle42, "cycle42_best_rmse", "phase_rmse_rad_mean"),
            "color": COLORS["blue"],
        },
        {
            "label": "Cycle44\n噪声增强",
            "params": 5.77,
            "strehl": float(clean_aug["strehl_ratio_mean"]),
            "eff": float(clean_aug["synthesis_efficiency_mean"]),
            "rmse": float(clean_aug["phase_rmse_rad_mean"]),
            "color": COLORS["green"],
        },
    ]
    df = pd.DataFrame(rows)
    x = np.arange(len(df))

    fig = plt.figure(figsize=(12.4, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, df["strehl"], marker="o", lw=2.2, color=COLORS["blue"], label="Strehl")
    ax1.plot(x, df["eff"], marker="s", lw=2.2, color=COLORS["green"], label="合成效率")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"])
    ax1.set_ylim(0.3, 0.86)
    ax1.set_ylabel("光束质量指标")
    ax1.set_title("(a) 主线模型光束质量演进", fontweight="bold")
    ax1.grid(axis="y", alpha=0.25, linestyle=":")
    ax1.legend(frameon=False, ncol=2)
    for idx, value in enumerate(df["strehl"]):
        ax1.text(idx, value + 0.016, f"{value:.3f}", ha="center", fontsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(x, df["rmse"], color=df["color"], alpha=0.78)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["label"], rotation=18)
    ax2.set_ylabel("残余相位 RMSE / rad")
    ax2.set_title("(b) 残余相位误差", fontweight="bold")
    ax2.grid(axis="y", alpha=0.25, linestyle=":")
    for bar, value in zip(bars, df["rmse"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    sizes = np.maximum(df["params"].to_numpy(), 1.0) * 42
    sc = ax3.scatter(df["params"], df["strehl"], s=sizes, c=x, cmap="viridis", alpha=0.82, edgecolor="white", linewidth=1.0)
    for _, row in df.iterrows():
        ax3.text(row["params"] + 0.18, row["strehl"], row["label"].replace("\n", " "), fontsize=8, va="center")
    ax3.set_xlabel("参数量 / M")
    ax3.set_ylabel("Strehl 比")
    ax3.set_title("(c) 参数效率", fontweight="bold")
    ax3.grid(alpha=0.25, linestyle=":")
    fig.colorbar(sc, ax=ax3, label="演进顺序")

    ax4 = fig.add_subplot(gs[1, 2])
    gains = (df["strehl"].to_numpy() / df.loc[0, "strehl"] - 1.0) * 100
    ax4.bar(x, gains, color=df["color"], alpha=0.78)
    ax4.set_xticks(x)
    ax4.set_xticklabels(df["label"], rotation=18)
    ax4.set_ylabel("Strehl 相对补偿前提升 / %")
    ax4.set_title("(d) 补偿收益", fontweight="bold")
    ax4.grid(axis="y", alpha=0.25, linestyle=":")
    for idx, value in enumerate(gains):
        ax4.text(idx, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("从补偿前到双分支融合与噪声增强的模型演进", fontweight="bold")
    save_figure(fig, output_dir, "add_fig9_model_evolution")


def fig_unit_circle_diagnostics(pred_labels: np.ndarray, true_labels: np.ndarray, output_dir: Path) -> None:
    pred_sin = pred_labels[:, 0::2]
    pred_cos = pred_labels[:, 1::2]
    true_sin = true_labels[:, 0::2]
    true_cos = true_labels[:, 1::2]
    pred_radius = np.sqrt(pred_sin**2 + pred_cos**2)
    true_radius = np.sqrt(true_sin**2 + true_cos**2)
    unit_error = pred_radius - 1.0

    fig = plt.figure(figsize=(12.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(pred_radius.ravel(), bins=30, color=COLORS["blue"], alpha=0.72, edgecolor="white", label="预测")
    ax1.hist(true_radius.ravel(), bins=16, color=COLORS["gray"], alpha=0.38, edgecolor="white", label="真实")
    ax1.axvline(1.0, color=COLORS["red"], linestyle="--", lw=1.1)
    ax1.set_xlabel(r"$\sqrt{\sin^2+\cos^2}$")
    ax1.set_ylabel("元素数")
    ax1.set_title("(a) sin/cos 半径分布", fontweight="bold")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    channel_radius = pred_radius.mean(axis=0)
    channel_std = pred_radius.std(axis=0)
    x = np.arange(1, 7)
    ax2.errorbar(x, channel_radius, yerr=channel_std, marker="o", lw=1.8, color=COLORS["green"], capsize=4)
    ax2.axhline(1.0, color=COLORS["red"], linestyle="--", lw=1.0)
    ax2.set_xticks(x)
    ax2.set_xlabel("相位通道")
    ax2.set_ylabel("预测半径均值")
    ax2.set_title("(b) 通道级单位圆偏差", fontweight="bold")
    ax2.grid(axis="y", alpha=0.25, linestyle=":")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.boxplot([unit_error[:, idx] for idx in range(6)], patch_artist=True, labels=[str(i) for i in range(1, 7)])
    for patch in ax3.artists:
        patch.set_facecolor(COLORS["cyan"])
    ax3.axhline(0, color=COLORS["red"], linestyle="--", lw=1.0)
    ax3.set_xlabel("相位通道")
    ax3.set_ylabel("半径 - 1")
    ax3.set_title("(c) 单位圆误差箱线图", fontweight="bold")
    ax3.grid(axis="y", alpha=0.25, linestyle=":")

    for idx in range(3):
        ax = fig.add_subplot(gs[1, idx])
        beam_idx = idx * 2
        ax.scatter(true_cos[:, beam_idx], true_sin[:, beam_idx], s=12, alpha=0.35, color=COLORS["gray"], label="真实")
        ax.scatter(pred_cos[:, beam_idx], pred_sin[:, beam_idx], s=12, alpha=0.55, color=COLORS["blue"], label="预测")
        theta = np.linspace(0, 2 * np.pi, 240)
        ax.plot(np.cos(theta), np.sin(theta), color=COLORS["red"], lw=1.0, linestyle="--")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_xlabel("cos")
        ax.set_ylabel("sin")
        ax.set_title(f"(d{idx + 1}) Beam {beam_idx + 1} 输出平面", fontweight="bold")
        if idx == 0:
            ax.legend(frameon=False, loc="upper right")
        ax.grid(alpha=0.2, linestyle=":")

    fig.suptitle("sin/cos 周期编码输出的单位圆几何诊断", fontweight="bold")
    save_figure(fig, output_dir, "add_fig10_unit_circle_diagnostics")


def fig_residual_phase_heatmap(residual: np.ndarray, errors: np.ndarray, output_dir: Path) -> None:
    sample_rmse = np.sqrt(np.mean(errors**2, axis=1))
    order = np.argsort(sample_rmse)
    sorted_residual = residual[order]
    channel_rmse = np.sqrt(np.mean(errors**2, axis=0))
    channel_bias = np.mean(errors, axis=0)

    fig = plt.figure(figsize=(12.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    ax1 = fig.add_subplot(gs[:, :2])
    im = ax1.imshow(sorted_residual, aspect="auto", cmap="coolwarm", vmin=-np.pi, vmax=np.pi)
    ax1.set_xticks(np.arange(6))
    ax1.set_xticklabels([f"B{i}" for i in range(1, 7)])
    ax1.set_xlabel("外圈光束通道")
    ax1.set_ylabel("按 RMSE 排序的样本")
    ax1.set_title("(a) 补偿后残余相位热图", fontweight="bold")
    fig.colorbar(im, ax=ax1, label="残余相位 / rad")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(np.sort(sample_rmse), color=COLORS["purple"], lw=2.0)
    ax2.set_xlabel("样本排序")
    ax2.set_ylabel("单样本 RMSE / rad")
    ax2.set_title("(b) 样本难度曲线", fontweight="bold")
    ax2.grid(alpha=0.25, linestyle=":")

    ax3 = fig.add_subplot(gs[0, 3])
    x = np.arange(1, 7)
    ax3.bar(x, channel_rmse, color=COLORS["orange"], alpha=0.78)
    ax3.set_xticks(x)
    ax3.set_xlabel("通道")
    ax3.set_ylabel("RMSE / rad")
    ax3.set_title("(c) 通道残余误差", fontweight="bold")
    ax3.grid(axis="y", alpha=0.25, linestyle=":")

    ax4 = fig.add_subplot(gs[1, 2:])
    width = 0.35
    ax4.bar(x - width / 2, channel_bias, width=width, color=COLORS["cyan"], alpha=0.78, label="均值偏差")
    ax4.bar(x + width / 2, np.mean(np.abs(errors), axis=0), width=width, color=COLORS["green"], alpha=0.78, label="MAE")
    ax4.axhline(0, color=COLORS["dark"], lw=0.9)
    ax4.set_xticks(x)
    ax4.set_xlabel("通道")
    ax4.set_ylabel("相位误差 / rad")
    ax4.set_title("(d) 通道偏置与绝对误差", fontweight="bold")
    ax4.legend(frameon=False)
    ax4.grid(axis="y", alpha=0.25, linestyle=":")

    fig.suptitle("补偿后残余相位的样本-通道结构", fontweight="bold")
    save_figure(fig, output_dir, "add_fig11_residual_phase_heatmap")


def fig_noise_gain_heatmap(noise_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(noise_csv)
    base = df[df["model"] == "cycle42_baseline"].set_index("noise_sigma")
    aug = df[df["model"] == "cycle44_noise_aug"].set_index("noise_sigma")
    sigmas = sorted(set(base.index).intersection(set(aug.index)))
    metric_defs = [
        ("main_lobe_ratio_mean", "主瓣能量", True),
        ("strehl_ratio_mean", "Strehl", True),
        ("synthesis_efficiency_mean", "合成效率", True),
        ("phase_rmse_rad_mean", "相位RMSE", False),
    ]

    matrix = []
    for metric, _, higher_is_better in metric_defs:
        row = []
        for sigma in sigmas:
            b = float(base.loc[sigma, metric])
            a = float(aug.loc[sigma, metric])
            value = (a - b) / b * 100.0
            if not higher_is_better:
                value = -value
            row.append(value)
        matrix.append(row)
    matrix = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(11.8, 5.6), constrained_layout=True)
    vmax = max(5.0, float(np.nanmax(np.abs(matrix))))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(sigmas)))
    ax.set_xticklabels([f"{sigma:g}" for sigma in sigmas])
    ax.set_yticks(np.arange(len(metric_defs)))
    ax.set_yticklabels([label for _, label, _ in metric_defs])
    ax.set_xlabel("输入噪声标准差")
    ax.set_title("Cycle44 噪声增强相对 Cycle42 的指标收益热图", fontweight="bold")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            text_color = "white" if abs(matrix[row, col]) > vmax * 0.55 else COLORS["dark"]
            ax.text(col, row, f"{matrix[row, col]:+.1f}%", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, label="收益 / %（相位RMSE按下降为正）")
    cbar.ax.axhline((0 + vmax) / (2 * vmax), color=COLORS["dark"], lw=0.7)
    save_figure(fig, output_dir, "add_fig12_noise_gain_heatmap")


def write_caption_sheet(output_dir: Path, paper_dir: Path) -> None:
    lines = [
        "# 新增论文图件清单与图注建议",
        "",
        "以下图件由 `train/generate_additional_manuscript_figures.py` 生成，PNG 与 PDF 均保存在 `result/figures/additional_manuscript/`。",
        "",
        "## 图 S1 多平面远场强度样例",
        "",
        "展示四个测试样本的焦平面、焦前平面及二者绝对差异。该图用于说明焦平面与焦前平面携带互补的干涉结构信息，为双分支融合网络提供直观依据。",
        "",
        "文件：`add_fig1_multiplane_data_examples.png`",
        "",
        "## 图 S2 双分支门控融合网络结构细图",
        "",
        "展示 Cycle42 `dual_plane_fusion_cnn` 的真实卷积、池化、残差块、门控融合、通道注意力和回归头结构，并标注主要张量尺寸和参数量。",
        "",
        "文件：`add_fig2_model_structure_detail.png`",
        "",
        "## 图 S3 六路相对相位预测一致性",
        "",
        "以周期对齐后的预测相位与真实相位作散点对比。虚线表示理想预测，子图标题给出各相位通道的 RMSE 与 MAE。",
        "",
        "文件：`add_fig3_phase_prediction_scatter.png`",
        "",
        "## 图 S4 相位误差统计诊断",
        "",
        "展示六路相位通道的周期误差分布、单样本 RMSE 直方图以及累计分布，用于补充平均 RMSE 之外的误差离散性分析。",
        "",
        "文件：`add_fig4_phase_error_distribution.png`",
        "",
        "## 图 S5 融合门控与通道注意力统计",
        "",
        "统计 256 个测试样本上的融合门控权重和通道注意力权重，展示样本级、通道级的焦平面/焦前平面利用比例及其与预测误差的关系。",
        "",
        "文件：`add_fig5_gate_attention_statistics.png`",
        "",
        "## 图 S6 典型样本远场光斑补偿前后对比",
        "",
        "选取低误差、中位误差和高误差样本，比较补偿前、Cycle42 补偿后和理想同相情况下的远场光斑。强度采用 log10 归一化显示。",
        "",
        "文件：`add_fig6_farfield_case_gallery.png`",
        "",
        "## 图 S7 相位误差与补偿后光束质量关系",
        "",
        "展示补偿前后 Strehl 分布、残余相位 RMSE 与 Strehl 的关系、主瓣能量统计和单样本 Strehl 提升分布，说明相位反演误差对下游光束质量的影响。",
        "",
        "文件：`add_fig7_metric_relationships.png`",
        "",
        "## 图 S8 动态噪声增强的鲁棒性收益拆解",
        "",
        "从 Strehl 保持率、Cycle44 相对 Cycle42 的百分比收益，以及噪声标准差 0.02 下核心指标对比三个角度展示噪声增强的效果。",
        "",
        "文件：`add_fig8_noise_augmentation_gain.png`",
        "",
        "## 图 S9 模型演进与参数效率",
        "",
        "汇总补偿前、Cycle30、Cycle41、Cycle42 和 Cycle44 的 Strehl 比、合成效率、残余相位 RMSE、参数量与相对提升，展示从深残差网络到双分支融合和噪声增强的主线收益。",
        "",
        "文件：`add_fig9_model_evolution.png`",
        "",
        "## 图 S10 sin/cos 周期编码输出几何诊断",
        "",
        "分析预测 sin/cos 对是否接近单位圆，并展示典型通道在 sin-cos 平面上的真实与预测分布，用于支撑周期编码的稳定性讨论。",
        "",
        "文件：`add_fig10_unit_circle_diagnostics.png`",
        "",
        "## 图 S11 残余相位样本-通道热图",
        "",
        "按单样本 RMSE 排序展示补偿后的六路残余相位，同时给出通道级 RMSE、偏置与 MAE，便于分析误差是否集中在特定光束通道。",
        "",
        "文件：`add_fig11_residual_phase_heatmap.png`",
        "",
        "## 图 S12 噪声增强收益热图",
        "",
        "以热图形式展示 Cycle44 相对 Cycle42 在不同噪声强度下对主瓣能量、Strehl、合成效率和相位 RMSE 的百分比收益，其中相位 RMSE 以下降为正收益。",
        "",
        "文件：`add_fig12_noise_gain_heatmap.png`",
        "",
    ]
    paper_dir.mkdir(parents=True, exist_ok=True)
    path = paper_dir / "additional_figures_caption_sheet.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate additional manuscript figures.")
    parser.add_argument(
        "--image-path",
        type=Path,
        default=REPO_ROOT / "dataset/seven_beam/multiplane_0_-0.07/images_multiplane_7cm.npy",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=REPO_ROOT / "dataset/seven_beam/multiplane_0_-0.07/labels_multiplane_7cm.npy",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "models/cycle42_dual_plane_fusion_7cm_best_rmse_30epoch.pth",
    )
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=REPO_ROOT / "result/metrics/cycle42_dual_plane_fusion_paired_detail.csv",
    )
    parser.add_argument(
        "--noise-csv",
        type=Path,
        default=REPO_ROOT / "result/metrics/cycle44_vs_cycle42_noise_comparison.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "result/figures/additional_manuscript",
    )
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_mmap = np.load(args.image_path, mmap_mode="r")
    labels_mmap = np.load(args.label_path, mmap_mode="r")
    max_samples = min(args.max_samples, images_mmap.shape[0])
    images = np.asarray(images_mmap[:max_samples], dtype=np.float32)
    labels = np.asarray(labels_mmap[:max_samples], dtype=np.float32)

    print(f"loaded images: {images.shape}, labels: {labels.shape}")
    fig_dataset_plane_examples(images, args.output_dir)
    fig_model_structure(args.output_dir)

    model, pred_labels, true_phases, pred_phases, errors, residual, device = run_inference(
        images=images,
        labels=labels,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(f"inference device: {device}")

    fig_phase_prediction(true_phases, pred_phases, errors, args.output_dir)
    fig_phase_error_distribution(errors, args.output_dir)

    gates, attentions = compute_gate_and_attention(model, images, args.batch_size, device)
    sample_rmse = np.sqrt(np.mean(errors**2, axis=1))
    fig_gate_statistics(gates, attentions, sample_rmse, args.output_dir)
    fig_farfield_case_gallery(true_phases, residual, errors, args.output_dir)
    fig_metric_relationships(args.detail_csv, args.output_dir)
    fig_noise_augmentation_gain(args.noise_csv, args.output_dir)
    fig_model_evolution(args.output_dir, args.noise_csv)
    fig_unit_circle_diagnostics(pred_labels, labels, args.output_dir)
    fig_residual_phase_heatmap(residual, errors, args.output_dir)
    fig_noise_gain_heatmap(args.noise_csv, args.output_dir)
    write_caption_sheet(args.output_dir, REPO_ROOT / "paper")


if __name__ == "__main__":
    main()
