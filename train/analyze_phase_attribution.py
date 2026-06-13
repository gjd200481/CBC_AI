"""Gradient attribution analysis for seven-beam phase inversion models."""

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

from train.models import DeepResidualPhaseCNN, MultiPlanePhaseCNN, build_phase_model


def load_model(checkpoint_path, model_name, image_size, output_dim, in_channels, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    if state_dict is None:
        raise KeyError("Checkpoint must contain 'model_state_dict' or 'model_state'.")

    config = checkpoint.get("config", {})
    model_label = model_name or checkpoint.get("model_name") or config.get("model_type") or "deep_residual_cnn"

    if model_label == "multiplane":
        model = MultiPlanePhaseCNN(image_size=image_size, output_dim=output_dim, num_planes=in_channels)
    elif model_label == "single":
        model = DeepResidualPhaseCNN(image_size=image_size, output_dim=output_dim, in_channels=1)
    else:
        model = build_phase_model(
            model_name=model_label,
            image_size=image_size,
            output_dim=output_dim,
            in_channels=in_channels,
        )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    display_name = checkpoint.get("model_name") or config.get("name") or model_label
    return model, display_name


def normalize_map(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float32)
    values = values - float(values.min())
    denom = float(values.max()) + eps
    return values / denom


def attribution_for_sample(model, image, channels, device):
    """Return saliency maps for selected phase channels.

    channels are 0-based phase indices.  Each channel score uses both the
    sin and cos outputs for that phase, so the map reflects sensitivity of
    the phase pair rather than one scalar logit.
    """
    x = torch.as_tensor(image, dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    x.requires_grad_(True)

    maps = {}
    plane_ratios = {}
    for channel in channels:
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        pred = model(x)
        pair = pred[0, 2 * channel:2 * channel + 2]
        score = torch.sum(pair * pair)
        score.backward(retain_graph=True)
        grad_abs = x.grad.detach().abs()[0].cpu().numpy()
        saliency = grad_abs.max(axis=0)
        maps[channel + 1] = normalize_map(saliency)

        plane_energy = grad_abs.reshape(grad_abs.shape[0], -1).sum(axis=1)
        total_energy = float(plane_energy.sum()) + 1e-12
        plane_ratios[channel + 1] = {
            f"plane_{index + 1}_energy_ratio": float(value / total_energy)
            for index, value in enumerate(plane_energy)
        }
    return maps, plane_ratios


def saliency_metrics(saliency, top_fraction=0.1):
    h, w = saliency.shape
    yy, xx = np.mgrid[0:h, 0:w]
    total = float(saliency.sum()) + 1e-12
    cy = float((yy * saliency).sum() / total)
    cx = float((xx * saliency).sum() / total)
    radius = np.sqrt((yy - h / 2 + 0.5) ** 2 + (xx - w / 2 + 0.5) ** 2)
    mean_radius = float((radius * saliency).sum() / total)

    threshold = np.quantile(saliency, 1.0 - top_fraction)
    top_mask = saliency >= threshold
    top_energy = float(saliency[top_mask].sum() / total)

    return {
        "center_y": cy,
        "center_x": cx,
        "mean_radius_px": mean_radius,
        "top_fraction": top_fraction,
        "top_energy_ratio": top_energy,
    }


def plot_attribution_grid(image, maps, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_image = image[0] if image.ndim == 3 else image
    ncols = len(maps) + 1
    plt.figure(figsize=(3.2 * ncols, 3.4))
    plt.subplot(1, ncols, 1)
    plt.imshow(display_image, cmap="gray")
    plt.title("input")
    plt.axis("off")

    for index, (channel, saliency) in enumerate(maps.items(), start=2):
        plt.subplot(1, ncols, index)
        plt.imshow(display_image, cmap="gray")
        plt.imshow(saliency, cmap="magma", alpha=0.65)
        plt.title(f"phase {channel}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--label-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--sample-indices", default="0,1,2")
    parser.add_argument("--channels", default="0,1,2,3,4,5")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--output-dim", type=int, default=12)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--figure-dir", type=Path, default=REPO_ROOT / "result/figures/cycle35_attribution")
    parser.add_argument("--summary-csv", type=Path, default=REPO_ROOT / "result/metrics/cycle35_attribution_summary.csv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    images = np.load(args.image_path)
    sample_indices = [int(item) for item in args.sample_indices.split(",") if item.strip()]
    channels = [int(item) for item in args.channels.split(",") if item.strip()]

    if images.ndim == 3:
        in_channels = 1
    elif images.ndim == 4:
        in_channels = images.shape[1]
    else:
        raise ValueError(f"Expected images [N,H,W] or [N,C,H,W], got {images.shape}")
    if args.in_channels is not None:
        in_channels = args.in_channels

    model, model_name = load_model(
        checkpoint_path=args.model_path,
        model_name=args.model_name,
        image_size=args.image_size,
        output_dim=args.output_dim,
        in_channels=in_channels,
        device=device,
    )

    rows = []
    for sample_index in sample_indices:
        image = images[sample_index]
        maps, plane_ratios = attribution_for_sample(model, image, channels, device)
        figure_path = args.figure_dir / f"{model_name}_sample{sample_index}.png"
        plot_attribution_grid(
            image=image,
            maps=maps,
            output_path=figure_path,
            title=f"{model_name} sample {sample_index}",
        )
        for channel, saliency in maps.items():
            rows.append(
                {
                    "model_name": model_name,
                    "sample_index": sample_index,
                    "channel": channel,
                    **saliency_metrics(saliency),
                    **plane_ratios[channel],
                    "figure_path": str(figure_path),
                }
            )

    save_csv(rows, args.summary_csv)
    print("Using device:", device)
    print("Model:", model_name)
    print("Samples:", sample_indices)
    print("Channels:", [channel + 1 for channel in channels])
    print("Summary saved to:", args.summary_csv)
    print("Figures saved to:", args.figure_dir)


if __name__ == "__main__":
    main()
