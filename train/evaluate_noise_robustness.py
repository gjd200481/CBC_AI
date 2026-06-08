import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.demo_evaluate_two_beam_model import evaluate, load_model
from train.data_utils import FarFieldPhaseDataset
from train.physics_loss import FarFieldConsistencyLoss, TwoBeamFourierOptics


def format_noise_tag(noise_sigma):
    if noise_sigma == 0:
        return "0"
    return str(noise_sigma).rstrip("0").rstrip(".")


def evaluate_farfield_loss(model, data_loader, device):
    """计算模型预测相位重建远场与输入远场之间的平均 MSE。"""
    loss_fn = FarFieldConsistencyLoss(
        optics_model=TwoBeamFourierOptics().to(device),
        loss_type="mse",
    )
    total_loss = 0.0
    total_samples = 0
    model.eval()

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            preds = model(images)
            loss = loss_fn(preds, images)
            batch_size = len(images)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def evaluate_model_on_noise_levels(model_name, model_path, dataset_dir, noise_levels, batch_size, device):
    model = load_model(model_path, device)
    rows = []

    for noise_sigma in noise_levels:
        noise_tag = format_noise_tag(noise_sigma)
        image_path = dataset_dir / f"images_noise_{noise_tag}.npy"
        label_path = dataset_dir / f"labels_noise_{noise_tag}.npy"

        dataset = FarFieldPhaseDataset(
            image_path=image_path,
            label_path=label_path,
            expected_size=(160, 160),
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        _, _, _, metrics = evaluate(model, data_loader, device)
        farfield_loss = evaluate_farfield_loss(model, data_loader, device)

        row = {
            "model": model_name,
            "model_path": str(model_path),
            "noise_sigma": noise_sigma,
            "samples": len(dataset),
            "rmse_rad": metrics["rmse_rad"],
            "rmse_deg": metrics["rmse_deg"],
            "mae_rad": metrics["mae_rad"],
            "mae_deg": metrics["mae_deg"],
            "mean_error_rad": metrics["mean_error_rad"],
            "mean_error_deg": metrics["mean_error_deg"],
            "farfield_mse": farfield_loss,
        }
        rows.append(row)
        print(
            f"{model_name} | noise={noise_sigma} | "
            f"RMSE={metrics['rmse_rad']:.6f} rad | farfield={farfield_loss:.6e}"
        )

    return rows


def save_rows(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_noise_results(rows, figure_path):
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for model_name in models:
        model_rows = [row for row in rows if row["model"] == model_name]
        plt.plot(
            [row["noise_sigma"] for row in model_rows],
            [row["rmse_rad"] for row in model_rows],
            marker="o",
            label=model_name,
        )
    plt.xlabel("Noise sigma")
    plt.ylabel("Phase RMSE(rad)")
    plt.title("Noise robustness")
    plt.legend()

    plt.subplot(1, 2, 2)
    for model_name in models:
        model_rows = [row for row in rows if row["model"] == model_name]
        plt.plot(
            [row["noise_sigma"] for row in model_rows],
            [row["farfield_mse"] for row in model_rows],
            marker="o",
            label=model_name,
        )
    plt.yscale("log")
    plt.xlabel("Noise sigma")
    plt.ylabel("Far-field MSE")
    plt.title("Far-field consistency")
    plt.legend()

    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print("Figure saved to:", figure_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate clean-trained CNN models on detector noise datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "two_beam" / "noise_robustness",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=REPO_ROOT / "models" / "baseline_cnn_main_clean.pth",
    )
    parser.add_argument(
        "--physics-model",
        type=Path,
        default=REPO_ROOT / "models" / "sweep_lambda_0.01_main_clean.pth",
    )
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0, 0.01, 0.03, 0.05, 0.08])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "result" / "metrics" / "cycle09_noise_robustness_2026-06-08.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT / "result" / "figures" / "cycle09_noise_robustness_2026-06-08.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    rows = []
    rows.extend(
        evaluate_model_on_noise_levels(
            model_name="baseline_cnn",
            model_path=args.baseline_model,
            dataset_dir=args.dataset_dir,
            noise_levels=args.noise_levels,
            batch_size=args.batch_size,
            device=device,
        )
    )
    rows.extend(
        evaluate_model_on_noise_levels(
            model_name="physics_cnn_lambda_0.01",
            model_path=args.physics_model,
            dataset_dir=args.dataset_dir,
            noise_levels=args.noise_levels,
            batch_size=args.batch_size,
            device=device,
        )
    )

    save_rows(rows, args.output_csv)
    plot_noise_results(rows, args.figure_path)
    print("Metrics saved to:", args.output_csv)


if __name__ == "__main__":
    main()
