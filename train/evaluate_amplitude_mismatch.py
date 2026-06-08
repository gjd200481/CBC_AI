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


def format_mismatch_tag(mismatch):
    if mismatch == 0:
        return "0"
    return str(mismatch).rstrip("0").rstrip(".")


def evaluate_model_on_mismatch_levels(model_name, model_path, dataset_dir, mismatch_levels, batch_size, device):
    model = load_model(model_path, device)
    rows = []

    for mismatch in mismatch_levels:
        mismatch_tag = format_mismatch_tag(mismatch)
        image_path = dataset_dir / f"images_amp_{mismatch_tag}.npy"
        label_path = dataset_dir / f"labels_amp_{mismatch_tag}.npy"

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
        row = {
            "model": model_name,
            "model_path": str(model_path),
            "mismatch_range": mismatch,
            "samples": len(dataset),
            "rmse_rad": metrics["rmse_rad"],
            "rmse_deg": metrics["rmse_deg"],
            "mae_rad": metrics["mae_rad"],
            "mae_deg": metrics["mae_deg"],
            "mean_error_rad": metrics["mean_error_rad"],
            "mean_error_deg": metrics["mean_error_deg"],
        }
        rows.append(row)
        print(f"{model_name} | mismatch={mismatch} | RMSE={metrics['rmse_rad']:.6f} rad")

    return rows


def save_rows(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, figure_path):
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = []
    for row in rows:
        if row["model"] not in model_names:
            model_names.append(row["model"])

    plt.figure(figsize=(6, 4))
    for model_name in model_names:
        model_rows = [row for row in rows if row["model"] == model_name]
        plt.plot(
            [row["mismatch_range"] for row in model_rows],
            [row["rmse_rad"] for row in model_rows],
            marker="o",
            label=model_name,
        )
    plt.xlabel("Amplitude mismatch range")
    plt.ylabel("Phase RMSE(rad)")
    plt.title("Amplitude mismatch robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print("Figure saved to:", figure_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate clean-trained CNN models on amplitude mismatch datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "two_beam" / "amplitude_mismatch",
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
    parser.add_argument("--mismatch-levels", nargs="+", type=float, default=[0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "result" / "metrics" / "cycle10_amplitude_mismatch_2026-06-08.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT / "result" / "figures" / "cycle10_amplitude_mismatch_2026-06-08.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    rows = []
    rows.extend(
        evaluate_model_on_mismatch_levels(
            model_name="baseline_cnn",
            model_path=args.baseline_model,
            dataset_dir=args.dataset_dir,
            mismatch_levels=args.mismatch_levels,
            batch_size=args.batch_size,
            device=device,
        )
    )
    rows.extend(
        evaluate_model_on_mismatch_levels(
            model_name="physics_cnn_lambda_0.01",
            model_path=args.physics_model,
            dataset_dir=args.dataset_dir,
            mismatch_levels=args.mismatch_levels,
            batch_size=args.batch_size,
            device=device,
        )
    )

    save_rows(rows, args.output_csv)
    plot_results(rows, args.figure_path)
    print("Metrics saved to:", args.output_csv)


if __name__ == "__main__":
    main()
