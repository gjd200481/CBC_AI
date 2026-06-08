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

from train.data_utils import FarFieldPhaseDataset
from train.evaluate_seven_beam_noise_robustness import load_seven_beam_model
from train.phase_metrics import phase_metrics_from_sin_cos
from train.physics_loss import FarFieldConsistencyLoss, SevenBeamFourierOptics
from train.train_seven_beam_baseline import channel_rmse_from_sin_cos


def format_float_tag(value):
    if value == 0:
        return "0"
    return str(value).rstrip("0").rstrip(".")


def format_offset_tag(offset_m):
    return f"{int(round(offset_m * 1e6))}um"


def evaluate_model(model, data_loader, farfield_loss_fn, device):
    pred_values = []
    true_values = []
    total_farfield_loss = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            farfield_loss = farfield_loss_fn(preds, images)

            batch_size = len(images)
            total_farfield_loss += farfield_loss.item() * batch_size
            total_samples += batch_size
            pred_values.append(preds.cpu().numpy())
            true_values.append(labels.cpu().numpy())

    pred_values = np.concatenate(pred_values, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    metrics = phase_metrics_from_sin_cos(pred_values, true_values)
    channel_rmse = channel_rmse_from_sin_cos(pred_values, true_values)
    metrics["farfield_mse"] = total_farfield_loss / max(total_samples, 1)
    return metrics, channel_rmse


def build_level_specs(amplitude_levels, position_levels):
    specs = []
    for value in amplitude_levels:
        tag = format_float_tag(value)
        specs.append(("amplitude", value, f"amplitude_{tag}", f"amplitude_{tag}"))
    for value in position_levels:
        tag = format_offset_tag(value)
        specs.append(("position", value, f"position_{tag}", f"position_{tag}"))
    return specs


def evaluate_model_on_specs(
    model_name,
    model_path,
    dataset_dir,
    specs,
    batch_size,
    farfield_loss_fn,
    device,
):
    model = load_seven_beam_model(model_path=model_path, device=device)
    rows = []

    for perturbation_type, level_value, level_name, file_prefix in specs:
        image_path = dataset_dir / f"images_{file_prefix}.npy"
        label_path = dataset_dir / f"labels_{file_prefix}.npy"

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
        metrics, channel_rmse = evaluate_model(
            model=model,
            data_loader=data_loader,
            farfield_loss_fn=farfield_loss_fn,
            device=device,
        )

        row = {
            "model": model_name,
            "model_path": str(model_path),
            "perturbation_type": perturbation_type,
            "level_name": level_name,
            "level_value": level_value,
            "samples": len(dataset),
            "rmse_rad": metrics["rmse_rad"],
            "rmse_deg": metrics["rmse_deg"],
            "mae_rad": metrics["mae_rad"],
            "mae_deg": metrics["mae_deg"],
            "mean_error_rad": metrics["mean_error_rad"],
            "mean_error_deg": metrics["mean_error_deg"],
            "farfield_mse": metrics["farfield_mse"],
        }
        for index, rmse in enumerate(channel_rmse, start=1):
            row[f"channel_{index}_rmse_rad"] = float(rmse)
        rows.append(row)

        print(
            f"{model_name} | {level_name} | "
            f"RMSE={metrics['rmse_rad']:.6f} rad | "
            f"farfield={metrics['farfield_mse']:.6e}"
        )

    return rows


def save_rows(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_improvement_rows(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_rows = {
        (row["perturbation_type"], row["level_name"]): row
        for row in rows
        if row["model"] == "baseline_cnn"
    }
    physics_rows = {
        (row["perturbation_type"], row["level_name"]): row
        for row in rows
        if row["model"] == "physics_cnn_lambda_0.1"
    }

    improvement_rows = []
    for key in baseline_rows:
        baseline = baseline_rows[key]
        physics = physics_rows[key]
        improvement_rows.append(
            {
                "perturbation_type": baseline["perturbation_type"],
                "level_name": baseline["level_name"],
                "level_value": baseline["level_value"],
                "baseline_rmse_rad": baseline["rmse_rad"],
                "physics_rmse_rad": physics["rmse_rad"],
                "rmse_change_percent": 100
                * (physics["rmse_rad"] - baseline["rmse_rad"])
                / baseline["rmse_rad"],
                "baseline_farfield_mse": baseline["farfield_mse"],
                "physics_farfield_mse": physics["farfield_mse"],
                "farfield_change_percent": 100
                * (physics["farfield_mse"] - baseline["farfield_mse"])
                / baseline["farfield_mse"],
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(improvement_rows[0].keys()))
        writer.writeheader()
        writer.writerows(improvement_rows)

    return improvement_rows


def plot_results(rows, figure_path):
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 8))
    plot_index = 1
    for perturbation_type in ["amplitude", "position"]:
        type_rows = [row for row in rows if row["perturbation_type"] == perturbation_type]
        models = []
        for row in type_rows:
            if row["model"] not in models:
                models.append(row["model"])

        plt.subplot(2, 2, plot_index)
        for model_name in models:
            model_rows = [row for row in type_rows if row["model"] == model_name]
            plt.plot(
                [row["level_value"] for row in model_rows],
                [row["rmse_rad"] for row in model_rows],
                marker="o",
                label=model_name,
            )
        plt.xlabel("Mismatch range" if perturbation_type == "amplitude" else "Offset range(m)")
        plt.ylabel("RMSE(rad)")
        plt.title(f"{perturbation_type} phase RMSE")
        plt.legend()

        plt.subplot(2, 2, plot_index + 2)
        for model_name in models:
            model_rows = [row for row in type_rows if row["model"] == model_name]
            plt.plot(
                [row["level_value"] for row in model_rows],
                [row["farfield_mse"] for row in model_rows],
                marker="o",
                label=model_name,
            )
        plt.xlabel("Mismatch range" if perturbation_type == "amplitude" else "Offset range(m)")
        plt.ylabel("Far-field MSE")
        plt.title(f"{perturbation_type} far-field consistency")
        plt.legend()
        plot_index += 1

    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print("Figure saved to:", figure_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate seven-beam CNN models on amplitude mismatch and position offset datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "seven_beam" / "complex_robustness",
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
    parser.add_argument("--amplitude-levels", nargs="+", type=float, default=[0, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--position-levels", nargs="+", type=float, default=[0, 1e-5, 2e-5, 5e-5, 1e-4])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "result" / "metrics" / "cycle16_seven_beam_complex_robustness_2026-06-08.csv",
    )
    parser.add_argument(
        "--improvement-csv",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "cycle16_seven_beam_complex_robustness_improvement_2026-06-08.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT / "result" / "figures" / "cycle16_seven_beam_complex_robustness_2026-06-08.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optics_model = SevenBeamFourierOptics().to(device)
    farfield_loss_fn = FarFieldConsistencyLoss(optics_model=optics_model, loss_type="mse")
    specs = build_level_specs(args.amplitude_levels, args.position_levels)

    print("Using device:", device)

    rows = []
    rows.extend(
        evaluate_model_on_specs(
            model_name="baseline_cnn",
            model_path=args.baseline_model,
            dataset_dir=args.dataset_dir,
            specs=specs,
            batch_size=args.batch_size,
            farfield_loss_fn=farfield_loss_fn,
            device=device,
        )
    )
    rows.extend(
        evaluate_model_on_specs(
            model_name="physics_cnn_lambda_0.1",
            model_path=args.physics_model,
            dataset_dir=args.dataset_dir,
            specs=specs,
            batch_size=args.batch_size,
            farfield_loss_fn=farfield_loss_fn,
            device=device,
        )
    )

    save_rows(rows, args.output_csv)
    save_improvement_rows(rows, args.improvement_csv)
    plot_results(rows, args.figure_path)
    print("Metrics saved to:", args.output_csv)
    print("Improvement saved to:", args.improvement_csv)


if __name__ == "__main__":
    main()
