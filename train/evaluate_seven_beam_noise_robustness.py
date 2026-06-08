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
from train.models import SimplePhaseCNN
from train.phase_metrics import phase_metrics_from_sin_cos
from train.physics_loss import FarFieldConsistencyLoss, SevenBeamFourierOptics
from train.train_seven_beam_baseline import channel_rmse_from_sin_cos


def format_noise_tag(noise_sigma):
    """将噪声强度转换为数据集文件标签。"""
    if noise_sigma == 0:
        return "0"
    return str(noise_sigma).rstrip("0").rstrip(".")


def load_seven_beam_model(model_path, device):
    """加载 12 维输出的 7 光束相位反演模型。"""
    checkpoint = torch.load(model_path, map_location=device)
    model = SimplePhaseCNN(image_size=160, output_dim=12).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_model(model, data_loader, farfield_loss_fn, device):
    """评估模型在一个噪声数据集上的相位误差与远场一致性误差。"""
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


def evaluate_model_on_noise_levels(
    model_name,
    model_path,
    dataset_dir,
    noise_levels,
    batch_size,
    farfield_loss_fn,
    device,
):
    """在多个噪声强度数据集上评估同一模型。"""
    model = load_seven_beam_model(model_path=model_path, device=device)
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

        metrics, channel_rmse = evaluate_model(
            model=model,
            data_loader=data_loader,
            farfield_loss_fn=farfield_loss_fn,
            device=device,
        )

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
            "farfield_mse": metrics["farfield_mse"],
        }
        for index, rmse in enumerate(channel_rmse, start=1):
            row[f"channel_{index}_rmse_rad"] = float(rmse)
        rows.append(row)

        print(
            f"{model_name} | noise={noise_sigma:g} | "
            f"RMSE={metrics['rmse_rad']:.6f} rad | "
            f"farfield={metrics['farfield_mse']:.6e}"
        )

    return rows


def save_rows(rows, output_path):
    """保存评估结果 CSV。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_improvement_rows(rows, output_path):
    """保存物理约束模型相对普通 CNN 的变化百分比。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_rows = {row["noise_sigma"]: row for row in rows if row["model"] == "baseline_cnn"}
    physics_rows = {row["noise_sigma"]: row for row in rows if row["model"] == "physics_cnn_lambda_0.1"}

    improvement_rows = []
    for noise_sigma in sorted(baseline_rows):
        baseline = baseline_rows[noise_sigma]
        physics = physics_rows[noise_sigma]
        improvement_rows.append(
            {
                "noise_sigma": noise_sigma,
                "baseline_rmse_rad": baseline["rmse_rad"],
                "physics_rmse_rad": physics["rmse_rad"],
                "rmse_change_percent": 100 * (physics["rmse_rad"] - baseline["rmse_rad"]) / baseline["rmse_rad"],
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


def plot_noise_results(rows, figure_path):
    """绘制噪声强度下的 RMSE 和远场 MSE 对比曲线。"""
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = []
    for row in rows:
        if row["model"] not in model_names:
            model_names.append(row["model"])

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    for model_name in model_names:
        model_rows = [row for row in rows if row["model"] == model_name]
        plt.plot(
            [row["noise_sigma"] for row in model_rows],
            [row["rmse_rad"] for row in model_rows],
            marker="o",
            label=model_name,
        )
    plt.xlabel("Noise sigma")
    plt.ylabel("Phase RMSE(rad)")
    plt.title("Seven-beam noise robustness")
    plt.legend()

    plt.subplot(2, 2, 2)
    for model_name in model_names:
        model_rows = [row for row in rows if row["model"] == model_name]
        plt.plot(
            [row["noise_sigma"] for row in model_rows],
            [row["mae_rad"] for row in model_rows],
            marker="o",
            label=model_name,
        )
    plt.xlabel("Noise sigma")
    plt.ylabel("Phase MAE(rad)")
    plt.title("Mean absolute error")
    plt.legend()

    plt.subplot(2, 2, 3)
    for model_name in model_names:
        model_rows = [row for row in rows if row["model"] == model_name]
        plt.plot(
            [row["noise_sigma"] for row in model_rows],
            [row["farfield_mse"] for row in model_rows],
            marker="o",
            label=model_name,
        )
    plt.xlabel("Noise sigma")
    plt.ylabel("Far-field MSE")
    plt.title("Far-field consistency")
    plt.legend()

    plt.subplot(2, 2, 4)
    channel_keys = [f"channel_{index}_rmse_rad" for index in range(1, 7)]
    physics_rows = [row for row in rows if row["model"] == "physics_cnn_lambda_0.1"]
    channel_values = np.array([[row[key] for key in channel_keys] for row in physics_rows])
    plt.imshow(channel_values, aspect="auto", cmap="viridis")
    plt.colorbar(label="RMSE(rad)")
    plt.xticks(np.arange(6), [f"ch{i}" for i in range(1, 7)])
    plt.yticks(np.arange(len(physics_rows)), [f"{row['noise_sigma']:g}" for row in physics_rows])
    plt.xlabel("Channel")
    plt.ylabel("Noise sigma")
    plt.title("Physics CNN channel RMSE")

    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print("Figure saved to:", figure_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate seven-beam CNN models on detector noise datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "seven_beam" / "noise_robustness",
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
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0, 0.01, 0.03, 0.05, 0.08])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "result" / "metrics" / "cycle15_seven_beam_noise_robustness_2026-06-08.csv",
    )
    parser.add_argument(
        "--improvement-csv",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "cycle15_seven_beam_noise_robustness_improvement_2026-06-08.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT / "result" / "figures" / "cycle15_seven_beam_noise_robustness_2026-06-08.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optics_model = SevenBeamFourierOptics().to(device)
    farfield_loss_fn = FarFieldConsistencyLoss(optics_model=optics_model, loss_type="mse")

    print("Using device:", device)

    rows = []
    rows.extend(
        evaluate_model_on_noise_levels(
            model_name="baseline_cnn",
            model_path=args.baseline_model,
            dataset_dir=args.dataset_dir,
            noise_levels=args.noise_levels,
            batch_size=args.batch_size,
            farfield_loss_fn=farfield_loss_fn,
            device=device,
        )
    )
    rows.extend(
        evaluate_model_on_noise_levels(
            model_name="physics_cnn_lambda_0.1",
            model_path=args.physics_model,
            dataset_dir=args.dataset_dir,
            noise_levels=args.noise_levels,
            batch_size=args.batch_size,
            farfield_loss_fn=farfield_loss_fn,
            device=device,
        )
    )

    save_rows(rows, args.output_csv)
    save_improvement_rows(rows, args.improvement_csv)
    plot_noise_results(rows, args.figure_path)
    print("Metrics saved to:", args.output_csv)
    print("Improvement saved to:", args.improvement_csv)


if __name__ == "__main__":
    main()
