import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.data_utils import build_dataloaders
from train.models import SimplePhaseCNN
from train.phase_metrics import decode_sin_cos, phase_metrics_from_sin_cos, wrap_phase_error
from train.physics_loss import FarFieldConsistencyLoss, SevenBeamFourierOptics
from train.train_seven_beam_baseline import channel_rmse_from_sin_cos


def save_history_csv(history, output_path):
    """保存物理约束训练过程中每个 epoch 的指标。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_total_loss",
        "train_phase_loss",
        "train_farfield_loss",
        "val_total_loss",
        "val_phase_loss",
        "val_farfield_loss",
        "val_rmse_rad",
        "val_rmse_deg",
        "val_mae_rad",
        "val_mae_deg",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_summary_csv(summary, output_path):
    """保存最终测试集汇总指标。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def compute_losses(preds, labels, images, phase_loss_fn, farfield_loss_fn, lambda_phy):
    """组合相位监督损失和远场物理一致性损失。"""
    phase_loss = phase_loss_fn(preds, labels)
    farfield_loss = farfield_loss_fn(preds, images)
    total_loss = phase_loss + lambda_phy * farfield_loss
    return total_loss, phase_loss, farfield_loss


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    phase_loss_fn,
    farfield_loss_fn,
    lambda_phy,
    device,
):
    """完成一轮 7 光束物理约束训练。"""
    model.train()
    totals = {
        "total_loss": 0.0,
        "phase_loss": 0.0,
        "farfield_loss": 0.0,
        "samples": 0,
    }

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        total_loss, phase_loss, farfield_loss = compute_losses(
            preds=preds,
            labels=labels,
            images=images,
            phase_loss_fn=phase_loss_fn,
            farfield_loss_fn=farfield_loss_fn,
            lambda_phy=lambda_phy,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_size = len(images)
        totals["total_loss"] += total_loss.item() * batch_size
        totals["phase_loss"] += phase_loss.item() * batch_size
        totals["farfield_loss"] += farfield_loss.item() * batch_size
        totals["samples"] += batch_size

    samples = max(totals["samples"], 1)
    return {
        "total_loss": totals["total_loss"] / samples,
        "phase_loss": totals["phase_loss"] / samples,
        "farfield_loss": totals["farfield_loss"] / samples,
    }


def evaluate_model(
    model,
    data_loader,
    phase_loss_fn,
    farfield_loss_fn,
    lambda_phy,
    device,
):
    """评估 7 光束物理约束模型。"""
    model.eval()
    totals = {
        "total_loss": 0.0,
        "phase_loss": 0.0,
        "farfield_loss": 0.0,
        "samples": 0,
    }
    pred_values = []
    true_values = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            total_loss, phase_loss, farfield_loss = compute_losses(
                preds=preds,
                labels=labels,
                images=images,
                phase_loss_fn=phase_loss_fn,
                farfield_loss_fn=farfield_loss_fn,
                lambda_phy=lambda_phy,
            )

            batch_size = len(images)
            totals["total_loss"] += total_loss.item() * batch_size
            totals["phase_loss"] += phase_loss.item() * batch_size
            totals["farfield_loss"] += farfield_loss.item() * batch_size
            totals["samples"] += batch_size
            pred_values.append(preds.cpu().numpy())
            true_values.append(labels.cpu().numpy())

    samples = max(totals["samples"], 1)
    pred_values = np.concatenate(pred_values, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    metrics = phase_metrics_from_sin_cos(pred_values, true_values)
    metrics.update(
        {
            "total_loss": totals["total_loss"] / samples,
            "phase_loss": totals["phase_loss"] / samples,
            "farfield_loss": totals["farfield_loss"] / samples,
        }
    )
    return metrics, pred_values, true_values


def plot_physics_training(history, pred_values, true_values, figure_path=None, show=True):
    """保存物理约束训练曲线和测试集逐通道误差。"""
    epochs = [item["epoch"] for item in history]
    train_total = [item["train_total_loss"] for item in history]
    train_phase = [item["train_phase_loss"] for item in history]
    train_farfield = [item["train_farfield_loss"] for item in history]
    val_total = [item["val_total_loss"] for item in history]
    val_rmse = [item["val_rmse_rad"] for item in history]

    pred_phase = decode_sin_cos(pred_values)
    true_phase = decode_sin_cos(true_values)
    errors = wrap_phase_error(pred_phase, true_phase)
    channel_rmse = channel_rmse_from_sin_cos(pred_values, true_values)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_total, label="train_total")
    plt.plot(epochs, val_total, label="val_total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_phase, label="phase")
    plt.plot(epochs, train_farfield, label="farfield")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train loss components")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_rmse)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE(rad)")
    plt.title("Validation phase RMSE")

    plt.subplot(2, 2, 4)
    plt.bar(np.arange(1, len(channel_rmse) + 1), channel_rmse)
    plt.xlabel("Outer beam channel")
    plt.ylabel("RMSE(rad)")
    plt.title(f"Test channel RMSE, mean={np.mean(np.abs(errors)):.4f}")

    plt.tight_layout()
    if figure_path is not None:
        figure_path = Path(figure_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path, dpi=200)
        print("Figure saved to:", figure_path)

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train a physics-constrained CNN for seven-beam phase inversion."
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
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models" / "physics_cnn_lambda_0.1_main_clean_seven_beam.pth",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "physics_cnn_lambda_0.1_main_clean_seven_beam.csv",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "physics_cnn_lambda_0.1_main_clean_seven_beam_summary.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "figures"
        / "physics_cnn_lambda_0.1_main_clean_seven_beam.png",
    )
    parser.add_argument("--lambda-phy", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loaders = build_dataloaders(
        image_path=args.image_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        expected_size=(args.image_size, args.image_size),
    )

    output_dim = loaders["dataset"].labels.shape[1]
    if output_dim != 12:
        raise ValueError(f"Seven-beam labels should have 12 columns, got {output_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePhaseCNN(image_size=args.image_size, output_dim=output_dim).to(device)

    optics_model = SevenBeamFourierOptics(
        num_points=args.num_points,
        window_size=args.window_size,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.image_size,
    ).to(device)
    farfield_loss_fn = FarFieldConsistencyLoss(optics_model=optics_model, loss_type="mse")
    phase_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Using device:", device)
    print("Images:", args.image_path)
    print("Labels:", args.label_path)
    print("Splits:", loaders["splits"])
    print("Output dim:", output_dim)
    print("lambda_phy:", args.lambda_phy)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            data_loader=loaders["train"],
            optimizer=optimizer,
            phase_loss_fn=phase_loss_fn,
            farfield_loss_fn=farfield_loss_fn,
            lambda_phy=args.lambda_phy,
            device=device,
        )
        val_metrics, _, _ = evaluate_model(
            model=model,
            data_loader=loaders["val"],
            phase_loss_fn=phase_loss_fn,
            farfield_loss_fn=farfield_loss_fn,
            lambda_phy=args.lambda_phy,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_phase_loss": train_metrics["phase_loss"],
            "train_farfield_loss": train_metrics["farfield_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_phase_loss": val_metrics["phase_loss"],
            "val_farfield_loss": val_metrics["farfield_loss"],
            "val_rmse_rad": val_metrics["rmse_rad"],
            "val_rmse_deg": val_metrics["rmse_deg"],
            "val_mae_rad": val_metrics["mae_rad"],
            "val_mae_deg": val_metrics["mae_deg"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_total={train_metrics['total_loss']:.6f} | "
            f"train_phase={train_metrics['phase_loss']:.6f} | "
            f"train_farfield={train_metrics['farfield_loss']:.6e} | "
            f"val_rmse={val_metrics['rmse_rad']:.6f} rad | "
            f"val_farfield={val_metrics['farfield_loss']:.6e}"
        )

    test_metrics, pred_values, true_values = evaluate_model(
        model=model,
        data_loader=loaders["test"],
        phase_loss_fn=phase_loss_fn,
        farfield_loss_fn=farfield_loss_fn,
        lambda_phy=args.lambda_phy,
        device=device,
    )
    channel_rmse = channel_rmse_from_sin_cos(pred_values, true_values)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_epochs": args.epochs,
            "history": history,
            "test_metrics": test_metrics,
            "channel_rmse_rad": channel_rmse.tolist(),
            "lambda_phy": args.lambda_phy,
            "image_path": str(args.image_path),
            "label_path": str(args.label_path),
            "splits": loaders["splits"],
            "seed": args.seed,
            "model_class": "SimplePhaseCNN",
            "output_format": "[sin(phi_1), cos(phi_1), ..., sin(phi_6), cos(phi_6)]",
            "physics_model": "SevenBeamFourierOptics",
        },
        args.model_path,
    )

    save_history_csv(history, args.metrics_path)

    summary = {
        "lambda_phy": args.lambda_phy,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_samples": loaders["splits"]["train"],
        "val_samples": loaders["splits"]["val"],
        "test_samples": loaders["splits"]["test"],
        **test_metrics,
    }
    for index, rmse in enumerate(channel_rmse, start=1):
        summary[f"channel_{index}_rmse_rad"] = float(rmse)
        summary[f"channel_{index}_rmse_deg"] = float(np.degrees(rmse))
    save_summary_csv(summary, args.summary_path)

    print("\nTest result:")
    print("RMSE(rad):", test_metrics["rmse_rad"])
    print("RMSE(deg):", test_metrics["rmse_deg"])
    print("MAE(rad):", test_metrics["mae_rad"])
    print("MAE(deg):", test_metrics["mae_deg"])
    print("Phase loss:", test_metrics["phase_loss"])
    print("Far-field loss:", test_metrics["farfield_loss"])
    print("Total loss:", test_metrics["total_loss"])
    for index, rmse in enumerate(channel_rmse, start=1):
        print(f"Channel {index} RMSE(rad): {rmse}")
    print("Model saved to:", args.model_path)
    print("Metrics saved to:", args.metrics_path)
    print("Summary saved to:", args.summary_path)

    plot_physics_training(
        history=history,
        pred_values=pred_values,
        true_values=true_values,
        figure_path=args.figure_path,
        show=not args.no_plot,
    )


if __name__ == "__main__":
    main()
