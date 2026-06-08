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


def evaluate_model(model, data_loader, device):
    """在一个 DataLoader 上计算平均监督损失和周期相位误差指标。"""
    criterion = nn.MSELoss()
    model.eval()

    total_loss = 0.0
    total_samples = 0
    pred_values = []
    true_values = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            batch_size = len(images)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            pred_values.append(preds.cpu().numpy())
            true_values.append(labels.cpu().numpy())

    pred_values = np.concatenate(pred_values, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    metrics = phase_metrics_from_sin_cos(pred_values, true_values)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics, pred_values, true_values


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    """训练一个 epoch，返回样本加权后的平均损失。"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = len(images)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_metrics_csv(history, output_path):
    """保存每个 epoch 的训练/验证指标。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_rmse_rad",
        "val_rmse_deg",
        "val_mae_rad",
        "val_mae_deg",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_summary_csv(test_metrics, output_path):
    """保存最终测试集指标。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in test_metrics.items():
            writer.writerow([key, value])


def plot_training(history, true_values, pred_values, figure_path=None, show=True):
    """显示或保存训练曲线和测试集相位预测效果。"""
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]

    true_phi = decode_sin_cos(true_values).reshape(-1)
    pred_phi = decode_sin_cos(pred_values).reshape(-1)
    errors = wrap_phase_error(pred_phi, true_phi)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(true_phi, pred_phi, s=5)
    plt.xlabel("True phi")
    plt.ylabel("Pred phi")
    plt.title("Pred vs True")

    plt.subplot(1, 3, 3)
    plt.hist(errors, bins=30)
    plt.xlabel("Error(rad)")
    plt.title("Error distribution")

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
        description="Train a supervised CNN baseline for two-beam phase inversion."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=REPO_ROOT / "dataset" / "two_beam" / "main_static" / "images_main_clean_two_beam.npy",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=REPO_ROOT / "dataset" / "two_beam" / "main_static" / "labels_main_clean_two_beam.npy",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models" / "two_beam_cnn_main_clean.pth",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=REPO_ROOT / "result" / "metrics" / "baseline_cnn_main_clean.csv",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=REPO_ROOT / "result" / "metrics" / "baseline_cnn_main_clean_summary.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT / "result" / "figures" / "baseline_cnn_main_clean.png",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--image-size", type=int, default=160)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePhaseCNN(
        image_size=args.image_size,
        output_dim=loaders["dataset"].labels.shape[1],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Using device:", device)
    print("Images:", args.image_path)
    print("Labels:", args.label_path)
    print("Splits:", loaders["splits"])
    print("Model output dim:", loaders["dataset"].labels.shape[1])

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_metrics, _, _ = evaluate_model(model, loaders["val"], device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_rmse_rad": val_metrics["rmse_rad"],
            "val_rmse_deg": val_metrics["rmse_deg"],
            "val_mae_rad": val_metrics["mae_rad"],
            "val_mae_deg": val_metrics["mae_deg"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_rmse={val_metrics['rmse_rad']:.6f} rad"
        )

    test_metrics, pred_values, true_values = evaluate_model(model, loaders["test"], device)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_epochs": args.epochs,
            "history": history,
            "test_metrics": test_metrics,
            "image_path": str(args.image_path),
            "label_path": str(args.label_path),
            "splits": loaders["splits"],
            "seed": args.seed,
            "model_class": "SimplePhaseCNN",
            "output_format": "[sin(phi), cos(phi)]",
        },
        args.model_path,
    )
    save_metrics_csv(history, args.metrics_path)
    save_summary_csv(test_metrics, args.summary_path)

    print("\nTest result:")
    print("RMSE(rad):", test_metrics["rmse_rad"])
    print("RMSE(deg):", test_metrics["rmse_deg"])
    print("MAE(rad):", test_metrics["mae_rad"])
    print("MAE(deg):", test_metrics["mae_deg"])
    print("Model saved to:", args.model_path)
    print("Metrics saved to:", args.metrics_path)
    print("Summary saved to:", args.summary_path)

    plot_training(
        history=history,
        true_values=true_values,
        pred_values=pred_values,
        figure_path=args.figure_path,
        show=not args.no_plot,
    )


if __name__ == "__main__":
    main()
