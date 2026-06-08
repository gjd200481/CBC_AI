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


def save_history_csv(history, output_path):
    """保存每一轮训练和验证指标，便于后续画收敛曲线。"""
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


def save_summary_csv(summary, output_path):
    """保存最终测试集汇总指标。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def train_one_epoch(model, data_loader, optimizer, loss_fn, device):
    """完成一轮普通监督训练。"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = len(images)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def evaluate_model(model, data_loader, loss_fn, device):
    """评估模型并返回整体相位指标和原始预测值。"""
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
            loss = loss_fn(preds, labels)

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


def channel_rmse_from_sin_cos(pred_values, true_values):
    """计算每一路相对相位的 RMSE，用于检查是否存在通道偏差。"""
    pred_phase = decode_sin_cos(pred_values)
    true_phase = decode_sin_cos(true_values)
    errors = wrap_phase_error(pred_phase, true_phase)
    return np.sqrt(np.mean(errors**2, axis=0))


def plot_training(history, pred_values, true_values, figure_path=None, show=True):
    """输出训练曲线、逐通道误差和预测散点图。"""
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    val_rmse = [item["val_rmse_rad"] for item in history]

    pred_phase = decode_sin_cos(pred_values)
    true_phase = decode_sin_cos(true_values)
    channel_rmse = channel_rmse_from_sin_cos(pred_values, true_values)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_rmse)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE(rad)")
    plt.title("Validation phase RMSE")

    plt.subplot(2, 2, 3)
    plt.bar(np.arange(1, len(channel_rmse) + 1), channel_rmse)
    plt.xlabel("Outer beam channel")
    plt.ylabel("RMSE(rad)")
    plt.title("Test RMSE by channel")

    plt.subplot(2, 2, 4)
    plt.scatter(true_phase.reshape(-1), pred_phase.reshape(-1), s=5)
    plt.xlabel("True phase(rad)")
    plt.ylabel("Pred phase(rad)")
    plt.title("Test pred vs true")

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
        description="Train a baseline CNN for seven-beam phase inversion."
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
        default=REPO_ROOT / "models" / "baseline_cnn_main_clean_seven_beam.pth",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "baseline_cnn_main_clean_seven_beam.csv",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "baseline_cnn_main_clean_seven_beam_summary.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "figures"
        / "baseline_cnn_main_clean_seven_beam.png",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260612)
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

    output_dim = loaders["dataset"].labels.shape[1]
    if output_dim != 12:
        raise ValueError(f"Seven-beam labels should have 12 columns, got {output_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePhaseCNN(image_size=args.image_size, output_dim=output_dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Using device:", device)
    print("Images:", args.image_path)
    print("Labels:", args.label_path)
    print("Splits:", loaders["splits"])
    print("Output dim:", output_dim)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=loaders["train"],
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        val_metrics, _, _ = evaluate_model(
            model=model,
            data_loader=loaders["val"],
            loss_fn=loss_fn,
            device=device,
        )

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

    test_metrics, pred_values, true_values = evaluate_model(
        model=model,
        data_loader=loaders["test"],
        loss_fn=loss_fn,
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
            "image_path": str(args.image_path),
            "label_path": str(args.label_path),
            "splits": loaders["splits"],
            "seed": args.seed,
            "model_class": "SimplePhaseCNN",
            "output_format": "[sin(phi_1), cos(phi_1), ..., sin(phi_6), cos(phi_6)]",
        },
        args.model_path,
    )

    save_history_csv(history, args.metrics_path)

    summary = {
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
    for index, rmse in enumerate(channel_rmse, start=1):
        print(f"Channel {index} RMSE(rad): {rmse}")
    print("Model saved to:", args.model_path)
    print("Metrics saved to:", args.metrics_path)
    print("Summary saved to:", args.summary_path)

    plot_training(
        history=history,
        pred_values=pred_values,
        true_values=true_values,
        figure_path=args.figure_path,
        show=not args.no_plot,
    )


if __name__ == "__main__":
    main()
