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
from train.physics_loss import FarFieldConsistencyLoss, SevenBeamFourierOptics
from train.train_seven_beam_baseline import channel_rmse_from_sin_cos
from train.train_seven_beam_physics_constrained_cnn import (
    evaluate_model,
    save_history_csv,
    train_one_epoch,
)


def lambda_tag(lambda_phy):
    """把 lambda 数值转换为适合文件名的短标签。"""
    text = f"{lambda_phy:g}"
    return text.replace(".", "p").replace("-", "m")


def save_sweep_summary(rows, output_path):
    """保存所有 lambda 的汇总结果。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "lambda_phy",
        "epochs",
        "batch_size",
        "learning_rate",
        "train_samples",
        "val_samples",
        "test_samples",
        "rmse_rad",
        "rmse_deg",
        "mae_rad",
        "mae_deg",
        "mean_error_rad",
        "mean_error_deg",
        "phase_loss",
        "farfield_loss",
        "total_loss",
        "channel_1_rmse_rad",
        "channel_2_rmse_rad",
        "channel_3_rmse_rad",
        "channel_4_rmse_rad",
        "channel_5_rmse_rad",
        "channel_6_rmse_rad",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows, figure_path=None, show=True):
    """绘制 lambda 权重对相位误差和远场误差的影响。"""
    lambdas = np.array([row["lambda_phy"] for row in rows], dtype=float)
    rmse = np.array([row["rmse_rad"] for row in rows], dtype=float)
    farfield = np.array([row["farfield_loss"] for row in rows], dtype=float)
    phase_loss = np.array([row["phase_loss"] for row in rows], dtype=float)

    labels = [f"{value:g}" for value in lambdas]
    x = np.arange(len(lambdas))

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x, rmse, marker="o")
    plt.xticks(x, labels)
    plt.xlabel("lambda_phy")
    plt.ylabel("RMSE(rad)")
    plt.title("Phase RMSE")

    plt.subplot(2, 2, 2)
    plt.plot(x, farfield, marker="o")
    plt.xticks(x, labels)
    plt.xlabel("lambda_phy")
    plt.ylabel("Far-field MSE")
    plt.title("Far-field consistency")

    plt.subplot(2, 2, 3)
    plt.plot(x, phase_loss, marker="o", label="phase")
    plt.plot(x, farfield, marker="o", label="farfield")
    plt.xticks(x, labels)
    plt.xlabel("lambda_phy")
    plt.ylabel("Loss")
    plt.title("Loss components")
    plt.legend()

    plt.subplot(2, 2, 4)
    channel_keys = [f"channel_{index}_rmse_rad" for index in range(1, 7)]
    channel_values = np.array([[row[key] for key in channel_keys] for row in rows])
    plt.imshow(channel_values, aspect="auto", cmap="viridis")
    plt.colorbar(label="RMSE(rad)")
    plt.xticks(np.arange(6), [f"ch{i}" for i in range(1, 7)])
    plt.yticks(x, labels)
    plt.xlabel("Channel")
    plt.ylabel("lambda_phy")
    plt.title("Channel RMSE")

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
        description="Sweep lambda_phy for seven-beam physics-constrained CNN."
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
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=REPO_ROOT / "result" / "metrics" / "cycle14_seven_beam_lambda_sweep",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "models" / "cycle14_seven_beam_lambda_sweep",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "cycle14_seven_beam_lambda_sweep_2026-06-08.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "figures"
        / "cycle14_seven_beam_lambda_sweep_2026-06-08.png",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

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
    optics_model = SevenBeamFourierOptics(
        num_points=args.num_points,
        window_size=args.window_size,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.image_size,
    ).to(device)
    farfield_loss_fn = FarFieldConsistencyLoss(optics_model=optics_model, loss_type="mse")
    phase_loss_fn = nn.MSELoss()

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Images:", args.image_path)
    print("Labels:", args.label_path)
    print("Splits:", loaders["splits"])
    print("Lambdas:", args.lambdas)

    rows = []
    for lambda_phy in args.lambdas:
        tag = lambda_tag(lambda_phy)
        print(f"\n=== lambda_phy={lambda_phy:g} ===")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = SimplePhaseCNN(image_size=args.image_size, output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        history = []
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                data_loader=loaders["train"],
                optimizer=optimizer,
                phase_loss_fn=phase_loss_fn,
                farfield_loss_fn=farfield_loss_fn,
                lambda_phy=lambda_phy,
                device=device,
            )
            val_metrics, _, _ = evaluate_model(
                model=model,
                data_loader=loaders["val"],
                phase_loss_fn=phase_loss_fn,
                farfield_loss_fn=farfield_loss_fn,
                lambda_phy=lambda_phy,
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
                f"train_farfield={train_metrics['farfield_loss']:.6e} | "
                f"val_rmse={val_metrics['rmse_rad']:.6f} rad"
            )

        test_metrics, pred_values, true_values = evaluate_model(
            model=model,
            data_loader=loaders["test"],
            phase_loss_fn=phase_loss_fn,
            farfield_loss_fn=farfield_loss_fn,
            lambda_phy=lambda_phy,
            device=device,
        )
        channel_rmse = channel_rmse_from_sin_cos(pred_values, true_values)

        history_path = args.metrics_dir / f"lambda_{tag}_history.csv"
        model_path = args.model_dir / f"lambda_{tag}.pth"
        save_history_csv(history, history_path)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_epochs": args.epochs,
                "history": history,
                "test_metrics": test_metrics,
                "channel_rmse_rad": channel_rmse.tolist(),
                "lambda_phy": lambda_phy,
                "image_path": str(args.image_path),
                "label_path": str(args.label_path),
                "splits": loaders["splits"],
                "seed": args.seed,
                "model_class": "SimplePhaseCNN",
                "physics_model": "SevenBeamFourierOptics",
            },
            model_path,
        )

        summary_row = {
            "lambda_phy": lambda_phy,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_samples": loaders["splits"]["train"],
            "val_samples": loaders["splits"]["val"],
            "test_samples": loaders["splits"]["test"],
            **test_metrics,
        }
        for index, rmse in enumerate(channel_rmse, start=1):
            summary_row[f"channel_{index}_rmse_rad"] = float(rmse)
        rows.append(summary_row)

        print(
            f"Test | rmse={test_metrics['rmse_rad']:.6f} rad | "
            f"farfield={test_metrics['farfield_loss']:.6e} | "
            f"model={model_path}"
        )

    save_sweep_summary(rows, args.summary_path)
    plot_sweep(rows, figure_path=args.figure_path, show=not args.no_plot)

    best_rmse = min(rows, key=lambda item: item["rmse_rad"])
    best_farfield = min(rows, key=lambda item: item["farfield_loss"])
    print("\nSweep complete.")
    print("Summary saved to:", args.summary_path)
    print("Best RMSE lambda:", best_rmse["lambda_phy"], best_rmse["rmse_rad"])
    print("Best far-field lambda:", best_farfield["lambda_phy"], best_farfield["farfield_loss"])


if __name__ == "__main__":
    main()
