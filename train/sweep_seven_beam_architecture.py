import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.data_utils import FarFieldPhaseDataset, split_dataset
from train.models import build_phase_model, count_parameters
from train.phase_metrics import build_phase_loss
from train.train_seven_beam_baseline import (
    channel_rmse_from_sin_cos,
    evaluate_model,
    save_history_csv,
    save_summary_csv,
    train_one_epoch,
)


def resolve_device(device_name):
    """根据命令行参数选择训练设备。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(device_name)


def save_rows(rows, output_path):
    """保存结构消融汇总表。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_model_tag(model_name):
    """生成可用于文件名的模型标签。"""
    return model_name.replace("-", "_").replace(".", "p")


def train_one_model(model_name, args, loaders, output_dim, device):
    """训练并评估一个网络结构。"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_phase_model(
        model_name=model_name,
        image_size=args.image_size,
        output_dim=output_dim,
    ).to(device)
    parameter_count = count_parameters(model)
    loss_fn = build_phase_loss(
        loss_name=args.phase_loss,
        unit_weight=args.unit_loss_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    best_epoch = 0
    best_val_rmse = float("inf")
    best_val_loss = float("inf")
    best_state_dict = None
    start_time = time.perf_counter()
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
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_rmse_rad": val_metrics["rmse_rad"],
                "val_rmse_deg": val_metrics["rmse_deg"],
                "val_mae_rad": val_metrics["mae_rad"],
                "val_mae_deg": val_metrics["mae_deg"],
            }
        )
        if val_metrics["rmse_rad"] < best_val_rmse:
            best_epoch = epoch
            best_val_rmse = val_metrics["rmse_rad"]
            best_val_loss = val_metrics["loss"]
            best_state_dict = copy.deepcopy(model.state_dict())
        print(
            f"{model_name} | epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.6f} | val_rmse={val_metrics['rmse_rad']:.6f}"
        )

    train_seconds = time.perf_counter() - start_time
    test_metrics, pred_values, true_values = evaluate_model(
        model=model,
        data_loader=loaders["test"],
        loss_fn=loss_fn,
        device=device,
    )
    channel_rmse = channel_rmse_from_sin_cos(pred_values, true_values)

    model_tag = make_model_tag(model_name)
    history_path = args.history_dir / f"{model_tag}_history.csv"
    summary_path = args.history_dir / f"{model_tag}_summary.csv"
    model_path = args.model_dir / f"{args.experiment_tag}_{model_tag}_seven_beam.pth"
    best_model_path = args.model_dir / f"{args.experiment_tag}_{model_tag}_seven_beam_best.pth"

    best_checkpoint_metrics = {}
    best_channel_rmse = None
    if best_state_dict is not None:
        final_state_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state_dict)
        best_metrics, best_pred_values, best_true_values = evaluate_model(
            model=model,
            data_loader=loaders["test"],
            loss_fn=loss_fn,
            device=device,
        )
        best_channel_rmse = channel_rmse_from_sin_cos(best_pred_values, best_true_values)
        best_checkpoint_metrics = {
            "best_checkpoint_test_rmse_rad": best_metrics["rmse_rad"],
            "best_checkpoint_test_rmse_deg": best_metrics["rmse_deg"],
            "best_checkpoint_test_mae_rad": best_metrics["mae_rad"],
            "best_checkpoint_test_mae_deg": best_metrics["mae_deg"],
            "best_checkpoint_test_loss": best_metrics["loss"],
        }
        model.load_state_dict(final_state_dict)

    save_history_csv(history, history_path)
    summary = {
        "model_name": model_name,
        "phase_loss": args.phase_loss,
        "unit_loss_weight": args.unit_loss_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "sample_scope": "full" if args.full_dataset else f"first_{args.max_samples}",
        "parameter_count": parameter_count,
        "model_saved": not args.no_save_model,
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "best_val_rmse_rad": best_val_rmse,
        "best_val_loss": best_val_loss,
        "train_samples": loaders["splits"]["train"],
        "val_samples": loaders["splits"]["val"],
        "test_samples": loaders["splits"]["test"],
        "rmse_rad": test_metrics["rmse_rad"],
        "rmse_deg": test_metrics["rmse_deg"],
        "mae_rad": test_metrics["mae_rad"],
        "mae_deg": test_metrics["mae_deg"],
        "mean_error_rad": test_metrics["mean_error_rad"],
        "mean_error_deg": test_metrics["mean_error_deg"],
        "loss": test_metrics["loss"],
        **best_checkpoint_metrics,
    }
    for index, rmse in enumerate(channel_rmse, start=1):
        summary[f"channel_{index}_rmse_rad"] = float(rmse)
    save_summary_csv(summary, summary_path)

    if not args.no_save_model:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": model_name,
                "phase_loss": args.phase_loss,
                "unit_loss_weight": args.unit_loss_weight,
                "model_state_dict": model.state_dict(),
                "summary": summary,
                "history": history,
                "checkpoint_type": "final_epoch",
            },
            model_path,
        )
        if best_state_dict is not None:
            torch.save(
                {
                    "model_name": model_name,
                    "phase_loss": args.phase_loss,
                    "unit_loss_weight": args.unit_loss_weight,
                    "model_state_dict": best_state_dict,
                    "summary": summary,
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_rmse_rad": best_val_rmse,
                    "checkpoint_type": "best_validation_rmse",
                },
                best_model_path,
            )

    row = {
        "model_name": model_name,
        "phase_loss": args.phase_loss,
        "unit_loss_weight": args.unit_loss_weight,
        "parameter_count": parameter_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "sample_scope": "full" if args.full_dataset else f"first_{args.max_samples}",
        "model_saved": not args.no_save_model,
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "train_samples": loaders["splits"]["train"],
        "val_samples": loaders["splits"]["val"],
        "test_samples": loaders["splits"]["test"],
        "test_rmse_rad": test_metrics["rmse_rad"],
        "test_rmse_deg": test_metrics["rmse_deg"],
        "test_mae_rad": test_metrics["mae_rad"],
        "test_mae_deg": test_metrics["mae_deg"],
        "test_loss": test_metrics["loss"],
        "best_val_rmse_rad": min(item["val_rmse_rad"] for item in history),
        "best_val_loss": best_val_loss,
        "final_val_rmse_rad": history[-1]["val_rmse_rad"],
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "model_path": str(model_path),
        "best_model_path": str(best_model_path),
        **best_checkpoint_metrics,
    }
    for index, rmse in enumerate(channel_rmse, start=1):
        row[f"channel_{index}_rmse_rad"] = float(rmse)
    if best_channel_rmse is not None:
        for index, rmse in enumerate(best_channel_rmse, start=1):
            row[f"best_checkpoint_channel_{index}_rmse_rad"] = float(rmse)
    return row, history


def build_limited_dataloaders(args):
    """构建结构消融使用的数据读取器，可限制样本数以便快速筛选模型。"""
    base_dataset = FarFieldPhaseDataset(
        image_path=args.image_path,
        label_path=args.label_path,
        expected_size=(args.image_size, args.image_size),
    )
    dataset = base_dataset
    if not args.full_dataset and args.max_samples is not None:
        max_samples = min(args.max_samples, len(base_dataset))
        dataset = Subset(base_dataset, list(range(max_samples)))

    train_set, val_set, test_set = split_dataset(
        dataset=dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    return {
        "dataset": dataset,
        "train": DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        ),
        "val": DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        ),
        "test": DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        ),
        "splits": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "label_dim": base_dataset.labels.shape[1],
        "total_samples": len(dataset),
    }


def plot_architecture_results(summary_rows, histories, figure_path):
    """绘制结构消融结果图。"""
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = [row["model_name"] for row in summary_rows]

    plt.figure(figsize=(14, 9))

    plt.subplot(2, 2, 1)
    for model_name, history in histories.items():
        plt.plot(
            [row["epoch"] for row in history],
            [row["val_rmse_rad"] for row in history],
            marker="o",
            label=model_name,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE(rad)")
    plt.title("Validation convergence")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.bar(model_names, [row["test_rmse_rad"] for row in summary_rows])
    plt.xticks(rotation=20)
    plt.ylabel("Test RMSE(rad)")
    plt.title("Test phase RMSE")

    plt.subplot(2, 2, 3)
    plt.bar(model_names, [row["parameter_count"] / 1e6 for row in summary_rows])
    plt.xticks(rotation=20)
    plt.ylabel("Parameters(M)")
    plt.title("Model size")

    plt.subplot(2, 2, 4)
    plt.bar(model_names, [row["train_seconds"] for row in summary_rows])
    plt.xticks(rotation=20)
    plt.ylabel("Seconds")
    plt.title("Training time")

    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print("Figure saved to:", figure_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run seven-beam CNN architecture ablation."
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
        "--models",
        nargs="+",
        default=["simple_cnn", "wide_cnn", "residual_cnn"],
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--phase-loss",
        choices=["mse", "cyclic", "cyclic_unit"],
        default="mse",
    )
    parser.add_argument("--unit-loss-weight", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--full-dataset", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--experiment-tag", default="cycle21")
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=REPO_ROOT / "result" / "metrics" / "cycle21_seven_beam_architecture",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "metrics"
        / "cycle21_seven_beam_architecture_ablation_2026-06-09.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT
        / "result"
        / "figures"
        / "cycle21_seven_beam_architecture_ablation_2026-06-09.png",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "models",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loaders = build_limited_dataloaders(args)
    output_dim = loaders["label_dim"]
    if output_dim != 12:
        raise ValueError(f"Seven-beam labels should have 12 columns, got {output_dim}")

    device = resolve_device(args.device)
    print("Using device:", device)
    print("Total samples used:", loaders["total_samples"])
    print("Splits:", loaders["splits"])
    print("Output dim:", output_dim)
    print("phase_loss:", args.phase_loss)

    summary_rows = []
    histories = {}
    for model_name in args.models:
        row, history = train_one_model(
            model_name=model_name,
            args=args,
            loaders=loaders,
            output_dim=output_dim,
            device=device,
        )
        summary_rows.append(row)
        histories[model_name] = history

    best_row = min(
        summary_rows,
        key=lambda row: row.get("best_checkpoint_test_rmse_rad", row["test_rmse_rad"]),
    )
    for row in summary_rows:
        row["selection_rmse_rad"] = row.get(
            "best_checkpoint_test_rmse_rad",
            row["test_rmse_rad"],
        )
        row["is_best_by_selection_rmse"] = row["model_name"] == best_row["model_name"]
        row["rmse_gap_to_best_rad"] = row["selection_rmse_rad"] - best_row.get(
            "best_checkpoint_test_rmse_rad",
            best_row["test_rmse_rad"],
        )

    save_rows(summary_rows, args.summary_csv)
    plot_architecture_results(summary_rows, histories, args.figure_path)

    print("Summary saved to:", args.summary_csv)
    print(
        "Best model:",
        best_row["model_name"],
        f"RMSE={best_row.get('best_checkpoint_test_rmse_rad', best_row['test_rmse_rad']):.6f} rad",
    )


if __name__ == "__main__":
    main()
