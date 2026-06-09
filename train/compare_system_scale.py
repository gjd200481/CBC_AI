import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_metric_value_csv(path):
    """读取 metric,value 格式的汇总 CSV。"""
    metrics = {}
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["metric"]] = parse_number(row["value"])
    return metrics


def read_named_metric_csv(path, value_columns):
    """读取 metric + 多个结果列格式的对比 CSV。"""
    rows = {}
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_name = row["metric"]
            rows[metric_name] = {
                column: parse_number(row[column])
                for column in value_columns
                if column in row and row[column] != ""
            }
    return rows


def parse_number(value):
    """尽量把 CSV 中的数字字符串转成 float，失败时保留原字符串。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def percent_change(new_value, old_value):
    """计算相对变化百分比，正值表示 new_value 高于 old_value。"""
    if abs(old_value) < 1e-12:
        return 0.0
    return 100 * (new_value - old_value) / old_value


def write_csv(rows, output_path):
    """保存字典列表为 CSV。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_scale_rows(two_beam, seven_beam):
    """构造双光束和 7 光束系统规模对比表。"""
    rows = []
    for item in [two_beam, seven_beam]:
        baseline_rmse = item["baseline"]["rmse_rad"]
        physics_rmse = item["physics"]["rmse_rad"]
        baseline_mae = item["baseline"]["mae_rad"]
        physics_mae = item["physics"]["mae_rad"]
        rows.append(
            {
                "system": item["system"],
                "num_beams": item["num_beams"],
                "predicted_phase_count": item["predicted_phase_count"],
                "network_output_dim": item["network_output_dim"],
                "main_dataset_samples": item["main_dataset_samples"],
                "train_samples": item["train_samples"],
                "test_samples": item["test_samples"],
                "baseline_epochs": item["baseline_epochs"],
                "physics_epochs": item["physics_epochs"],
                "physics_lambda": item["physics_lambda"],
                "baseline_rmse_rad": baseline_rmse,
                "physics_rmse_rad": physics_rmse,
                "physics_rmse_change_percent": percent_change(physics_rmse, baseline_rmse),
                "baseline_mae_rad": baseline_mae,
                "physics_mae_rad": physics_mae,
                "physics_mae_change_percent": percent_change(physics_mae, baseline_mae),
                "baseline_farfield_mse": item["baseline_farfield_mse"],
                "physics_farfield_mse": item["physics_farfield_mse"],
                "farfield_mse_change_percent": percent_change(
                    item["physics_farfield_mse"],
                    item["baseline_farfield_mse"],
                ),
                "channel_rmse_std_rad": item["channel_rmse_std_rad"],
                "note": item["note"],
            }
        )
    return rows


def make_ratio_rows(scale_rows):
    """构造 7 光束相对双光束的规模变化表。"""
    two = next(row for row in scale_rows if row["system"] == "two_beam")
    seven = next(row for row in scale_rows if row["system"] == "seven_beam")

    ratio_specs = [
        ("num_beams", "光束数量"),
        ("predicted_phase_count", "待预测相位数量"),
        ("network_output_dim", "网络输出维度"),
        ("main_dataset_samples", "主数据集样本数"),
        ("train_samples", "训练样本数"),
        ("test_samples", "测试样本数"),
        ("baseline_rmse_rad", "普通 CNN 相位 RMSE"),
        ("physics_rmse_rad", "物理约束 CNN 相位 RMSE"),
        ("baseline_mae_rad", "普通 CNN 相位 MAE"),
        ("physics_mae_rad", "物理约束 CNN 相位 MAE"),
        ("physics_farfield_mse", "物理约束 CNN 远场 MSE"),
    ]

    rows = []
    for key, label in ratio_specs:
        two_value = two[key]
        seven_value = seven[key]
        rows.append(
            {
                "metric": key,
                "description": label,
                "two_beam": two_value,
                "seven_beam": seven_value,
                "seven_vs_two_factor": seven_value / two_value if abs(two_value) > 1e-12 else 0.0,
                "absolute_delta": seven_value - two_value,
            }
        )
    return rows


def plot_scale_comparison(scale_rows, ratio_rows, figure_path):
    """绘制系统规模、相位误差和物理约束收益对比图。"""
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    systems = [row["system"] for row in scale_rows]
    x = np.arange(len(systems))
    width = 0.35

    plt.figure(figsize=(14, 9))

    plt.subplot(2, 2, 1)
    plt.bar(x - width / 2, [row["num_beams"] for row in scale_rows], width, label="beams")
    plt.bar(
        x + width / 2,
        [row["predicted_phase_count"] for row in scale_rows],
        width,
        label="predicted phases",
    )
    plt.xticks(x, systems)
    plt.ylabel("Count")
    plt.title("System scale")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.bar(x - width / 2, [row["baseline_rmse_rad"] for row in scale_rows], width, label="baseline")
    plt.bar(x + width / 2, [row["physics_rmse_rad"] for row in scale_rows], width, label="physics")
    plt.yscale("log")
    plt.xticks(x, systems)
    plt.ylabel("Phase RMSE(rad), log scale")
    plt.title("Phase inversion difficulty")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.bar(
        systems,
        [row["physics_rmse_change_percent"] for row in scale_rows],
        color=["#2563EB", "#DC2626"],
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("RMSE change vs baseline (%)")
    plt.title("Physics constraint effect")

    plt.subplot(2, 2, 4)
    selected = [
        row
        for row in ratio_rows
        if row["metric"]
        in ["predicted_phase_count", "network_output_dim", "baseline_rmse_rad", "physics_rmse_rad"]
    ]
    plt.bar(
        [row["metric"] for row in selected],
        [row["seven_vs_two_factor"] for row in selected],
        color="#059669",
    )
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Seven-beam / two-beam factor")
    plt.title("Scale-up factors")

    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print("Figure saved to:", figure_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two-beam and seven-beam CBC phase inversion results."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "result" / "metrics" / "cycle20_system_scale_comparison_2026-06-09.csv",
    )
    parser.add_argument(
        "--ratio-csv",
        type=Path,
        default=REPO_ROOT / "result" / "metrics" / "cycle20_system_scale_ratio_2026-06-09.csv",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=REPO_ROOT / "result" / "figures" / "cycle20_system_scale_comparison_2026-06-09.png",
    )
    args = parser.parse_args()

    two_baseline = read_metric_value_csv(
        REPO_ROOT / "result" / "metrics" / "baseline_cnn_main_clean_summary_2026-06-07.csv"
    )
    two_physics = read_metric_value_csv(
        REPO_ROOT / "result" / "metrics" / "sweep_lambda_0.01_main_clean_summary_2026-06-07.csv"
    )
    two_sweep = read_metric_value_csv(
        REPO_ROOT / "result" / "metrics" / "sweep_lambda_0_main_clean_summary_2026-06-07.csv"
    )

    seven_baseline = read_metric_value_csv(
        REPO_ROOT / "result" / "metrics" / "baseline_cnn_main_clean_seven_beam_summary_2026-06-08.csv"
    )
    seven_physics = read_metric_value_csv(
        REPO_ROOT
        / "result"
        / "metrics"
        / "physics_cnn_lambda_0.1_main_clean_seven_beam_summary_2026-06-08.csv"
    )
    seven_compare = read_named_metric_csv(
        REPO_ROOT / "result" / "metrics" / "cycle13_seven_beam_physics_vs_baseline_2026-06-08.csv",
        value_columns=["baseline_cnn", "physics_cnn_lambda_0.1"],
    )

    seven_channel_rmse = [
        seven_baseline[f"channel_{index}_rmse_rad"]
        for index in range(1, 7)
    ]

    two_beam = {
        "system": "two_beam",
        "num_beams": 2,
        "predicted_phase_count": 1,
        "network_output_dim": 2,
        "main_dataset_samples": 2000,
        "train_samples": 1400,
        "test_samples": 300,
        "baseline_epochs": int(two_baseline.get("epochs", 20)),
        "physics_epochs": int(two_physics.get("epochs", 8)),
        "physics_lambda": two_physics["lambda_phy"],
        "baseline": two_baseline,
        "physics": two_physics,
        "baseline_farfield_mse": two_sweep["farfield_loss"],
        "physics_farfield_mse": two_physics["farfield_loss"],
        "channel_rmse_std_rad": 0.0,
        "note": "低维方法验证；物理约束模型使用 Cycle 08 中 lambda=0.01 候选。",
    }
    seven_beam = {
        "system": "seven_beam",
        "num_beams": 7,
        "predicted_phase_count": 6,
        "network_output_dim": 12,
        "main_dataset_samples": 1024,
        "train_samples": int(seven_baseline["train_samples"]),
        "test_samples": int(seven_baseline["test_samples"]),
        "baseline_epochs": int(seven_baseline["epochs"]),
        "physics_epochs": int(seven_physics["epochs"]),
        "physics_lambda": seven_physics["lambda_phy"],
        "baseline": seven_baseline,
        "physics": seven_physics,
        "baseline_farfield_mse": seven_compare["farfield_loss"]["baseline_cnn"],
        "physics_farfield_mse": seven_compare["farfield_loss"]["physics_cnn_lambda_0.1"],
        "channel_rmse_std_rad": float(np.std(seven_channel_rmse)),
        "note": "论文主系统；中心光束为参考，外圈 6 路相位同时反演。",
    }

    scale_rows = make_scale_rows(two_beam, seven_beam)
    ratio_rows = make_ratio_rows(scale_rows)

    write_csv(scale_rows, args.output_csv)
    write_csv(ratio_rows, args.ratio_csv)
    plot_scale_comparison(scale_rows, ratio_rows, args.figure_path)

    print("Scale comparison saved to:", args.output_csv)
    print("Scale ratio saved to:", args.ratio_csv)
    for row in scale_rows:
        print(
            f"{row['system']} | phases={row['predicted_phase_count']} | "
            f"baseline_rmse={row['baseline_rmse_rad']:.6f} | "
            f"physics_rmse={row['physics_rmse_rad']:.6f} | "
            f"physics_change={row['physics_rmse_change_percent']:.2f}%"
        )
    for row in ratio_rows:
        if row["metric"] in ["predicted_phase_count", "network_output_dim", "baseline_rmse_rad", "physics_rmse_rad"]:
            print(
                f"factor {row['metric']}: seven/two = {row['seven_vs_two_factor']:.2f}"
            )


if __name__ == "__main__":
    main()
