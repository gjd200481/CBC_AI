"""使用修正后的多平面数据快速训练 Cycle 42 模型进行验证

目标：
- 验证修正后的多平面数据是否能提升性能
- 对比修正前后的 RMSE、Strehl、主瓣能量
- 重新评估 attribution 和双分支融合的有效性
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 复用现有训练脚本，只需更改数据路径
from train.train_multiplane import main as train_main


def quick_verify_corrected_data(
    data_dir="dataset/seven_beam/multiplane_corrected_f1.0_d0.05",
    prefix="multiplane_corrected_10k",
    epochs=15,
    output_tag="corrected_quick",
):
    """快速验证训练"""

    print("="*60)
    print("CORRECTED DATA QUICK VERIFICATION")
    print("="*60)
    print(f"Data dir: {data_dir}")
    print(f"Prefix: {prefix}")
    print(f"Epochs: {epochs}")
    print(f"Output tag: {output_tag}")
    print("="*60)
    print()

    # 构造参数
    args = [
        "--data-dir", data_dir,
        "--prefix", prefix,
        "--epochs", str(epochs),
        "--batch-size", "32",
        "--learning-rate", "0.001",
        "--model-path", f"models/cycle_corrected_{output_tag}.pth",
        "--comp-model-path", f"models/cycle_corrected_{output_tag}_comp.pth",
        "--strehl-model-path", f"models/cycle_corrected_{output_tag}_strehl.pth",
        "--main-lobe-model-path", f"models/cycle_corrected_{output_tag}_main_lobe.pth",
        "--log-file", f"result/logs/cycle_corrected_{output_tag}.md",
        "--lambda-comp", "0.5",
        "--lambda-phy", "0.05",
        "--lambda-unit", "0.01",
        "--comp-warmup-epochs", "5",
        "--augment-mode", "noise",
        "--seed", "20260615",
    ]

    # 调用训练
    import sys
    old_argv = sys.argv
    sys.argv = ["train_multiplane.py"] + args

    try:
        train_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="修正数据快速验证训练")
    parser.add_argument("--data-dir", type=str,
                       default="dataset/seven_beam/multiplane_corrected_f1.0_d0.05")
    parser.add_argument("--prefix", type=str, default="multiplane_corrected_10k")
    parser.add_argument("--epochs", type=int, default=15,
                       help="快速验证用15 epoch，正式训练用30")
    parser.add_argument("--output-tag", type=str, default="corrected_quick")

    args = parser.parse_args()

    quick_verify_corrected_data(
        data_dir=args.data_dir,
        prefix=args.prefix,
        epochs=args.epochs,
        output_tag=args.output_tag,
    )
