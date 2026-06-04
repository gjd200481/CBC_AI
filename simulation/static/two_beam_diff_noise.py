import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# 该脚本位于 simulation/static/ 下。直接运行时需要手动把项目根目录加入 sys.path，
# 否则无法导入 simulation.common.two_beam_core。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.two_beam_core import (
    dataset_config,
    generate_two_beam_dataset,
    save_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the default noisy two-beam dataset used by training."
    )
    # 默认参数保持当前训练脚本使用的 noise_0.05 数据设置。
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--noise-sigma", type=float, default=0.05)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/two_beam"))
    # prefix 默认是 noise_0.05，因此输出文件会与 train/evaluate_two_beam.py 的路径一致。
    parser.add_argument("--prefix", default="noise_0.05")
    parser.add_argument("--save-phases", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # 例如 prefix=noise_0.05 时，输出 images_noise_0.05.npy / labels_noise_0.05.npy。
    image_name = f"images_{args.prefix}.npy"
    label_name = f"labels_{args.prefix}.npy"
    phase_name = f"phases_{args.prefix}.npy" if args.save_phases else None
    config_name = f"config_{args.prefix}.json"

    # 生成双光束远场图像和相位标签。
    images, labels, phases = generate_two_beam_dataset(
        num_samples=args.num_samples,
        noise_sigma=args.noise_sigma,
        num_points=args.num_points,
        window_size=args.window_size,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.crop_size,
        seed=args.seed,
    )

    image_path = args.output_dir / image_name
    label_path = args.output_dir / label_name
    phase_path = args.output_dir / phase_name if phase_name is not None else None
    # 保存完整生成参数，避免以后忘记某个数据集对应的噪声强度、随机种子或裁剪尺寸。
    config = dataset_config(
        num_samples=args.num_samples,
        noise_sigma=args.noise_sigma,
        num_points=args.num_points,
        window_size=args.window_size,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.crop_size,
        seed=args.seed,
        image_path=image_path,
        label_path=label_path,
        phase_path=phase_path,
    )

    # 统一保存图像、标签、可选 phase 和配置文件。
    image_path, label_path, config_path = save_dataset(
        images=images,
        labels=labels,
        output_dir=args.output_dir,
        image_name=image_name,
        label_name=label_name,
        config_name=config_name,
        config=config,
        phases=phases if args.save_phases else None,
        phase_name=phase_name,
    )

    print("Dataset generated successfully!")
    print("Images:", image_path, images.shape, images.dtype)
    print("Labels:", label_path, labels.shape, labels.dtype)
    print("Config:", config_path)
    if phase_path is not None:
        print("Phases:", phase_path, phases.shape, phases.dtype)

    if args.show:
        # 人工检查样本时开启；批量生成训练数据时默认关闭。
        plt.imshow(images[0], cmap="jet")
        plt.title("Sample far field")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()
