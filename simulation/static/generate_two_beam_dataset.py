import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# 当直接运行 `python simulation/static/generate_two_beam_dataset.py` 时，
# Python 默认只把 simulation/static/ 加入模块搜索路径，找不到项目根目录。
# 这里把仓库根目录加入 sys.path，保证可以导入 simulation.common.two_beam_core。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.two_beam_core import (
    dataset_config,
    generate_two_beam_dataset,
    save_dataset,
)


def format_noise_tag(noise_sigma):
    """把噪声强度转换为适合文件名的字符串。"""
    if noise_sigma == 0:
        return "0"
    return str(noise_sigma).rstrip("0").rstrip(".")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a reproducible two-beam far-field phase dataset."
    )
    # 样本数量和物理仿真参数。
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--crop-size", type=int, default=160)
    # 当前脚本专门生成双光束数据。保留 num-beams 参数用于配置记录和未来多光束扩展。
    parser.add_argument("--num-beams", type=int, default=2, choices=[2])
    parser.add_argument("--phase-min", type=float, default=-3.141592653589793)
    parser.add_argument("--phase-max", type=float, default=3.141592653589793)
    # seed 用于保证每次生成的数据完全可复现。
    parser.add_argument("--seed", type=int, default=20260604)
    # 输出目录默认放在 dataset/two_beam 下；该目录被 .gitignore 忽略，避免大数据误提交。
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/two_beam"))
    # prefix 控制输出文件名。若不指定，则根据噪声强度自动生成，例如 noise_0.02。
    parser.add_argument("--prefix", default=None)
    # phases 是原始相位数组，训练不必需，但对调试很方便。
    parser.add_argument("--save-phases", action="store_true")
    # show 只用于人工检查样本图，不建议批量生成数据时开启。
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    prefix = args.prefix
    if prefix is None:
        noise_tag = format_noise_tag(args.noise_sigma)
        prefix = f"noise_{noise_tag}"

    # 统一命名规则，便于后续训练脚本按文件名查找图像、标签和配置。
    image_name = f"images_{prefix}.npy"
    label_name = f"labels_{prefix}.npy"
    phase_name = f"phases_{prefix}.npy" if args.save_phases else None
    config_name = f"config_{prefix}.json"

    # 调用核心函数生成图像、sin/cos 标签和可选原始相位。
    images, labels, phases = generate_two_beam_dataset(
        num_samples=args.num_samples,
        noise_sigma=args.noise_sigma,
        num_points=args.num_points,
        window_size=args.window_size,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.crop_size,
        phase_min=args.phase_min,
        phase_max=args.phase_max,
        seed=args.seed,
    )

    image_path = args.output_dir / image_name
    label_path = args.output_dir / label_name
    phase_path = args.output_dir / phase_name if phase_name is not None else None
    # 将本次生成数据所用的参数也保存下来，保证实验可追溯。
    config = dataset_config(
        num_samples=args.num_samples,
        noise_sigma=args.noise_sigma,
        num_points=args.num_points,
        window_size=args.window_size,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.crop_size,
        phase_min=args.phase_min,
        phase_max=args.phase_max,
        seed=args.seed,
        image_path=image_path,
        label_path=label_path,
        phase_path=phase_path,
    )

    # 统一保存 .npy 数据和 .json 配置。
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
        # 显示第一张样本图，用于快速检查远场条纹和噪声水平是否合理。
        plt.imshow(images[0], cmap="jet")
        plt.title("Sample far field")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()
