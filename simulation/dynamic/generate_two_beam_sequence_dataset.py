import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# 直接运行本脚本时，将仓库根目录加入 sys.path，保证能导入 simulation.common.two_beam_core。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.common.two_beam_core import (
    generate_two_beam_sequence_dataset,
    save_sequence_dataset,
    sequence_dataset_config,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate two-beam far-field sequence data for CNN+LSTM."
    )

    # 序列任务参数：模型读取 input_length 帧，预测 predict_steps 之后的相位。
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--input-length", type=int, default=8)
    parser.add_argument("--predict-steps", type=int, default=1)

    # 相位动态模式。mixed 会在每条序列中随机选择一种模式。
    parser.add_argument(
        "--phase-mode",
        choices=["random_walk", "sine", "step", "drift", "mixed"],
        default="random_walk",
    )

    # 光学仿真参数，与静态数据生成脚本保持一致。
    parser.add_argument("--noise-sigma", type=float, default=0.02)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=10e-3)
    parser.add_argument("--waist", type=float, default=0.5e-3)
    parser.add_argument("--beam-distance", type=float, default=1.5e-3)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260608)

    # 动态相位模型参数。
    parser.add_argument("--step-sigma", type=float, default=0.08)
    parser.add_argument("--sine-amplitude", type=float, default=1.0)
    parser.add_argument("--sine-frequency", type=float, default=0.04)
    parser.add_argument("--drift-velocity", type=float, default=0.03)
    parser.add_argument("--step-probability", type=float, default=0.08)
    parser.add_argument("--step-scale", type=float, default=0.6)

    # 输出参数。dataset/ 默认被 .gitignore 忽略，避免大数据集误提交。
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/two_beam_sequence"),
    )
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.prefix is None:
        args.prefix = (
            f"{args.phase_mode}_T{args.input_length}_P{args.predict_steps}_"
            f"noise{args.noise_sigma}"
        )

    (
        images,
        labels,
        input_phases,
        target_phases,
        all_phases,
        modes,
    ) = generate_two_beam_sequence_dataset(
        num_sequences=args.num_sequences,
        input_length=args.input_length,
        predict_steps=args.predict_steps,
        phase_mode=args.phase_mode,
        noise_sigma=args.noise_sigma,
        num_points=args.num_points,
        window_size=args.window_size,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.crop_size,
        seed=args.seed,
        step_sigma=args.step_sigma,
        sine_amplitude=args.sine_amplitude,
        sine_frequency=args.sine_frequency,
        drift_velocity=args.drift_velocity,
        step_probability=args.step_probability,
        step_scale=args.step_scale,
    )

    image_path = args.output_dir / f"images_{args.prefix}.npy"
    label_path = args.output_dir / f"labels_{args.prefix}.npy"
    input_phase_path = args.output_dir / f"input_phases_{args.prefix}.npy"
    target_phase_path = args.output_dir / f"target_phases_{args.prefix}.npy"
    all_phase_path = args.output_dir / f"all_phases_{args.prefix}.npy"
    mode_path = args.output_dir / f"modes_{args.prefix}.npy"

    config = sequence_dataset_config(
        num_sequences=args.num_sequences,
        input_length=args.input_length,
        predict_steps=args.predict_steps,
        phase_mode=args.phase_mode,
        noise_sigma=args.noise_sigma,
        num_points=args.num_points,
        window_size=args.window_size,
        waist=args.waist,
        beam_distance=args.beam_distance,
        crop_size=args.crop_size,
        seed=args.seed,
        image_path=image_path,
        label_path=label_path,
        input_phase_path=input_phase_path,
        target_phase_path=target_phase_path,
        all_phase_path=all_phase_path,
        mode_path=mode_path,
        step_sigma=args.step_sigma,
        sine_amplitude=args.sine_amplitude,
        sine_frequency=args.sine_frequency,
        drift_velocity=args.drift_velocity,
        step_probability=args.step_probability,
        step_scale=args.step_scale,
    )

    paths = save_sequence_dataset(
        images=images,
        labels=labels,
        input_phases=input_phases,
        target_phases=target_phases,
        all_phases=all_phases,
        modes=modes,
        output_dir=args.output_dir,
        prefix=args.prefix,
        config=config,
    )

    print("Sequence dataset generated successfully!")
    print("Images:", paths["image_path"], images.shape, images.dtype)
    print("Labels:", paths["label_path"], labels.shape, labels.dtype)
    print("Input phases:", paths["input_phase_path"], input_phases.shape, input_phases.dtype)
    print("Target phases:", paths["target_phase_path"], target_phases.shape, target_phases.dtype)
    print("All phases:", paths["all_phase_path"], all_phases.shape, all_phases.dtype)
    print("Modes:", paths["mode_path"], modes.shape, modes.dtype)
    print("Config:", paths["config_path"])

    if args.show:
        # 左侧显示输入序列最后一帧远场图，右侧显示该序列完整相位轨迹。
        plt.figure(figsize=(9, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(images[0, -1], cmap="jet")
        plt.title("Last input frame")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.plot(all_phases[0], marker="o")
        plt.axvline(args.input_length - 1, color="r", linestyle="--", label="last input")
        plt.xlabel("Time index")
        plt.ylabel("Phase(rad)")
        plt.title(f"Phase mode: {modes[0]}")
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
