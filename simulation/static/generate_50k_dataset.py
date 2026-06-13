"""
50k大规模数据集生成脚本

相比10k数据集：
1. 5倍样本量，验证模型容量上限
2. 支持分批生成（避免内存溢出）
3. 自动验证数据质量
"""

import argparse
import numpy as np
import json
from pathlib import Path
import sys
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simulation.common.multi_beam_core import (
    create_grid, seven_beam_near_field, far_field_intensity, crop_center
)


def generate_batch(batch_size, num_points, window_size, waist, beam_distance,
                  crop_size, phase_range, seed_offset=0):
    """生成一批数据"""
    np.random.seed(20260613 + seed_offset)

    # 创建网格
    x_grid, y_grid = create_grid(num_points, window_size)

    # 生成两个平面的数据
    images_focal = []
    images_befocal = []
    labels = []
    phases_array = []

    for i in tqdm(range(batch_size), desc=f"Batch {seed_offset//1000}", leave=False):
        # 随机生成6个相位（中心beam_0固定为0）
        phases = np.random.uniform(phase_range[0], phase_range[1], size=6)

        # 生成焦平面远场图像
        near_field_focal = seven_beam_near_field(
            x_grid, y_grid, phases=phases,
            waist=waist, beam_distance=beam_distance,
            amplitudes=np.ones(7)
        )
        far_field_focal = far_field_intensity(near_field_focal)
        far_field_focal_crop = crop_center(far_field_focal, crop_size)
        far_field_focal_norm = far_field_focal_crop / far_field_focal_crop.max()

        # 生成焦前平面图像（z = -0.07m）
        # 简化处理：添加轻微模糊模拟焦前效果
        from scipy.ndimage import gaussian_filter
        far_field_befocal_norm = gaussian_filter(far_field_focal_norm, sigma=0.5)

        images_focal.append(far_field_focal_norm)
        images_befocal.append(far_field_befocal_norm)

        # 生成sin/cos标签
        sin_cos_label = np.zeros(12)
        for j in range(6):
            sin_cos_label[2*j] = np.sin(phases[j])
            sin_cos_label[2*j+1] = np.cos(phases[j])

        labels.append(sin_cos_label)
        phases_array.append(phases)

    return np.array(images_focal), np.array(images_befocal), np.array(labels), np.array(phases_array)


def main():
    parser = argparse.ArgumentParser(description='生成50k大规模数据集')

    # 数据参数
    parser.add_argument('--num-samples', type=int, default=50000,
                       help='总样本数')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='每批生成样本数（避免内存溢出）')

    # 物理参数
    parser.add_argument('--num-points', type=int, default=256,
                       help='近场网格点数')
    parser.add_argument('--window-size', type=float, default=0.01,
                       help='近场窗口尺寸 (m)')
    parser.add_argument('--waist', type=float, default=0.0005,
                       help='光束腰斑半径 (m)')
    parser.add_argument('--beam-distance', type=float, default=0.0015,
                       help='光束间距 (m)')
    parser.add_argument('--crop-size', type=int, default=160,
                       help='远场裁剪尺寸')
    parser.add_argument('--phase-min', type=float, default=-np.pi,
                       help='相位最小值')
    parser.add_argument('--phase-max', type=float, default=np.pi,
                       help='相位最大值')

    # 输出参数
    parser.add_argument('--output-dir', type=str,
                       default='dataset/seven_beam/multiplane_50k',
                       help='输出目录')
    parser.add_argument('--prefix', type=str, default='multiplane_50k',
                       help='文件前缀')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"生成50k大规模数据集")
    print(f"{'='*70}\n")
    print(f"总样本数: {args.num_samples}")
    print(f"批次大小: {args.batch_size}")
    print(f"预计批次数: {args.num_samples // args.batch_size}")
    print()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 分批生成
    n_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    all_images_focal = []
    all_images_befocal = []
    all_labels = []
    all_phases = []

    print("开始生成数据...\n")

    for batch_idx in range(n_batches):
        # 计算当前批次大小
        current_batch_size = min(args.batch_size,
                                 args.num_samples - batch_idx * args.batch_size)

        # 生成当前批次
        images_focal, images_befocal, labels, phases = generate_batch(
            current_batch_size,
            args.num_points,
            args.window_size,
            args.waist,
            args.beam_distance,
            args.crop_size,
            (args.phase_min, args.phase_max),
            seed_offset=batch_idx * 1000
        )

        all_images_focal.append(images_focal)
        all_images_befocal.append(images_befocal)
        all_labels.append(labels)
        all_phases.append(phases)

        print(f"批次 {batch_idx+1}/{n_batches} 完成")

    # 合并所有批次
    print("\n合并数据...")
    all_images_focal = np.concatenate(all_images_focal, axis=0)
    all_images_befocal = np.concatenate(all_images_befocal, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_phases = np.concatenate(all_phases, axis=0)

    # 堆叠焦平面和焦前平面
    all_images = np.stack([all_images_focal, all_images_befocal], axis=1)

    print(f"  最终图像形状: {all_images.shape}")
    print(f"  最终标签形状: {all_labels.shape}")
    print(f"  最终相位形状: {all_phases.shape}")

    # 数据验证
    print("\n验证数据质量...")
    print(f"  图像范围: [{all_images.min():.4f}, {all_images.max():.4f}]")
    print(f"  标签范围: [{all_labels.min():.4f}, {all_labels.max():.4f}]")
    print(f"  相位范围: [{all_phases.min():.4f}, {all_phases.max():.4f}]")

    # sin^2 + cos^2 = 1 验证
    sin_cos_check = []
    for i in range(6):
        sin_val = all_labels[:, 2*i]
        cos_val = all_labels[:, 2*i+1]
        unit_circle_error = np.abs(sin_val**2 + cos_val**2 - 1.0)
        sin_cos_check.append(unit_circle_error.max())

    print(f"  sin^2+cos^2=1 最大误差: {max(sin_cos_check):.2e}")

    # 保存数据
    print("\n保存数据...")
    image_path = output_dir / f"images_{args.prefix}.npy"
    label_path = output_dir / f"labels_{args.prefix}.npy"
    phase_path = output_dir / f"phases_{args.prefix}.npy"

    np.save(image_path, all_images)
    np.save(label_path, all_labels)
    np.save(phase_path, all_phases)

    print(f"  图像保存至: {image_path}")
    print(f"  标签保存至: {label_path}")
    print(f"  相位保存至: {phase_path}")

    # 保存配置
    config = {
        'num_samples': int(args.num_samples),
        'batch_size': int(args.batch_size),
        'num_points': int(args.num_points),
        'window_size': float(args.window_size),
        'waist': float(args.waist),
        'beam_distance': float(args.beam_distance),
        'crop_size': int(args.crop_size),
        'phase_range': [float(args.phase_min), float(args.phase_max)],
        'image_shape': list(all_images.shape),
        'label_shape': list(all_labels.shape),
        'phase_shape': list(all_phases.shape)
    }

    config_path = output_dir / f"config_{args.prefix}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  配置保存至: {config_path}")

    # 数据统计
    print("\n数据统计:")
    print(f"  文件总大小: ~{(all_images.nbytes + all_labels.nbytes + all_phases.nbytes) / 1e9:.2f} GB")
    print(f"  单样本大小: ~{all_images[0].nbytes / 1e3:.2f} KB")

    print(f"\n{'='*70}")
    print(f"50k数据集生成完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
