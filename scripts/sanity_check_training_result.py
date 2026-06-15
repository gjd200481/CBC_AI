"""验证训练结果的合理性 - 诊断脚本"""
import numpy as np
from pathlib import Path

print("="*60)
print("TRAINING RESULT SANITY CHECK")
print("="*60)

# 1. 检查数据集本身
print("\n1. 检查数据集:")
data_path = Path("dataset/seven_beam/multiplane_corrected_f1.0_d0.05")
images = np.load(data_path / "images_multiplane_corrected_10k.npy")
labels = np.load(data_path / "labels_multiplane_corrected_10k.npy")
phases = np.load(data_path / "phases_multiplane_corrected_10k.npy")

print(f"   Images shape: {images.shape}")
print(f"   Labels shape: {labels.shape}")
print(f"   Phases shape: {phases.shape}")

# 2. 检查数据范围
print(f"\n2. 数据范围:")
print(f"   Images: [{images.min():.3f}, {images.max():.3f}]")
print(f"   Labels: [{labels.min():.3f}, {labels.max():.3f}]")
print(f"   Phases: [{phases.min():.3f}, {phases.max():.3f}] rad")

# 3. 检查两通道差异
plane0 = images[:, 0]
plane1 = images[:, 1]
max_diff = np.max(np.abs(plane0 - plane1))
mean_diff = np.mean(np.abs(plane0 - plane1))
identical_count = sum(np.allclose(plane0[i], plane1[i], atol=1e-10) for i in range(min(10, len(images))))

print(f"\n3. 两通道差异验证:")
print(f"   Max diff: {max_diff:.6f}")
print(f"   Mean diff: {mean_diff:.6f}")
print(f"   Identical in first 10: {identical_count}/10")

if identical_count >= 5:
    print(f"   [WARNING] 两通道可能退化！")
else:
    print(f"   [OK] 两通道有显著差异")

# 4. 检查标签编码正确性
print(f"\n4. 标签编码验证:")
for i in range(min(5, len(labels))):
    sin_cos = labels[i]
    # 每对 sin/cos 应满足 sin²+cos²=1
    errors = []
    for j in range(0, 12, 2):
        sin_val = sin_cos[j]
        cos_val = sin_cos[j+1]
        norm = sin_val**2 + cos_val**2
        errors.append(abs(norm - 1.0))
    max_error = max(errors)
    if max_error > 0.01:
        print(f"   Sample {i}: [WARNING] sin²+cos² error = {max_error:.6f}")
    elif i == 0:
        print(f"   Sample 0: sin²+cos² error = {max_error:.6e} [OK]")

# 5. 检查相位与标签一致性
print(f"\n5. 相位与标签一致性:")
for i in range(min(3, len(phases))):
    phase = phases[i]
    label = labels[i]
    # 解码 sin/cos 回相位
    decoded = []
    for j in range(0, 12, 2):
        decoded_phase = np.arctan2(label[j], label[j+1])
        decoded.append(decoded_phase)
    decoded = np.array(decoded)

    # 比较
    error = np.max(np.abs(phase - decoded))
    if error > 1e-5:
        print(f"   Sample {i}: [WARNING] Phase decode error = {error:.6f}")
    elif i == 0:
        print(f"   Sample 0: Phase decode error = {error:.6e} [OK]")

# 6. 可能导致性能"虚高"的问题
print(f"\n6. 常见陷阱检查:")

# 6.1 数据是否有重复？
unique_count = len(np.unique(images.reshape(len(images), -1), axis=0))
print(f"   Unique samples: {unique_count}/{len(images)}")
if unique_count < len(images):
    print(f"   [WARNING] 存在重复样本！")
else:
    print(f"   [OK] 无重复样本")

# 6.2 标签是否退化为常数？
phase_std = np.std(phases, axis=0)
print(f"   Phase std across samples: min={phase_std.min():.3f}, max={phase_std.max():.3f}")
if phase_std.min() < 0.5:
    print(f"   [WARNING] 某些通道相位变化太小！")
else:
    print(f"   [OK] 相位分布合理")

# 6.3 图像是否退化为常数？
image_std_per_sample = np.std(images, axis=(2, 3))  # [N, 2]
mean_std = np.mean(image_std_per_sample)
print(f"   Image std per sample: mean={mean_std:.3f}")
if mean_std < 0.01:
    print(f"   [WARNING] 图像强度变化太小！")
else:
    print(f"   [OK] 图像对比度合理")

print("\n" + "="*60)
print("SANITY CHECK COMPLETE")
print("="*60)
