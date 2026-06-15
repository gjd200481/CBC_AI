"""可视化修正前后的多平面图像差异"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 读取旧数据（退化的）
old_path = Path("dataset/seven_beam/multiplane_0_-0.05")
old_images = np.load(old_path / "images_multiplane_5cm.npy")

# 读取新数据（修正的）
new_path = Path("dataset/seven_beam/multiplane_corrected_smoke")
new_images = np.load(new_path / "images_multiplane_corrected_smoke.npy")

# 选择第一个样本
sample_idx = 0
old_focal = old_images[sample_idx, 0]
old_defocus = old_images[sample_idx, 1]
new_focal = new_images[sample_idx, 0]
new_defocus = new_images[sample_idx, 1]

# 计算差异
old_diff = np.abs(old_focal - old_defocus)
new_diff = np.abs(new_focal - new_defocus)

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 旧数据（退化的）
axes[0, 0].imshow(old_focal, cmap='hot')
axes[0, 0].set_title('Old: Focal Plane', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(old_defocus, cmap='hot')
axes[0, 1].set_title('Old: Defocus -5cm', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(old_diff, cmap='viridis')
axes[0, 2].set_title(f'Old: Diff (max={np.max(old_diff):.2e})', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

axes[0, 3].text(0.5, 0.5, f'OLD DATA\n\n'
                          f'Max diff: {np.max(old_diff):.2e}\n'
                          f'Mean diff: {np.mean(old_diff):.2e}\n'
                          f'Allclose: {np.allclose(old_focal, old_defocus, atol=1e-10)}\n\n'
                          f'Problem: Degenerate!',
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0, 3].axis('off')

# 新数据（修正的）
axes[1, 0].imshow(new_focal, cmap='hot')
axes[1, 0].set_title('New: Focal Plane', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(new_defocus, cmap='hot')
axes[1, 1].set_title('New: Defocus -5cm', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(new_diff, cmap='viridis', vmax=0.5)
axes[1, 2].set_title(f'New: Diff (max={np.max(new_diff):.2f})', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

axes[1, 3].text(0.5, 0.5, f'NEW DATA\n\n'
                          f'Max diff: {np.max(new_diff):.2f}\n'
                          f'Mean diff: {np.mean(new_diff):.4f}\n'
                          f'Allclose: {np.allclose(new_focal, new_defocus, atol=1e-10)}\n\n'
                          f'Fixed: Real difference!',
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
axes[1, 3].axis('off')

plt.suptitle('Multiplane Data Correction: Before vs After', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path("result/figures/multiplane_correction_comparison.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {output_path}")

plt.show()
