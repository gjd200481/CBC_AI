"""验证多平面数据的两通道是否退化相同"""
import numpy as np
from pathlib import Path

# 检查三个数据集
datasets = [
    "dataset/seven_beam/multiplane_0_-0.03",
    "dataset/seven_beam/multiplane_0_-0.05",
    "dataset/seven_beam/multiplane_0_-0.07",
]

for dataset_path in datasets:
    path = Path(dataset_path)
    if not path.exists():
        print(f"[!] {dataset_path} 不存在")
        continue

    # 读取数据
    images_file = list(path.glob("images_*.npy"))[0]
    images = np.load(images_file)

    print(f"\n{'='*60}")
    print(f"数据集: {dataset_path}")
    print(f"  Shape: {images.shape}")

    # 检查前10个样本的两个通道
    num_check = min(10, images.shape[0])

    identical_count = 0
    max_diff_list = []

    for i in range(num_check):
        plane0 = images[i, 0]  # 焦平面
        plane1 = images[i, 1]  # 焦前平面

        # 检查是否完全相同
        is_identical = np.allclose(plane0, plane1, atol=1e-10, rtol=1e-10)
        max_diff = np.max(np.abs(plane0 - plane1))

        if is_identical:
            identical_count += 1

        max_diff_list.append(max_diff)

    print(f"\n  前{num_check}个样本检查:")
    print(f"    完全相同(allclose): {identical_count}/{num_check}")
    print(f"    最大差异范围: {min(max_diff_list):.2e} ~ {max(max_diff_list):.2e}")

    if identical_count == num_check:
        print(f"  [X] 确认问题: 两通道完全退化相同")
    elif identical_count > num_check // 2:
        print(f"  [!] 大部分样本两通道相同")
    else:
        print(f"  [OK] 两通道有差异")
