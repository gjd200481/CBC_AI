from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class FarFieldPhaseDataset(Dataset):
    """读取远场光强图像和相位 sin/cos 标签。

    数据约定：
    - images: [num_samples, height, width]
    - labels: [num_samples, 2 * num_phases]

    对双光束任务来说，num_phases=1，标签就是 [sin(phi), cos(phi)]。
    后续扩展到多束时，可以使用 [sin(phi_1), cos(phi_1), ...] 的形式。
    """

    def __init__(self, image_path, label_path, expected_size=None, augment=False):
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)
        self.augment = augment

        self.images = np.load(self.image_path)
        self.labels = np.load(self.label_path)

        self._validate(expected_size=expected_size)

    def _validate(self, expected_size):
        # 支持单平面 [N,H,W] 或多平面 [N,P,H,W]
        if self.images.ndim not in (3, 4):
            raise ValueError(
                f"Expected images with shape [N,H,W] or [N,P,H,W], got {self.images.shape}"
            )
        if self.labels.ndim != 2:
            raise ValueError(
                f"Expected labels with shape [N, 2 * num_phases], got {self.labels.shape}"
            )
        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Images and labels have different lengths: "
                f"{len(self.images)} vs {len(self.labels)}"
            )
        if self.labels.shape[1] % 2 != 0:
            raise ValueError(
                f"Label dimension must be even for sin/cos pairs, got {self.labels.shape[1]}"
            )
        if expected_size is not None:
            actual_size = self.images.shape[-2:] if self.images.ndim == 4 else self.images.shape[1:]
            if actual_size != tuple(expected_size):
                raise ValueError(
                    f"Expected image size {tuple(expected_size)}, got {actual_size}"
                )


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].copy()
        label = self.labels[index].copy()
        
        if self.augment and np.random.rand() > 0.5:
            # 随机噪声增强
            noise = np.random.randn(*image.shape) * 0.01
            image = image + noise
            image = np.clip(image, 0, None)
        
        # 单平面 [H,W] -> [1,H,W], 多平面 [P,H,W] 保持
        if image.ndim == 2:
            image = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
        else:
            image = torch.as_tensor(image, dtype=torch.float32)
        
        label = torch.as_tensor(label, dtype=torch.float32)
        return image, label

    @property
    def image_size(self):
        return self.images.shape[1:]

    @property
    def num_phases(self):
        return self.labels.shape[1] // 2


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=20260608):
    """按固定随机种子划分训练集、验证集和测试集。"""
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "Expected train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1"
        )

    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


def build_dataloaders(
    image_path,
    label_path,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=20260608,
    expected_size=(160, 160),
    num_workers=0,
    augment_train=False,
):
    """构建训练、验证、测试 DataLoader。"""
    dataset = FarFieldPhaseDataset(
        image_path=image_path,
        label_path=label_path,
        expected_size=expected_size,
        augment=False,
    )
    train_set, val_set, test_set = split_dataset(
        dataset=dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    
    if augment_train:
        train_set.dataset.augment = True

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {
        "dataset": dataset,
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "splits": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
    }
