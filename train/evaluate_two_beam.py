import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


# =====================================
# Dataset
# =====================================


class Dataset2Beam(Dataset):
    def __init__(self, image_path, label_path):
        # 从 .npy 文件读取远场光强图像和相位标签。
        # images 保存双光束干涉图，labels 保存 [sin(phi), cos(phi)]。
        self.images = np.load(image_path)
        self.labels = np.load(label_path)
        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Images and labels have different lengths: "
                f"{len(self.images)} vs {len(self.labels)}"
            )
        if self.images.shape[1:] != (160, 160):
            raise ValueError(
                f"Expected images with shape (num_samples, 160, 160), "
                f"got {self.images.shape}"
            )
        if self.labels.shape[1:] != (2,):
            raise ValueError(
                f"Expected labels with shape (num_samples, 2) for "
                f"[sin(phi), cos(phi)], got {self.labels.shape}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Conv2d 输入格式为 [C, H, W]，因此给灰度图增加通道维度 C=1。
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# =====================================
# CNN
# =====================================


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 三层卷积提取远场光斑特征；每次池化后图像尺寸减半。
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 输入 160x160，三次 2x2 池化后为 20x20，输出 [sin(phi), cos(phi)]。
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20 * 20, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# =====================================
# load
# =====================================


image_path = "dataset/two_beam/images_noise_0.05.npy"
label_path = "dataset/two_beam/labels_noise_0.05.npy"
model_dir = "models"
model_path = os.path.join(model_dir, "two_beam_cnn__noise_0.05.pth")

os.makedirs(model_dir, exist_ok=True)

dataset = Dataset2Beam(image_path, label_path)

# 将数据集按 80% / 20% 划分为训练集和测试集。
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)


# =====================================
# train
# =====================================


losses = []

for epoch in range(20):
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    losses.append(total_loss)


# =====================================
# save model
# =====================================


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_epochs": 20,
        "losses": losses,
        "image_path": image_path,
        "label_path": label_path,
        "model_class": "CNN",
        "output_format": "[sin(phi), cos(phi)]",
    },
    model_path,
)

print(f"Model saved to: {model_path}")


# =====================================
# evaluate
# =====================================


model.eval()

pred_phi = []
true_phi = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        pred = model(images).cpu().numpy()
        labels = labels.numpy()

        pred_angle = np.arctan2(pred[:, 0], pred[:, 1])
        true_angle = np.arctan2(labels[:, 0], labels[:, 1])

        pred_phi.extend(pred_angle)
        true_phi.extend(true_angle)

pred_phi = np.array(pred_phi)
true_phi = np.array(true_phi)

# 相位是周期变量，需要把误差折回 [-pi, pi]，避免 pi 边界附近出现虚假大误差。
error = np.arctan2(
    np.sin(pred_phi - true_phi),
    np.cos(pred_phi - true_phi),
)
rmse = np.sqrt(np.mean(error**2))

print("RMSE=", rmse)


# =====================================
# plot
# =====================================


plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(losses)
plt.title("Loss")

plt.subplot(132)
plt.scatter(true_phi, pred_phi, s=5)
plt.xlabel("True")
plt.ylabel("Pred")
plt.title("Pred vs True")

plt.subplot(133)
plt.hist(error, bins=30)
plt.title("Error distribution")

plt.tight_layout()
plt.show()
