import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# =====================================
# Dataset
# =====================================

class Dataset2Beam(Dataset):
    def __init__(self, image_path, label_path):
        # 从 .npy 文件读取输入图像和对应标签。
        # images 保存双光束干涉图，labels 保存每张图对应的目标值。
        self.images = np.load(image_path)
        self.labels = np.load(label_path)

    def __len__(self):
        # DataLoader 会调用这个函数来获得数据集大小。
        return len(self.images)

    def __getitem__(self, idx):
        # 取出第 idx 张图像，并转换成 float32 的 PyTorch 张量。
        # 原始灰度图一般是 [H, W]，Conv2d 需要 [C, H, W]，
        # 所以用 unsqueeze(0) 增加一个通道维度 C=1。
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


# =====================================
# CNN
# =====================================


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积特征提取部分。
        # 每一组结构都是：卷积层 -> ReLU 激活函数 -> 最大池化层。
        self.features = nn.Sequential(
            # 第 1 层卷积：输入 1 个通道，输出 16 个特征图，卷积核大小 3x3。
            # padding=1 可以保持卷积前后的图像高宽不变。
            nn.Conv2d(1, 16, 3, padding=1),
            # ReLU 激活函数：把负值变成 0，保留正值，用来增加模型的非线性表达能力。
            nn.ReLU(),
            # 2x2 最大池化：把特征图的高和宽都缩小一半，通道数不变。
            nn.MaxPool2d(2),
            # 第 2 层卷积：特征图数量从 16 增加到 32。
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 第 3 层卷积：特征图数量从 32 增加到 64。
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 全连接回归部分。
        # 这里把卷积提取到的二维特征转换成最终的 2 个输出值。
        self.fc = nn.Sequential(
            # 把 [通道数, 高, 宽] 展平成一维向量。
            nn.Flatten(),
            # 输入图像尺寸是 160x160，经过三次 2x2 池化后变成 20x20。
            # 最后一层卷积有 64 个通道，所以展开后的长度是 64*20*20。
            nn.Linear(64 * 20 * 20, 128),
            nn.ReLU(),
            # 输出 2 个数，用 arctan2 可以还原成相位角。
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # 前向传播：先经过卷积层提取特征，再经过全连接层输出预测结果。
        x = self.features(x)
        x = self.fc(x)
        return x


# =====================================
# load
# =====================================


dataset = Dataset2Beam(
    "dataset/two_beam/images.npy",
    "dataset/two_beam/labels.npy",
)

# 将数据集按 80% / 20% 划分为训练集和测试集。
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader 用来按 batch 加载数据。
# 训练集 shuffle=True 表示每轮训练前打乱顺序，有助于模型训练。
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

# 如果有可用的 NVIDIA GPU，就使用 cuda；否则使用 cpu。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建 CNN 模型，并把模型放到对应设备上。
model = CNN().to(device)

# MSELoss 是均方误差损失，适合预测连续数值的回归任务。
criterion = nn.MSELoss()

# Adam 优化器负责根据梯度更新模型参数。
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
        # 输入图像和标签必须和模型在同一个设备上。
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播得到预测值，然后计算预测值和真实标签之间的损失。
        pred = model(images)
        loss = criterion(pred, labels)

        # 训练三步：
        # 1. 清空上一轮残留的梯度；
        # 2. 反向传播计算当前梯度；
        # 3. 根据梯度更新模型参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 保存每个 epoch 的总损失，后面用于画训练损失曲线。
    losses.append(total_loss)


# =====================================
# evaluate
# =====================================


model.eval()

# 保存测试集上的预测相位和真实相位。
pred_phi = []
true_phi = []

# 测试/评估时不需要计算梯度，可以减少内存占用并加快速度。
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        pred = model(images)
        # 把预测值从 GPU/CPU 张量转换成 numpy 数组，方便后续计算。
        pred = pred.cpu().numpy()
        labels = labels.numpy()

        # labels 和 pred 都是两个分量。
        # arctan2(y, x) 可以把两个分量转换成相位角。
        pred_angle = np.arctan2(pred[:, 0], pred[:, 1])
        true_angle = np.arctan2(labels[:, 0], labels[:, 1])

        pred_phi.extend(pred_angle)
        true_phi.extend(true_angle)

pred_phi = np.array(pred_phi)
true_phi = np.array(true_phi)

# 计算预测相位和真实相位的误差。
error = pred_phi - true_phi
# RMSE 是均方根误差，用来衡量整体预测误差大小。
rmse = np.sqrt(np.mean(error**2))

print("RMSE=", rmse)


# =====================================
# plot
# =====================================


plt.figure(figsize=(12, 4))

# 第 1 张图：训练损失随 epoch 的变化。
plt.subplot(131)
plt.plot(losses)
plt.title("Loss")

# 第 2 张图：真实相位和预测相位的散点图。
# 如果预测很好，散点应该接近 y=x 的对角线。
plt.subplot(132)
plt.scatter(true_phi, pred_phi, s=5)
plt.xlabel("True")
plt.ylabel("Pred")
plt.title("Pred vs True")

# 第 3 张图：误差分布图。
# 理想情况下，误差应该集中在 0 附近。
plt.subplot(133)
plt.hist(error, bins=30)
plt.title("Error distribution")

plt.tight_layout()
plt.show()
