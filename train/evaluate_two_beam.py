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
        self.images = np.load(image_path)
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


# =====================================
# CNN
# =====================================


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

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

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20 * 20, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
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
# evaluate
# =====================================


model.eval()

pred_phi = []
true_phi = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        pred = model(images)
        pred = pred.cpu().numpy()
        labels = labels.numpy()

        pred_angle = np.arctan2(pred[:, 0], pred[:, 1])
        true_angle = np.arctan2(labels[:, 0], labels[:, 1])

        pred_phi.extend(pred_angle)
        true_phi.extend(true_angle)

pred_phi = np.array(pred_phi)
true_phi = np.array(true_phi)

error = pred_phi - true_phi
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
