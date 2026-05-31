import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# ======================
# Dataset
# ======================

class TwoBeamDataset(Dataset):
    def __init__(self, image_path, label_path):
        self.images = np.load(image_path)
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


# ======================
# CNN Model
# ======================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20 * 20, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# ======================
# Main
# ======================

image_path = "dataset/two_beam/images.npy"
label_path = "dataset/two_beam/labels.npy"

dataset = TwoBeamDataset(image_path, label_path)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

model = SimpleCNN().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

num_epochs = 20

train_losses = []

for epoch in range(num_epochs):
    model.train()

    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}"
    )


# ======================
# Test one sample
# ======================

model.eval()

images, labels = next(iter(test_loader))

images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    preds = model(images)

pred = preds[0].cpu().numpy()
true = labels[0].cpu().numpy()

pred_phi = np.arctan2(pred[0], pred[1])
true_phi = np.arctan2(true[0], true[1])

print("\nExample prediction:")
print("Pred sin/cos:", pred)
print("True sin/cos:", true)
print("Pred phi:", pred_phi)
print("True phi:", true_phi)
print("Error:", pred_phi - true_phi)


# ======================
# Plot loss
# ======================

plt.figure()
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid()
plt.show()