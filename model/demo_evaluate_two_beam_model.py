import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class TwoBeamDataset(Dataset):
    """Load two-beam far-field images and sin/cos phase labels from .npy files."""

    def __init__(self, image_path, label_path):
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

    def __getitem__(self, index):
        image = torch.tensor(self.images[index], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, label


class SimpleCNN(nn.Module):
    """Same network structure as train/evaluate_two_beam.py."""

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
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


def default_model_path(repo_root):
    """Prefer the noisy model if it exists, otherwise fall back to the original name."""
    noisy_model = repo_root / "models" / "two_beam_cnn__noise_0.05.pth"
    clean_model = repo_root / "models" / "two_beam_cnn.pth"

    if noisy_model.exists():
        return noisy_model

    return clean_model


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    model = SimpleCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def angle_error(pred_phi, true_phi):
    """Wrap phase error into [-pi, pi] so boundary cases do not inflate RMSE."""
    return np.arctan2(
        np.sin(pred_phi - true_phi),
        np.cos(pred_phi - true_phi),
    )


def evaluate(model, data_loader, device):
    pred_phi = []
    true_phi = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            preds = model(images).cpu().numpy()
            labels = labels.numpy()

            pred_angle = np.arctan2(preds[:, 0], preds[:, 1])
            true_angle = np.arctan2(labels[:, 0], labels[:, 1])

            pred_phi.extend(pred_angle)
            true_phi.extend(true_angle)

    pred_phi = np.array(pred_phi)
    true_phi = np.array(true_phi)
    errors = angle_error(pred_phi, true_phi)
    rmse = np.sqrt(np.mean(errors**2))

    return pred_phi, true_phi, errors, rmse


def main():
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Load a trained two-beam CNN model and evaluate phase RMSE."
    )
    parser.add_argument(
        "--model-path",
        default=default_model_path(repo_root),
        type=Path,
        help="Path to the trained .pth model file.",
    )
    parser.add_argument(
        "--image-path",
        default=repo_root / "dataset" / "two_beam" / "images_noise_0.05.npy",
        type=Path,
        help="Path to evaluation images saved as .npy.",
    )
    parser.add_argument(
        "--label-path",
        default=repo_root / "dataset" / "two_beam" / "labels_noise_0.05.npy",
        type=Path,
        help="Path to evaluation labels saved as .npy.",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print metrics and skip matplotlib figures.",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    if not args.label_path.exists():
        raise FileNotFoundError(f"Label file not found: {args.label_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Model:", args.model_path)
    print("Images:", args.image_path)
    print("Labels:", args.label_path)

    dataset = TwoBeamDataset(args.image_path, args.label_path)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = load_model(args.model_path, device)
    pred_phi, true_phi, errors, rmse = evaluate(model, data_loader, device)

    print("\nEvaluation result:")
    print("Samples:", len(dataset))
    print("RMSE(rad):", rmse)
    print("RMSE(deg):", np.degrees(rmse))
    print("Mean error(rad):", np.mean(errors))
    print("Mean error(deg):", np.degrees(np.mean(errors)))

    if args.no_plot:
        return

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(true_phi, pred_phi, s=5)
    plt.xlabel("True phi")
    plt.ylabel("Pred phi")
    plt.title("Predicted vs True Phase")

    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=30)
    plt.xlabel("Error(rad)")
    plt.title("Phase Error Distribution")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
