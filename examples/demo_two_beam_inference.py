import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """训练时使用的 CNN 结构，必须和 train/evaluate_two_beam.py 保持一致。"""

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


def load_model(model_path, device):
    """加载训练好的模型权重，并切换到 eval 推理模式。"""
    checkpoint = torch.load(model_path, map_location=device)

    model = SimpleCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_phi(model, image, device):
    """输入一张 160x160 远场图，输出预测相位。"""
    image_tensor = torch.tensor(image, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image_tensor)[0].cpu().numpy()

    pred_sin = pred[0]
    pred_cos = pred[1]
    pred_phi = np.arctan2(pred_sin, pred_cos)

    return pred_sin, pred_cos, pred_phi


def main():
    repo_root = Path(__file__).resolve().parents[1]

    default_model_path = repo_root / "models" / "two_beam_cnn__noise_0.05.pth"
    default_image_path = repo_root / "dataset" / "two_beam" / "images_noise_0.05.npy"
    default_label_path = repo_root / "dataset" / "two_beam" / "labels_noise_0.05.npy"

    parser = argparse.ArgumentParser(
        description="Load a trained two-beam CNN model and predict one phase."
    )
    parser.add_argument(
        "--model-path",
        default=default_model_path,
        type=Path,
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--image-path",
        default=default_image_path,
        type=Path,
        help="Path to the .npy image data.",
    )
    parser.add_argument(
        "--label-path",
        default=default_label_path,
        type=Path,
        help="Path to the .npy label data, used only for comparison.",
    )
    parser.add_argument(
        "--index",
        default=0,
        type=int,
        help="Sample index selected from the image array.",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model_path}\n"
            "Please run train/evaluate_two_beam.py first."
        )

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, checkpoint = load_model(args.model_path, device)
    images = np.load(args.image_path)
    if images.shape[1:] != (160, 160):
        raise ValueError(
            f"Expected images with shape (num_samples, 160, 160), "
            f"got {images.shape}"
        )

    image = images[args.index]
    pred_sin, pred_cos, pred_phi = predict_phi(model, image, device)

    print("\nModel loaded from:", args.model_path)
    print("Model class:", checkpoint.get("model_class", "SimpleCNN"))
    print("Output format:", checkpoint.get("output_format", "[sin(phi), cos(phi)]"))

    print("\nPrediction:")
    print("Index:", args.index)
    print("Pred sin(phi):", pred_sin)
    print("Pred cos(phi):", pred_cos)
    print("Pred phi(rad):", pred_phi)
    print("Pred phi(deg):", np.degrees(pred_phi))

    if args.label_path.exists():
        labels = np.load(args.label_path)
        if labels.shape[1:] != (2,):
            raise ValueError(
                f"Expected labels with shape (num_samples, 2) for "
                f"[sin(phi), cos(phi)], got {labels.shape}"
            )
        true = labels[args.index]
        true_phi = np.arctan2(true[0], true[1])
        wrapped_error = np.arctan2(
            np.sin(pred_phi - true_phi),
            np.cos(pred_phi - true_phi),
        )

        print("\nGround truth:")
        print("True sin/cos:", true)
        print("True phi(rad):", true_phi)
        print("True phi(deg):", np.degrees(true_phi))
        print("Wrapped error(rad):", wrapped_error)
        print("Wrapped error(deg):", np.degrees(wrapped_error))


if __name__ == "__main__":
    main()
