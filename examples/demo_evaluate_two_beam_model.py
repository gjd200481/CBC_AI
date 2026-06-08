import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.data_utils import FarFieldPhaseDataset
from train.models import SimplePhaseCNN
from train.phase_metrics import (
    decode_sin_cos,
    phase_metrics_from_sin_cos,
    wrap_phase_error,
)


def default_model_path(repo_root):
    """Prefer the noisy model if it exists, otherwise fall back to the original name."""
    noisy_model = repo_root / "models" / "two_beam_cnn__noise_0.05.pth"
    clean_model = repo_root / "models" / "two_beam_cnn.pth"

    if noisy_model.exists():
        return noisy_model

    return clean_model


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    model = SimplePhaseCNN(image_size=160, output_dim=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def evaluate(model, data_loader, device):
    pred_values = []
    true_values = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            preds = model(images).cpu().numpy()
            labels = labels.numpy()

            pred_values.append(preds)
            true_values.append(labels)

    pred_values = np.concatenate(pred_values, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    pred_phi = decode_sin_cos(pred_values).reshape(-1)
    true_phi = decode_sin_cos(true_values).reshape(-1)
    errors = wrap_phase_error(pred_phi, true_phi).reshape(-1)
    metrics = phase_metrics_from_sin_cos(pred_values, true_values)

    return pred_phi, true_phi, errors, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Load a trained two-beam CNN model and evaluate phase RMSE."
    )
    parser.add_argument(
        "--model-path",
        default=default_model_path(REPO_ROOT),
        type=Path,
        help="Path to the trained .pth model file.",
    )
    parser.add_argument(
        "--image-path",
        default=REPO_ROOT / "dataset" / "two_beam" / "images_noise_0.05.npy",
        type=Path,
        help="Path to evaluation images saved as .npy.",
    )
    parser.add_argument(
        "--label-path",
        default=REPO_ROOT / "dataset" / "two_beam" / "labels_noise_0.05.npy",
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

    dataset = FarFieldPhaseDataset(
        args.image_path,
        args.label_path,
        expected_size=(160, 160),
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.model_path, device)
    pred_phi, true_phi, errors, metrics = evaluate(model, data_loader, device)

    print("\nEvaluation result:")
    print("Samples:", len(dataset))
    print("RMSE(rad):", metrics["rmse_rad"])
    print("RMSE(deg):", metrics["rmse_deg"])
    print("MAE(rad):", metrics["mae_rad"])
    print("MAE(deg):", metrics["mae_deg"])
    print("Mean error(rad):", metrics["mean_error_rad"])
    print("Mean error(deg):", metrics["mean_error_deg"])

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
