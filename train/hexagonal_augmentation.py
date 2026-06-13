"""Hexagonal symmetry augmentation for seven-beam CBC phase labels.

The seven outer beams are ordered counter-clockwise at angles
0, 60, 120, 180, 240, and 300 degrees.  When the far-field image is
rotated or mirrored, the six phase labels must be permuted in the same
physical way.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def _validate_seven_beam_label(label):
    label = np.asarray(label)
    if label.shape[-1] != 12:
        raise ValueError(f"Expected seven-beam sin/cos label with 12 values, got {label.shape}")
    return label.reshape(6, 2)


def rotate_seven_beam_label(label, steps):
    """Permute labels after rotating the image by steps * 60 degrees CCW."""
    pairs = _validate_seven_beam_label(label)
    steps = int(steps) % 6
    if steps == 0:
        return pairs.reshape(12).copy()
    return np.roll(pairs, shift=steps, axis=0).reshape(12).copy()


def mirror_seven_beam_label(label):
    """Permute labels after a left-right image mirror."""
    pairs = _validate_seven_beam_label(label)
    mirror_indices = np.array([3, 2, 1, 0, 5, 4])
    return pairs[mirror_indices].reshape(12).copy()


def _as_plane_batch(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return image[None, :, :], True
    if image.ndim == 3:
        return image, False
    raise ValueError(f"Expected image with shape [H,W] or [P,H,W], got {image.shape}")


def rotate_image_by_60_steps(image, steps):
    """Rotate an image or plane stack by steps * 60 degrees CCW.

    Uses bilinear sampling and keeps the original image size.  Works for
    single-plane [H, W] and multi-plane [P, H, W] arrays.
    """
    steps = int(steps) % 6
    if steps == 0:
        return np.asarray(image, dtype=np.float32).copy()

    planes, squeeze = _as_plane_batch(image)
    tensor = torch.from_numpy(planes).unsqueeze(0)
    angle = -steps * math.pi / 3.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    theta = torch.tensor(
        [[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]],
        dtype=tensor.dtype,
    )
    grid = F.affine_grid(theta, size=tensor.shape, align_corners=False)
    rotated = F.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0].numpy()
    if squeeze:
        return rotated[0]
    return rotated


def apply_hexagonal_augmentation(image, label, rng=None):
    """Apply a random D6 symmetry transform to image and seven-beam label."""
    rng = rng or np.random
    steps = int(rng.randint(0, 6))
    mirror = bool(rng.randint(0, 2))

    aug_image = rotate_image_by_60_steps(image, steps)
    aug_label = rotate_seven_beam_label(label, steps)

    if mirror:
        aug_image = np.flip(aug_image, axis=-1).copy()
        aug_label = mirror_seven_beam_label(aug_label)

    return aug_image.astype(np.float32, copy=False), aug_label.astype(np.float32, copy=False)
