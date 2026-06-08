import torch
import torch.nn as nn
import torch.nn.functional as F

from train.phase_metrics import decode_sin_cos


def crop_center_torch(images, crop_size):
    """从批量图像中心裁剪正方形区域。

    支持形状：
    - [B, H, W]
    - [B, C, H, W]
    """
    if images.ndim not in (3, 4):
        raise ValueError(f"Expected images with 3 or 4 dims, got {images.shape}")

    height = images.shape[-2]
    width = images.shape[-1]
    if crop_size > min(height, width):
        raise ValueError(f"crop_size={crop_size} is larger than image shape {images.shape}")

    center_y = height // 2
    center_x = width // 2
    half = crop_size // 2

    if crop_size % 2 == 0:
        return images[..., center_y - half:center_y + half, center_x - half:center_x + half]

    return images[
        ...,
        center_y - half:center_y + half + 1,
        center_x - half:center_x + half + 1,
    ]


def normalize_intensity(intensity, eps=1e-12):
    """按每张图最大值归一化远场光强。"""
    if intensity.ndim < 2:
        raise ValueError(f"Expected intensity image tensor, got {intensity.shape}")

    reduce_dims = tuple(range(1, intensity.ndim))
    max_values = intensity.amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    return intensity / max_values


class TwoBeamFourierOptics(nn.Module):
    """双光束傅里叶光学可微分前向模型。

    该模块用于物理一致性损失：

    1. 将网络输出的 [sin(phi), cos(phi)] 解码为相位 phi。
    2. 根据 phi 重建双光束近场复振幅。
    3. 通过 torch.fft.fft2 计算远场光强。
    4. 按每张图最大值归一化，并裁剪中心区域。

    当前模型与 simulation/common/two_beam_core.py 中的双光束仿真保持同一参数默认值。
    """

    def __init__(
        self,
        num_points=256,
        window_size=10e-3,
        waist=0.5e-3,
        beam_distance=1.5e-3,
        crop_size=160,
        eps=1e-12,
        dtype=torch.float32,
    ):
        super().__init__()
        self.num_points = num_points
        self.window_size = window_size
        self.waist = waist
        self.beam_distance = beam_distance
        self.crop_size = crop_size
        self.eps = eps

        x = torch.linspace(-window_size / 2, window_size / 2, num_points, dtype=dtype)
        x_grid, y_grid = torch.meshgrid(x, x, indexing="xy")

        envelope_1 = self._gaussian_envelope(
            x_grid=x_grid,
            y_grid=y_grid,
            center_x=-beam_distance / 2,
            center_y=0.0,
            waist=waist,
        )
        envelope_2 = self._gaussian_envelope(
            x_grid=x_grid,
            y_grid=y_grid,
            center_x=beam_distance / 2,
            center_y=0.0,
            waist=waist,
        )

        self.register_buffer("envelope_1", envelope_1)
        self.register_buffer("envelope_2", envelope_2)

    @staticmethod
    def _gaussian_envelope(x_grid, y_grid, center_x, center_y, waist):
        return torch.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / waist**2)

    def reconstruct_from_phase(self, phase):
        """根据相位重建裁剪后的归一化远场光强。"""
        if phase.ndim == 2 and phase.shape[-1] == 1:
            phase = phase[:, 0]
        if phase.ndim != 1:
            raise ValueError(f"Expected phase with shape [B] or [B, 1], got {phase.shape}")

        batch_size = phase.shape[0]
        envelope_1 = self.envelope_1.unsqueeze(0).expand(batch_size, -1, -1)
        envelope_2 = self.envelope_2.unsqueeze(0).expand(batch_size, -1, -1)

        reference_phase = torch.zeros_like(envelope_1)
        beam_1 = torch.polar(envelope_1, reference_phase)

        phase_map = phase[:, None, None].expand_as(envelope_2)
        beam_2 = torch.polar(envelope_2, phase_map)
        near_field = beam_1 + beam_2

        far_field = torch.fft.fftshift(torch.fft.fft2(near_field), dim=(-2, -1))
        intensity = torch.abs(far_field) ** 2
        intensity = normalize_intensity(intensity, eps=self.eps)
        return crop_center_torch(intensity, self.crop_size).to(dtype=torch.float32)

    def reconstruct_from_sin_cos(self, sin_cos_values):
        """根据 [sin(phi), cos(phi)] 重建裁剪后的归一化远场光强。"""
        phase = decode_sin_cos(sin_cos_values)
        return self.reconstruct_from_phase(phase)

    def forward(self, sin_cos_values):
        return self.reconstruct_from_sin_cos(sin_cos_values)


class FarFieldConsistencyLoss(nn.Module):
    """远场物理一致性损失。

    输入：
    - pred_sin_cos: 网络预测的 [sin(phi), cos(phi)]。
    - target_images: 输入远场光强图，形状 [B, 1, H, W] 或 [B, H, W]。

    输出：
    - MSE(reconstructed_far_field, target_images)
    """

    def __init__(self, optics_model=None, loss_type="mse"):
        super().__init__()
        self.optics_model = optics_model or TwoBeamFourierOptics()
        self.loss_type = loss_type

    def forward(self, pred_sin_cos, target_images):
        reconstructed = self.optics_model(pred_sin_cos)

        if target_images.ndim == 4:
            target_images = target_images[:, 0]
        if target_images.shape != reconstructed.shape:
            raise ValueError(
                f"Target image shape {target_images.shape} does not match "
                f"reconstructed shape {reconstructed.shape}"
            )

        target_images = target_images.to(device=reconstructed.device, dtype=reconstructed.dtype)

        if self.loss_type == "mse":
            return F.mse_loss(reconstructed, target_images)
        if self.loss_type == "l1":
            return F.l1_loss(reconstructed, target_images)

        raise ValueError(f"Unknown loss_type: {self.loss_type}")
