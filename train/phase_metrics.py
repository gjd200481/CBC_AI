import numpy as np
import torch
import torch.nn as nn


def decode_sin_cos(values):
    """把 sin/cos 编码解码为相位角。

    输入最后一维必须是偶数，按 [sin(phi), cos(phi)] 成对排列。
    返回相位单位为 rad，范围为 [-pi, pi]。
    """
    if values.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {values.shape[-1]}")

    if torch.is_tensor(values):
        sin_values = values[..., 0::2]
        cos_values = values[..., 1::2]
        return torch.atan2(sin_values, cos_values)

    values = np.asarray(values)
    sin_values = values[..., 0::2]
    cos_values = values[..., 1::2]
    return np.arctan2(sin_values, cos_values)


def wrap_phase_error(pred_phase, true_phase):
    """把相位误差折回 [-pi, pi]，避免周期边界导致虚假大误差。"""
    diff = pred_phase - true_phase

    if torch.is_tensor(diff):
        return torch.atan2(torch.sin(diff), torch.cos(diff))

    diff = np.asarray(diff)
    return np.arctan2(np.sin(diff), np.cos(diff))


def phase_rmse_from_angles(pred_phase, true_phase):
    """根据预测相位和真实相位计算周期 RMSE。"""
    errors = wrap_phase_error(pred_phase, true_phase)

    if torch.is_tensor(errors):
        return torch.sqrt(torch.mean(errors**2))

    return float(np.sqrt(np.mean(errors**2)))


def phase_rmse_from_sin_cos(pred_values, true_values):
    """根据 sin/cos 编码直接计算周期相位 RMSE。"""
    pred_phase = decode_sin_cos(pred_values)
    true_phase = decode_sin_cos(true_values)
    return phase_rmse_from_angles(pred_phase, true_phase)


def phase_metrics_from_sin_cos(pred_values, true_values):
    """返回常用相位误差指标，便于训练和评估脚本统一记录。"""
    pred_phase = decode_sin_cos(pred_values)
    true_phase = decode_sin_cos(true_values)
    errors = wrap_phase_error(pred_phase, true_phase)

    if torch.is_tensor(errors):
        rmse = torch.sqrt(torch.mean(errors**2)).item()
        mae = torch.mean(torch.abs(errors)).item()
        mean_error = torch.mean(errors).item()
    else:
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        mean_error = float(np.mean(errors))

    return {
        "rmse_rad": rmse,
        "rmse_deg": float(np.degrees(rmse)),
        "mae_rad": mae,
        "mae_deg": float(np.degrees(mae)),
        "mean_error_rad": mean_error,
        "mean_error_deg": float(np.degrees(mean_error)),
    }


def _split_and_normalize_sin_cos(values, eps=1e-8):
    if values.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {values.shape[-1]}")

    sin_values = values[..., 0::2]
    cos_values = values[..., 1::2]
    norm = torch.sqrt(sin_values**2 + cos_values**2 + eps)
    return sin_values / norm, cos_values / norm


def cyclic_phase_loss_from_sin_cos(pred_values, true_values, eps=1e-8):
    """Xie-style periodic phase loss for [sin(phi), cos(phi)] targets.

    The paper uses 2 - 2*cos(theta - phi). With sin/cos labels this is the
    same as 2 - 2*(sin(theta)sin(phi) + cos(theta)cos(phi)).
    """
    pred_sin, pred_cos = _split_and_normalize_sin_cos(pred_values, eps=eps)
    true_sin, true_cos = _split_and_normalize_sin_cos(true_values, eps=eps)
    cos_delta = pred_sin * true_sin + pred_cos * true_cos
    cos_delta = torch.clamp(cos_delta, min=-1.0, max=1.0)
    return torch.mean(2.0 - 2.0 * cos_delta)


def unit_circle_loss_from_sin_cos(pred_values):
    """Keep raw network outputs close to valid sin/cos pairs."""
    if pred_values.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {pred_values.shape[-1]}")
    sin_values = pred_values[..., 0::2]
    cos_values = pred_values[..., 1::2]
    return torch.mean((sin_values**2 + cos_values**2 - 1.0) ** 2)


class CyclicPhaseLoss(nn.Module):
    """Periodic phase loss with an optional unit-circle regularizer."""

    def __init__(self, unit_weight=0.0, eps=1e-8):
        super().__init__()
        self.unit_weight = unit_weight
        self.eps = eps

    def forward(self, pred_values, true_values):
        loss = cyclic_phase_loss_from_sin_cos(
            pred_values=pred_values,
            true_values=true_values,
            eps=self.eps,
        )
        if self.unit_weight > 0:
            loss = loss + self.unit_weight * unit_circle_loss_from_sin_cos(pred_values)
        return loss


def build_phase_loss(loss_name="mse", unit_weight=0.0):
    """Build supervised phase loss by name."""
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "cyclic":
        return CyclicPhaseLoss(unit_weight=unit_weight)
    if loss_name == "cyclic_unit":
        if unit_weight == 0.0:
            unit_weight = 0.01
        return CyclicPhaseLoss(unit_weight=unit_weight)
    raise ValueError("Unknown phase loss: " f"{loss_name}")
