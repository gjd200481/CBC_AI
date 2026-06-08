import numpy as np
import torch


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
