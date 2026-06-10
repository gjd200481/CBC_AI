import torch
import torch.nn as nn


class SimplePhaseCNN(nn.Module):
    """用于相干合成相位反演的基础 CNN。

    输入为单通道远场光强图像，默认尺寸 160 x 160。
    双光束任务输出 [sin(phi), cos(phi)]。
    7 光束任务可设置 output_dim=12，输出 6 路相对相位的 sin/cos 编码。
    """

    def __init__(self, image_size=160, output_dim=2):
        super().__init__()

        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8 for three pooling layers")

        feature_size = image_size // 8

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
            nn.Linear(64 * feature_size * feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


class WidePhaseCNN(nn.Module):
    """通道数更宽的 CNN，用于检查模型容量对 7 光束相位反演的影响。"""

    def __init__(self, image_size=160, output_dim=2):
        super().__init__()

        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8 for three pooling layers")

        feature_size = image_size // 8

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


class ResidualBlock(nn.Module):
    """两层卷积残差块，输入输出通道数保持一致。"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResidualPhaseCNN(nn.Module):
    """带残差连接和全局池化的 CNN，用于提升多路相位特征提取稳定性。"""

    def __init__(self, image_size=160, output_dim=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(32),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.regressor(x)


class SpatialChannelGate(nn.Module):
    """Light attention gate for CBC far-field fringe maps."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        channel_weight = self.channel_gate(x)
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        spatial_weight = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
        return x * channel_weight * spatial_weight


class SeparableResidualBlock(nn.Module):
    """Depthwise-separable residual block with optional dilation."""

    def __init__(self, input_channels, output_channels, stride=1, dilation=1):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.GELU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            SpatialChannelGate(output_channels),
        )
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class MultiScalePhaseHead(nn.Module):
    """Pool local and global fringe features before phase regression."""

    def __init__(self, channels, output_dim):
        super().__init__()
        self.pool_sizes = (1, 2, 4)
        feature_dim = channels * sum(size * size for size in self.pool_sizes)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        pooled = [
            nn.functional.adaptive_avg_pool2d(x, (size, size)).flatten(1)
            for size in self.pool_sizes
        ]
        return self.regressor(torch.cat(pooled, dim=1))


class CBCPhaseLiteCNN(nn.Module):
    """Project-specific lightweight CNN for seven-beam phase inversion."""

    def __init__(self, image_size=160, output_dim=2):
        super().__init__()
        del image_size

        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.features = nn.Sequential(
            SeparableResidualBlock(24, 32, stride=1, dilation=1),
            SeparableResidualBlock(32, 32, stride=1, dilation=2),
            SeparableResidualBlock(32, 48, stride=2, dilation=1),
            SeparableResidualBlock(48, 48, stride=1, dilation=2),
            SeparableResidualBlock(48, 80, stride=2, dilation=1),
            SeparableResidualBlock(80, 80, stride=1, dilation=2),
            SeparableResidualBlock(80, 128, stride=2, dilation=1),
            SeparableResidualBlock(128, 128, stride=1, dilation=2),
        )
        self.regressor = MultiScalePhaseHead(channels=128, output_dim=output_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.regressor(x)


def build_phase_model(model_name, image_size=160, output_dim=2):
    """按名称构建相位反演网络，便于训练脚本做结构消融。"""
    model_classes = {
        "simple_cnn": SimplePhaseCNN,
        "wide_cnn": WidePhaseCNN,
        "residual_cnn": ResidualPhaseCNN,
        "cbc_lite_cnn": CBCPhaseLiteCNN,
    }
    if model_name not in model_classes:
        raise ValueError(f"Unknown model_name={model_name}. Expected one of {sorted(model_classes)}")
    return model_classes[model_name](image_size=image_size, output_dim=output_dim)


def count_parameters(model):
    """统计模型可训练参数量。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
