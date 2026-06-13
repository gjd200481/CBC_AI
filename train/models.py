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


class DeepResidualPhaseCNN(nn.Module):
    """深度残差网络 + 通道注意力，用于高精度相位反演。"""
    
    def __init__(self, image_size=160, output_dim=2, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()
        )
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
        layers = [BasicResBlock(in_ch, out_ch, stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicResBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        att = self.channel_attention(x).unsqueeze(-1).unsqueeze(-1)
        x = x * att
        
        return self.fc(x)


class BasicResBlock(nn.Module):
    """基础残差块"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return nn.functional.relu(out)


class PlaneFeatureEncoder(nn.Module):
    """Compact residual encoder for one optical observation plane."""

    def __init__(self, input_channels=1, base_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        self.out_channels = base_channels * 8

    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
        layers = [BasicResBlock(in_ch, out_ch, stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicResBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class DualPlaneFusionPhaseCNN(nn.Module):
    """Two-branch focal/befocal fusion network for seven-beam phase inversion."""

    def __init__(self, image_size=160, output_dim=2, in_channels=2, base_channels=32):
        super().__init__()
        del image_size
        if in_channels < 2:
            raise ValueError("DualPlaneFusionPhaseCNN expects at least two input planes")

        self.focal_encoder = PlaneFeatureEncoder(input_channels=1, base_channels=base_channels)
        self.befocal_encoder = PlaneFeatureEncoder(input_channels=1, base_channels=base_channels)
        feature_channels = self.focal_encoder.out_channels

        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels * 2, feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels // 2, feature_channels),
            nn.Sigmoid(),
        )
        attention_hidden = max(16, feature_channels // 16)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels, attention_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(attention_hidden, feature_channels),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        focal = self.focal_encoder(x[:, 0:1])
        befocal = self.befocal_encoder(x[:, 1:2])
        gate = self.fusion_gate(torch.cat([focal, befocal], dim=1)).unsqueeze(-1).unsqueeze(-1)
        fused = gate * focal + (1.0 - gate) * befocal
        attention = self.channel_attention(fused).unsqueeze(-1).unsqueeze(-1)
        return self.fc(fused * attention)


class MultiPlanePhaseCNN(nn.Module):
    """多平面输入深度残差网络"""
    
    def __init__(self, image_size=160, output_dim=2, num_planes=2):
        super().__init__()
        # 使用DeepResidualPhaseCNN架构，但输入通道改为num_planes
        self.base = DeepResidualPhaseCNN(image_size, output_dim, in_channels=num_planes)
    
    def forward(self, x):
        # x: [B, num_planes, H, W]
        return self.base(x)


def build_phase_model(model_name, image_size=160, output_dim=2, in_channels=1):
    """按名称构建相位反演网络，便于训练脚本做结构消融。"""
    model_classes = {
        "simple_cnn": SimplePhaseCNN,
        "wide_cnn": WidePhaseCNN,
        "residual_cnn": ResidualPhaseCNN,
        "cbc_lite_cnn": CBCPhaseLiteCNN,
        "deep_residual_cnn": DeepResidualPhaseCNN,
        "dual_plane_fusion_cnn": DualPlaneFusionPhaseCNN,
        "multiplane_cnn": MultiPlanePhaseCNN,
    }
    if model_name not in model_classes:
        raise ValueError(f"Unknown model_name={model_name}. Expected one of {sorted(model_classes)}")
    
    if model_name in {"deep_residual_cnn", "dual_plane_fusion_cnn"}:
        return model_classes[model_name](image_size=image_size, output_dim=output_dim, in_channels=in_channels)
    else:
        return model_classes[model_name](image_size=image_size, output_dim=output_dim)


def count_parameters(model):
    """统计模型可训练参数量。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
