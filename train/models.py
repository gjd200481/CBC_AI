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


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        activation_layer=nn.ReLU,
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation_layer(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, squeeze_factor=4):
        super().__init__()
        squeeze_channels = max(8, channels // squeeze_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, kernel_size=1)
        self.scale_activation = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


class MobileNetV3Block(nn.Module):
    def __init__(
        self,
        input_channels,
        expanded_channels,
        output_channels,
        kernel_size,
        stride,
        use_se,
        activation_layer,
    ):
        super().__init__()
        layers = []
        if expanded_channels != input_channels:
            layers.append(
                ConvBNActivation(
                    input_channels,
                    expanded_channels,
                    kernel_size=1,
                    activation_layer=activation_layer,
                )
            )
        layers.append(
            ConvBNActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=expanded_channels,
                activation_layer=activation_layer,
            )
        )
        if use_se:
            layers.append(SqueezeExcitation(expanded_channels))
        layers.extend(
            [
                nn.Conv2d(expanded_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and input_channels == output_channels

    def forward(self, x):
        result = self.block(x)
        if self.use_residual:
            result = result + x
        return result


class MobileNetV3SmallPhaseCNN(nn.Module):
    """MobileNetV3-Small inspired phase regressor for seven-beam CBC."""

    def __init__(self, image_size=160, output_dim=2):
        super().__init__()
        del image_size

        relu = nn.ReLU
        hswish = nn.Hardswish
        self.features = nn.Sequential(
            ConvBNActivation(1, 16, kernel_size=3, stride=2, activation_layer=hswish),
            MobileNetV3Block(16, 16, 16, kernel_size=3, stride=2, use_se=True, activation_layer=relu),
            MobileNetV3Block(16, 72, 24, kernel_size=3, stride=2, use_se=False, activation_layer=relu),
            MobileNetV3Block(24, 88, 24, kernel_size=3, stride=1, use_se=False, activation_layer=relu),
            MobileNetV3Block(24, 96, 40, kernel_size=5, stride=2, use_se=True, activation_layer=hswish),
            MobileNetV3Block(40, 240, 40, kernel_size=5, stride=1, use_se=True, activation_layer=hswish),
            MobileNetV3Block(40, 240, 40, kernel_size=5, stride=1, use_se=True, activation_layer=hswish),
            MobileNetV3Block(40, 120, 48, kernel_size=5, stride=1, use_se=True, activation_layer=hswish),
            MobileNetV3Block(48, 144, 48, kernel_size=5, stride=1, use_se=True, activation_layer=hswish),
            MobileNetV3Block(48, 288, 96, kernel_size=5, stride=2, use_se=True, activation_layer=hswish),
            MobileNetV3Block(96, 576, 96, kernel_size=5, stride=1, use_se=True, activation_layer=hswish),
            MobileNetV3Block(96, 576, 96, kernel_size=5, stride=1, use_se=True, activation_layer=hswish),
            ConvBNActivation(96, 576, kernel_size=1, activation_layer=hswish),
        )
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


def build_phase_model(model_name, image_size=160, output_dim=2):
    """按名称构建相位反演网络，便于训练脚本做结构消融。"""
    model_classes = {
        "simple_cnn": SimplePhaseCNN,
        "wide_cnn": WidePhaseCNN,
        "residual_cnn": ResidualPhaseCNN,
        "mobilenetv3_small": MobileNetV3SmallPhaseCNN,
    }
    if model_name not in model_classes:
        raise ValueError(f"Unknown model_name={model_name}. Expected one of {sorted(model_classes)}")
    return model_classes[model_name](image_size=image_size, output_dim=output_dim)


def count_parameters(model):
    """统计模型可训练参数量。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
