import torch.nn as nn


class SimplePhaseCNN(nn.Module):
    """用于双光束相位反演的基础 CNN。

    输入为单通道远场光强图像，默认尺寸 160 x 160。
    输出为 [sin(phi), cos(phi)]。后续多光束扩展时，可通过 output_dim 调整输出维度。
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
