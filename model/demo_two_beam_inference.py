import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """训练时使用的 CNN 结构，必须和 train/train_two_beam_cnn.py 中保持一致。"""

    def __init__(self):
        super().__init__()

        # ======================
        # 卷积特征提取部分
        # ======================
        # 输入图像尺寸为 160 x 160，通道数为 1。
        # 每经过一次 MaxPool2d(2)，图像宽高都会减半：
        # 160 -> 80 -> 40 -> 20。
        self.features = nn.Sequential(
            # 第 1 层卷积：输入 1 个灰度通道，输出 16 个特征通道。
            # padding=1 可以让 3x3 卷积前后的图像尺寸保持不变。
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第 2 层卷积：通道数从 16 增加到 32。
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第 3 层卷积：通道数从 32 增加到 64。
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # ======================
        # 回归输出部分
        # ======================
        # 三次池化后，特征图尺寸为 64 x 20 x 20。
        # Flatten 后送入全连接层，最终输出两个数：[sin(phi), cos(phi)]。
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20 * 20, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """前向传播：输入图像 -> 卷积特征 -> 全连接回归 -> 相位的 sin/cos。"""
        x = self.features(x)
        x = self.regressor(x)
        return x


def load_model(model_path, device):
    """加载训练好的模型权重，并切换到 eval 推理模式。"""
    # torch.load 读回 train/train_two_beam_cnn.py 中 torch.save 保存的字典。
    # map_location=device 可以让模型在当前机器可用的 CPU/GPU 上加载。
    checkpoint = torch.load(model_path, map_location=device)

    # 先创建同样结构的模型，再把保存好的参数填进去。
    model = SimpleCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # eval() 会关闭 dropout/batchnorm 的训练行为。
    # 虽然这个模型里没有 dropout/batchnorm，但推理时保持这个习惯是好的。
    model.eval()

    return model, checkpoint


def predict_phi(model, image, device):
    """输入一张 160 x 160 的远场图，输出预测相位。"""
    # image 来自 images.npy，通常形状是 [height, width]。
    image_tensor = torch.tensor(image, dtype=torch.float32)

    # PyTorch 的 Conv2d 输入格式必须是 [batch, channel, height, width]。
    # 单张灰度图需要补两个维度：
    # [160, 160] -> [1, 1, 160, 160]。
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)

    # 推理不需要计算梯度，用 no_grad 可以减少显存/内存占用并加快速度。
    with torch.no_grad():
        pred = model(image_tensor)[0].cpu().numpy()

    # 模型输出不是直接的 phi，而是 [sin(phi), cos(phi)]。
    # 用 arctan2(sin, cos) 可以把两个分量还原成相位角。
    pred_sin = pred[0]
    pred_cos = pred[1]
    pred_phi = np.arctan2(pred_sin, pred_cos)

    return pred_sin, pred_cos, pred_phi


def main():
    # 当前文件在 CBC_AI/model/ 下。
    # parents[1] 表示回到项目根目录 CBC_AI，方便构造默认路径。
    repo_root = Path(__file__).resolve().parents[1]

    # 命令行参数允许你替换模型、数据路径或选择不同样本。
    parser = argparse.ArgumentParser(
        description="Demo: load two_beam_cnn.pth and predict phase from one image."
    )
    parser.add_argument(
        "--model-path",
        default=repo_root / "models" / "two_beam_cnn.pth",
        type=Path,
        help="训练后保存的模型文件路径。"
    )
    parser.add_argument(
        "--image-path",
        default=repo_root / "dataset" / "two_beam" / "images.npy",
        type=Path,
        help="待预测图像数据 .npy 文件路径。"
    )
    parser.add_argument(
        "--label-path",
        default=repo_root / "dataset" / "two_beam" / "labels.npy",
        type=Path,
        help="标签 .npy 文件路径；只用于 demo 中对比真实相位。"
    )
    parser.add_argument(
        "--index",
        default=0,
        type=int,
        help="从 images.npy 中选择第几张图做预测。"
    )
    args = parser.parse_args()

    # 训练脚本会把模型保存到 models/two_beam_cnn.pth。
    # 如果这里找不到，说明需要先运行 train/train_two_beam_cnn.py。
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{args.model_path}\n"
            "请先运行 train/train_two_beam_cnn.py 生成 models/two_beam_cnn.pth"
        )

    if not args.image_path.exists():
        raise FileNotFoundError(f"找不到图像文件：{args.image_path}")

    # 优先使用 CUDA GPU；如果没有 GPU，则自动使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 加载模型和测试图像。
    model, checkpoint = load_model(args.model_path, device)
    images = np.load(args.image_path)

    # 取出指定编号的单张图像，进行相位预测。
    image = images[args.index]
    pred_sin, pred_cos, pred_phi = predict_phi(model, image, device)

    # 打印模型信息，确认加载的是哪个文件、输出格式是什么。
    print("\nModel loaded from:", args.model_path)
    print("Model class:", checkpoint.get("model_class", "SimpleCNN"))
    print("Output format:", checkpoint.get("output_format", "[sin(phi), cos(phi)]"))

    # 打印预测结果。
    # rad 是弧度，deg 是角度，二者只是单位不同。
    print("\nPrediction:")
    print("Index:", args.index)
    print("Pred sin(phi):", pred_sin)
    print("Pred cos(phi):", pred_cos)
    print("Pred phi(rad):", pred_phi)
    print("Pred phi(deg):", np.degrees(pred_phi))

    # 如果标签文件存在，就顺便读出真实值，方便检查模型预测误差。
    if args.label_path.exists():
        labels = np.load(args.label_path)
        true = labels[args.index]
        true_phi = np.arctan2(true[0], true[1])

        print("\nGround truth:")
        print("True sin/cos:", true)
        print("True phi(rad):", true_phi)
        print("True phi(deg):", np.degrees(true_phi))
        print("Error(rad):", pred_phi - true_phi)
        print("Error(deg):", np.degrees(pred_phi - true_phi))


if __name__ == "__main__":
    main()
