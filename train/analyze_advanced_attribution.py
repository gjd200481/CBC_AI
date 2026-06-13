"""
高级解释性分析工具：Integrated Gradients + Grad-CAM

相比简单梯度，这些方法提供更准确的特征归因：
1. Integrated Gradients (IG): 沿路径积分，消除梯度饱和问题
2. Grad-CAM: 类激活映射，定位关键特征区域
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train.models import build_phase_model


class IntegratedGradients:
    """Integrated Gradients实现

    相比简单梯度的优势：
    1. 满足敏感性公理（sensitivity axiom）
    2. 满足实现不变性（implementation invariance）
    3. 消除梯度饱和导致的伪影
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def compute(self, input_tensor, target_channel, baseline=None, steps=50):
        """
        计算Integrated Gradients

        Args:
            input_tensor: [1, C, H, W]
            target_channel: 目标输出通道索引
            baseline: 基线输入（默认为全零）
            steps: 积分步数

        Returns:
            integrated_grads: [C, H, W]
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # 生成插值路径
        alphas = torch.linspace(0, 1, steps + 1, device=self.device)

        integrated_grads = torch.zeros_like(input_tensor)

        for alpha in alphas:
            # 插值输入
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)

            # 前向传播
            output = self.model(interpolated)
            target_output = output[0, target_channel]

            # 反向传播
            self.model.zero_grad()
            target_output.backward(retain_graph=True)

            # 累积梯度
            integrated_grads += interpolated.grad / (steps + 1)

        # 乘以输入差值
        integrated_grads = integrated_grads * (input_tensor - baseline)

        return integrated_grads[0].detach().cpu().numpy()


class GradCAM:
    """Grad-CAM实现

    优势：
    1. 可视化卷积层学到的特征
    2. 定位对预测最重要的空间区域
    3. 与网络结构无关
    """

    def __init__(self, model, target_layer_name, device='cpu'):
        """
        Args:
            model: 神经网络模型
            target_layer_name: 目标层名称（通常选择最后一个卷积层）
            device: 设备
        """
        self.model = model
        self.device = device
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        """注册前向和反向钩子"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 找到目标层
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break

        if target_layer is None:
            raise ValueError(f"Layer {self.target_layer_name} not found")

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def compute(self, input_tensor, target_channel):
        """
        计算Grad-CAM

        Args:
            input_tensor: [1, C, H, W]
            target_channel: 目标输出通道

        Returns:
            cam: [H, W] 热图
        """
        # 前向传播
        output = self.model(input_tensor)
        target_output = output[0, target_channel]

        # 反向传播
        self.model.zero_grad()
        target_output.backward(retain_graph=True)

        # 计算权重（全局平均池化梯度）
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 加权组合激活
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU
        cam = F.relu(cam)

        # 上采样到输入尺寸
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                           mode='bilinear', align_corners=False)

        # 归一化
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def analyze_sample(model, image, target_channels, device='cpu',
                  use_ig=True, use_gradcam=True, ig_steps=50):
    """
    分析单个样本

    Args:
        model: 模型
        image: [C, H, W] 或 [H, W]
        target_channels: 要分析的输出通道列表
        device: 设备
        use_ig: 是否使用Integrated Gradients
        use_gradcam: 是否使用Grad-CAM
        ig_steps: IG积分步数

    Returns:
        results: 字典，包含IG和Grad-CAM结果
    """
    # 准备输入
    if len(image.shape) == 2:
        image = image[np.newaxis, :]
    input_tensor = torch.FloatTensor(image).unsqueeze(0).to(device)

    results = {}

    # Integrated Gradients
    if use_ig:
        ig = IntegratedGradients(model, device)
        ig_results = {}

        for ch in target_channels:
            ig_attr = ig.compute(input_tensor, ch, steps=ig_steps)
            ig_results[f'channel_{ch}'] = ig_attr

        results['integrated_gradients'] = ig_results

    # Grad-CAM
    if use_gradcam:
        # 找到最后一个卷积层
        last_conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = name

        if last_conv_layer:
            gradcam = GradCAM(model, last_conv_layer, device)
            gradcam_results = {}

            for ch in target_channels:
                cam = gradcam.compute(input_tensor, ch)
                gradcam_results[f'channel_{ch}'] = cam

            results['grad_cam'] = gradcam_results
            results['target_layer'] = last_conv_layer
        else:
            print("Warning: No convolutional layer found for Grad-CAM")

    return results


def visualize_attribution(image, ig_attr, gradcam, channel_idx,
                         output_path=None, title=None):
    """
    可视化归因结果

    Args:
        image: 原始图像 [C, H, W]
        ig_attr: IG归因 [C, H, W]
        gradcam: Grad-CAM热图 [H, W]
        channel_idx: 通道索引
        output_path: 保存路径
        title: 标题
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 选择显示的平面（如果是多平面）
    if len(image.shape) == 3 and image.shape[0] > 1:
        focal_plane = image[0]
        befocal_plane = image[1]
    else:
        focal_plane = image[0] if len(image.shape) == 3 else image
        befocal_plane = None

    # (a) 原始图像（焦平面）
    ax = axes[0]
    im0 = ax.imshow(focal_plane, cmap='gray', origin='lower')
    ax.set_title(f'(a) Focal Plane Image', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im0, ax=ax, fraction=0.046)

    # (b) Integrated Gradients
    ax = axes[1]
    ig_magnitude = np.sqrt(np.sum(ig_attr**2, axis=0))
    im1 = ax.imshow(ig_magnitude, cmap='jet', origin='lower')
    ax.set_title(f'(b) Integrated Gradients\nChannel {channel_idx}', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im1, ax=ax, fraction=0.046)

    # (c) Grad-CAM
    ax = axes[2]
    im2 = ax.imshow(focal_plane, cmap='gray', origin='lower', alpha=0.5)
    im3 = ax.imshow(gradcam, cmap='jet', origin='lower', alpha=0.5)
    ax.set_title(f'(c) Grad-CAM Overlay\nChannel {channel_idx}', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im3, ax=ax, fraction=0.046)

    # (d) Grad-CAM热图
    ax = axes[3]
    im4 = ax.imshow(gradcam, cmap='hot', origin='lower')
    ax.set_title(f'(d) Grad-CAM Heatmap\nChannel {channel_idx}', fontweight='bold')
    ax.axis('off')
    plt.colorbar(im4, ax=ax, fraction=0.046)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='高级解释性分析')

    # 数据参数
    parser.add_argument('--image-path', type=str,
                       default='dataset/seven_beam/multiplane_0_-0.07/images_multiplane_7cm.npy')
    parser.add_argument('--model-path', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='分析样本数')
    parser.add_argument('--target-channels', type=int, nargs='+',
                       default=[0, 2, 4], help='目标输出通道（相位索引）')

    # 方法参数
    parser.add_argument('--use-ig', action='store_true', default=True,
                       help='使用Integrated Gradients')
    parser.add_argument('--use-gradcam', action='store_true', default=True,
                       help='使用Grad-CAM')
    parser.add_argument('--ig-steps', type=int, default=50,
                       help='IG积分步数')

    # 输出参数
    parser.add_argument('--output-prefix', type=str, default='advanced_attribution')
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*70}")
    print(f"高级解释性分析：Integrated Gradients + Grad-CAM")
    print(f"{'='*70}\n")
    print(f"设备: {device}")
    print(f"模型: {args.model_path}")
    print(f"分析方法: ", end="")
    methods = []
    if args.use_ig:
        methods.append("Integrated Gradients")
    if args.use_gradcam:
        methods.append("Grad-CAM")
    print(", ".join(methods))
    print()

    # 加载数据
    print("加载数据...")
    images = np.load(args.image_path)[:args.num_samples]
    print(f"  图像形状: {images.shape}\n")

    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(args.model_path, map_location=device)

    model = build_phase_model(
        checkpoint['model_name'],
        image_size=images.shape[-1],
        output_dim=12,
        in_channels=images.shape[1] if len(images.shape) == 4 else 1
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  模型类型: {checkpoint['model_name']}\n")

    # 创建输出目录
    output_dir = REPO_ROOT / "result" / "figures" / args.output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    # 分析每个样本
    print(f"开始分析 {args.num_samples} 个样本...")

    summary_data = []

    for i in tqdm(range(args.num_samples)):
        image = images[i]

        # 对每个目标通道进行分析
        for ch in args.target_channels:
            # IG和Grad-CAM对应的是sin/cos输出，需要转换到相位通道
            # sin(phi_i) 在索引 2*i, cos(phi_i) 在索引 2*i+1
            sin_ch = 2 * ch
            cos_ch = 2 * ch + 1

            # 分析sin通道
            results_sin = analyze_sample(
                model, image, [sin_ch], device,
                use_ig=args.use_ig,
                use_gradcam=args.use_gradcam,
                ig_steps=args.ig_steps
            )

            # 分析cos通道
            results_cos = analyze_sample(
                model, image, [cos_ch], device,
                use_ig=args.use_ig,
                use_gradcam=args.use_gradcam,
                ig_steps=args.ig_steps
            )

            # 可视化sin通道
            if args.use_ig and args.use_gradcam:
                ig_attr_sin = results_sin['integrated_gradients'][f'channel_{sin_ch}']
                gradcam_sin = results_sin['grad_cam'][f'channel_{sin_ch}']

                output_path = output_dir / f"sample{i}_phase{ch}_sin.png"
                visualize_attribution(
                    image, ig_attr_sin, gradcam_sin, sin_ch,
                    output_path=output_path,
                    title=f"Sample {i}, Phase Channel {ch} (sin)"
                )

                # 统计数据
                ig_energy = np.sum(ig_attr_sin**2)
                gradcam_peak = gradcam.max()

                summary_data.append({
                    'sample': i,
                    'phase_channel': ch,
                    'component': 'sin',
                    'ig_energy': ig_energy,
                    'gradcam_peak': gradcam_peak
                })

    print(f"\n分析完成！输出目录: {output_dir}")

    # 保存统计数据
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = REPO_ROOT / "result" / "metrics" / f"{args.output_prefix}_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"统计数据保存至: {summary_csv}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
