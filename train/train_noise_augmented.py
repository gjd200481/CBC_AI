"""
噪声增强训练 - 改善σ=0.002局部退化问题

策略：
1. 训练时动态添加随机噪声（σ=0~0.005范围）
2. 使用课程学习：前期干净数据，后期逐渐增加噪声
3. 对比三种策略：无噪声、固定噪声、动态噪声
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train.models import build_phase_model, count_parameters
from train.phase_metrics import phase_rmse_from_sin_cos, decode_sin_cos, wrap_phase_error
from train.physics_loss import SevenBeamFourierOptics


class NoisyMultiPlaneDataset(Dataset):
    """支持动态噪声增强的多平面数据集"""

    def __init__(self, images, labels, noise_mode='none', noise_sigma_range=(0, 0.005),
                 transform=None):
        """
        noise_mode: 'none', 'fixed', 'dynamic', 'curriculum'
        noise_sigma_range: (min_sigma, max_sigma)
        """
        self.images = torch.FloatTensor(images)
        self.labels = torch.FloatTensor(labels)
        self.noise_mode = noise_mode
        self.noise_sigma_range = noise_sigma_range
        self.transform = transform
        self.epoch = 0  # 用于课程学习

    def set_epoch(self, epoch):
        """设置当前epoch（用于课程学习）"""
        self.epoch = epoch

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].clone()
        label = self.labels[idx]

        # 添加噪声
        if self.noise_mode == 'fixed':
            # 固定噪声强度
            sigma = self.noise_sigma_range[1]
            if sigma > 0:
                noise = torch.randn_like(image) * sigma
                image = image + noise

        elif self.noise_mode == 'dynamic':
            # 动态随机噪声
            sigma = np.random.uniform(*self.noise_sigma_range)
            if sigma > 0:
                noise = torch.randn_like(image) * sigma
                image = image + noise

        elif self.noise_mode == 'curriculum':
            # 课程学习：逐渐增加噪声
            # 前10个epoch无噪声，之后线性增加到最大值
            if self.epoch < 10:
                sigma = 0
            else:
                progress = min((self.epoch - 10) / 20, 1.0)  # 20个epoch内增加到最大
                sigma = self.noise_sigma_range[0] + progress * (
                    self.noise_sigma_range[1] - self.noise_sigma_range[0]
                )

            if sigma > 0:
                noise = torch.randn_like(image) * sigma
                image = image + noise

        return image, label


def train_one_epoch(model, dataloader, criterion, optimizer, device, physics_loss_fn=None,
                   lambda_phy=0.05, lambda_comp=0.5, epoch=0):
    """训练一个epoch"""
    model.train()

    # 如果是课程学习，更新epoch
    if hasattr(dataloader.dataset, 'set_epoch'):
        dataloader.dataset.set_epoch(epoch)

    total_loss = 0
    total_phase_loss = 0
    total_phy_loss = 0
    total_comp_loss = 0

    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 相位损失
        phase_loss = criterion(outputs, labels)
        total_loss_batch = phase_loss

        # 物理一致性损失
        phy_loss = 0
        if physics_loss_fn is not None and lambda_phy > 0:
            phy_loss = physics_loss_fn(outputs, images[:, 0:1])  # 使用焦平面
            total_loss_batch = total_loss_batch + lambda_phy * phy_loss

        # 补偿质量损失
        comp_loss = 0
        if physics_loss_fn is not None and lambda_comp > 0:
            with torch.no_grad():
                pred_phases = decode_sin_cos(outputs.cpu().numpy())

            # 计算补偿后的Strehl（简化版，实际训练中使用完整版）
            strehl_batch = []
            for i in range(len(pred_phases)):
                # 这里简化处理，实际应该用完整的补偿质量计算
                phase_var = np.var(pred_phases[i])
                approx_strehl = np.exp(-phase_var)
                strehl_batch.append(approx_strehl)

            comp_loss = -torch.tensor(np.mean(strehl_batch), device=device)
            total_loss_batch = total_loss_batch + lambda_comp * comp_loss

        # 反向传播
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()
        total_phase_loss += phase_loss.item()
        if isinstance(phy_loss, torch.Tensor):
            total_phy_loss += phy_loss.item()
        if isinstance(comp_loss, torch.Tensor):
            total_comp_loss += comp_loss.item()

    n_batches = len(dataloader)
    return {
        'total': total_loss / n_batches,
        'phase': total_phase_loss / n_batches,
        'phy': total_phy_loss / n_batches,
        'comp': total_comp_loss / n_batches
    }


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算RMSE
    rmse = phase_rmse_from_sin_cos(all_preds, all_labels)

    return rmse, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='噪声增强训练')

    # 数据参数
    parser.add_argument('--image-path', type=str,
                       default='dataset/seven_beam/multiplane_0_-0.07/images_multiplane_7cm.npy')
    parser.add_argument('--label-path', type=str,
                       default='dataset/seven_beam/multiplane_0_-0.07/labels_multiplane_7cm.npy')
    parser.add_argument('--max-samples', type=int, default=10000)

    # 噪声增强参数
    parser.add_argument('--noise-mode', type=str, default='dynamic',
                       choices=['none', 'fixed', 'dynamic', 'curriculum'],
                       help='噪声增强模式')
    parser.add_argument('--noise-sigma-min', type=float, default=0.0)
    parser.add_argument('--noise-sigma-max', type=float, default=0.005,
                       help='最大噪声强度（针对σ=0.002退化）')

    # 训练参数
    parser.add_argument('--model-name', type=str, default='dual_plane_fusion_cnn')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--lambda-phy', type=float, default=0.05)
    parser.add_argument('--lambda-comp', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=20260613)

    # 输出参数
    parser.add_argument('--output-prefix', type=str, default='cycle44_noise_aug')
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*70}")
    print(f"噪声增强训练 - 应对σ=0.002局部退化")
    print(f"{'='*70}\n")
    print(f"噪声模式: {args.noise_mode}")
    print(f"噪声范围: [{args.noise_sigma_min}, {args.noise_sigma_max}]")
    print(f"设备: {device}\n")

    # 加载数据
    print("加载数据...")
    images = np.load(args.image_path)[:args.max_samples]
    labels = np.load(args.label_path)[:args.max_samples]

    print(f"  图像形状: {images.shape}")
    print(f"  标签形状: {labels.shape}")

    # 数据划分
    n_total = len(images)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    train_images = images[:n_train]
    train_labels = labels[:n_train]
    val_images = images[n_train:n_train+n_val]
    val_labels = labels[n_train:n_train+n_val]
    test_images = images[n_train+n_val:]
    test_labels = labels[n_train+n_val:]

    print(f"  训练集: {len(train_images)}")
    print(f"  验证集: {len(val_images)}")
    print(f"  测试集: {len(test_images)}\n")

    # 创建数据集（训练集使用噪声增强）
    train_dataset = NoisyMultiPlaneDataset(
        train_images, train_labels,
        noise_mode=args.noise_mode,
        noise_sigma_range=(args.noise_sigma_min, args.noise_sigma_max)
    )
    val_dataset = NoisyMultiPlaneDataset(val_images, val_labels, noise_mode='none')
    test_dataset = NoisyMultiPlaneDataset(test_images, test_labels, noise_mode='none')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    print("创建模型...")
    model = build_phase_model(
        args.model_name,
        image_size=images.shape[-1],
        output_dim=12,
        input_channels=images.shape[1] if len(images.shape) == 4 else 1
    ).to(device)

    n_params = count_parameters(model)
    print(f"  模型: {args.model_name}")
    print(f"  参数量: {n_params/1e6:.2f}M\n")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 物理损失
    physics_loss_fn = SevenBeamFourierOptics(
        num_points=256, window_size=0.01,
        waist=0.0005, beam_distance=0.0015,
        crop_size=images.shape[-1]
    ).to(device) if args.lambda_phy > 0 else None

    # 训练循环
    print("开始训练...\n")
    history = []
    best_val_rmse = float('inf')
    best_model_path = None

    for epoch in range(args.epochs):
        # 训练
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            physics_loss_fn, args.lambda_phy, args.lambda_comp, epoch
        )

        # 验证
        val_rmse, _, _ = evaluate(model, val_loader, device)

        # 学习率调度
        scheduler.step()

        # 记录
        history.append({
            'epoch': epoch + 1,
            'train_total': train_losses['total'],
            'train_phase': train_losses['phase'],
            'train_phy': train_losses['phy'],
            'train_comp': train_losses['comp'],
            'val_rmse': val_rmse,
            'lr': optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_losses['total']:.4f}, "
              f"Val RMSE: {val_rmse:.4f}")

        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_path = REPO_ROOT / f"models/{args.output_prefix}_{args.noise_mode}_best.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': args.model_name,
                'epoch': epoch + 1,
                'val_rmse': val_rmse,
                'args': vars(args)
            }, best_model_path)

    # 测试最佳模型
    print(f"\n加载最佳模型进行测试...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_rmse, test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"测试集 RMSE: {test_rmse:.4f} rad\n")

    # 保存结果
    output_dir = REPO_ROOT / "result" / "metrics"

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_csv = output_dir / f"{args.output_prefix}_{args.noise_mode}_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"训练历史保存至: {history_csv}")

    # 保存测试结果
    summary = {
        'noise_mode': args.noise_mode,
        'noise_sigma_range': f"[{args.noise_sigma_min}, {args.noise_sigma_max}]",
        'best_val_rmse': best_val_rmse,
        'test_rmse': test_rmse,
        'parameters': n_params,
        'model_path': str(best_model_path)
    }

    summary_df = pd.DataFrame([summary])
    summary_csv = output_dir / f"{args.output_prefix}_{args.noise_mode}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"测试结果保存至: {summary_csv}")

    print(f"\n{'='*70}")
    print(f"训练完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
