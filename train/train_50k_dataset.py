"""
50k大规模数据集训练脚本

特点：
1. 支持大规模数据训练
2. 梯度累积（模拟更大batch size）
3. 混合精度训练（节省显存）
4. 定期保存checkpoint
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from tqdm import tqdm
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train.models import build_phase_model, count_parameters
from train.phase_metrics import phase_rmse_from_sin_cos
from train.physics_loss import FarFieldConsistencyLoss, SevenBeamFourierOptics


class MultiPlaneDataset(Dataset):
    """多平面数据集"""

    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                   physics_loss_fn=None, lambda_phy=0.05, lambda_comp=0.5,
                   use_amp=True, accumulation_steps=1, scaler=None):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_phase_loss = 0
    total_phy_loss = 0

    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)

        # 混合精度训练
        with autocast(enabled=use_amp):
            # 前向传播
            outputs = model(images)

            # 相位损失
            phase_loss = criterion(outputs, labels)
            total_loss_batch = phase_loss

            # 物理一致性损失
            if physics_loss_fn is not None and lambda_phy > 0:
                phy_loss = physics_loss_fn(outputs, images[:, 0:1])
                total_loss_batch = total_loss_batch + lambda_phy * phy_loss
            else:
                phy_loss = 0

            # 梯度累积
            total_loss_batch = total_loss_batch / accumulation_steps

        # 反向传播
        if use_amp and scaler is not None:
            scaler.scale(total_loss_batch).backward()
        else:
            total_loss_batch.backward()

        # 梯度更新
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += total_loss_batch.item() * accumulation_steps
        total_phase_loss += phase_loss.item()
        if isinstance(phy_loss, torch.Tensor):
            total_phy_loss += phy_loss.item()

    n_batches = len(dataloader)
    return {
        'total': total_loss / n_batches,
        'phase': total_phase_loss / n_batches,
        'phy': total_phy_loss / n_batches
    }


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    rmse = phase_rmse_from_sin_cos(all_preds, all_labels)

    return rmse


def main():
    parser = argparse.ArgumentParser(description='50k大规模数据集训练')

    # 数据参数
    parser.add_argument('--image-path', type=str,
                       default='dataset/seven_beam/multiplane_50k/images_multiplane_50k.npy')
    parser.add_argument('--label-path', type=str,
                       default='dataset/seven_beam/multiplane_50k/labels_multiplane_50k.npy')

    # 模型参数
    parser.add_argument('--model-name', type=str, default='dual_plane_fusion_cnn')
    parser.add_argument('--pretrained-model', type=str, default=None,
                       help='预训练模型路径（可选）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32,
                       help='实际batch size')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                       help='梯度累积步数（有效batch size = batch_size * accumulation_steps）')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--lambda-phy', type=float, default=0.05)
    parser.add_argument('--lambda-comp', type=float, default=0.5)
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='使用混合精度训练')

    # 数据加载参数
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pin-memory', action='store_true', default=True)

    # 输出参数
    parser.add_argument('--output-prefix', type=str, default='cycle45_50k')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='每N个epoch保存一次checkpoint')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=20260613)

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
    print(f"50k大规模数据集训练")
    print(f"{'='*70}\n")
    print(f"设备: {device}")
    print(f"混合精度: {args.use_amp}")
    print(f"有效batch size: {args.batch_size * args.accumulation_steps}")
    print()

    # 加载数据
    print("加载数据...")
    images = np.load(args.image_path)
    labels = np.load(args.label_path)

    print(f"  图像形状: {images.shape}")
    print(f"  标签形状: {labels.shape}")
    print(f"  数据大小: {images.nbytes / 1e9:.2f} GB")

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

    # 创建数据集
    train_dataset = MultiPlaneDataset(train_images, train_labels)
    val_dataset = MultiPlaneDataset(val_images, val_labels)
    test_dataset = MultiPlaneDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             pin_memory=args.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory)

    # 创建模型
    print("创建模型...")
    model = build_phase_model(
        args.model_name,
        image_size=images.shape[-1],
        output_dim=12,
        in_channels=images.shape[1] if len(images.shape) == 4 else 1
    ).to(device)

    # 加载预训练权重（可选）
    if args.pretrained_model:
        print(f"加载预训练模型: {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    n_params = count_parameters(model)
    print(f"  模型: {args.model_name}")
    print(f"  参数量: {n_params/1e6:.2f}M\n")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 混合精度训练
    scaler = GradScaler() if args.use_amp else None

    # 物理损失
    if args.lambda_phy > 0:
        optics_model = SevenBeamFourierOptics(
            num_points=256, window_size=0.01,
            waist=0.0005, beam_distance=0.0015,
            crop_size=images.shape[-1]
        ).to(device)
        physics_loss_fn = FarFieldConsistencyLoss(optics_model=optics_model).to(device)
    else:
        physics_loss_fn = None

    # 训练循环
    print("开始训练...\n")
    history = []
    best_val_rmse = float('inf')
    best_model_path = None

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # 训练
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            physics_loss_fn, args.lambda_phy, args.lambda_comp,
            args.use_amp, args.accumulation_steps, scaler
        )

        # 验证
        val_rmse = evaluate(model, val_loader, device)

        # 学习率调度
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 记录
        history.append({
            'epoch': epoch + 1,
            'train_total': train_losses['total'],
            'train_phase': train_losses['phase'],
            'train_phy': train_losses['phy'],
            'val_rmse': val_rmse,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        })

        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_losses['total']:.4f}, "
              f"Val RMSE: {val_rmse:.4f}")

        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_path = REPO_ROOT / f"models/{args.output_prefix}_best.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': args.model_name,
                'epoch': epoch + 1,
                'val_rmse': val_rmse,
                'args': vars(args)
            }, best_model_path)

        # 定期保存checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = REPO_ROOT / f"models/{args.output_prefix}_epoch{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': args.model_name,
                'epoch': epoch + 1,
                'val_rmse': val_rmse,
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    total_time = time.time() - start_time

    # 测试最佳模型
    print(f"\n加载最佳模型进行测试...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_rmse = evaluate(model, test_loader, device)
    print(f"测试集 RMSE: {test_rmse:.4f} rad\n")

    # 保存结果
    output_dir = REPO_ROOT / "result" / "metrics"

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_csv = output_dir / f"{args.output_prefix}_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"训练历史保存至: {history_csv}")

    # 保存摘要
    summary = {
        'dataset_size': n_total,
        'train_size': n_train,
        'val_size': n_val,
        'test_size': len(test_images),
        'best_val_rmse': best_val_rmse,
        'test_rmse': test_rmse,
        'total_time_hours': total_time / 3600,
        'parameters': n_params,
        'model_path': str(best_model_path)
    }

    summary_df = pd.DataFrame([summary])
    summary_csv = output_dir / f"{args.output_prefix}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"训练摘要保存至: {summary_csv}")

    print(f"\n{'='*70}")
    print(f"50k训练完成！")
    print(f"总耗时: {total_time/3600:.2f} 小时")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
