"""七光束物理约束 + 补偿质量损失训练脚本。

新增补偿质量损失，直接优化Strehl比和主瓣能量占比。
总损失 = L_phase + lambda_phy * L_farfield + lambda_comp * L_compensation
"""
import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.data_utils import build_dataloaders
from train.models import build_phase_model, count_parameters
from train.phase_metrics import build_phase_loss, phase_metrics_from_sin_cos
from train.physics_loss import (
    CompensationQualityLoss,
    FarFieldConsistencyLoss,
    SevenBeamFourierOptics,
)
from train.train_seven_beam_baseline import channel_rmse_from_sin_cos


def train_one_epoch(model, loader, optimizer, phase_loss_fn, farfield_loss_fn, 
                    comp_loss_fn, lambda_phy, lambda_comp, device):
    model.train()
    totals = {'phase': 0.0, 'farfield': 0.0, 'comp': 0.0, 'total': 0.0, 'samples': 0}
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        
        loss_phase = phase_loss_fn(preds, labels)
        loss_farfield = farfield_loss_fn(preds, images)
        loss_comp = comp_loss_fn(preds, labels)
        loss_total = loss_phase + lambda_phy * loss_farfield + lambda_comp * loss_comp
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        batch_size = images.size(0)
        totals['phase'] += loss_phase.item() * batch_size
        totals['farfield'] += loss_farfield.item() * batch_size
        totals['comp'] += loss_comp.item() * batch_size
        totals['total'] += loss_total.item() * batch_size
        totals['samples'] += batch_size
    
    return {k: v / totals['samples'] if k != 'samples' else v for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, phase_loss_fn, farfield_loss_fn, comp_loss_fn, 
             lambda_phy, lambda_comp, device):
    model.eval()
    totals = {'phase': 0.0, 'farfield': 0.0, 'comp': 0.0, 'total': 0.0, 'samples': 0}
    all_preds, all_labels = [], []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        
        loss_phase = phase_loss_fn(preds, labels)
        loss_farfield = farfield_loss_fn(preds, images)
        loss_comp = comp_loss_fn(preds, labels)
        loss_total = loss_phase + lambda_phy * loss_farfield + lambda_comp * loss_comp
        
        batch_size = images.size(0)
        totals['phase'] += loss_phase.item() * batch_size
        totals['farfield'] += loss_farfield.item() * batch_size
        totals['comp'] += loss_comp.item() * batch_size
        totals['total'] += loss_total.item() * batch_size
        totals['samples'] += batch_size
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = phase_metrics_from_sin_cos(all_preds, all_labels)
    
    return {
        'phase_loss': totals['phase'] / totals['samples'],
        'farfield_loss': totals['farfield'] / totals['samples'],
        'comp_loss': totals['comp'] / totals['samples'],
        'total_loss': totals['total'] / totals['samples'],
        'rmse_rad': metrics['rmse_rad'],
        'rmse_deg': metrics['rmse_deg'],
        'mae_rad': metrics['mae_rad'],
        'mae_deg': metrics['mae_deg'],
    }, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=Path, 
                       default=REPO_ROOT / "dataset/seven_beam/main_static_10k/images_main_clean_seven_beam_10k.npy")
    parser.add_argument("--label-path", type=Path,
                       default=REPO_ROOT / "dataset/seven_beam/main_static_10k/labels_main_clean_seven_beam_10k.npy")
    parser.add_argument("--model-path", type=Path,
                       default=REPO_ROOT / "models/cycle29_comp_loss_10k.pth")
    parser.add_argument("--history-csv", type=Path,
                       default=REPO_ROOT / "result/metrics/cycle29_comp_loss_10k_history.csv")
    parser.add_argument("--summary-csv", type=Path,
                       default=REPO_ROOT / "result/metrics/cycle29_comp_loss_10k_summary.csv")
    parser.add_argument("--model-name", default="residual_cnn")
    parser.add_argument("--lambda-phy", type=float, default=0.05)
    parser.add_argument("--lambda-comp", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")
    
    loaders = build_dataloaders(args.image_path, args.label_path, args.batch_size,
                                train_ratio=0.7, val_ratio=0.15, seed=args.seed,
                                expected_size=(160, 160), num_workers=args.num_workers)
    
    model = build_phase_model(args.model_name, 160, 12).to(device)
    print(f"Model: {args.model_name}, Parameters: {count_parameters(model)}")
    
    optics = SevenBeamFourierOptics(crop_size=160).to(device)
    phase_loss_fn = build_phase_loss("mse")
    farfield_loss_fn = FarFieldConsistencyLoss(optics)
    comp_loss_fn = CompensationQualityLoss(optics, main_lobe_radius=3)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print(f"lambda_phy={args.lambda_phy}, lambda_comp={args.lambda_comp}")
    
    history = []
    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders['train'], optimizer, phase_loss_fn,
                                       farfield_loss_fn, comp_loss_fn, args.lambda_phy, 
                                       args.lambda_comp, device)
        val_metrics, _, _ = evaluate(model, loaders['val'], phase_loss_fn, farfield_loss_fn,
                                     comp_loss_fn, args.lambda_phy, args.lambda_comp, device)
        scheduler.step()
        
        history.append({
            'epoch': epoch,
            'train_total': train_metrics['total'],
            'train_phase': train_metrics['phase'],
            'train_farfield': train_metrics['farfield'],
            'train_comp': train_metrics['comp'],
            'val_total': val_metrics['total_loss'],
            'val_rmse_rad': val_metrics['rmse_rad'],
        })
        
        if val_metrics['rmse_rad'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse_rad']
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        
        print(f"Epoch {epoch:03d} | train_total={train_metrics['total']:.6f} | "
              f"train_comp={train_metrics['comp']:.6f} | val_rmse={val_metrics['rmse_rad']:.6f} rad")
    
    # 测试最佳模型
    model.load_state_dict(best_state)
    test_metrics, preds, labels = evaluate(model, loaders['test'], phase_loss_fn, farfield_loss_fn,
                                          comp_loss_fn, args.lambda_phy, args.lambda_comp, device)
    channel_rmse = channel_rmse_from_sin_cos(preds, labels)
    
    print(f"\nBest epoch: {best_epoch}")
    print(f"Test RMSE: {test_metrics['rmse_rad']:.6f} rad ({test_metrics['rmse_deg']:.2f} deg)")
    print(f"Test MAE: {test_metrics['mae_rad']:.6f} rad")
    
    # 保存
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': best_state,
        'test_rmse_rad': test_metrics['rmse_rad'],
        'best_epoch': best_epoch,
        'lambda_phy': args.lambda_phy,
        'lambda_comp': args.lambda_comp,
        'model_name': args.model_name,
        'channel_rmse_rad': channel_rmse.tolist(),
    }, args.model_path)
    print(f"Model saved to: {args.model_path}")
    
    args.history_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.history_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"History saved to: {args.history_csv}")


if __name__ == "__main__":
    main()
