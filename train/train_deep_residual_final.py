"""深度残差网络 + 补偿质量损失 + 学习率衰减 + 数据增强的完整训练脚本。"""
import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.data_utils import build_dataloaders
from train.models import build_phase_model, count_parameters
from train.phase_metrics import build_phase_loss, phase_metrics_from_sin_cos, unit_circle_loss_from_sin_cos
from train.physics_loss import CompensationQualityLoss, FarFieldConsistencyLoss, SevenBeamFourierOptics
from train.train_seven_beam_baseline import channel_rmse_from_sin_cos


def train_epoch(model, loader, optimizer, phase_fn, farfield_fn, comp_fn, lam_phy, lam_comp, lam_unit, device):
    model.train()
    totals = {'phase': 0.0, 'farfield': 0.0, 'comp': 0.0, 'unit': 0.0, 'total': 0.0, 'n': 0}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        l_phase = phase_fn(preds, labels)
        l_far = farfield_fn(preds, images)
        l_comp = comp_fn(preds, labels)
        l_unit = unit_circle_loss_from_sin_cos(preds)
        loss = l_phase + lam_phy * l_far + lam_comp * l_comp + lam_unit * l_unit
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch = images.size(0)
        totals['phase'] += l_phase.item() * batch
        totals['farfield'] += l_far.item() * batch
        totals['comp'] += l_comp.item() * batch
        totals['unit'] += l_unit.item() * batch
        totals['total'] += loss.item() * batch
        totals['n'] += batch
    return {k: v / totals['n'] for k, v in totals.items() if k != 'n'}


@torch.no_grad()
def eval_epoch(model, loader, phase_fn, farfield_fn, comp_fn, lam_phy, lam_comp, lam_unit, device):
    model.eval()
    totals = {'phase': 0.0, 'farfield': 0.0, 'comp': 0.0, 'unit': 0.0, 'total': 0.0, 'n': 0}
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        l_phase = phase_fn(preds, labels)
        l_far = farfield_fn(preds, images)
        l_comp = comp_fn(preds, labels)
        l_unit = unit_circle_loss_from_sin_cos(preds)
        loss = l_phase + lam_phy * l_far + lam_comp * l_comp + lam_unit * l_unit
        batch = images.size(0)
        totals['phase'] += l_phase.item() * batch
        totals['farfield'] += l_far.item() * batch
        totals['comp'] += l_comp.item() * batch
        totals['unit'] += l_unit.item() * batch
        totals['total'] += loss.item() * batch
        totals['n'] += batch
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = phase_metrics_from_sin_cos(all_preds, all_labels)
    return {
        'phase_loss': totals['phase'] / totals['n'],
        'farfield_loss': totals['farfield'] / totals['n'],
        'comp_loss': totals['comp'] / totals['n'],
        'unit_loss': totals['unit'] / totals['n'],
        'total_loss': totals['total'] / totals['n'],
        'rmse_rad': metrics['rmse_rad'],
        'mae_rad': metrics['mae_rad'],
    }, all_preds, all_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-path", type=Path, default=REPO_ROOT / "dataset/seven_beam/main_static_10k/images_main_clean_seven_beam_10k.npy")
    p.add_argument("--label-path", type=Path, default=REPO_ROOT / "dataset/seven_beam/main_static_10k/labels_main_clean_seven_beam_10k.npy")
    p.add_argument("--model-path", type=Path, default=REPO_ROOT / "models/cycle30_deep_comp_10k.pth")
    p.add_argument("--history-csv", type=Path, default=REPO_ROOT / "result/metrics/cycle30_deep_comp_10k_history.csv")
    p.add_argument("--model-name", default="deep_residual_cnn")
    p.add_argument("--lambda-phy", type=float, default=0.05)
    p.add_argument("--lambda-comp", type=float, default=0.5)
    p.add_argument("--comp-warmup-epochs", type=int, default=0)
    p.add_argument("--lambda-unit", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20260615)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--augment-mode", choices=["none", "noise", "hex"], default="noise")
    args = p.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")
    
    augment_train = args.augment_mode != "none"
    loaders = build_dataloaders(
        args.image_path,
        args.label_path,
        args.batch_size,
        0.7,
        0.15,
        args.seed,
        (160, 160),
        args.num_workers,
        augment_train=augment_train,
        augment_mode=args.augment_mode if augment_train else "noise",
    )
    model = build_phase_model(args.model_name, 160, 12).to(device)
    params = count_parameters(model)
    print(f"Model: {args.model_name}, Parameters: {params}")
    
    optics = SevenBeamFourierOptics(crop_size=160).to(device)
    phase_fn = build_phase_loss("mse")
    farfield_fn = FarFieldConsistencyLoss(optics)
    comp_fn = CompensationQualityLoss(optics, main_lobe_radius=3)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print(
        f"lambda_phy={args.lambda_phy}, lambda_comp={args.lambda_comp}, "
        f"comp_warmup_epochs={args.comp_warmup_epochs}, lambda_unit={args.lambda_unit}, "
        f"augment_mode={args.augment_mode}"
    )
    
    history = []
    best_rmse = float('inf')
    best_epoch = 0
    best_state = None
    
    for epoch in range(1, args.epochs + 1):
        if args.comp_warmup_epochs > 0:
            comp_scale = min(1.0, epoch / args.comp_warmup_epochs)
        else:
            comp_scale = 1.0
        active_lambda_comp = args.lambda_comp * comp_scale

        train_m = train_epoch(
            model,
            loaders['train'],
            optimizer,
            phase_fn,
            farfield_fn,
            comp_fn,
            args.lambda_phy,
            active_lambda_comp,
            args.lambda_unit,
            device,
        )
        val_m, _, _ = eval_epoch(
            model,
            loaders['val'],
            phase_fn,
            farfield_fn,
            comp_fn,
            args.lambda_phy,
            active_lambda_comp,
            args.lambda_unit,
            device,
        )
        scheduler.step()
        
        history.append({
            'epoch': epoch,
            'lambda_comp_active': active_lambda_comp,
            'train_total': train_m['total'],
            'train_phase': train_m['phase'],
            'train_farfield': train_m['farfield'],
            'train_comp': train_m['comp'],
            'train_unit': train_m['unit'],
            'val_total': val_m['total_loss'],
            'val_rmse_rad': val_m['rmse_rad'],
            'val_mae_rad': val_m['mae_rad'],
            'val_unit': val_m['unit_loss'],
        })
        
        if val_m['rmse_rad'] < best_rmse:
            best_rmse = val_m['rmse_rad']
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        
        print(
            f"Epoch {epoch:03d} | train={train_m['total']:.6f} | "
            f"lambda_comp={active_lambda_comp:.4f} | val_rmse={val_m['rmse_rad']:.6f} | "
            f"unit={val_m['unit_loss']:.6f} | lr={scheduler.get_last_lr()[0]:.2e}"
        )
    
    model.load_state_dict(best_state)
    test_m, preds, labels = eval_epoch(
        model,
        loaders['test'],
        phase_fn,
        farfield_fn,
        comp_fn,
        args.lambda_phy,
        args.lambda_comp,
        args.lambda_unit,
        device,
    )
    channel_rmse = channel_rmse_from_sin_cos(preds, labels)
    
    print(f"\nBest epoch: {best_epoch}")
    print(f"Test RMSE: {test_m['rmse_rad']:.6f} rad")
    print(f"Test MAE: {test_m['mae_rad']:.6f} rad")
    for i, r in enumerate(channel_rmse):
        print(f"Channel {i+1} RMSE: {r:.6f} rad")
    
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': best_state, 'test_rmse_rad': test_m['rmse_rad'], 'best_epoch': best_epoch,
                'model_name': args.model_name, 'channel_rmse_rad': channel_rmse.tolist(),
                'parameters': params, 'augment_mode': args.augment_mode,
                'lambda_phy': args.lambda_phy, 'lambda_comp': args.lambda_comp,
                'comp_warmup_epochs': args.comp_warmup_epochs,
                'lambda_unit': args.lambda_unit}, args.model_path)
    print(f"Model saved: {args.model_path}")
    
    args.history_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.history_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    print(f"History saved: {args.history_csv}")


if __name__ == "__main__":
    main()
