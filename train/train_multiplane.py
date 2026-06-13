"""多平面输入三通道训练脚本。"""
import argparse
import copy
import csv
from pathlib import Path
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.models import build_phase_model, count_parameters
from train.phase_metrics import build_phase_loss, phase_metrics_from_sin_cos, unit_circle_loss_from_sin_cos
from train.physics_loss import CompensationQualityLoss, FarFieldConsistencyLoss, SevenBeamFourierOptics
from train.train_seven_beam_baseline import channel_rmse_from_sin_cos


class MultiPlaneDataset(Dataset):
    def __init__(self, image_path, label_path):
        self.images = np.load(image_path)  # [N, C, H, W]
        self.labels = np.load(label_path)  # [N, 12]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.images[idx]), torch.FloatTensor(self.labels[idx])


def train_epoch(model, loader, optimizer, phase_fn, farfield_fn, comp_fn, lam_phy, lam_comp, lam_unit, focal_plane_index, device):
    model.train()
    totals = {'phase': 0.0, 'farfield': 0.0, 'comp': 0.0, 'unit': 0.0, 'total': 0.0, 'n': 0}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        l_phase = phase_fn(preds, labels)
        l_far = farfield_fn(preds, images[:, focal_plane_index:focal_plane_index + 1])
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
def eval_epoch(model, loader, phase_fn, farfield_fn, comp_fn, lam_phy, lam_comp, lam_unit, focal_plane_index, device):
    model.eval()
    totals = {'phase': 0.0, 'farfield': 0.0, 'comp': 0.0, 'unit': 0.0, 'total': 0.0, 'n': 0}
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        l_phase = phase_fn(preds, labels)
        l_far = farfield_fn(preds, images[:, focal_plane_index:focal_plane_index + 1])
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
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
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


@torch.no_grad()
def compensation_metrics_from_predictions(optics, pred_values, true_values, main_lobe_radius=3, batch_size=256, device="cpu"):
    pred = torch.as_tensor(pred_values, dtype=torch.float32, device=device)
    true = torch.as_tensor(true_values, dtype=torch.float32, device=device)

    crop_size = optics.crop_size
    yy, xx = torch.meshgrid(
        torch.arange(crop_size, dtype=torch.float32, device=device),
        torch.arange(crop_size, dtype=torch.float32, device=device),
        indexing="ij",
    )
    center = crop_size // 2
    radius = torch.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    main_lobe_mask = (radius <= main_lobe_radius).float()
    ideal_farfield = optics.reconstruct_from_phase(torch.zeros(1, 6, device=device), normalize=False)
    ideal_peak = ideal_farfield.max().clamp_min(1e-8)
    ideal_main_lobe_energy = (ideal_farfield * main_lobe_mask).sum().clamp_min(1e-8)

    peaks = []
    main_ratios = []
    synthesis_efficiencies = []
    for start in range(0, len(pred), batch_size):
        pred_batch = pred[start:start + batch_size]
        true_batch = true[start:start + batch_size]
        pred_phases = torch.atan2(pred_batch[:, 0::2], pred_batch[:, 1::2])
        true_phases = torch.atan2(true_batch[:, 0::2], true_batch[:, 1::2])
        residual = torch.atan2(
            torch.sin(true_phases - pred_phases),
            torch.cos(true_phases - pred_phases),
        )
        farfield = optics.reconstruct_from_phase(residual, normalize=False)
        peak = farfield.amax(dim=(1, 2)) / ideal_peak
        main_energy = (farfield * main_lobe_mask).sum(dim=(1, 2))
        total_energy = farfield.sum(dim=(1, 2)).clamp_min(1e-8)
        peaks.append(peak.detach().cpu())
        main_ratios.append((main_energy / total_energy).detach().cpu())
        synthesis_efficiencies.append((main_energy / ideal_main_lobe_energy).detach().cpu())

    strehl = torch.cat(peaks)
    main_lobe = torch.cat(main_ratios)
    synthesis = torch.cat(synthesis_efficiencies)
    return {
        "strehl_ratio": float(strehl.mean().item()),
        "main_lobe_ratio": float(main_lobe.mean().item()),
        "synthesis_efficiency": float(synthesis.mean().item()),
    }


def save_selected_checkpoint(
    path,
    state,
    selection_metric,
    selection_value,
    selection_epoch,
    model,
    test_loader,
    phase_fn,
    farfield_fn,
    comp_fn,
    args,
    device,
    in_channels,
    params,
):
    model.load_state_dict(state)
    test_m, preds, labels = eval_epoch(
        model, test_loader, phase_fn, farfield_fn, comp_fn,
        args.lambda_phy, args.lambda_comp, args.lambda_unit, args.focal_plane_index, device
    )
    channel_rmse = channel_rmse_from_sin_cos(preds, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': state,
        'test_rmse_rad': test_m['rmse_rad'],
        'best_epoch': selection_epoch,
        'selection_metric': selection_metric,
        'selection_value': selection_value,
        'model_name': args.model_name,
        'in_channels': in_channels,
        'focal_plane_index': args.focal_plane_index,
        'channel_rmse_rad': channel_rmse.tolist(),
        'parameters': params,
        'lambda_phy': args.lambda_phy,
        'lambda_comp': args.lambda_comp,
        'comp_warmup_epochs': args.comp_warmup_epochs,
        'lambda_unit': args.lambda_unit,
    }, path)
    return test_m, channel_rmse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-path", type=Path, default=REPO_ROOT / "dataset/seven_beam/multiplane_10k/images_multiplane_seven_beam_10k.npy")
    p.add_argument("--label-path", type=Path, default=REPO_ROOT / "dataset/seven_beam/multiplane_10k/labels_multiplane_seven_beam_10k.npy")
    p.add_argument("--model-path", type=Path, default=REPO_ROOT / "models/cycle31_multiplane_10k.pth")
    p.add_argument("--comp-model-path", type=Path, default=None)
    p.add_argument("--strehl-model-path", type=Path, default=None)
    p.add_argument("--main-lobe-model-path", type=Path, default=None)
    p.add_argument("--history-csv", type=Path, default=REPO_ROOT / "result/metrics/cycle31_multiplane_history.csv")
    p.add_argument("--model-name", default="deep_residual_cnn")
    p.add_argument("--lambda-phy", type=float, default=0.05)
    p.add_argument("--lambda-comp", type=float, default=0.5)
    p.add_argument("--comp-warmup-epochs", type=int, default=0)
    p.add_argument("--lambda-unit", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20260616)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--focal-plane-index", type=int, default=0)
    args = p.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    
    dataset = MultiPlaneDataset(args.image_path, args.label_path)
    if dataset.images.ndim != 4:
        raise ValueError(f"Expected image array [N,C,H,W], got {dataset.images.shape}")
    in_channels = int(dataset.images.shape[1])
    if not 0 <= args.focal_plane_index < in_channels:
        raise ValueError(f"--focal-plane-index must be in [0, {in_channels - 1}], got {args.focal_plane_index}")

    train_set, val_set, test_set = random_split(dataset, [7000, 1500, 1500], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = build_phase_model(args.model_name, 160, 12, in_channels=in_channels).to(device)
    params = count_parameters(model)
    print(f"Model: {args.model_name}({in_channels}-channel), Parameters: {params}")
    
    optics = SevenBeamFourierOptics(crop_size=160).to(device)
    phase_fn = build_phase_loss("mse")
    farfield_fn = FarFieldConsistencyLoss(optics)
    comp_fn = CompensationQualityLoss(optics)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print(
        f"Multiplane training: lambda_phy={args.lambda_phy}, "
        f"lambda_comp={args.lambda_comp}, comp_warmup_epochs={args.comp_warmup_epochs}, "
        f"lambda_unit={args.lambda_unit}, focal_plane_index={args.focal_plane_index}"
    )
    
    history = []
    best_rmse = float('inf')
    best_epoch = 0
    best_state = None
    best_comp_loss = float('inf')
    best_comp_epoch = 0
    best_comp_state = None
    best_strehl = -float('inf')
    best_strehl_epoch = 0
    best_strehl_state = None
    best_main_lobe = -float('inf')
    best_main_lobe_epoch = 0
    best_main_lobe_state = None
    
    for epoch in range(1, args.epochs + 1):
        if args.comp_warmup_epochs > 0:
            comp_scale = min(1.0, epoch / args.comp_warmup_epochs)
        else:
            comp_scale = 1.0
        active_lambda_comp = args.lambda_comp * comp_scale

        train_m = train_epoch(
            model, train_loader, optimizer, phase_fn, farfield_fn, comp_fn,
            args.lambda_phy, active_lambda_comp, args.lambda_unit, args.focal_plane_index, device
        )
        val_m, val_preds, val_labels = eval_epoch(
            model, val_loader, phase_fn, farfield_fn, comp_fn,
            args.lambda_phy, active_lambda_comp, args.lambda_unit, args.focal_plane_index, device
        )
        val_comp_metrics = compensation_metrics_from_predictions(
            optics=optics,
            pred_values=val_preds,
            true_values=val_labels,
            device=device,
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
            'val_strehl_ratio': val_comp_metrics['strehl_ratio'],
            'val_main_lobe_ratio': val_comp_metrics['main_lobe_ratio'],
            'val_synthesis_efficiency': val_comp_metrics['synthesis_efficiency'],
        })
        
        if val_m['rmse_rad'] < best_rmse:
            best_rmse = val_m['rmse_rad']
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if val_m['comp_loss'] < best_comp_loss:
            best_comp_loss = val_m['comp_loss']
            best_comp_epoch = epoch
            best_comp_state = copy.deepcopy(model.state_dict())
        if val_comp_metrics['strehl_ratio'] > best_strehl:
            best_strehl = val_comp_metrics['strehl_ratio']
            best_strehl_epoch = epoch
            best_strehl_state = copy.deepcopy(model.state_dict())
        if val_comp_metrics['main_lobe_ratio'] > best_main_lobe:
            best_main_lobe = val_comp_metrics['main_lobe_ratio']
            best_main_lobe_epoch = epoch
            best_main_lobe_state = copy.deepcopy(model.state_dict())
        
        print(
            f"Epoch {epoch:03d} | train={train_m['total']:.6f} | "
            f"lambda_comp={active_lambda_comp:.4f} | val_rmse={val_m['rmse_rad']:.6f} | "
            f"strehl={val_comp_metrics['strehl_ratio']:.6f} | "
            f"main={val_comp_metrics['main_lobe_ratio']:.6f} | "
            f"eff={val_comp_metrics['synthesis_efficiency']:.6f} | "
            f"unit={val_m['unit_loss']:.6f} | lr={scheduler.get_last_lr()[0]:.2e}"
        )
    
    test_m, channel_rmse = save_selected_checkpoint(
        path=args.model_path,
        state=best_state,
        selection_metric='val_rmse_rad',
        selection_value=best_rmse,
        selection_epoch=best_epoch,
        model=model,
        test_loader=test_loader,
        phase_fn=phase_fn,
        farfield_fn=farfield_fn,
        comp_fn=comp_fn,
        args=args,
        device=device,
        in_channels=in_channels,
        params=params,
    )
    print(f"\nBest RMSE epoch: {best_epoch}")
    print(f"Best RMSE Test RMSE: {test_m['rmse_rad']:.6f} rad")
    print(f"Best RMSE Test MAE: {test_m['mae_rad']:.6f} rad")
    for i, r in enumerate(channel_rmse):
        print(f"Channel {i+1} RMSE: {r:.6f} rad")
    print(f"Best-RMSE model saved: {args.model_path}")

    comp_model_path = args.comp_model_path or args.model_path.with_name(f"{args.model_path.stem}_best_comp{args.model_path.suffix}")
    comp_test_m, _ = save_selected_checkpoint(
        path=comp_model_path,
        state=best_comp_state,
        selection_metric='val_comp_loss',
        selection_value=best_comp_loss,
        selection_epoch=best_comp_epoch,
        model=model,
        test_loader=test_loader,
        phase_fn=phase_fn,
        farfield_fn=farfield_fn,
        comp_fn=comp_fn,
        args=args,
        device=device,
        in_channels=in_channels,
        params=params,
    )
    print(f"Best comp epoch: {best_comp_epoch}")
    print(f"Best comp Test RMSE: {comp_test_m['rmse_rad']:.6f} rad")
    print(f"Best-comp model saved: {comp_model_path}")

    strehl_model_path = args.strehl_model_path or args.model_path.with_name(f"{args.model_path.stem}_best_strehl{args.model_path.suffix}")
    strehl_test_m, _ = save_selected_checkpoint(
        path=strehl_model_path,
        state=best_strehl_state,
        selection_metric='val_strehl_ratio',
        selection_value=best_strehl,
        selection_epoch=best_strehl_epoch,
        model=model,
        test_loader=test_loader,
        phase_fn=phase_fn,
        farfield_fn=farfield_fn,
        comp_fn=comp_fn,
        args=args,
        device=device,
        in_channels=in_channels,
        params=params,
    )
    print(f"Best Strehl epoch: {best_strehl_epoch}")
    print(f"Best Strehl Test RMSE: {strehl_test_m['rmse_rad']:.6f} rad")
    print(f"Best-Strehl model saved: {strehl_model_path}")

    main_lobe_model_path = args.main_lobe_model_path or args.model_path.with_name(f"{args.model_path.stem}_best_main_lobe{args.model_path.suffix}")
    main_lobe_test_m, _ = save_selected_checkpoint(
        path=main_lobe_model_path,
        state=best_main_lobe_state,
        selection_metric='val_main_lobe_ratio',
        selection_value=best_main_lobe,
        selection_epoch=best_main_lobe_epoch,
        model=model,
        test_loader=test_loader,
        phase_fn=phase_fn,
        farfield_fn=farfield_fn,
        comp_fn=comp_fn,
        args=args,
        device=device,
        in_channels=in_channels,
        params=params,
    )
    print(f"Best main-lobe epoch: {best_main_lobe_epoch}")
    print(f"Best main-lobe Test RMSE: {main_lobe_test_m['rmse_rad']:.6f} rad")
    print(f"Best-main-lobe model saved: {main_lobe_model_path}")
    
    args.history_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.history_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)


if __name__ == "__main__":
    main()
