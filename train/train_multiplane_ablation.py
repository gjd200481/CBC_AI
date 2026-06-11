"""多平面七光束训练脚本 + 消融实验"""
import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.data_utils import build_dataloaders
from train.models import MultiPlanePhaseCNN, DeepResidualPhaseCNN, count_parameters
from train.phase_metrics import phase_metrics_from_sin_cos
from train.physics_loss import CompensationQualityLoss, FarFieldConsistencyLoss, SevenBeamFourierOptics


def train_epoch(model, loader, optimizer, phase_fn, farfield_fn, comp_fn, lam_phy, lam_comp, device):
    model.train()
    totals = {'phase': 0.0, 'farfield': 0.0, 'comp': 0.0, 'total': 0.0, 'n': 0}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        l_phase = phase_fn(preds, labels)
        # 物理损失只用焦平面图像（第一个平面）
        focal_plane = images[:, 0:1] if images.shape[1] > 1 else images
        l_far = farfield_fn(preds, focal_plane)
        l_comp = comp_fn(preds, labels)
        loss = l_phase + lam_phy * l_far + lam_comp * l_comp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch = images.size(0)
        totals['phase'] += l_phase.item() * batch
        totals['farfield'] += l_far.item() * batch
        totals['comp'] += l_comp.item() * batch
        totals['total'] += loss.item() * batch
        totals['n'] += batch
    return {k: v / totals['n'] for k, v in totals.items() if k != 'n'}


@torch.no_grad()
def eval_epoch(model, loader, phase_fn, farfield_fn, comp_fn, lam_phy, lam_comp, device):
    model.eval()
    totals = {'phase': 0.0, 'farfield': 0.0, 'comp': 0.0, 'total': 0.0, 'n': 0}
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        l_phase = phase_fn(preds, labels)
        focal_plane = images[:, 0:1] if images.shape[1] > 1 else images
        l_far = farfield_fn(preds, focal_plane)
        l_comp = comp_fn(preds, labels)
        loss = l_phase + lam_phy * l_far + lam_comp * l_comp
        batch = images.size(0)
        totals['phase'] += l_phase.item() * batch
        totals['farfield'] += l_far.item() * batch
        totals['comp'] += l_comp.item() * batch
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
        'total_loss': totals['total'] / totals['n'],
        'rmse_rad': metrics['rmse_rad'],
        'mae_rad': metrics['mae_rad'],
    }, all_preds, all_labels


def train_single_config(config, output_dir):
    """训练单个配置"""
    print(f"\n{'='*60}")
    print(f"Config: {config['name']}")
    print(f"{'='*60}")
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device(config['device'])
    
    # 数据
    loaders = build_dataloaders(
        config['image_path'], config['label_path'], 
        config['batch_size'], 0.7, 0.15, config['seed'], 
        (160, 160), config['num_workers'], augment_train=config['augment']
    )
    
    # 模型
    if config['model_type'] == 'multiplane':
        model = MultiPlanePhaseCNN(160, 12, num_planes=config['num_planes'])
    else:
        model = DeepResidualPhaseCNN(160, 12, in_channels=1)
    
    model = model.to(device)
    params = count_parameters(model)
    print(f"Model: {config['model_type']}, Params: {params/1e6:.2f}M")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    # 损失函数
    phase_fn = nn.MSELoss()
    farfield_fn = FarFieldConsistencyLoss(SevenBeamFourierOptics(), loss_type='mse')
    comp_fn = CompensationQualityLoss(SevenBeamFourierOptics())
    
    # 训练
    history = []
    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    
    start_time = time.time()
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, loaders['train'], optimizer, phase_fn, farfield_fn, 
                                comp_fn, config['lambda_phy'], config['lambda_comp'], device)
        val_metrics, _, _ = eval_epoch(model, loaders['val'], phase_fn, farfield_fn, 
                                      comp_fn, config['lambda_phy'], config['lambda_comp'], device)
        scheduler.step()
        
        if val_metrics['rmse_rad'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse_rad']
            best_epoch = epoch
            best_state = model.state_dict().copy()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss['total'],
            'val_loss': val_metrics['total_loss'],
            'val_rmse': val_metrics['rmse_rad'],
            'lr': scheduler.get_last_lr()[0]
        })
        
        if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"val_rmse={val_metrics['rmse_rad']:.4f}, "
                  f"best={best_val_rmse:.4f}@{best_epoch+1}")
    
    train_time = time.time() - start_time
    
    # 测试最佳模型
    model.load_state_dict(best_state)
    test_metrics, test_preds, test_labels = eval_epoch(
        model, loaders['test'], phase_fn, farfield_fn, comp_fn, 
        config['lambda_phy'], config['lambda_comp'], device
    )
    
    print(f"\nBest model (epoch {best_epoch+1}): test_rmse={test_metrics['rmse_rad']:.4f}")
    print(f"Training time: {train_time:.1f}s")
    
    # 保存
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state': best_state,
        'config': config,
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
    }, output_dir / f"{config['name']}_best.pth")
    
    with open(output_dir / f"{config['name']}_history.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_rmse', 'lr'])
        writer.writeheader()
        writer.writerows(history)
    
    return {
        'name': config['name'],
        'model_type': config['model_type'],
        'num_planes': config.get('num_planes', 1),
        'params_M': params / 1e6,
        'best_epoch': best_epoch + 1,
        'test_rmse': test_metrics['rmse_rad'],
        'test_mae': test_metrics['mae_rad'],
        'train_time_s': train_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['smoke', 'ablation'], default='smoke',
                       help="smoke: 快速1k测试; ablation: 完整消融实验")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, 
                       default=REPO_ROOT / "result/cycle31_multiplane")
    args = parser.parse_args()
    
    # 配置
    if args.mode == 'smoke':
        # 快速smoke测试: 1k数据, 10 epoch
        configs = [
            {
                'name': 'baseline_single_focal',
                'model_type': 'single',
                'image_path': REPO_ROOT / 'dataset/seven_beam/main_static/images_main_clean_seven_beam.npy',
                'label_path': REPO_ROOT / 'dataset/seven_beam/main_static/labels_main_clean_seven_beam.npy',
                'num_planes': 1,
                'epochs': 10,
                'batch_size': 32,
                'lr': 1e-3,
                'lambda_phy': 0.05,
                'lambda_comp': 0.5,
                'augment': True,
                'seed': 20260616,
                'device': args.device,
                'num_workers': args.num_workers,
            },
            {
                'name': 'multiplane_dual',
                'model_type': 'multiplane',
                'image_path': REPO_ROOT / 'dataset/seven_beam/multiplane/images_multiplane_seven_beam.npy',
                'label_path': REPO_ROOT / 'dataset/seven_beam/multiplane/labels_multiplane_seven_beam.npy',
                'num_planes': 2,
                'epochs': 10,
                'batch_size': 32,
                'lr': 1e-3,
                'lambda_phy': 0.05,
                'lambda_comp': 0.5,
                'augment': True,
                'seed': 20260616,
                'device': args.device,
                'num_workers': args.num_workers,
            },
        ]
    else:
        # 完整消融实验
        configs = []
        # 1. 单平面baseline (10k数据)
        configs.append({
            'name': 'baseline_single_10k',
            'model_type': 'single',
            'image_path': REPO_ROOT / 'dataset/seven_beam/main_static_10k/images_main_clean_seven_beam_10k.npy',
            'label_path': REPO_ROOT / 'dataset/seven_beam/main_static_10k/labels_main_clean_seven_beam_10k.npy',
            'num_planes': 1,
            'epochs': 30,
            'batch_size': 32,
            'lr': 1e-3,
            'lambda_phy': 0.05,
            'lambda_comp': 0.5,
            'augment': True,
            'seed': 20260616,
            'device': args.device,
            'num_workers': args.num_workers,
        })
        
        # 2. 双平面 (焦平面 + 焦前5cm)
        for dist_str, dist_val in [('5cm', '0,-0.05'), ('3cm', '0,-0.03'), ('7cm', '0,-0.07')]:
            configs.append({
                'name': f'multiplane_befocal_{dist_str}_10k',
                'model_type': 'multiplane',
                'image_path': REPO_ROOT / f'dataset/seven_beam/multiplane_{dist_val.replace(",","_")}/images_multiplane.npy',
                'label_path': REPO_ROOT / f'dataset/seven_beam/multiplane_{dist_val.replace(",","_")}/labels_multiplane.npy',
                'num_planes': 2,
                'epochs': 30,
                'batch_size': 32,
                'lr': 1e-3,
                'lambda_phy': 0.05,
                'lambda_comp': 0.5,
                'augment': True,
                'seed': 20260616,
                'device': args.device,
                'num_workers': args.num_workers,
            })
    
    # 执行训练
    results = []
    for config in configs:
        result = train_single_config(config, args.output_dir)
        results.append(result)
    
    # 保存汇总
    with open(args.output_dir / f'summary_{args.mode}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['name']:30s} test_rmse={r['test_rmse']:.4f} rad")


if __name__ == "__main__":
    main()
