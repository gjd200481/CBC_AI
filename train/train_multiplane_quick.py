"""快速消融实验：焦前距离对比（减少epoch加速）"""
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


def train_and_eval(config):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device(config['device'])
    
    loaders = build_dataloaders(
        config['image_path'], config['label_path'], 
        config['batch_size'], 0.7, 0.15, config['seed'], 
        (160, 160), config['num_workers'], augment_train=True
    )
    
    if config['model_type'] == 'multiplane':
        model = MultiPlanePhaseCNN(160, 12, num_planes=2)
    else:
        model = DeepResidualPhaseCNN(160, 12, in_channels=1)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    phase_fn = nn.MSELoss()
    farfield_fn = FarFieldConsistencyLoss(SevenBeamFourierOptics(), loss_type='mse')
    comp_fn = CompensationQualityLoss(SevenBeamFourierOptics())
    
    best_val_rmse = float('inf')
    best_state = None
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        for images, labels in loaders['train']:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            focal_plane = images[:, 0:1] if images.shape[1] > 1 else images
            loss = phase_fn(preds, labels) + 0.05 * farfield_fn(preds, focal_plane) + 0.5 * comp_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loaders['val']:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        metrics = phase_metrics_from_sin_cos(all_preds, all_labels)
        
        if metrics['rmse_rad'] < best_val_rmse:
            best_val_rmse = metrics['rmse_rad']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: val_rmse={metrics['rmse_rad']:.4f}, best={best_val_rmse:.4f}")
    
    # Test
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loaders['test']:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    test_metrics = phase_metrics_from_sin_cos(all_preds, all_labels)
    
    return test_metrics['rmse_rad'], test_metrics['mae_rad']


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [
        {
            'name': 'baseline_single_10k',
            'model_type': 'single',
            'image_path': REPO_ROOT / 'dataset/seven_beam/main_static_10k/images_main_clean_seven_beam_10k.npy',
            'label_path': REPO_ROOT / 'dataset/seven_beam/main_static_10k/labels_main_clean_seven_beam_10k.npy',
            'epochs': 15,
            'batch_size': 32,
            'lr': 1e-3,
            'seed': 20260616,
            'device': device,
            'num_workers': 2,
        },
        {
            'name': 'multiplane_3cm_10k',
            'model_type': 'multiplane',
            'image_path': REPO_ROOT / 'dataset/seven_beam/multiplane_0_-0.03/images_multiplane_3cm.npy',
            'label_path': REPO_ROOT / 'dataset/seven_beam/multiplane_0_-0.03/labels_multiplane_3cm.npy',
            'epochs': 15,
            'batch_size': 32,
            'lr': 1e-3,
            'seed': 20260616,
            'device': device,
            'num_workers': 2,
        },
        {
            'name': 'multiplane_5cm_10k',
            'model_type': 'multiplane',
            'image_path': REPO_ROOT / 'dataset/seven_beam/multiplane_0_-0.05/images_multiplane_5cm.npy',
            'label_path': REPO_ROOT / 'dataset/seven_beam/multiplane_0_-0.05/labels_multiplane_5cm.npy',
            'epochs': 15,
            'batch_size': 32,
            'lr': 1e-3,
            'seed': 20260616,
            'device': device,
            'num_workers': 2,
        },
        {
            'name': 'multiplane_7cm_10k',
            'model_type': 'multiplane',
            'image_path': REPO_ROOT / 'dataset/seven_beam/multiplane_0_-0.07/images_multiplane_7cm.npy',
            'label_path': REPO_ROOT / 'dataset/seven_beam/multiplane_0_-0.07/labels_multiplane_7cm.npy',
            'epochs': 15,
            'batch_size': 32,
            'lr': 1e-3,
            'seed': 20260616,
            'device': device,
            'num_workers': 2,
        },
    ]
    
    results = []
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training: {config['name']}")
        print(f"{'='*60}")
        start = time.time()
        test_rmse, test_mae = train_and_eval(config)
        elapsed = time.time() - start
        print(f"Result: test_rmse={test_rmse:.4f} rad, time={elapsed:.1f}s")
        results.append({
            'name': config['name'],
            'model_type': config['model_type'],
            'test_rmse_rad': test_rmse,
            'test_mae_rad': test_mae,
            'train_time_s': elapsed,
        })
    
    output_dir = REPO_ROOT / 'result/cycle31_multiplane_quick'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    baseline_rmse = results[0]['test_rmse_rad']
    for r in results:
        improvement = (baseline_rmse - r['test_rmse_rad']) / baseline_rmse * 100
        print(f"{r['name']:25s} RMSE={r['test_rmse_rad']:.4f} rad  Improvement={improvement:+.1f}%")


if __name__ == "__main__":
    main()
