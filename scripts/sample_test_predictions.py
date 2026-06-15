"""随机抽样验证：展示模型实际预测效果"""
import numpy as np
import torch
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.models import build_phase_model
from train.phase_metrics import phase_metrics_from_sin_cos

# 1. 加载数据
print("="*60)
print("RANDOM SAMPLING FROM TEST SET")
print("="*60)

data_path = Path("dataset/seven_beam/multiplane_corrected_f1.0_d0.05")
images = np.load(data_path / "images_multiplane_corrected_10k.npy")
labels = np.load(data_path / "labels_multiplane_corrected_10k.npy")
phases = np.load(data_path / "phases_multiplane_corrected_10k.npy")

# 2. 模拟测试集划分（使用第一次运行的种子）
torch.manual_seed(20260615)
indices = torch.randperm(10000).tolist()
test_indices = indices[8500:]  # 最后 1500 个样本

# 3. 加载模型
print("\nLoading model...")
model = build_phase_model('dual_plane_fusion_cnn', image_size=160, output_dim=12, in_channels=2)
checkpoint = torch.load('models/cycle_corrected_full_30epoch_best_rmse.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded: {checkpoint.get('epoch', 'unknown')} epoch")

# 4. 随机抽取 10 个测试样本
np.random.seed(42)
sample_indices = np.random.choice(test_indices, size=10, replace=False)

print(f"\n{'='*60}")
print(f"SAMPLING 10 RANDOM TEST SAMPLES")
print(f"{'='*60}\n")

total_error = 0
for i, idx in enumerate(sample_indices):
    # 真实值
    true_phase = phases[idx]  # [6]
    true_label = labels[idx]  # [12]

    # 预测
    image = torch.FloatTensor(images[idx]).unsqueeze(0)  # [1, 2, H, W]
    with torch.no_grad():
        pred_label = model(image).numpy()[0]  # [12]

    # 解码相位
    pred_phase = np.arctan2(pred_label[0::2], pred_label[1::2])

    # 计算误差
    error = np.arctan2(np.sin(true_phase - pred_phase), np.cos(true_phase - pred_phase))
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))

    total_error += rmse

    print(f"Sample {i+1} (Index {idx}):")
    print(f"  True phases (rad):  [{', '.join(f'{p:+.3f}' for p in true_phase)}]")
    print(f"  Pred phases (rad):  [{', '.join(f'{p:+.3f}' for p in pred_phase)}]")
    print(f"  Errors (rad):       [{', '.join(f'{e:+.3f}' for e in error)}]")
    print(f"  RMSE: {rmse:.4f} rad ({rmse*180/np.pi:.2f} deg)")
    print(f"  Max error: {max_error:.4f} rad ({max_error*180/np.pi:.2f} deg)")
    print()

avg_rmse = total_error / 10
print(f"{'='*60}")
print(f"AVERAGE RMSE OVER 10 SAMPLES: {avg_rmse:.4f} rad ({avg_rmse*180/np.pi:.2f} deg)")
print(f"{'='*60}")

# 5. 展示最好和最差的样本
errors_list = []
for idx in test_indices[:100]:  # 检查前 100 个测试样本
    true_phase = phases[idx]
    image = torch.FloatTensor(images[idx]).unsqueeze(0)
    with torch.no_grad():
        pred_label = model(image).numpy()[0]
    pred_phase = np.arctan2(pred_label[0::2], pred_label[1::2])
    error = np.arctan2(np.sin(true_phase - pred_phase), np.cos(true_phase - pred_phase))
    rmse = np.sqrt(np.mean(error**2))
    errors_list.append((idx, rmse, true_phase, pred_phase, error))

errors_list.sort(key=lambda x: x[1])

print(f"\n{'='*60}")
print(f"BEST CASE (lowest RMSE in first 100 test samples):")
print(f"{'='*60}")
idx, rmse, true_p, pred_p, err = errors_list[0]
print(f"Sample Index: {idx}")
print(f"  True phases: [{', '.join(f'{p:+.3f}' for p in true_p)}]")
print(f"  Pred phases: [{', '.join(f'{p:+.3f}' for p in pred_p)}]")
print(f"  Errors:      [{', '.join(f'{e:+.3f}' for e in err)}]")
print(f"  RMSE: {rmse:.4f} rad ({rmse*180/np.pi:.2f} deg)")

print(f"\n{'='*60}")
print(f"WORST CASE (highest RMSE in first 100 test samples):")
print(f"{'='*60}")
idx, rmse, true_p, pred_p, err = errors_list[-1]
print(f"Sample Index: {idx}")
print(f"  True phases: [{', '.join(f'{p:+.3f}' for p in true_p)}]")
print(f"  Pred phases: [{', '.join(f'{p:+.3f}' for p in pred_p)}]")
print(f"  Errors:      [{', '.join(f'{e:+.3f}' for e in err)}]")
print(f"  RMSE: {rmse:.4f} rad ({rmse*180/np.pi:.2f} deg)")

print(f"\n{'='*60}")
print("SAMPLING COMPLETE")
print(f"{'='*60}")
