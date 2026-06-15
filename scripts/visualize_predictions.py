"""可视化模型预测效果：输入图像、真实相位、预测相位、补偿后远场"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.models import build_phase_model
from train.physics_loss import SevenBeamFourierOptics

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("Loading data and model...")

# 1. 加载数据
data_path = Path("dataset/seven_beam/multiplane_corrected_f1.0_d0.05")
images = np.load(data_path / "images_multiplane_corrected_10k.npy")
labels = np.load(data_path / "labels_multiplane_corrected_10k.npy")
phases = np.load(data_path / "phases_multiplane_corrected_10k.npy")

# 2. 模拟测试集划分
torch.manual_seed(20260615)
indices = torch.randperm(10000).tolist()
test_indices = indices[8500:]

# 3. 加载模型
model = build_phase_model('dual_plane_fusion_cnn', image_size=160, output_dim=12, in_channels=2)
checkpoint = torch.load('models/cycle_corrected_full_30epoch_best_rmse.pth',
                       map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 4. 初始化光学系统
optics = SevenBeamFourierOptics(
    num_points=256,
    window_size=10e-3,
    waist=0.5e-3,
    beam_distance=1.5e-3,
    crop_size=160
)

# 5. 选择样本（最好、平均、最差各一个，plus 1个随机）
np.random.seed(42)
sample_cases = []

# 计算前100个测试样本的误差
for idx in test_indices[:100]:
    true_phase = phases[idx]
    image = torch.FloatTensor(images[idx]).unsqueeze(0)
    with torch.no_grad():
        pred_label = model(image).numpy()[0]
    pred_phase = np.arctan2(pred_label[0::2], pred_label[1::2])
    error = np.arctan2(np.sin(true_phase - pred_phase), np.cos(true_phase - pred_phase))
    rmse = np.sqrt(np.mean(error**2))
    sample_cases.append((idx, rmse, true_phase, pred_phase))

sample_cases.sort(key=lambda x: x[1])

# 选择4个代表性样本
selected = [
    ("Best", sample_cases[0]),
    ("Good", sample_cases[len(sample_cases)//3]),
    ("Average", sample_cases[len(sample_cases)//2]),
    ("Worst", sample_cases[-1]),
]

print(f"Selected samples:")
for label, (idx, rmse, _, _) in selected:
    print(f"  {label}: Index {idx}, RMSE {rmse:.4f} rad ({rmse*180/np.pi:.2f} deg)")

# 6. 生成大图
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.4)

for row, (case_label, (idx, rmse, true_phase, pred_phase)) in enumerate(selected):
    # 加载图像
    img = images[idx]  # [2, 160, 160]

    # 预测
    with torch.no_grad():
        pred_label = model(torch.FloatTensor(img).unsqueeze(0)).numpy()[0]
    pred_phase_tensor = np.arctan2(pred_label[0::2], pred_label[1::2])

    # 计算补偿后远场
    true_phase_torch = torch.FloatTensor(true_phase).unsqueeze(0)
    pred_phase_torch = torch.FloatTensor(pred_phase_tensor).unsqueeze(0)

    # 残余相位
    residual = torch.atan2(
        torch.sin(true_phase_torch - pred_phase_torch),
        torch.cos(true_phase_torch - pred_phase_torch)
    )

    # 远场
    farfield_before = optics.reconstruct_from_phase(true_phase_torch, normalize=True)
    farfield_after = optics.reconstruct_from_phase(residual, normalize=True)
    farfield_ideal = optics.reconstruct_from_phase(torch.zeros_like(residual), normalize=True)

    # 转为numpy
    farfield_before = farfield_before[0].numpy()
    farfield_after = farfield_after[0].numpy()
    farfield_ideal = farfield_ideal[0].numpy()

    # 计算指标
    error = np.arctan2(np.sin(true_phase - pred_phase_tensor),
                      np.cos(true_phase - pred_phase_tensor))
    rmse_val = np.sqrt(np.mean(error**2))

    peak_before = farfield_before.max()
    peak_after = farfield_after.max()
    peak_ideal = farfield_ideal.max()
    strehl = peak_after / peak_ideal

    # Col 0: 焦平面图像
    ax = fig.add_subplot(gs[row, 0])
    im = ax.imshow(img[0], cmap='hot', aspect='auto')
    ax.set_title(f'{case_label}: Focal Plane', fontsize=10, fontweight='bold')
    ax.set_xlabel('Pixel', fontsize=8)
    ax.set_ylabel('Pixel', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Col 1: 焦前图像
    ax = fig.add_subplot(gs[row, 1])
    im = ax.imshow(img[1], cmap='hot', aspect='auto')
    ax.set_title(f'Defocus Plane (-5cm)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Pixel', fontsize=8)
    ax.set_ylabel('Pixel', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Col 2: 真实相位
    ax = fig.add_subplot(gs[row, 2])
    x = np.arange(6)
    ax.bar(x - 0.2, true_phase, 0.4, label='True', alpha=0.7, color='blue')
    ax.bar(x + 0.2, pred_phase_tensor, 0.4, label='Pred', alpha=0.7, color='orange')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_xlabel('Beam Channel', fontsize=8)
    ax.set_ylabel('Phase (rad)', fontsize=8)
    ax.set_title(f'True vs Pred Phase', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Col 3: 相位误差
    ax = fig.add_subplot(gs[row, 3])
    colors = ['green' if abs(e) < 0.05 else 'orange' if abs(e) < 0.1 else 'red' for e in error]
    bars = ax.bar(x, error, color=colors, alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Beam Channel', fontsize=8)
    ax.set_ylabel('Error (rad)', fontsize=8)
    ax.set_title(f'Phase Error\nRMSE={rmse_val:.3f} rad ({rmse_val*180/np.pi:.1f}°)',
                fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Col 4: 补偿前远场
    ax = fig.add_subplot(gs[row, 4])
    im = ax.imshow(np.log10(farfield_before + 1e-8), cmap='hot', aspect='auto')
    ax.set_title(f'Before Compensation\nPeak={peak_before:.3f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Pixel', fontsize=8)
    ax.set_ylabel('Pixel', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, label='log10(I)')

    # Col 5: 补偿后远场
    ax = fig.add_subplot(gs[row, 5])
    im = ax.imshow(np.log10(farfield_after + 1e-8), cmap='hot', aspect='auto')
    ax.set_title(f'After Compensation\nStrehl={strehl:.4f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Pixel', fontsize=8)
    ax.set_ylabel('Pixel', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, label='log10(I)')

plt.suptitle('Model Prediction Visualization: From Input to Compensated Far-field',
            fontsize=16, fontweight='bold', y=0.995)

# 保存
output_path = Path("result/figures/model_prediction_visualization.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_path}")

# 同时保存PDF
output_pdf = output_path.with_suffix('.pdf')
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Saved: {output_pdf}")

print("\nVisualization complete!")
