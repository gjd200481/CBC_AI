"""生成 30 epoch 修正数据训练曲线图"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 读取训练历史
history_path = Path("result/metrics/cycle_corrected_full_30epoch_history.csv")
df = pd.read_csv(history_path)

# 创建 6 子图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('30-Epoch Training: Corrected Multi-Plane Data (Breakthrough Results)',
             fontsize=16, fontweight='bold')

# 1. RMSE vs Epoch
ax = axes[0, 0]
ax.plot(df['epoch'], df['val_rmse_rad'], 'b-', linewidth=2, label='Val RMSE')
best_rmse_epoch = df['val_rmse_rad'].idxmin() + 1
best_rmse = df.loc[best_rmse_epoch-1, 'val_rmse_rad']
ax.axvline(best_rmse_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best @ Epoch {best_rmse_epoch}')
ax.scatter([best_rmse_epoch], [best_rmse], color='r', s=100, zorder=5)
ax.text(best_rmse_epoch, best_rmse, f'  {best_rmse:.4f} rad', fontsize=10, va='center')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('RMSE (rad)', fontsize=12)
ax.set_title('Phase RMSE', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Strehl Ratio vs Epoch
ax = axes[0, 1]
ax.plot(df['epoch'], df['val_strehl_ratio'], 'g-', linewidth=2, label='Val Strehl')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Ideal = 1.0')
best_strehl_epoch = df['val_strehl_ratio'].idxmax() + 1
best_strehl = df.loc[best_strehl_epoch-1, 'val_strehl_ratio']
ax.axvline(best_strehl_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best @ Epoch {best_strehl_epoch}')
ax.scatter([best_strehl_epoch], [best_strehl], color='r', s=100, zorder=5)
ax.text(best_strehl_epoch, best_strehl, f'  {best_strehl:.4f}', fontsize=10, va='center')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Strehl Ratio', fontsize=12)
ax.set_title('Strehl Ratio (Higher is Better)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Main Lobe Energy vs Epoch
ax = axes[0, 2]
ax.plot(df['epoch'], df['val_main_lobe_ratio'], 'm-', linewidth=2, label='Val Main Lobe')
ax.axhline(0.651, color='gray', linestyle=':', alpha=0.5, label='Theoretical Max = 0.651')
best_lobe_epoch = df['val_main_lobe_ratio'].idxmax() + 1
best_lobe = df.loc[best_lobe_epoch-1, 'val_main_lobe_ratio']
ax.axvline(best_lobe_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best @ Epoch {best_lobe_epoch}')
ax.scatter([best_lobe_epoch], [best_lobe], color='r', s=100, zorder=5)
ax.text(best_lobe_epoch, best_lobe, f'  {best_lobe:.4f}', fontsize=10, va='center')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Main Lobe Energy Ratio', fontsize=12)
ax.set_title('Main Lobe Energy', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Synthesis Efficiency vs Epoch
ax = axes[1, 0]
ax.plot(df['epoch'], df['val_synthesis_efficiency'], 'c-', linewidth=2, label='Val Syn Eff')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Ideal = 1.0')
best_eff_epoch = df['val_synthesis_efficiency'].idxmax() + 1
best_eff = df.loc[best_eff_epoch-1, 'val_synthesis_efficiency']
ax.axvline(best_eff_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best @ Epoch {best_eff_epoch}')
ax.scatter([best_eff_epoch], [best_eff], color='r', s=100, zorder=5)
ax.text(best_eff_epoch, best_eff, f'  {best_eff:.4f}', fontsize=10, va='center')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Synthesis Efficiency', fontsize=12)
ax.set_title('Synthesis Efficiency', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Loss Components vs Epoch
ax = axes[1, 1]
ax.plot(df['epoch'], df['train_phase'], label='Phase Loss', linewidth=1.5, alpha=0.8)
ax.plot(df['epoch'], df['train_farfield'], label='Farfield Loss', linewidth=1.5, alpha=0.8)
ax.plot(df['epoch'], df['train_comp'], label='Comp Loss', linewidth=1.5, alpha=0.8)
ax.plot(df['epoch'], df['train_unit'], label='Unit Loss', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss Value', fontsize=12)
ax.set_title('Training Loss Components', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('symlog')

# 6. Learning Rate Schedule (if available)
ax = axes[1, 2]
if 'lr' in df.columns:
    ax.plot(df['epoch'], df['lr'], 'orange', linewidth=2, label='Learning Rate')
    ax.axvline(5, color='gray', linestyle='--', alpha=0.3, label='Warmup End')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (CosineAnnealing)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
else:
    # 如果没有lr列，显示总损失
    ax.plot(df['epoch'], df['train_total'], 'orange', linewidth=2, label='Train Total Loss')
    ax.plot(df['epoch'], df['val_total'], 'blue', linewidth=2, label='Val Total Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Total Loss (Train vs Val)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# 保存
output_png = Path("result/figures/cycle_corrected_30epoch_training_curves.png")
output_pdf = Path("result/figures/cycle_corrected_30epoch_training_curves.pdf")
output_png.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"Training curves saved:")
print(f"  PNG: {output_png}")
print(f"  PDF: {output_pdf}")

# 打印关键指标
print(f"\n{'='*60}")
print(f"KEY METRICS FROM 30-EPOCH TRAINING")
print(f"{'='*60}")
print(f"Best RMSE:          {best_rmse:.6f} rad @ Epoch {best_rmse_epoch}")
print(f"Best Strehl:        {best_strehl:.6f} @ Epoch {best_strehl_epoch}")
print(f"Best Main Lobe:     {best_lobe:.6f} @ Epoch {best_lobe_epoch}")
print(f"Best Syn Eff:       {best_eff:.6f} @ Epoch {best_eff_epoch}")
print(f"{'='*60}")

# 与 15 epoch 对比
print(f"\nCOMPARISON WITH 15-EPOCH QUICK VERIFICATION:")
print(f"15-epoch RMSE:  0.074 rad")
print(f"30-epoch RMSE:  {best_rmse:.3f} rad")
improvement = (0.074 - best_rmse) / 0.074 * 100
if improvement > 0:
    print(f"Improvement:    {improvement:.1f}% (better)")
else:
    print(f"Change:         {improvement:.1f}% (similar)")

plt.show()
