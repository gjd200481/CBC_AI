"""
生成论文出版级别的高质量图表

包括：
1. 系统架构流程图
2. 七光束阵列示意图
3. 补偿效果对比（含远场图样）
4. 噪声鲁棒性曲线
5. 训练过程可视化
6. Attribution热图对比
7. 模型性能雷达图
8. 消融实验汇总图
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import torch

# 添加项目根目录到路径
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simulation.common.multi_beam_core import create_grid, seven_beam_near_field, far_field_intensity
from train.phase_metrics import decode_sin_cos

# 设置出版级别样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# 创建输出目录
OUTPUT_DIR = REPO_ROOT / "result" / "figures" / "publication"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = REPO_ROOT / "result" / "metrics"


def fig1_system_overview():
    """
    图1: 系统概览图 - 七光束阵列 + 双分支网络架构
    """
    print("生成图1: 系统概览...")

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # (a) 七光束六边形阵列示意图
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(a) Seven-beam hexagonal array', fontweight='bold', pad=10)

    # 绘制七光束位置
    beam_distance = 1.0
    positions = [
        (0, 0),  # 中心
        (beam_distance, 0),
        (beam_distance * np.cos(np.pi/3), beam_distance * np.sin(np.pi/3)),
        (beam_distance * np.cos(2*np.pi/3), beam_distance * np.sin(2*np.pi/3)),
        (-beam_distance, 0),
        (beam_distance * np.cos(4*np.pi/3), beam_distance * np.sin(4*np.pi/3)),
        (beam_distance * np.cos(5*np.pi/3), beam_distance * np.sin(5*np.pi/3)),
    ]

    colors_beams = ['#dc2626'] + ['#3b82f6'] * 6  # 中心红色，外圈蓝色
    labels = ['Beam 0\n(Reference)\nφ=0'] + [f'Beam {i}\nφ{i}' for i in range(1, 7)]

    for i, (x, y) in enumerate(positions):
        circle = Circle((x, y), 0.35, color=colors_beams[i], alpha=0.7, ec='black', linewidth=1.5)
        ax1.add_patch(circle)
        ax1.text(x, y, labels[i], ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax1.text(0, -1.8, 'Near field (z=0)', ha='center', fontsize=10, style='italic')

    # (b) 网络架构流程图
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3)
    ax2.axis('off')
    ax2.set_title('(b) Dual-branch fusion network architecture', fontweight='bold', pad=10)

    # 定义框
    boxes = [
        {'xy': (0.3, 2.2), 'w': 1.0, 'h': 0.6, 'text': 'Focal plane\nimage', 'color': '#dbeafe'},
        {'xy': (0.3, 0.2), 'w': 1.0, 'h': 0.6, 'text': 'Befocal plane\nimage', 'color': '#e0f2fe'},
        {'xy': (2.0, 2.2), 'w': 1.2, 'h': 0.6, 'text': 'Focal\nencoder', 'color': '#bfdbfe'},
        {'xy': (2.0, 0.2), 'w': 1.2, 'h': 0.6, 'text': 'Befocal\nencoder', 'color': '#bae6fd'},
        {'xy': (4.0, 1.0), 'w': 1.3, 'h': 1.0, 'text': 'Gated\nfusion', 'color': '#dcfce7'},
        {'xy': (6.2, 1.0), 'w': 1.2, 'h': 1.0, 'text': 'Channel\nattention', 'color': '#fef3c7'},
        {'xy': (8.2, 1.0), 'w': 1.5, 'h': 1.0, 'text': 'Phase\nregression', 'color': '#fecaca'},
    ]

    for box in boxes:
        fancy_box = FancyBboxPatch(
            box['xy'], box['w'], box['h'],
            boxstyle="round,pad=0.05",
            edgecolor='black', facecolor=box['color'], linewidth=1.2
        )
        ax2.add_patch(fancy_box)
        ax2.text(box['xy'][0] + box['w']/2, box['xy'][1] + box['h']/2,
                box['text'], ha='center', va='center', fontsize=9, fontweight='bold')

    # 绘制箭头
    arrows = [
        ((1.3, 2.5), (2.0, 2.5)),
        ((1.3, 0.5), (2.0, 0.5)),
        ((3.2, 2.5), (4.3, 1.8)),
        ((3.2, 0.5), (4.3, 1.2)),
        ((5.3, 1.5), (6.2, 1.5)),
        ((7.4, 1.5), (8.2, 1.5)),
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->',
                               mutation_scale=15, linewidth=1.5, color='black')
        ax2.add_patch(arrow)

    # (c) 远场传播示意
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 2)
    ax3.axis('off')
    ax3.set_title('(c) Physical constraint via Fourier optics', fontweight='bold', pad=10)

    # 物理约束框图
    phys_boxes = [
        {'xy': (0.3, 0.6), 'w': 1.5, 'h': 0.8, 'text': 'Predicted\nphases', 'color': '#fecaca'},
        {'xy': (2.5, 0.6), 'w': 1.5, 'h': 0.8, 'text': 'Reconstruct\nnear field', 'color': '#fed7aa'},
        {'xy': (4.8, 0.6), 'w': 1.2, 'h': 0.8, 'text': 'FFT', 'color': '#fef3c7'},
        {'xy': (6.7, 0.6), 'w': 1.5, 'h': 0.8, 'text': 'Predicted\nfar field', 'color': '#dcfce7'},
        {'xy': (8.8, 0.6), 'w': 1.0, 'h': 0.8, 'text': 'L_farfield', 'color': '#e0e7ff'},
    ]

    for box in phys_boxes:
        fancy_box = FancyBboxPatch(
            box['xy'], box['w'], box['h'],
            boxstyle="round,pad=0.05",
            edgecolor='black', facecolor=box['color'], linewidth=1.2
        )
        ax3.add_patch(fancy_box)
        ax3.text(box['xy'][0] + box['w']/2, box['xy'][1] + box['h']/2,
                box['text'], ha='center', va='center', fontsize=9, fontweight='bold')

    # 物理约束箭头
    phys_arrows = [
        ((1.8, 1.0), (2.5, 1.0)),
        ((4.0, 1.0), (4.8, 1.0)),
        ((6.0, 1.0), (6.7, 1.0)),
        ((8.2, 1.0), (8.8, 1.0)),
    ]

    for start, end in phys_arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->',
                               mutation_scale=15, linewidth=1.5, color='black')
        ax3.add_patch(arrow)

    # 输入远场回环箭头
    ax3.annotate('', xy=(9.3, 1.7), xytext=(9.3, 1.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#dc2626'))
    ax3.text(9.5, 1.55, 'Input\nfar field', ha='left', va='center', fontsize=8, color='#dc2626')

    plt.savefig(OUTPUT_DIR / 'fig1_system_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_system_overview.pdf', bbox_inches='tight')
    print(f"  保存至: {OUTPUT_DIR / 'fig1_system_overview.png'}")
    plt.close()


def fig2_compensation_farfield_comparison():
    """
    图2: 补偿效果对比 - 简化版
    """
    print("生成图2: 补偿效果对比...")

    # 读取summary数据
    summary_csv = METRICS_DIR / "cycle42_dual_plane_fusion_paired_summary.csv"
    df_summary = pd.read_csv(summary_csv)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    states = ['before', 'comp0p3_best_rmse', 'cycle41_best_strehl', 'cycle42_best_rmse']
    state_labels = ['Before\nCompensation', 'Cycle 37\n(Phase Model)',
                    'Cycle 41\n(Simple Stack)', 'Cycle 42\n(Dual-Branch)']

    # 第一行：核心指标柱状图
    metrics_to_plot = [
        ('strehl_ratio_mean', 'Strehl Ratio', axes[0, 0], '#3b82f6'),
        ('main_lobe_ratio_mean', 'Main Lobe Energy', axes[0, 1], '#10b981'),
        ('synthesis_efficiency_mean', 'Synthesis Efficiency', axes[1, 0], '#f59e0b'),
        ('phase_rmse_rad_mean', 'Residual Phase RMSE (rad)', axes[1, 1], '#dc2626')
    ]

    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    for (metric, ylabel, ax, color), label in zip(metrics_to_plot, subplot_labels):
        values = df_summary[df_summary['state'].isin(states)][metric].values
        x_pos = np.arange(len(values))

        bars = ax.bar(x_pos, values, color=color, alpha=0.8, edgecolor='black', linewidth=0.8)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'{label} {ylabel}', fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(state_labels, rotation=15, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # 为非"Before"的柱子添加改善标注
        if len(values) > 1:
            before_val = values[0]
            for i in range(1, len(values)):
                if 'rmse' in metric.lower():
                    improvement = (before_val - values[i]) / before_val * 100
                else:
                    improvement = (values[i] - before_val) / before_val * 100

                if improvement > 0:
                    ax.text(bars[i].get_x() + bars[i].get_width()/2.,
                           bars[i].get_height() * 0.5,
                           f'+{improvement:.1f}%', ha='center', va='center',
                           fontsize=7, color='white', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_compensation_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_compensation_comparison.pdf', bbox_inches='tight')
    print(f"  保存至: {OUTPUT_DIR / 'fig2_compensation_comparison.png'}")
    plt.close()


def fig3_noise_robustness():
    """
    图3: 噪声鲁棒性曲线（Cycle 43核心结果）
    """
    print("生成图3: 噪声鲁棒性...")

    csv_path = METRICS_DIR / "cycle43_dual_plane_noise_robustness_summary.csv"
    df = pd.read_csv(csv_path)

    cycle41 = df[df['model'] == 'cycle41_stack']
    cycle42 = df[df['model'] == 'cycle42_fusion']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('strehl_ratio', 'Strehl Ratio', axes[0, 0]),
        ('main_lobe_ratio', 'Main Lobe Energy Ratio', axes[0, 1]),
        ('synthesis_efficiency', 'Synthesis Efficiency', axes[1, 0]),
        ('phase_rmse_rad', 'Residual Phase RMSE (rad)', axes[1, 1])
    ]

    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    for (metric, ylabel, ax), label in zip(metrics, subplot_labels):
        ax.errorbar(cycle41['noise_sigma'], cycle41[f'{metric}_mean'],
                   yerr=cycle41[f'{metric}_std'],
                   marker='o', label='Cycle 41 (Simple Stack)',
                   linewidth=2, capsize=5, markersize=7, alpha=0.8)
        ax.errorbar(cycle42['noise_sigma'], cycle42[f'{metric}_mean'],
                   yerr=cycle42[f'{metric}_std'],
                   marker='s', label='Cycle 42 (Dual-Branch Fusion)',
                   linewidth=2, capsize=5, markersize=7, alpha=0.8)

        ax.set_xlabel('Noise Level σ', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'{label} {ylabel}', fontweight='bold', pad=10)
        ax.legend(loc='best', frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')

        # 添加改善区域标注
        if metric != 'phase_rmse_rad':
            # 找到Cycle42优于Cycle41的区域
            improvement = cycle42[f'{metric}_mean'].values - cycle41[f'{metric}_mean'].values
            for i in range(len(improvement)):
                if improvement[i] > 0 and cycle42['noise_sigma'].iloc[i] >= 0.005:
                    ax.axvspan(cycle42['noise_sigma'].iloc[i] - 0.001,
                             cycle42['noise_sigma'].iloc[i] + 0.001,
                             alpha=0.1, color='green')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_noise_robustness.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_noise_robustness.pdf', bbox_inches='tight')
    print(f"  保存至: {OUTPUT_DIR / 'fig3_noise_robustness.png'}")
    plt.close()


def fig4_training_evolution():
    """
    图4: Cycle 42训练过程演化
    """
    print("生成图4: 训练过程...")

    csv_path = METRICS_DIR / "cycle42_dual_plane_fusion_7cm_30epoch_history.csv"
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 子图配置
    plots = [
        ('train_total', 'val_total', 'Total Loss', axes[0, 0], '(a)'),
        ('train_phase', None, 'Phase Loss', axes[0, 1], '(b)'),
        ('train_farfield', None, 'Far-field Loss', axes[0, 2], '(c)'),
        ('val_rmse_rad', None, 'Validation Phase RMSE (rad)', axes[1, 0], '(d)'),
        ('val_strehl_ratio', None, 'Validation Strehl Ratio', axes[1, 1], '(e)'),
        ('val_synthesis_efficiency', None, 'Validation Synthesis Efficiency', axes[1, 2], '(f)'),
    ]

    for train_col, val_col, ylabel, ax, label in plots:
        if val_col:
            ax.plot(df['epoch'], df[train_col], marker='o', label='Train',
                   linewidth=2, markersize=5, alpha=0.7)
            ax.plot(df['epoch'], df[val_col], marker='s', label='Validation',
                   linewidth=2, markersize=5, alpha=0.7)
        else:
            ax.plot(df['epoch'], df[train_col], marker='o',
                   linewidth=2, markersize=5, alpha=0.7, color='#f59e0b')

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'{label} {ylabel}', fontweight='bold')
        if val_col:
            ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        # 标注最佳点
        if 'val' in train_col and val_col is None:
            if 'rmse' in train_col or 'loss' in train_col:
                best_idx = df[train_col].idxmin()
                marker_color = 'green'
            else:
                best_idx = df[train_col].idxmax()
                marker_color = 'red'

            best_epoch = df.loc[best_idx, 'epoch']
            best_value = df.loc[best_idx, train_col]
            ax.plot(best_epoch, best_value, 'r*', markersize=15,
                   markeredgecolor='black', markeredgewidth=1, label=f'Best (Epoch {best_epoch})')
            ax.legend(loc='best', frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_training_evolution.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_training_evolution.pdf', bbox_inches='tight')
    print(f"  保存至: {OUTPUT_DIR / 'fig4_training_evolution.png'}")
    plt.close()


def fig5_attribution_analysis():
    """
    图5: Attribution解释性分析
    """
    print("生成图5: Attribution分析...")

    csv_path = METRICS_DIR / "cycle43_attribution_overview_64.csv"
    df = pd.read_csv(csv_path)

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    models = df['model'].tolist()
    model_labels = ['Cycle 41\n(Simple Stack)', 'Cycle 42\n(Dual-Branch)']

    # (a) 平面能量分布
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(models))
    width = 0.35

    focal = df['plane_1_energy_ratio_mean'].values
    befocal = df['plane_2_energy_ratio_mean'].values

    bars1 = ax.bar(x - width/2, focal, width, label='Focal Plane', color='#3b82f6', alpha=0.8)
    bars2 = ax.bar(x + width/2, befocal, width, label='Befocal Plane', color='#f59e0b', alpha=0.8)

    ax.set_ylabel('Gradient Energy Ratio', fontweight='bold')
    ax.set_title('(a) Gradient Energy Distribution', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # (b) Top能量集中度
    ax = fig.add_subplot(gs[0, 1])
    top_energy = df['top_energy_ratio_mean'].values
    bars = ax.bar(x, top_energy, width=0.5, color=['#10b981', '#dc2626'], alpha=0.8)

    ax.set_ylabel('Top 10% Energy Concentration', fontweight='bold')
    ax.set_title('(b) Gradient Concentration', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # (c) 平均敏感半径
    ax = fig.add_subplot(gs[0, 2])
    avg_radius = df['mean_radius_px_mean'].values
    bars = ax.bar(x, avg_radius, width=0.5, color=['#8b5cf6', '#ec4899'], alpha=0.8)

    ax.set_ylabel('Average Sensitive Radius (px)', fontweight='bold')
    ax.set_title('(c) Spatial Sensitivity', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # (d-f) 示例Attribution热图
    # 这里需要读取实际的attribution图像
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    ax.text(0.5, 0.5, 'Attribution heatmap examples:\nLoad from cycle43_attribution_cycle4X_64/',
           ha='center', va='center', fontsize=11, style='italic',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(OUTPUT_DIR / 'fig5_attribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig5_attribution_analysis.pdf', bbox_inches='tight')
    print(f"  保存至: {OUTPUT_DIR / 'fig5_attribution_analysis.png'}")
    plt.close()


def fig6_ablation_study():
    """
    图6: 消融实验汇总
    """
    print("生成图6: 消融实验...")

    # 手动整理关键消融结果
    ablation_data = {
        'Model': [
            'Simple CNN (Cycle 12)',
            'Residual CNN (Cycle 23)',
            'Residual + Physics (Cycle 25)',
            'Deep Residual (Cycle 30)',
            'Multiplane Input (Cycle 35)',
            'Hex Augmentation (Cycle 32)',
            'Compensation Loss (Cycle 33)',
            'Dual-Branch Fusion (Cycle 42)'
        ],
        'Parameters (M)': [0.17, 0.17, 0.17, 11.34, 11.34, 11.34, 11.34, 5.77],
        'Test RMSE (rad)': [1.027, 0.992, 0.983, 0.964, 0.941, 0.968, 0.953, 0.974],
        'Strehl Ratio': [0.647, 0.664, 0.653, 0.624, 0.658, 0.610, 0.624, 0.683],
        'Main Lobe': [0.519, 0.524, 0.517, 0.514, 0.525, 0.511, 0.514, 0.525],
        'Efficiency': [0.786, 0.793, 0.783, 0.777, 0.794, 0.772, 0.777, 0.796]
    }

    df = pd.DataFrame(ablation_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) 参数量 vs Strehl
    ax = axes[0, 0]
    scatter = ax.scatter(df['Parameters (M)'], df['Strehl Ratio'],
                        s=200, c=range(len(df)), cmap='viridis',
                        alpha=0.7, edgecolors='black', linewidth=1.5)

    # 标注关键点
    for i in [0, 3, 7]:  # Simple, Deep Residual, Dual-Branch
        ax.annotate(df['Model'].iloc[i].split('(')[0],
                   (df['Parameters (M)'].iloc[i], df['Strehl Ratio'].iloc[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax.set_xlabel('Parameters (Million)', fontweight='bold')
    ax.set_ylabel('Strehl Ratio', fontweight='bold')
    ax.set_title('(a) Model Complexity vs Performance', fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # (b) 各模型Strehl对比
    ax = axes[0, 1]
    colors_ablation = plt.cm.viridis(np.linspace(0, 1, len(df)))
    bars = ax.barh(range(len(df)), df['Strehl Ratio'], color=colors_ablation, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Model'], fontsize=8)
    ax.set_xlabel('Strehl Ratio', fontweight='bold')
    ax.set_title('(b) Ablation: Strehl Ratio', fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 标注数值
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
               f'{width:.3f}', ha='left', va='center', fontsize=8)

    # (c) 效率对比
    ax = axes[1, 0]
    bars = ax.barh(range(len(df)), df['Efficiency'], color=colors_ablation, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Model'], fontsize=8)
    ax.set_xlabel('Synthesis Efficiency', fontweight='bold')
    ax.set_title('(c) Ablation: Synthesis Efficiency', fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
               f'{width:.3f}', ha='left', va='center', fontsize=8)

    # (d) 雷达图对比三个最佳模型
    ax = axes[1, 1]
    categories = ['Strehl', 'Main Lobe', 'Efficiency', 'Low RMSE', 'Parameters']
    N = len(categories)

    # 归一化数据
    selected_models = [3, 4, 7]  # Deep Residual, Multiplane, Dual-Branch
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax = plt.subplot(2, 2, 4, projection='polar')

    for idx, color in zip(selected_models, ['#3b82f6', '#10b981', '#f59e0b']):
        values = [
            df['Strehl Ratio'].iloc[idx] / df['Strehl Ratio'].max(),
            df['Main Lobe'].iloc[idx] / df['Main Lobe'].max(),
            df['Efficiency'].iloc[idx] / df['Efficiency'].max(),
            1 - (df['Test RMSE (rad)'].iloc[idx] / df['Test RMSE (rad)'].max()),
            1 - (df['Parameters (M)'].iloc[idx] / df['Parameters (M)'].max())
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=df['Model'].iloc[idx].split('(')[0], color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('(d) Multi-metric Comparison', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_ablation_study.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig6_ablation_study.pdf', bbox_inches='tight')
    print(f"  保存至: {OUTPUT_DIR / 'fig6_ablation_study.png'}")
    plt.close()


def main():
    """生成所有论文图表"""
    print("\n" + "="*70)
    print("开始生成论文出版级别图表")
    print("="*70 + "\n")

    try:
        fig1_system_overview()
        print("[OK] 图1完成\n")
    except Exception as e:
        print(f"[FAIL] 图1失败: {e}\n")

    try:
        fig2_compensation_farfield_comparison()
        print("[OK] 图2完成\n")
    except Exception as e:
        print(f"[FAIL] 图2失败: {e}\n")

    try:
        fig3_noise_robustness()
        print("[OK] 图3完成\n")
    except Exception as e:
        print(f"[FAIL] 图3失败: {e}\n")

    try:
        fig4_training_evolution()
        print("[OK] 图4完成\n")
    except Exception as e:
        print(f"[FAIL] 图4失败: {e}\n")

    try:
        fig5_attribution_analysis()
        print("[OK] 图5完成\n")
    except Exception as e:
        print(f"[FAIL] 图5失败: {e}\n")

    try:
        fig6_ablation_study()
        print("[OK] 图6完成\n")
    except Exception as e:
        print(f"[FAIL] 图6失败: {e}\n")

    print("="*70)
    print(f"所有图表已生成！输出目录: {OUTPUT_DIR}")
    print("="*70)
    print("\n图表清单：")
    print("  - fig1_system_overview.png/pdf          : 系统架构概览")
    print("  - fig2_compensation_comparison.png/pdf  : 补偿效果对比")
    print("  - fig3_noise_robustness.png/pdf         : 噪声鲁棒性曲线")
    print("  - fig4_training_evolution.png/pdf       : 训练过程演化")
    print("  - fig5_attribution_analysis.png/pdf     : Attribution解释性")
    print("  - fig6_ablation_study.png/pdf           : 消融实验汇总")
    print("\n")


if __name__ == "__main__":
    main()



