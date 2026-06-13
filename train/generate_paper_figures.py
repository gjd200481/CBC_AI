"""
生成论文主图脚本

生成高质量的论文图表，包括：
1. 系统架构示意图
2. 训练曲线对比
3. 补偿效果对比
4. 噪声鲁棒性曲线
5. 消融分析表
6. Attribution热图
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

# 设置论文级别的图表样式
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 路径配置
BASE_DIR = Path(__file__).parent.parent
RESULT_DIR = BASE_DIR / "result"
METRICS_DIR = RESULT_DIR / "metrics"
FIGURES_DIR = RESULT_DIR / "figures"
PAPER_FIGURES_DIR = FIGURES_DIR / "paper_figures"
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def figure1_noise_robustness():
    """
    图1: 噪声鲁棒性对比 (Cycle 43核心结果)
    4个子图: Strehl比、主瓣能量、合成效率、残余相位RMSE
    """
    print("生成图1: 噪声鲁棒性对比...")
    
    # 读取Cycle 43噪声鲁棒性数据
    csv_path = METRICS_DIR / "cycle43_dual_plane_noise_robustness_summary.csv"
    df = pd.read_csv(csv_path)
    
    # 分离两个模型的数据
    cycle41 = df[df['model'] == 'cycle41_stack']
    cycle42 = df[df['model'] == 'cycle42_fusion']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 子图1: Strehl比
    ax = axes[0, 0]
    ax.errorbar(cycle41['noise_sigma'], cycle41['strehl_ratio_mean'], 
                yerr=cycle41['strehl_ratio_std'], 
                marker='o', label='Cycle 41 (Simple Stack)', linewidth=2, capsize=5)
    ax.errorbar(cycle42['noise_sigma'], cycle42['strehl_ratio_mean'], 
                yerr=cycle42['strehl_ratio_std'], 
                marker='s', label='Cycle 42 (Dual-Branch Fusion)', linewidth=2, capsize=5)
    ax.set_xlabel('Noise Level σ')
    ax.set_ylabel('Strehl Ratio')
    ax.set_title('(a) Strehl Ratio vs Noise', fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # 子图2: 主瓣能量
    ax = axes[0, 1]
    ax.errorbar(cycle41['noise_sigma'], cycle41['main_lobe_ratio_mean'], 
                yerr=cycle41['main_lobe_ratio_std'], 
                marker='o', label='Cycle 41', linewidth=2, capsize=5)
    ax.errorbar(cycle42['noise_sigma'], cycle42['main_lobe_ratio_mean'], 
                yerr=cycle42['main_lobe_ratio_std'], 
                marker='s', label='Cycle 42', linewidth=2, capsize=5)
    ax.set_xlabel('Noise Level σ')
    ax.set_ylabel('Main Lobe Energy Ratio')
    ax.set_title('(b) Main Lobe Energy vs Noise', fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # 子图3: 合成效率
    ax = axes[1, 0]
    ax.errorbar(cycle41['noise_sigma'], cycle41['synthesis_efficiency_mean'], 
                yerr=cycle41['synthesis_efficiency_std'], 
                marker='o', label='Cycle 41', linewidth=2, capsize=5)
    ax.errorbar(cycle42['noise_sigma'], cycle42['synthesis_efficiency_mean'], 
                yerr=cycle42['synthesis_efficiency_std'], 
                marker='s', label='Cycle 42', linewidth=2, capsize=5)
    ax.set_xlabel('Noise Level σ')
    ax.set_ylabel('Synthesis Efficiency')
    ax.set_title('(c) Synthesis Efficiency vs Noise', fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # 子图4: 残余相位RMSE
    ax = axes[1, 1]
    ax.errorbar(cycle41['noise_sigma'], cycle41['phase_rmse_rad_mean'], 
                yerr=cycle41['phase_rmse_rad_std'], 
                marker='o', label='Cycle 41', linewidth=2, capsize=5)
    ax.errorbar(cycle42['noise_sigma'], cycle42['phase_rmse_rad_mean'], 
                yerr=cycle42['phase_rmse_rad_std'], 
                marker='s', label='Cycle 42', linewidth=2, capsize=5)
    ax.set_xlabel('Noise Level σ')
    ax.set_ylabel('Residual Phase RMSE (rad)')
    ax.set_title('(d) Residual Phase RMSE vs Noise', fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = PAPER_FIGURES_DIR / "figure1_noise_robustness.png"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))
    print(f"  保存至: {output_path}")
    plt.close()


def figure2_compensation_comparison():
    """
    图2: 补偿效果对比
    柱状图展示补偿前、Cycle37、Cycle41、Cycle42的主要指标
    """
    print("生成图2: 补偿效果对比...")
    
    # 读取Cycle 42 paired评估数据
    csv_path = METRICS_DIR / "cycle42_dual_plane_fusion_paired_summary.csv"
    df = pd.read_csv(csv_path)
    
    # 选择关键模型进行展示
    selected_states = ['before', 'comp0p3_best_rmse', 'cycle41_best_strehl', 'cycle42_best_rmse']
    df_selected = df[df['state'].isin(selected_states)]
    
    # 提取关键指标
    models = ['Before\nCompensation', 'Cycle 37\n(Phase Model)', 'Cycle 41\n(Simple Stack)', 'Cycle 42\n(Dual-Branch)']
    strehl = df_selected['strehl_ratio_mean'].tolist()
    main_lobe = df_selected['main_lobe_ratio_mean'].tolist()
    efficiency = df_selected['synthesis_efficiency_mean'].tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(models))
    width = 0.6
    
    # 子图1: Strehl比
    ax = axes[0]
    bars = ax.bar(x, strehl, width, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    ax.set_ylabel('Strehl Ratio')
    ax.set_title('(a) Strehl Ratio Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{strehl[i]:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 子图2: 主瓣能量
    ax = axes[1]
    bars = ax.bar(x, main_lobe, width, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    ax.set_ylabel('Main Lobe Energy Ratio')
    ax.set_title('(b) Main Lobe Energy Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{main_lobe[i]:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 子图3: 合成效率
    ax = axes[2]
    bars = ax.bar(x, efficiency, width, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    ax.set_ylabel('Synthesis Efficiency')
    ax.set_title('(c) Synthesis Efficiency Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{efficiency[i]:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = PAPER_FIGURES_DIR / "figure2_compensation_comparison.png"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))
    print(f"  保存至: {output_path}")
    plt.close()


def figure3_training_curves():
    """
    图3: Cycle 42训练曲线
    展示训练/验证损失、相位RMSE、Strehl比的变化
    """
    print("生成图3: 训练曲线...")
    
    # 读取Cycle 42训练历史
    csv_path = METRICS_DIR / "cycle42_dual_plane_fusion_7cm_30epoch_history.csv"
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = df['epoch']
    
    # 子图1: 总损失
    ax = axes[0, 0]
    ax.plot(epochs, df['train_total'], marker='o', label='Train Loss', linewidth=2, markersize=4)
    ax.plot(epochs, df['val_total'], marker='s', label='Val Loss', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('(a) Training and Validation Loss', fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # 子图2: 相位RMSE
    ax = axes[0, 1]
    ax.plot(epochs, df['val_rmse_rad'], marker='o', label='Validation RMSE', 
            linewidth=2, markersize=4, color='#ff7f0e')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Phase RMSE (rad)')
    ax.set_title('(b) Validation Phase RMSE', fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # 子图3: Strehl比
    ax = axes[1, 0]
    ax.plot(epochs, df['val_strehl_ratio'], marker='s', label='Validation Strehl', 
            linewidth=2, markersize=4, color='#2ca02c')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Strehl Ratio')
    ax.set_title('(c) Validation Strehl Ratio', fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # 子图4: 合成效率
    ax = axes[1, 1]
    ax.plot(epochs, df['val_synthesis_efficiency'], marker='^', label='Validation Efficiency', 
            linewidth=2, markersize=4, color='#d62728')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Synthesis Efficiency')
    ax.set_title('(d) Validation Synthesis Efficiency', fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = PAPER_FIGURES_DIR / "figure3_training_curves.png"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))
    print(f"  保存至: {output_path}")
    plt.close()


def figure4_attribution_comparison():
    """
    图4: Attribution对比
    展示Cycle 41和Cycle 42的梯度能量分布
    """
    print("生成图4: Attribution对比...")
    
    # 读取attribution概览数据
    csv_path = METRICS_DIR / "cycle43_attribution_overview_64.csv"
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = df['model'].tolist()
    focal_energy = df['plane_1_energy_ratio_mean'].tolist()
    defocus_energy = df['plane_2_energy_ratio_mean'].tolist()
    
    # 子图1: 平面能量占比
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, focal_energy, width, label='Focal Plane', color='#1f77b4')
    bars2 = ax.bar(x + width/2, defocus_energy, width, label='Defocus Plane', color='#ff7f0e')
    
    ax.set_ylabel('Energy Ratio')
    ax.set_title('(a) Gradient Energy Distribution', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Cycle 41\n(Simple Stack)', 'Cycle 42\n(Dual-Branch)'])
    ax.legend(loc='best', frameon=True)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 子图2: Top能量集中度
    ax = axes[1]
    top_energy = df['top_energy_ratio_mean'].tolist()
    colors = ['#2ca02c', '#d62728']
    bars = ax.bar(x, top_energy, width=0.6, color=colors)
    
    ax.set_ylabel('Top Energy Concentration')
    ax.set_title('(b) Gradient Concentration', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Cycle 41\n(Simple Stack)', 'Cycle 42\n(Dual-Branch)'])
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = PAPER_FIGURES_DIR / "figure4_attribution_comparison.png"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))
    print(f"  保存至: {output_path}")
    plt.close()


def generate_all_figures():
    """生成所有论文主图"""
    print("\n" + "="*60)
    print("开始生成论文主图...")
    print("="*60 + "\n")
    
    figure1_noise_robustness()
    figure2_compensation_comparison()
    figure3_training_curves()
    figure4_attribution_comparison()
    
    print("\n" + "="*60)
    print("所有论文主图生成完成！")
    print(f"输出目录: {PAPER_FIGURES_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    generate_all_figures()
