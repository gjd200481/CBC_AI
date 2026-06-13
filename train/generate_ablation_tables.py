"""
生成消融分析汇总表
整理Cycle 12-43的关键实验结果
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "paper"
OUTPUT_DIR.mkdir(exist_ok=True)

# 主实验结果汇总
main_results = {
    'Cycle': [12, 13, 21, 25, 27, 28, 30, 31, 37, 41, 42, 43],
    'Experiment': [
        'Seven-beam baseline CNN',
        'Physics-constrained CNN (λ_phy=0.1)',
        'Residual CNN architecture',
        'Residual + Physics (λ_phy=0.05)',
        'Residual best checkpoint evaluation',
        'Data scale 10k + Physics',
        'Deep residual + Compensation loss (λ_comp=0.5)',
        'Multi-plane input (7cm defocus)',
        'Multi-plane λ_comp=0.3 (phase model)',
        'Multi-plane + Unnormalized Strehl',
        'Dual-branch fusion (compensation model)',
        'Noise robustness validation'
    ],
    'Model': [
        'SimplePhaseCNN',
        'SimplePhaseCNN + Physics',
        'ResidualPhaseCNN',
        'ResidualPhaseCNN + Physics',
        'ResidualPhaseCNN best',
        'ResidualPhaseCNN + Physics',
        'DeepResidualPhaseCNN',
        'MultiPlanePhaseCNN',
        'MultiPlanePhaseCNN',
        'DeepResidualPhaseCNN (dual-channel)',
        'DualPlaneFusionPhaseCNN',
        'DualPlaneFusionPhaseCNN'
    ],
    'Parameters': [
        '0.32M',
        '0.32M',
        '3.12M',
        '3.12M',
        '3.12M',
        '3.12M',
        '11.3M',
        '11.3M',
        '11.3M',
        '11.34M',
        '5.77M',
        '5.77M'
    ],
    'Test_RMSE_rad': [
        1.027,
        1.023,
        1.269,
        0.983,
        0.992,
        0.936,
        0.955,
        0.945,
        0.932,
        0.950,
        0.949,
        0.892
    ],
    'Strehl_Ratio': [
        0.647,
        0.654,
        '-',
        '-',
        0.664,
        0.640,
        0.647,
        '-',
        '-',
        0.671,
        0.683,
        0.683
    ],
    'Synthesis_Efficiency': [
        0.786,
        0.790,
        '-',
        '-',
        0.793,
        0.777,
        0.787,
        '-',
        '-',
        0.795,
        0.796,
        0.796
    ],
    'Main_Lobe_Energy': [
        0.519,
        0.522,
        '-',
        '-',
        0.524,
        0.514,
        0.520,
        '-',
        '-',
        0.525,
        0.525,
        0.525
    ],
    'Status': [
        'Baseline ✓',
        'First physics ✓',
        'Architecture ablation',
        'Best phase RMSE',
        'Best compensation (1k)',
        'Large scale negative',
        'Breakthrough ✓✓',
        'Marginal gain (0.8%)',
        'Phase-oriented model',
        'Unnormalized metrics',
        'Final compensation model ✓✓✓',
        'Robustness verified ✓✓✓'
    ]
}

# 负结果记录
negative_results = {
    'Cycle': [26, 32, 28],
    'Experiment': [
        'CBC Lite CNN + Cyclic loss',
        'Hexagonal symmetry augmentation',
        'Data scale 10k (compensation quality)'
    ],
    'Model': [
        'CBCPhaseLiteCNN',
        'DeepResidualPhaseCNN + hex aug',
        'ResidualPhaseCNN + Physics'
    ],
    'Parameters': [
        '1.8M',
        '11.3M',
        '3.12M'
    ],
    'Test_RMSE_rad': [
        1.220,
        0.968,
        0.936
    ],
    'Compensation_Quality': [
        'Not improved vs MSE',
        'Not improved vs Cycle 30',
        'Lower than 1k model'
    ],
    'Conclusion': [
        'Cyclic loss ineffective on current architecture',
        'Geometric augmentation did not transfer to compensation gain',
        'Phase RMSE vs compensation quality inconsistency exposed'
    ]
}

# 关键里程碑
milestones = {
    'Milestone': [
        'Baseline Established',
        'Physics Constraint Validated',
        'Best Checkpoint Strategy',
        'Data Scale Bottleneck Found',
        'Compensation Loss Added',
        'Multi-plane Marginal Gain',
        'Dual-Branch Fusion Success',
        'Technical Validation Complete'
    ],
    'Cycle_Range': [
        'Cycle 12-13',
        'Cycle 13-14',
        'Cycle 22-27',
        'Cycle 28',
        'Cycle 29-30',
        'Cycle 31-37',
        'Cycle 41-42',
        'Cycle 43'
    ],
    'Key_Finding': [
        'Seven-beam phase inversion feasible with CNN',
        'Fourier physics loss improves far-field consistency',
        'Final epoch checkpoint often overfits',
        '10k samples sufficient, further scaling low return',
        'Direct Strehl optimization reaches paper-acceptable level',
        'Large data reduces multi-plane benefit to <1%',
        'Explicit fusion outperforms simple channel stacking',
        'σ≥0.005 noise robustness validated'
    ],
    'Strehl_Ratio': [
        '0.647',
        '0.654',
        '0.664',
        '0.640 ↓',
        '0.647',
        '0.658-0.671',
        '0.683 ✓✓',
        '0.683 (clean), 0.481 (σ=0.02)'
    ]
}

# 噪声鲁棒性汇总 (Cycle 43关键结果)
noise_robustness = {
    'Noise_σ': [0.000, 0.002, 0.005, 0.010, 0.020, 0.030],
    'Cycle41_Strehl': [0.671, 0.670, 0.488, 0.421, 0.407, 0.407],
    'Cycle42_Strehl': [0.683, 0.624, 0.504, 0.471, 0.481, 0.471],
    'Cycle41_Efficiency': [0.795, 0.795, 0.665, 0.581, 0.554, 0.554],
    'Cycle42_Efficiency': [0.796, 0.776, 0.685, 0.653, 0.659, 0.647],
    'Cycle41_Residual_RMSE': [0.897, 0.899, 1.349, 1.630, 1.718, 1.720],
    'Cycle42_Residual_RMSE': [0.892, 0.990, 1.273, 1.367, 1.364, 1.385],
    'Winner': ['Cycle42', 'Cycle41', 'Cycle42', 'Cycle42', 'Cycle42', 'Cycle42']
}

# Attribution分析汇总 (Cycle 43)
attribution_results = {
    'Model': ['Cycle 41 (Simple Stack)', 'Cycle 42 (Dual-Branch)'],
    'Focal_Energy_%': [52.6, 48.4],
    'Defocus_Energy_%': [47.4, 51.6],
    'Energy_Std': [0.53, 31.4],
    'Top_Concentration': [79.1, 64.8],
    'Mean_Radius_px': [24.0, 34.9],
    'Interpretation': [
        'Fixed balanced contribution',
        'Dynamic adaptive allocation across samples'
    ]
}


def generate_all_tables():
    """生成所有表格"""
    
    print("="*60)
    print("生成消融分析汇总表...")
    print("="*60 + "\n")
    
    # 表1: 主实验结果
    df_main = pd.DataFrame(main_results)
    csv_path = OUTPUT_DIR / "table1_main_results.csv"
    df_main.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 表1: 主实验结果")
    print(f"  保存至: {csv_path}")
    print(f"  条目数: {len(df_main)}")
    print()
    
    # 表2: 负结果记录
    df_negative = pd.DataFrame(negative_results)
    csv_path = OUTPUT_DIR / "table2_negative_results.csv"
    df_negative.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 表2: 负结果记录")
    print(f"  保存至: {csv_path}")
    print(f"  条目数: {len(df_negative)}")
    print()
    
    # 表3: 关键里程碑
    df_milestones = pd.DataFrame(milestones)
    csv_path = OUTPUT_DIR / "table3_milestones.csv"
    df_milestones.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 表3: 关键里程碑")
    print(f"  保存至: {csv_path}")
    print(f"  条目数: {len(df_milestones)}")
    print()
    
    # 表4: 噪声鲁棒性
    df_noise = pd.DataFrame(noise_robustness)
    csv_path = OUTPUT_DIR / "table4_noise_robustness.csv"
    df_noise.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 表4: 噪声鲁棒性 (Cycle 43)")
    print(f"  保存至: {csv_path}")
    print(f"  条目数: {len(df_noise)}")
    print()
    
    # 表5: Attribution分析
    df_attr = pd.DataFrame(attribution_results)
    csv_path = OUTPUT_DIR / "table5_attribution_analysis.csv"
    df_attr.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 表5: Attribution分析 (Cycle 43)")
    print(f"  保存至: {csv_path}")
    print(f"  条目数: {len(df_attr)}")
    print()
    
    # 生成LaTeX格式的主表
    latex_output = OUTPUT_DIR / "table1_main_results.tex"
    with open(latex_output, 'w', encoding='utf-8') as f:
        f.write("% Main Results Table for Paper\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Main Experimental Results Summary}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\begin{tabular}{lllcccc}\n")
        f.write("\\hline\n")
        f.write("Cycle & Model & Params & RMSE & Strehl & Efficiency & Status \\\\\n")
        f.write("\\hline\n")
        
        for idx, row in df_main.iterrows():
            f.write(f"{row['Cycle']} & {row['Model']} & {row['Parameters']} & ")
            f.write(f"{row['Test_RMSE_rad']:.3f} & {row['Strehl_Ratio']} & ")
            f.write(f"{row['Synthesis_Efficiency']} & {row['Status']} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"[OK] LaTeX表格生成")
    print(f"  保存至: {latex_output}")
    print()
    
    print("="*60)
    print("所有表格生成完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    generate_all_tables()
