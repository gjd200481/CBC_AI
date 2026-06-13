"""
生成论文LaTeX表格

包括：
1. 消融实验对比表
2. 噪声鲁棒性表
3. 模型性能汇总表
"""

import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "result" / "metrics"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def table1_ablation_study():
    """表1: 消融实验对比"""
    print("生成表1: 消融实验...")

    # 手动整理关键消融结果
    data = {
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
        'Params (M)': [0.17, 0.17, 0.17, 11.34, 11.34, 11.34, 11.34, 5.77],
        'Test RMSE (rad)': [1.027, 0.992, 0.983, 0.964, 0.941, 0.968, 0.953, 0.974],
        'Strehl Ratio': [0.647, 0.664, 0.653, 0.624, 0.658, 0.610, 0.624, 0.683],
        'Main Lobe': [0.519, 0.524, 0.517, 0.514, 0.525, 0.511, 0.514, 0.525],
        'Efficiency': [0.786, 0.793, 0.783, 0.777, 0.794, 0.772, 0.777, 0.796]
    }

    df = pd.DataFrame(data)

    # 生成LaTeX表格
    latex = r"""\begin{table}[htbp]
\centering
\caption{Ablation Study: Model Configurations and Performance}
\label{tab:ablation}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{Test RMSE} & \textbf{Strehl} & \textbf{Main Lobe} & \textbf{Efficiency} \\
 & (M) & (rad) & Ratio & Energy & \\
\midrule
"""

    for _, row in df.iterrows():
        model_name = row['Model'].replace('(', '\\textit{(').replace(')', ')}')
        latex += f"{model_name} & {row['Params (M)']:.2f} & {row['Test RMSE (rad)']:.3f} & "
        latex += f"{row['Strehl Ratio']:.3f} & {row['Main Lobe']:.3f} & {row['Efficiency']:.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path = OUTPUT_DIR / "table1_ablation.tex"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f"  保存至: {output_path}")
    return latex


def table2_noise_robustness():
    """表2: 噪声鲁棒性对比"""
    print("生成表2: 噪声鲁棒性...")

    csv_path = METRICS_DIR / "cycle43_dual_plane_noise_robustness_summary.csv"
    df = pd.read_csv(csv_path)

    cycle41 = df[df['model'] == 'cycle41_stack']
    cycle42 = df[df['model'] == 'cycle42_fusion']

    latex = r"""\begin{table}[htbp]
\centering
\caption{Noise Robustness: Cycle 41 (Simple Stack) vs Cycle 42 (Dual-Branch Fusion)}
\label{tab:noise_robustness}
\begin{tabular}{ccccccc}
\toprule
\multirow{2}{*}{\textbf{Noise $\sigma$}} & \multicolumn{2}{c}{\textbf{Strehl Ratio}} & \multicolumn{2}{c}{\textbf{Main Lobe Energy}} & \multicolumn{2}{c}{\textbf{Efficiency}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
 & C41 & C42 & C41 & C42 & C41 & C42 \\
\midrule
"""

    for i in range(len(cycle41)):
        sigma = cycle41.iloc[i]['noise_sigma']
        c41_strehl = cycle41.iloc[i]['strehl_ratio_mean']
        c42_strehl = cycle42.iloc[i]['strehl_ratio_mean']
        c41_main = cycle41.iloc[i]['main_lobe_ratio_mean']
        c42_main = cycle42.iloc[i]['main_lobe_ratio_mean']
        c41_eff = cycle41.iloc[i]['synthesis_efficiency_mean']
        c42_eff = cycle42.iloc[i]['synthesis_efficiency_mean']

        # 高亮Cycle42优于Cycle41的情况
        strehl_better = r"\textbf{" if c42_strehl > c41_strehl else ""
        main_better = r"\textbf{" if c42_main > c41_main else ""
        eff_better = r"\textbf{" if c42_eff > c41_eff else ""
        close_tag = "}" if any([c42_strehl > c41_strehl, c42_main > c41_main, c42_eff > c41_eff]) else ""

        latex += f"{sigma:.3f} & {c41_strehl:.3f} & {strehl_better}{c42_strehl:.3f}{close_tag} & "
        latex += f"{c41_main:.3f} & {main_better}{c42_main:.3f}{close_tag} & "
        latex += f"{c41_eff:.3f} & {eff_better}{c42_eff:.3f}{close_tag} \\\\\n"

    latex += r"""\bottomrule
\multicolumn{7}{l}{\textbf{Bold} values indicate where Cycle 42 outperforms Cycle 41.} \\
\end{tabular}
\end{table}
"""

    output_path = OUTPUT_DIR / "table2_noise_robustness.tex"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f"  保存至: {output_path}")
    return latex


def table3_main_results():
    """表3: 主要结果汇总"""
    print("生成表3: 主要结果...")

    csv_path = METRICS_DIR / "cycle42_dual_plane_fusion_paired_summary.csv"
    df = pd.read_csv(csv_path)

    states = ['before', 'comp0p3_best_rmse', 'cycle41_best_strehl', 'cycle42_best_rmse']
    state_names = ['Before Compensation', 'Cycle 37 (Phase Model)',
                   'Cycle 41 (Simple Stack)', 'Cycle 42 (Dual-Branch)']

    latex = r"""\begin{table}[htbp]
\centering
\caption{Main Results: Compensation Performance Comparison}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Strehl} & \textbf{Main Lobe} & \textbf{Efficiency} & \textbf{Residual RMSE} \\
 & Ratio & Energy & & (rad) \\
\midrule
"""

    for state, name in zip(states, state_names):
        row = df[df['state'] == state].iloc[0]
        strehl = row['strehl_ratio_mean']
        main = row['main_lobe_ratio_mean']
        eff = row['synthesis_efficiency_mean']
        rmse = row['phase_rmse_rad_mean']

        # 高亮最佳值
        best_strehl = strehl == df[df['state'].isin(states)]['strehl_ratio_mean'].max()
        best_main = main == df[df['state'].isin(states)]['main_lobe_ratio_mean'].max()
        best_eff = eff == df[df['state'].isin(states)]['synthesis_efficiency_mean'].max()
        best_rmse = rmse == df[df['state'].isin(states)]['phase_rmse_rad_mean'].min()

        strehl_str = f"\\textbf{{{strehl:.3f}}}" if best_strehl else f"{strehl:.3f}"
        main_str = f"\\textbf{{{main:.3f}}}" if best_main else f"{main:.3f}"
        eff_str = f"\\textbf{{{eff:.3f}}}" if best_eff else f"{eff:.3f}"
        rmse_str = f"\\textbf{{{rmse:.3f}}}" if best_rmse else f"{rmse:.3f}"

        latex += f"{name} & {strehl_str} & {main_str} & {eff_str} & {rmse_str} \\\\\n"

    latex += r"""\bottomrule
\multicolumn{5}{l}{\textbf{Bold} values indicate the best performance for each metric.} \\
\end{tabular}
\end{table}
"""

    output_path = OUTPUT_DIR / "table3_main_results.tex"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f"  保存至: {output_path}")
    return latex


def main():
    """生成所有LaTeX表格"""
    print("\n" + "="*70)
    print("开始生成论文LaTeX表格")
    print("="*70 + "\n")

    table1_ablation_study()
    print()

    table2_noise_robustness()
    print()

    table3_main_results()
    print()

    print("="*70)
    print(f"所有表格已生成！输出目录: {OUTPUT_DIR}")
    print("="*70)
    print("\n表格清单：")
    print("  - table1_ablation.tex         : 消融实验对比表")
    print("  - table2_noise_robustness.tex : 噪声鲁棒性对比表")
    print("  - table3_main_results.tex     : 主要结果汇总表")
    print("\n")


if __name__ == "__main__":
    main()
