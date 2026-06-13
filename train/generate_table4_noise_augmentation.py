"""
生成Table 4: 噪声增强对比表
"""

import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
csv_path = REPO_ROOT / "result" / "metrics" / "cycle44_vs_cycle42_noise_comparison.csv"
df = pd.read_csv(csv_path)

# 选择关键噪声等级
key_noise_levels = [0.0, 0.002, 0.005, 0.02]

latex = r"""\begin{table*}[htbp]
\centering
\caption{Noise Augmentation Training Results: Cycle 42 (Baseline) vs Cycle 44 (Noise-Augmented)}
\label{tab:noise_augmentation}
\begin{tabular}{cccccccc}
\toprule
\multirow{2}{*}{\textbf{Noise $\sigma$}} & \multirow{2}{*}{\textbf{Model}} & \textbf{Strehl} & \textbf{Main Lobe} & \textbf{Synthesis} & \textbf{Residual RMSE} & \multirow{2}{*}{\textbf{Improvement}} \\
 &  & \textbf{Ratio} & \textbf{Energy} & \textbf{Efficiency} & \textbf{(rad)} & \\
\midrule
"""

for noise_sigma in key_noise_levels:
    # Cycle 42 baseline
    c42_row = df[(df['model'] == 'cycle42_baseline') & (df['noise_sigma'] == noise_sigma)].iloc[0]
    # Cycle 44 noise-augmented
    c44_row = df[(df['model'] == 'cycle44_noise_aug') & (df['noise_sigma'] == noise_sigma)].iloc[0]

    c42_strehl = c42_row['strehl_ratio_mean']
    c44_strehl = c44_row['strehl_ratio_mean']
    c42_main = c42_row['main_lobe_ratio_mean']
    c44_main = c44_row['main_lobe_ratio_mean']
    c42_eff = c42_row['synthesis_efficiency_mean']
    c44_eff = c44_row['synthesis_efficiency_mean']
    c42_rmse = c42_row['phase_rmse_rad_mean']
    c44_rmse = c44_row['phase_rmse_rad_mean']

    # 计算改善率（Strehl）
    improvement = (c44_strehl - c42_strehl) / c42_strehl * 100

    # Cycle 42行
    latex += f"{noise_sigma:.3f} & Cycle 42 & {c42_strehl:.3f} & {c42_main:.3f} & {c42_eff:.3f} & {c42_rmse:.3f} & \\\\\n"

    # Cycle 44行（高亮更好的值）
    strehl_str = f"\\textbf{{{c44_strehl:.3f}}}" if c44_strehl > c42_strehl else f"{c44_strehl:.3f}"
    main_str = f"\\textbf{{{c44_main:.3f}}}" if c44_main > c42_main else f"{c44_main:.3f}"
    eff_str = f"\\textbf{{{c44_eff:.3f}}}" if c44_eff > c42_eff else f"{c44_eff:.3f}"
    rmse_str = f"\\textbf{{{c44_rmse:.3f}}}" if c44_rmse < c42_rmse else f"{c44_rmse:.3f}"

    improvement_str = f"{improvement:+.1f}\\%" if improvement > 0 else f"{improvement:.1f}\\%"
    if abs(improvement) > 5:
        improvement_str = f"\\textbf{{{improvement_str}}}"

    latex += f" & Cycle 44 & {strehl_str} & {main_str} & {eff_str} & {rmse_str} & {improvement_str} \\\\\n"

    if noise_sigma != key_noise_levels[-1]:
        latex += "\\midrule\n"

latex += r"""\bottomrule
\multicolumn{7}{l}{\textbf{Bold} values indicate superior performance. Improvement computed on Strehl ratio.} \\
\multicolumn{7}{l}{Cycle 44 trained with dynamic noise augmentation $\sigma \sim \text{Uniform}(0, 0.005)$.} \\
\end{tabular}
\end{table*}
"""

output_path = REPO_ROOT / "paper" / "tables" / "table4_noise_augmentation.tex"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(latex)

print(f"Table 4 saved to: {output_path}")
print("\nPreview:")
print(latex[:1000] + "...")
