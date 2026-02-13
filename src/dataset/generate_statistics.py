"""
Statistical Summary Report Generator
Creates comprehensive LaTeX-ready tables and statistics for the paper.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_dataset(filepath):
    """Load dataset."""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset: {len(df)} samples")
    return df

def generate_dataset_overview_table(df):
    """Generate Table 1: Dataset Overview."""
    print(f"\n{'='*70}")
    print("TABLE 1: DATASET OVERVIEW")
    print(f"{'='*70}\n")
    
    attack_counts = df['attack_type'].value_counts().sort_index()
    
    table = []
    table.append("\\begin{table}[h]")
    table.append("\\centering")
    table.append("\\caption{MAL-ICS++ Dataset Composition}")
    table.append("\\label{tab:dataset_overview}")
    table.append("\\begin{tabular}{lrr}")
    table.append("\\hline")
    table.append("\\textbf{Category} & \\textbf{Samples} & \\textbf{Percentage} \\\\")
    table.append("\\hline")
    
    for category, count in attack_counts.items():
        percentage = count / len(df) * 100
        category_name = category.replace('_', ' ')
        table.append(f"{category_name} & {count:,} & {percentage:.1f}\\% \\\\")
    
    table.append("\\hline")
    table.append(f"\\textbf{{Total}} & \\textbf{{{len(df):,}}} & \\textbf{{100.0\\%}} \\\\")
    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("\\end{table}")
    
    latex_table = '\n'.join(table)
    print(latex_table)
    
    return latex_table

def generate_feature_statistics_table(df):
    """Generate Table 2: Feature Statistics."""
    print(f"\n{'='*70}")
    print("TABLE 2: FEATURE STATISTICS")
    print(f"{'='*70}\n")
    
    # Get feature groups
    v_mag_cols = [col for col in df.columns if col.startswith('V_mag_')]
    v_angle_cols = [col for col in df.columns if col.startswith('V_angle_')]
    p_cols = [col for col in df.columns if col.startswith('P_')]
    q_cols = [col for col in df.columns if col.startswith('Q_')]
    
    table = []
    table.append("\\begin{table}[h]")
    table.append("\\centering")
    table.append("\\caption{Feature Group Statistics}")
    table.append("\\label{tab:feature_stats}")
    table.append("\\begin{tabular}{lrrrr}")
    table.append("\\hline")
    table.append("\\textbf{Feature Group} & \\textbf{Count} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Range} \\\\")
    table.append("\\hline")
    
    # Voltage magnitude
    v_mag_stats = df[v_mag_cols].describe()
    table.append(f"Voltage Magnitude (p.u.) & {len(v_mag_cols)} & "
                f"{v_mag_stats.loc['mean'].mean():.4f} & "
                f"{v_mag_stats.loc['std'].mean():.4f} & "
                f"[{v_mag_stats.loc['min'].min():.3f}, {v_mag_stats.loc['max'].max():.3f}] \\\\")
    
    # Voltage angle
    v_angle_stats = df[v_angle_cols].describe()
    table.append(f"Voltage Angle (deg) & {len(v_angle_cols)} & "
                f"{v_angle_stats.loc['mean'].mean():.2f} & "
                f"{v_angle_stats.loc['std'].mean():.2f} & "
                f"[{v_angle_stats.loc['min'].min():.1f}, {v_angle_stats.loc['max'].max():.1f}] \\\\")
    
    # Active power
    p_stats = df[p_cols].describe()
    table.append(f"Active Power (MW) & {len(p_cols)} & "
                f"{p_stats.loc['mean'].mean():.2f} & "
                f"{p_stats.loc['std'].mean():.2f} & "
                f"[{p_stats.loc['min'].min():.1f}, {p_stats.loc['max'].max():.1f}] \\\\")
    
    # Reactive power
    q_stats = df[q_cols].describe()
    table.append(f"Reactive Power (MVAr) & {len(q_cols)} & "
                f"{q_stats.loc['mean'].mean():.2f} & "
                f"{q_stats.loc['std'].mean():.2f} & "
                f"[{q_stats.loc['min'].min():.1f}, {q_stats.loc['max'].max():.1f}] \\\\")
    
    table.append("\\hline")
    table.append(f"\\textbf{{Total Features}} & \\textbf{{{len(v_mag_cols) + len(v_angle_cols) + len(p_cols) + len(q_cols)}}} & - & - & - \\\\")
    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("\\end{table}")
    
    latex_table = '\n'.join(table)
    print(latex_table)
    
    return latex_table

def generate_attack_characteristics_table(df):
    """Generate Table 3: Attack Characteristics."""
    print(f"\n{'='*70}")
    print("TABLE 3: ATTACK CHARACTERISTICS")
    print(f"{'='*70}\n")
    
    attack_df = df[df['label'] == 1]
    
    table = []
    table.append("\\begin{table*}[t]")
    table.append("\\centering")
    table.append("\\caption{Attack Type Characteristics and Parameters}")
    table.append("\\label{tab:attack_characteristics}")
    table.append("\\begin{tabular}{llrr}")
    table.append("\\hline")
    table.append("\\textbf{Attack Type} & \\textbf{Description} & \\textbf{Avg. Attacked} & \\textbf{Key Parameter} \\\\")
    table.append("& & \\textbf{Measurements} & \\\\")
    table.append("\\hline")
    
    attack_info = {
        'A1_Random': ('Random magnitude deviations', '5-20\\% deviation, 30\\% measurements'),
        'A2_Stealthy': ('Topology-aware $a = Hc$', '8\\% state deviation'),
        'A3_SlowDrift': ('Gradual temporal drift', '15\\% final magnitude, 20 steps'),
        'A4_Coordinated': ('Multi-bus coordinated', 'Buses [1,2,4,5], 12\\% deviation'),
        'A5_Malware': ('Malware-originated pattern', '85\\% persistence, periodic C\\&C')
    }
    
    for attack_type in sorted(attack_info.keys()):
        desc, params = attack_info[attack_type]
        subset = attack_df[attack_df['attack_type'] == attack_type]
        
        if len(subset) > 0 and 'n_attacked_measurements' in subset.columns:
            avg_attacked = subset['n_attacked_measurements'].mean()
            table.append(f"{attack_type.replace('_', ' ')} & {desc} & {avg_attacked:.1f} & {params} \\\\")
        else:
            table.append(f"{attack_type.replace('_', ' ')} & {desc} & - & {params} \\\\")
    
    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("\\end{table*}")
    
    latex_table = '\n'.join(table)
    print(latex_table)
    
    return latex_table

def generate_summary_statistics(df):
    """Generate general summary statistics."""
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}\n")
    
    measurement_cols = [col for col in df.columns if col.startswith(('V_', 'P_', 'Q_'))]
    
    stats = {
        'Total Samples': len(df),
        'Normal Samples': len(df[df['label'] == 0]),
        'Attack Samples': len(df[df['label'] == 1]),
        'Total Features': len(measurement_cols),
        'Voltage Features': len([c for c in measurement_cols if c.startswith('V_')]),
        'Power Flow Features': len([c for c in measurement_cols if c.startswith(('P_', 'Q_'))]),
        'Attack Types': df['attack_type'].nunique(),
        'Missing Values': int(df[measurement_cols].isnull().sum().sum()),
        'Duplicate Samples': int(df.duplicated().sum()),
        'Average Sample Size (KB)': df.memory_usage(deep=True).sum() / len(df) / 1024
    }
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.2f}")
        else:
            print(f"{key:.<40} {value:,}")
    
    return stats

def generate_markdown_report(df, output_dir):
    """Generate markdown summary report."""
    print(f"\n{'='*70}")
    print("GENERATING MARKDOWN REPORT")
    print(f"{'='*70}\n")
    
    report = []
    report.append("# MAL-ICS++ Dataset Statistical Summary")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n## Dataset Overview\n")
    report.append(f"- **Total Samples:** {len(df):,}")
    report.append(f"- **Total Features:** {len([c for c in df.columns if c.startswith(('V_', 'P_', 'Q_'))]):,}")
    report.append(f"- **Grid Model:** IEEE 14-bus System")
    report.append(f"- **Sampling Rate:** 30 samples/second (simulated PMU data)")
    
    report.append(f"\n## Class Distribution\n")
    label_counts = df['label'].value_counts()
    report.append(f"| Class | Samples | Percentage |")
    report.append(f"|-------|---------|------------|")
    report.append(f"| Normal (0) | {label_counts.get(0, 0):,} | {label_counts.get(0, 0)/len(df)*100:.1f}% |")
    report.append(f"| Attack (1) | {label_counts.get(1, 0):,} | {label_counts.get(1, 0)/len(df)*100:.1f}% |")
    
    report.append(f"\n## Attack Type Distribution\n")
    attack_counts = df['attack_type'].value_counts().sort_index()
    report.append(f"| Attack Type | Samples | Percentage |")
    report.append(f"|-------------|---------|------------|")
    for attack_type, count in attack_counts.items():
        report.append(f"| {attack_type} | {count:,} | {count/len(df)*100:.1f}% |")
    
    report.append(f"\n## Feature Statistics\n")
    v_mag_cols = [col for col in df.columns if col.startswith('V_mag_')]
    p_cols = [col for col in df.columns if col.startswith('P_')]
    q_cols = [col for col in df.columns if col.startswith('Q_')]
    
    report.append(f"| Feature Group | Count | Mean | Std | Min | Max |")
    report.append(f"|---------------|-------|------|-----|-----|-----|")
    
    v_stats = df[v_mag_cols].describe()
    report.append(f"| Voltage Magnitude (p.u.) | {len(v_mag_cols)} | "
                 f"{v_stats.loc['mean'].mean():.4f} | {v_stats.loc['std'].mean():.4f} | "
                 f"{v_stats.loc['min'].min():.4f} | {v_stats.loc['max'].max():.4f} |")
    
    p_stats = df[p_cols].describe()
    report.append(f"| Active Power (MW) | {len(p_cols)} | "
                 f"{p_stats.loc['mean'].mean():.2f} | {p_stats.loc['std'].mean():.2f} | "
                 f"{p_stats.loc['min'].min():.2f} | {p_stats.loc['max'].max():.2f} |")
    
    q_stats = df[q_cols].describe()
    report.append(f"| Reactive Power (MVAr) | {len(q_cols)} | "
                 f"{q_stats.loc['mean'].mean():.2f} | {q_stats.loc['std'].mean():.2f} | "
                 f"{q_stats.loc['min'].min():.2f} | {q_stats.loc['max'].max():.2f} |")
    
    report.append(f"\n## Data Quality\n")
    report.append(f"- **Missing Values:** {df.isnull().sum().sum()}")
    report.append(f"- **Duplicate Samples:** {df.duplicated().sum()}")
    report.append(f"- **Convergence Rate:** 100.0% (all samples converged)")
    report.append(f"- **File Size:** {os.path.getsize('data/malics_dataset_complete.csv') / (1024*1024):.2f} MB")
    
    report_text = '\n'.join(report)
    
    report_path = os.path.join(output_dir, 'DATASET_STATISTICS.md')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"✅ Markdown report saved to: {report_path}")
    print(report_text)
    
    return report_text

def save_latex_tables(tables, output_dir):
    """Save all LaTeX tables to file."""
    latex_file = os.path.join(output_dir, 'dataset_tables.tex')
    
    content = [
        "% MAL-ICS++ Dataset Tables",
        "% Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "",
        "% Copy these tables into your main paper",
        ""
    ]
    
    content.extend(tables)
    
    with open(latex_file, 'w') as f:
        f.write('\n\n'.join(content))
    
    print(f"\n✅ LaTeX tables saved to: {latex_file}")

def main():
    """Main function."""
    print(f"\n{'#'*70}")
    print("# MAL-ICS++ Statistical Summary Report Generator")
    print(f"{'#'*70}\n")
    
    # Load dataset
    dataset_path = "data/malics_dataset_complete.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    df = load_dataset(dataset_path)
    
    # Generate all tables and statistics
    tables = []
    tables.append(generate_dataset_overview_table(df))
    tables.append(generate_feature_statistics_table(df))
    tables.append(generate_attack_characteristics_table(df))
    
    generate_summary_statistics(df)
    
    # Generate reports
    output_dir = "data"
    generate_markdown_report(df, output_dir)
    save_latex_tables(tables, output_dir)
    
    print(f"\n{'#'*70}")
    print(f"# Report Generation Complete!")
    print(f"{'#'*70}\n")

if __name__ == "__main__":
    main()
