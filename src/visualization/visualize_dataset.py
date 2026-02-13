"""
Visualization Suite for MAL-ICS++ Dataset
Generates comprehensive plots for paper figures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

def load_dataset(filepath):
    """Load dataset."""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset: {len(df)} samples, {df.shape[1]} features")
    return df

def plot_class_distribution(df, output_dir):
    """Plot class and attack type distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Label distribution
    label_counts = df['label'].value_counts().sort_index()
    axes[0].bar(['Normal', 'Attack'], label_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Binary Classification Distribution')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(label_counts.values):
        axes[0].text(i, v + 100, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Attack type distribution
    attack_counts = df['attack_type'].value_counts().sort_index()
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#e67e22']
    axes[1].bar(range(len(attack_counts)), attack_counts.values, color=colors)
    axes[1].set_xticks(range(len(attack_counts)))
    axes[1].set_xticklabels(attack_counts.index, rotation=45, ha='right')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Attack Type Distribution')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_voltage_profiles(df, output_dir):
    """Plot voltage magnitude profiles for different attack types."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    attack_types = df['attack_type'].unique()
    v_mag_cols = [col for col in df.columns if col.startswith('V_mag_')]
    
    for idx, attack_type in enumerate(sorted(attack_types)):
        if idx >= 6:
            break
        
        subset = df[df['attack_type'] == attack_type]
        
        # Sample 100 random instances
        if len(subset) > 100:
            subset = subset.sample(n=100, random_state=42)
        
        # Plot voltage profiles
        for _, row in subset.iterrows():
            voltages = [row[col] for col in v_mag_cols]
            bus_ids = range(len(voltages))
            axes[idx].plot(bus_ids, voltages, alpha=0.3, linewidth=0.5, color='blue')
        
        # Add mean profile
        mean_voltages = subset[v_mag_cols].mean()
        axes[idx].plot(range(len(mean_voltages)), mean_voltages.values, 
                      color='red', linewidth=2, label='Mean')
        
        axes[idx].set_xlabel('Bus ID')
        axes[idx].set_ylabel('Voltage Magnitude (p.u.)')
        axes[idx].set_title(f'{attack_type}')
        axes[idx].grid(alpha=0.3)
        axes[idx].set_ylim([0.95, 1.15])
        axes[idx].legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'voltage_profiles.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_power_flow_distributions(df, output_dir):
    """Plot power flow distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    p_cols = [col for col in df.columns if col.startswith('P_line_')]
    q_cols = [col for col in df.columns if col.startswith('Q_line_')]
    
    # Active power distribution by attack type
    for attack_type in df['attack_type'].unique():
        subset = df[df['attack_type'] == attack_type]
        p_values = subset[p_cols].values.flatten()
        axes[0].hist(p_values, bins=50, alpha=0.5, label=attack_type, density=True)
    
    axes[0].set_xlabel('Active Power (MW)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Active Power Flow Distribution')
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    
    # Reactive power distribution by attack type
    for attack_type in df['attack_type'].unique():
        subset = df[df['attack_type'] == attack_type]
        q_values = subset[q_cols].values.flatten()
        axes[1].hist(q_values, bins=50, alpha=0.5, label=attack_type, density=True)
    
    axes[1].set_xlabel('Reactive Power (MVAr)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Reactive Power Flow Distribution')
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'power_flow_distributions.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_attack_signatures(df, output_dir):
    """Plot attack signatures showing deviation patterns."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Get measurement columns
    measurement_cols = [col for col in df.columns if col.startswith(('V_mag_', 'P_line_'))][:20]
    
    normal_df = df[df['attack_type'] == 'Normal']
    normal_mean = normal_df[measurement_cols].mean()
    
    attack_types = [at for at in df['attack_type'].unique() if at != 'Normal']
    
    for idx, attack_type in enumerate(sorted(attack_types)):
        if idx >= 6:
            break
        
        attack_df = df[df['attack_type'] == attack_type]
        attack_mean = attack_df[measurement_cols].mean()
        
        # Calculate deviation from normal
        deviation = ((attack_mean - normal_mean) / normal_mean * 100).values
        
        axes[idx].bar(range(len(deviation)), deviation, color='red', alpha=0.7)
        axes[idx].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[idx].set_xlabel('Measurement Index')
        axes[idx].set_ylabel('Deviation from Normal (%)')
        axes[idx].set_title(f'{attack_type} Attack Signature')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim([-30, 30])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'attack_signatures.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_feature_correlation(df, output_dir):
    """Plot correlation heatmap of selected features."""
    # Select subset of features for visualization
    v_mag_cols = [col for col in df.columns if col.startswith('V_mag_')][:10]
    p_cols = [col for col in df.columns if col.startswith('P_line_')][:10]
    selected_cols = v_mag_cols + p_cols
    
    # Calculate correlation matrix
    corr_matrix = df[selected_cols].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=[c.replace('V_mag_', 'V').replace('P_line_', 'P') for c in selected_cols],
                yticklabels=[c.replace('V_mag_', 'V').replace('P_line_', 'P') for c in selected_cols])
    plt.title('Feature Correlation Heatmap (Selected Features)')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'feature_correlation.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_temporal_drift_analysis(df, output_dir):
    """Plot temporal drift patterns for A3 attack."""
    a3_df = df[df['attack_type'] == 'A3_SlowDrift']
    
    if len(a3_df) == 0 or 'drift_time_step' not in a3_df.columns:
        print("⚠️  Skipping temporal drift analysis (no A3 data)")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Group by time step and calculate mean drift magnitude
    drift_by_time = a3_df.groupby('drift_time_step')['drift_magnitude'].mean()
    
    plt.plot(drift_by_time.index, drift_by_time.values, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Time Step')
    plt.ylabel('Average Drift Magnitude')
    plt.title('A3 Slow-Drift Attack: Temporal Progression')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'temporal_drift_analysis.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def main():
    """Main visualization function."""
    print(f"\n{'#'*70}")
    print("# MAL-ICS++ Dataset Visualization Suite")
    print(f"{'#'*70}\n")
    
    # Load dataset
    dataset_path = "data/malics_dataset_complete.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    df = load_dataset(dataset_path)
    
    # Create output directory for figures
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate all plots
    print("\nGenerating visualizations...")
    plot_class_distribution(df, output_dir)
    plot_voltage_profiles(df, output_dir)
    plot_power_flow_distributions(df, output_dir)
    plot_attack_signatures(df, output_dir)
    plot_feature_correlation(df, output_dir)
    plot_temporal_drift_analysis(df, output_dir)
    
    print(f"\n{'#'*70}")
    print(f"# Visualization Complete!")
    print(f"# Figures saved to: {output_dir}/")
    print(f"{'#'*70}\n")

if __name__ == "__main__":
    main()
