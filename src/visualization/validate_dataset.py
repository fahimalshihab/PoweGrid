"""
Dataset Quality Validation Tool
Checks convergence rates, feature distributions, class balance, and data integrity.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

def load_dataset(filepath):
    """Load the combined dataset."""
    if not os.path.exists(filepath):
        print(f"❌ Dataset not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset: {filepath}")
    print(f"   Shape: {df.shape}")
    return df

def check_class_balance(df):
    """Check class distribution and balance."""
    print(f"\n{'='*70}")
    print("CLASS BALANCE ANALYSIS")
    print(f"{'='*70}\n")
    
    # Overall label distribution
    label_counts = df['label'].value_counts()
    print("Label Distribution:")
    print(f"  Normal (0): {label_counts.get(0, 0):,} samples ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Attack (1): {label_counts.get(1, 0):,} samples ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # Attack type distribution
    print("\nAttack Type Distribution:")
    attack_counts = df['attack_type'].value_counts().sort_index()
    for attack_type, count in attack_counts.items():
        print(f"  {attack_type}: {count:,} samples ({count/len(df)*100:.1f}%)")
    
    # Check balance
    balance_ratio = label_counts.min() / label_counts.max()
    print(f"\nBalance Ratio: {balance_ratio:.2f}")
    if balance_ratio > 0.8:
        print("✅ Classes are well balanced")
    elif balance_ratio > 0.5:
        print("⚠️  Classes are moderately imbalanced")
    else:
        print("❌ Classes are severely imbalanced")

def check_feature_statistics(df):
    """Check feature distributions and statistics."""
    print(f"\n{'='*70}")
    print("FEATURE STATISTICS")
    print(f"{'='*70}\n")
    
    # Get measurement columns
    measurement_cols = [col for col in df.columns if col.startswith(('V_', 'P_', 'Q_'))]
    print(f"Total measurement features: {len(measurement_cols)}")
    
    # Voltage magnitude statistics
    v_mag_cols = [col for col in measurement_cols if col.startswith('V_mag_')]
    print(f"\nVoltage Magnitude Features: {len(v_mag_cols)}")
    v_mag_stats = df[v_mag_cols].describe()
    print(f"  Mean range: [{v_mag_stats.loc['mean'].min():.4f}, {v_mag_stats.loc['mean'].max():.4f}]")
    print(f"  Std range:  [{v_mag_stats.loc['std'].min():.4f}, {v_mag_stats.loc['std'].max():.4f}]")
    print(f"  Min value:  {v_mag_stats.loc['min'].min():.4f}")
    print(f"  Max value:  {v_mag_stats.loc['max'].max():.4f}")
    
    # Check for anomalies
    if v_mag_stats.loc['min'].min() < 0.5 or v_mag_stats.loc['max'].max() > 1.5:
        print("  ⚠️  Warning: Voltage values outside typical range [0.5, 1.5] p.u.")
    else:
        print("  ✅ Voltage values within acceptable range")
    
    # Power flow statistics
    p_cols = [col for col in measurement_cols if col.startswith('P_')]
    print(f"\nActive Power Features: {len(p_cols)}")
    p_stats = df[p_cols].describe()
    print(f"  Mean range: [{p_stats.loc['mean'].min():.2f}, {p_stats.loc['mean'].max():.2f}] MW")
    print(f"  Std range:  [{p_stats.loc['std'].min():.2f}, {p_stats.loc['std'].max():.2f}] MW")
    
    # Check for missing values
    missing = df[measurement_cols].isnull().sum().sum()
    print(f"\nMissing Values: {missing}")
    if missing == 0:
        print("✅ No missing values detected")
    else:
        print(f"❌ Found {missing} missing values")

def check_attack_characteristics(df):
    """Analyze attack-specific characteristics."""
    print(f"\n{'='*70}")
    print("ATTACK CHARACTERISTICS ANALYSIS")
    print(f"{'='*70}\n")
    
    attack_df = df[df['label'] == 1]
    
    if 'n_attacked_measurements' in attack_df.columns:
        print("Attacked Measurements Statistics:")
        stats = attack_df['n_attacked_measurements'].describe()
        print(f"  Mean: {stats['mean']:.1f}")
        print(f"  Median: {stats['50%']:.1f}")
        print(f"  Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
    
    # Attack-specific features
    if 'drift_magnitude' in attack_df.columns:
        a3_df = attack_df[attack_df['attack_type'] == 'A3_SlowDrift']
        if len(a3_df) > 0:
            print(f"\nA3 Slow-Drift Attack:")
            print(f"  Mean drift magnitude: {a3_df['drift_magnitude'].mean():.4f}")
            print(f"  Max drift magnitude: {a3_df['drift_magnitude'].max():.4f}")
    
    if 'malware_persistent' in attack_df.columns:
        a5_df = attack_df[attack_df['attack_type'] == 'A5_Malware']
        if len(a5_df) > 0:
            persistence_rate = a5_df['malware_persistent'].mean()
            print(f"\nA5 Malware Attack:")
            print(f"  Persistence rate: {persistence_rate*100:.1f}%")
            print(f"  Process injection rate: {a5_df['process_injection'].mean()*100:.1f}%")

def check_data_integrity(df):
    """Check for data integrity issues."""
    print(f"\n{'='*70}")
    print("DATA INTEGRITY CHECKS")
    print(f"{'='*70}\n")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=[col for col in df.columns if col not in ['sample_id', 'timestamp']]).sum()
    print(f"Duplicate samples: {duplicates}")
    if duplicates == 0:
        print("✅ No duplicate samples found")
    else:
        print(f"⚠️  Found {duplicates} duplicate samples")
    
    # Check sample IDs are unique
    unique_ids = df['sample_id'].nunique()
    print(f"\nUnique sample IDs: {unique_ids}/{len(df)}")
    if unique_ids == len(df):
        print("✅ All sample IDs are unique")
    else:
        print("❌ Duplicate sample IDs found")
    
    # Check for infinite values
    measurement_cols = [col for col in df.columns if col.startswith(('V_', 'P_', 'Q_'))]
    inf_count = np.isinf(df[measurement_cols]).sum().sum()
    print(f"\nInfinite values: {inf_count}")
    if inf_count == 0:
        print("✅ No infinite values found")
    else:
        print(f"❌ Found {inf_count} infinite values")

def generate_validation_report(df, output_dir):
    """Generate comprehensive validation report."""
    report = {
        'validation_date': datetime.now().isoformat(),
        'total_samples': len(df),
        'total_features': len([col for col in df.columns if col.startswith(('V_', 'P_', 'Q_'))]),
        'class_distribution': df['label'].value_counts().to_dict(),
        'attack_type_distribution': df['attack_type'].value_counts().to_dict(),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_samples': int(df.duplicated().sum()),
        'integrity_status': 'PASSED' if df.isnull().sum().sum() == 0 and df.duplicated().sum() == 0 else 'FAILED'
    }
    
    # Add feature statistics
    measurement_cols = [col for col in df.columns if col.startswith(('V_', 'P_', 'Q_'))]
    report['feature_statistics'] = {
        'mean': float(df[measurement_cols].mean().mean()),
        'std': float(df[measurement_cols].std().mean()),
        'min': float(df[measurement_cols].min().min()),
        'max': float(df[measurement_cols].max().max())
    }
    
    report_path = os.path.join(output_dir, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Validation report saved to: {report_path}")
    
    return report

def main():
    """Main validation function."""
    print(f"\n{'#'*70}")
    print("# MAL-ICS++ Dataset Quality Validation")
    print(f"{'#'*70}\n")
    
    # Load dataset
    dataset_path = "data/malics_dataset_complete.csv"
    df = load_dataset(dataset_path)
    
    if df is None:
        print("\n❌ Validation failed: Could not load dataset")
        return
    
    # Run validation checks
    check_class_balance(df)
    check_feature_statistics(df)
    check_attack_characteristics(df)
    check_data_integrity(df)
    
    # Generate report
    report = generate_validation_report(df, "data")
    
    print(f"\n{'#'*70}")
    print(f"# Validation Complete!")
    print(f"# Status: {report['integrity_status']}")
    print(f"{'#'*70}\n")

if __name__ == "__main__":
    main()
