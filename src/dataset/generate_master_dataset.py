"""
Master Dataset Generator for MAL-ICS++ Framework
Generates complete 10,000-sample dataset with all attack types:
- 5,000 Normal operation samples
- 1,000 Attack A1 (Random FDIA)
- 1,000 Attack A2 (Stealthy FDIA)
- 1,000 Attack A3 (Slow-drift FDIA)
- 1,000 Attack A4 (Coordinated FDIA)
- 1,000 Attack A5 (Malware-originated FDIA)
"""

import subprocess
import pandas as pd
import os
from datetime import datetime
import sys

# Configuration
OUTPUT_DIR = "data"
COMBINED_OUTPUT = "malics_dataset_complete.csv"

def run_generator(script_name):
    """Run a generator script and return success status."""
    print(f"\n{'#'*70}")
    print(f"# Running: {script_name}")
    print(f"{'#'*70}\n")
    
    try:
        result = subprocess.run(
            ['python3', script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}")
        print(e.stdout)
        print(e.stderr)
        return False

def combine_datasets():
    """Combine all generated datasets into one master file."""
    print(f"\n{'='*70}")
    print(f"Combining all datasets...")
    print(f"{'='*70}\n")
    
    # Find all CSV files in data directory
    csv_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv') and f != COMBINED_OUTPUT]
    
    if not csv_files:
        print("❌ No CSV files found to combine!")
        return False
    
    print(f"Found {len(csv_files)} dataset files:")
    for f in csv_files:
        file_path = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(file_path) / (1024*1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
    
    # Load and combine datasets
    dfs = []
    total_samples = 0
    
    for csv_file in csv_files:
        file_path = os.path.join(OUTPUT_DIR, csv_file)
        df = pd.read_csv(file_path)
        dfs.append(df)
        total_samples += len(df)
        print(f"✅ Loaded {csv_file}: {len(df)} samples")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Reset sample IDs to be sequential
    combined_df['sample_id'] = range(len(combined_df))
    
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save combined dataset
    output_path = os.path.join(OUTPUT_DIR, COMBINED_OUTPUT)
    combined_df.to_csv(output_path, index=False)
    
    file_size = os.path.getsize(output_path) / (1024*1024)
    
    print(f"\n{'='*70}")
    print(f"✅ Combined Dataset Created!")
    print(f"{'='*70}")
    print(f"Output file: {output_path}")
    print(f"Total samples: {len(combined_df)}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Shape: {combined_df.shape}")
    print(f"\nClass distribution:")
    print(combined_df['attack_type'].value_counts().sort_index())
    print(f"\nLabel distribution:")
    print(combined_df['label'].value_counts())
    print(f"{'='*70}\n")
    
    return True

def generate_metadata():
    """Generate metadata file with dataset information."""
    metadata = {
        'generation_date': datetime.now().isoformat(),
        'framework': 'MAL-ICS++',
        'grid_model': 'IEEE 14-bus',
        'total_samples': 10000,
        'sample_distribution': {
            'Normal': 5000,
            'A1_Random': 1000,
            'A2_Stealthy': 1000,
            'A3_SlowDrift': 1000,
            'A4_Coordinated': 1000,
            'A5_Malware': 1000
        },
        'features': {
            'voltage_magnitudes': 14,
            'voltage_angles': 14,
            'line_power_flows': 60,  # 15 lines × 4 measurements
            'transformer_power_flows': 20  # 5 transformers × 4 measurements
        },
        'total_features': 112,
        'attack_parameters': {
            'A1': 'Random magnitude (5-20%), 30% measurements',
            'A2': 'Stealthy (a=Hc), 8% deviation',
            'A3': 'Slow-drift (20 steps), 15% final magnitude',
            'A4': 'Coordinated on buses [1,2,4,5], 12% deviation',
            'A5': 'Malware (85% persistence, periodic C&C)'
        }
    }
    
    import json
    metadata_path = os.path.join(OUTPUT_DIR, 'dataset_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: {metadata_path}\n")

def main():
    """Main execution function."""
    print(f"\n{'#'*70}")
    print(f"# MAL-ICS++ Master Dataset Generator")
    print(f"# Target: 10,000 samples across 6 categories")
    print(f"# IEEE 14-bus Power Grid System")
    print(f"{'#'*70}\n")
    
    start_time = datetime.now()
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # List of generator scripts
    generators = [
        'generate_normal_samples.py',
        'generate_attack_a1.py',
        'generate_attack_a2.py',
        'generate_attack_a3.py',
        'generate_attack_a4.py',
        'generate_attack_a5.py'
    ]
    
    # Track success
    success_count = 0
    
    # Run each generator
    for generator in generators:
        if run_generator(generator):
            success_count += 1
        else:
            print(f"⚠️  Warning: {generator} failed, continuing...")
    
    print(f"\n{'='*70}")
    print(f"Generation Summary: {success_count}/{len(generators)} successful")
    print(f"{'='*70}\n")
    
    # Combine datasets
    if success_count > 0:
        combine_datasets()
        generate_metadata()
    else:
        print("❌ No datasets generated successfully!")
        sys.exit(1)
    
    # Calculate duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'#'*70}")
    print(f"# Dataset Generation Complete!")
    print(f"# Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"# Output directory: {OUTPUT_DIR}")
    print(f"{'#'*70}\n")

if __name__ == "__main__":
    main()
