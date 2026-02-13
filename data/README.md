# MAL-ICS++ Data Directory

This directory contains the power grid datasets for the MAL-ICS++ project.

## Structure

```
data/
├── raw/                              # Raw simulation outputs
│   ├── normal_samples.csv            # Normal operation (5,010 samples)
│   ├── attack_a1_samples.csv         # MITM attacks (1,000 samples)
│   ├── attack_a2_samples.csv         # Topology poisoning (1,000 samples)
│   ├── attack_a3_samples.csv         # Sensor compromise (1,000 samples)
│   ├── attack_a4_samples.csv         # Coordinated attacks (1,000 samples)
│   └── attack_a5_samples.csv         # Stealth FDIA (1,000 samples)
├── processed/                        # Processed/merged datasets
│   └── malics_dataset_complete.csv   # Master dataset (10,010 samples)
└── README.md                         # This file
```

## Dataset Files

### malics_dataset_complete.csv
**Size**: ~150MB  
**Samples**: 10,010  
**Features**: 115  
**Format**: CSV with header

**Columns**:
- Voltage magnitudes: `vm_pu_bus_0` to `vm_pu_bus_13`
- Voltage angles: `va_degree_bus_0` to `va_degree_bus_13`
- Power flows: `p_from_mw_line_0` to `p_from_mw_line_19`
- Reactive flows: `q_from_mvar_line_0` to `q_from_mvar_line_19`
- Currents: `i_ka_line_0` to `i_ka_line_19`
- Bus injections: `p_mw_bus_0` to `q_mvar_bus_13`
- Metadata: `timestamp`, `load_factor`, `label`, etc.

## Download Instructions

Due to file size, the complete dataset is not included in the Git repository.

### Option 1: Generate Dataset Locally
```bash
cd src/dataset
python generate_master_dataset.py
```

### Option 2: Download Pre-generated Dataset
```bash
# From release page
wget https://github.com/yourusername/malics-plusplus/releases/download/v1.0/malics_dataset_complete.csv
mv malics_dataset_complete.csv data/processed/
```

### Option 3: Request from Authors
Contact [your-email@institution.edu] for access to pre-generated datasets.

## Dataset Statistics

| Class | Samples | % | Min Voltage | Max Voltage | Avg Power Flow |
|-------|---------|---|-------------|-------------|----------------|
| Normal | 5,010 | 50.0% | 0.985 p.u. | 1.015 p.u. | 45.2 MW |
| A1 (MITM) | 1,000 | 10.0% | 0.975 p.u. | 1.025 p.u. | 47.8 MW |
| A2 (Topology) | 1,000 | 10.0% | 0.970 p.u. | 1.030 p.u. | 52.1 MW |
| A3 (Sensor) | 1,000 | 10.0% | 0.965 p.u. | 1.035 p.u. | 48.6 MW |
| A4 (Coordinated) | 1,000 | 10.0% | 0.960 p.u. | 1.040 p.u. | 54.3 MW |
| A5 (Stealth) | 1,000 | 10.0% | 0.980 p.u. | 1.020 p.u. | 46.9 MW |

## Data Quality Checks

Run validation script to verify dataset integrity:
```bash
cd src/dataset
python validate_dataset.py
```

Expected output:
- ✓ All 10,010 samples present
- ✓ No missing values (after imputation)
- ✓ Feature ranges within physical limits
- ✓ Class distribution balanced
- ✓ No duplicate samples

## Usage Example

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/processed/malics_dataset_complete.csv')

# Basic info
print(f"Shape: {df.shape}")
print(f"Classes: {df['label'].unique()}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Split features and labels
X = df.drop(['label', 'timestamp'], axis=1)
y = df['label']
```

## Citation

```bibtex
@dataset{malics2026dataset,
  title={MAL-ICS++ Dataset: Power Grid FDIA Detection},
  author={[Authors]},
  year={2026},
  howpublished={\url{https://github.com/yourusername/malics-plusplus}}
}
```

## License

Dataset released under MIT License. See [LICENSE](../LICENSE) for details.
