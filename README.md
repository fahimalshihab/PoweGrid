# MAL-ICS++: Malware-Resilient FDIA Detection Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE](https://img.shields.io/badge/IEEE-Transactions-green.svg)](https://ieeexplore.ieee.org/)

Official implementation of **"A Multi-Layer Malware-Resilient Detection Framework for False Data Injection Attacks in Smart Grids"** submitted to IEEE Transactions.

## 🎯 Overview

MAL-ICS++ is a comprehensive cybersecurity framework for detecting False Data Injection Attacks (FDIAs) in smart grid systems. It addresses the critical challenge of malware-facilitated attacks that evade traditional state estimation defenses.

### Key Features

- **🛡️ Multi-Layer Defense**: Four-layer architecture (Hardware Trust, Control Fingerprinting, Federated Learning, Digital Twin)
- **📊 High Performance**: 99.90% detection accuracy with 0.20% false positive rate
- **⚡ Real-time**: Sub-second inference latency (<1s)
- **🔒 Malware-Resilient**: TPM-based firmware verification and behavioral fingerprinting
- **🌐 Federated Learning**: Distributed anomaly detection with privacy preservation
- **🔬 Physics-Aware**: Digital twin validation for power flow consistency




## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/malics-plusplus.git
cd malics-plusplus
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Generate Dataset

Run the complete pipeline to generate the MAL-ICS++ dataset (10,010 samples):

```bash
cd src/dataset
python generate_master_dataset.py
```

This creates:
- 5,010 normal operation samples
- 1,000 samples per attack type (A1-A5)
- 115 features per sample (108 power grid measurements + 7 metadata)

### Train Models

Train all four classifiers (SVM, Random Forest, Gradient Boosting, Logistic Regression):

```bash
cd src/models
python train_classifiers.py
```

### Generate Results

Create all figures and performance metrics:

```bash
cd src/visualization
python generate_figures.py
```

## 📊 Dataset Specification

**MAL-ICS++ Dataset v1.0**

- **Total Samples**: 10,010
- **Features**: 115 (108 measurements + 7 metadata)
- **Classes**: 6 (Normal + 5 attack types)
- **Power Grid**: IEEE 14-bus system
- **Simulator**: Pandapower 3.2.1

### Attack Types

| Type | Name | Description | Samples |
|------|------|-------------|---------|
| A1 | MITM Attack | Man-in-the-Middle with measurement tampering | 1,000 |
| A2 | Topology Poisoning | Circuit breaker state manipulation | 1,000 |
| A3 | Sensor Compromise | Direct sensor value injection | 1,000 |
| A4 | Coordinated Multi-Bus | Simultaneous multi-location attacks | 1,000 |
| A5 | Stealth FDIA | Low-magnitude stealthy injections | 1,000 |
| Normal | Normal Operation | Baseline power grid operation | 5,010 |

### Feature Categories

1. **Voltage Magnitudes**: Bus voltages (pu) - 14 features
2. **Voltage Angles**: Phase angles (degrees) - 14 features
3. **Active Power**: Real power flows (MW) - 20 features
4. **Reactive Power**: Reactive flows (MVAr) - 20 features
5. **Line Currents**: Branch currents (kA) - 20 features
6. **Bus Injections**: Net power at buses (MW/MVAr) - 20 features
7. **Metadata**: Timestamps, load factors, attack labels - 7 features

## 🔬 Methodology

### Four-Layer Architecture

1. **Hardware Trust Layer**
   - TPM-based firmware verification
   - Secure boot validation
   - Attestation protocols

2. **Control Fingerprinting Layer**
   - Timing analysis (IEC 61850 latency patterns)
   - Response behavior profiling
   - Command sequence validation

3. **Federated Learning Layer**
   - Distributed model training across substations
   - Privacy-preserving aggregation
   - Adversarial robustness mechanisms

4. **Digital Twin Validation Layer**
   - Physics-based power flow modeling
   - Measurement residual analysis
   - Topology-aware anomaly detection

### Machine Learning Pipeline

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Best-performing model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train on IEEE 14-bus dataset
pipeline.fit(X_train, y_train)
```

## 📖 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{malics2026,
  title={A Multi-Layer Malware-Resilient Detection Framework for False Data Injection Attacks in Smart Grids},
  author={[Authors]},
  journal={IEEE Transactions on [Venue]},
  year={2026},
  publisher={IEEE}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

For questions or collaboration:
- **Email**: [your-email@institution.edu]
- **Issues**: [GitHub Issues](https://github.com/yourusername/malics-plusplus/issues)

## 🙏 Acknowledgments

- IEEE 14-bus test system from Pandapower library
- Scikit-learn for machine learning implementations
- Research supported by [Funding Agency]

## 📚 Related Publications

1. Original MAL-ICS framework (predecessor work)
2. Survey on FDIA detection methods
3. Federated learning for smart grid security

---

**Note**: This repository contains the research implementation. For production deployment, additional hardening and compliance validation is required.
