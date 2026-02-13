# MAL-ICS++ Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/malics-plusplus.git
cd malics-plusplus

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
cd src/dataset
python generate_master_dataset.py
```

This will create `data/processed/malics_dataset_complete.csv` with 10,010 samples.

### 3. Train Models

```bash
cd src/models
python train_classifiers.py
```

Trains 4 classifiers and saves results to `results/metrics/`.

### 4. Generate Visualizations

```bash
cd src/visualization
python generate_figures.py
```

Creates figures in `results/figures/`.

## Detailed Workflows

### Dataset Generation Pipeline

#### Step 1: Generate Normal Samples
```bash
cd src/dataset
python generate_normal_samples.py
```

Output: `data/raw/normal_samples.csv` (5,010 samples)

#### Step 2: Generate Attack Samples
```bash
python generate_attack_a1.py  # MITM attacks
python generate_attack_a2.py  # Topology poisoning
python generate_attack_a3.py  # Sensor compromise
python generate_attack_a4.py  # Coordinated attacks
python generate_attack_a5.py  # Stealth FDIA
```

Output: Individual attack CSVs in `data/raw/`

#### Step 3: Merge into Master Dataset
```bash
python generate_master_dataset.py
```

Output: `data/processed/malics_dataset_complete.csv`

### Model Training

#### Training Individual Models

```python
from src.models.train_classifiers import create_pipeline, load_dataset
from sklearn.ensemble import RandomForestClassifier

# Load data
df = load_dataset('../../data/processed/malics_dataset_complete.csv')
X = df.drop(['label', 'timestamp'], axis=1)
y = df['label']

# Train Random Forest
rf_pipeline = create_pipeline(RandomForestClassifier(n_estimators=200))
rf_pipeline.fit(X_train, y_train)

# Predict
predictions = rf_pipeline.predict(X_test)
```

#### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    rf_pipeline, param_grid, cv=5, 
    scoring='f1_weighted', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

### Real-time Detection Deployment

#### Load Trained Model

```python
import joblib

# Load best model
model = joblib.load('results/models/random_forest.joblib')

# Incoming measurement vector
new_sample = [...]  # 115 features

# Predict
prediction = model.predict([new_sample])[0]
confidence = model.predict_proba([new_sample]).max()

print(f"Prediction: {prediction} (confidence: {confidence:.2%})")
```

#### Streaming Detection

```python
import pandas as pd
import time

def monitor_scada_stream():
    """Real-time SCADA data monitoring."""
    model = joblib.load('results/models/random_forest.joblib')
    
    while True:
        # Fetch latest measurements from SCADA
        measurements = fetch_scada_data()  # Your data source
        
        # Preprocess
        X = preprocess_measurements(measurements)
        
        # Detect
        prediction = model.predict(X)[0]
        
        if prediction != 'Normal':
            alert_operator(prediction, measurements)
        
        time.sleep(1)  # 1 Hz sampling
```

### Custom Attack Generation

#### Create New Attack Type

```python
import pandapower as pp
import pandas as pd

def generate_custom_attack(net, num_samples=1000):
    """Generate custom attack scenario."""
    samples = []
    
    for i in range(num_samples):
        # Run base power flow
        pp.runpp(net)
        
        # Custom injection logic
        net.res_bus.at[5, 'vm_pu'] *= 1.15  # 15% voltage boost at bus 5
        
        # Re-run power flow
        pp.runpp(net)
        
        # Extract features
        features = extract_features(net)
        features['label'] = 'A6_custom'
        samples.append(features)
    
    return pd.DataFrame(samples)
```

### Evaluation and Analysis

#### Per-Class Performance

```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

#### ROC Analysis

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_proba = model.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

#### Feature Importance

```python
# Get Random Forest feature importances
importances = model.named_steps['classifier'].feature_importances_
feature_names = X.columns

# Sort and plot
indices = np.argsort(importances)[::-1][:20]
plt.barh(range(20), importances[indices])
plt.yticks(range(20), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()
```

## Advanced Usage

### Federated Learning Setup

```python
# Split dataset across multiple substations
substation1_data = df.iloc[:2000]
substation2_data = df.iloc[2000:4000]
substation3_data = df.iloc[4000:6000]

# Train local models
local_models = []
for data in [substation1_data, substation2_data, substation3_data]:
    model = train_local_model(data)
    local_models.append(model)

# Aggregate (simple averaging)
global_model = aggregate_models(local_models)
```

### Digital Twin Integration

```python
import pandapower as pp

def validate_measurement(measurement, net):
    """Use digital twin for consistency check."""
    # Set measurements in network
    net.load.p_mw = measurement['load_p']
    net.load.q_mvar = measurement['load_q']
    
    # Run power flow
    pp.runpp(net)
    
    # Compare predicted vs measured voltages
    predicted_vm = net.res_bus.vm_pu.values
    measured_vm = measurement['voltage_values']
    
    residual = np.abs(predicted_vm - measured_vm)
    
    if residual.max() > 0.05:  # 5% threshold
        return "ANOMALY_DETECTED"
    return "NORMAL"
```

## Troubleshooting

### Issue: "Module 'pandapower' not found"
```bash
pip install pandapower>=3.2.0
```

### Issue: Dataset generation fails
- Check Python version (3.8+ required)
- Ensure sufficient disk space (500MB+)
- Verify pandapower installation: `python -c "import pandapower"`

### Issue: Out of memory during training
- Reduce dataset size
- Use mini-batch training
- Increase swap space

### Issue: Poor model performance
- Check class balance
- Verify feature scaling
- Tune hyperparameters
- Increase training data

## Performance Optimization

### GPU Acceleration
```python
# For deep learning extensions
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Parallel Processing
```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(train_model)(data_chunk) for data_chunk in chunks
)
```

## Contact & Support

- **Documentation**: [GitHub Wiki](https://github.com/yourusername/malics-plusplus/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/malics-plusplus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/malics-plusplus/discussions)
- **Email**: [your-email@institution.edu]
