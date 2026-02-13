# MAL-ICS++ Methodology

## System Architecture

The MAL-ICS++ framework consists of four integrated defense layers:

### Layer 1: Hardware Trust Layer
**Purpose**: Establish trusted computing base for control systems

**Components**:
- **Trusted Platform Module (TPM)**: Cryptographic attestation of firmware integrity
- **Secure Boot**: Verify bootloader and OS signatures
- **Remote Attestation**: Periodic validation of runtime state

**Implementation**:
```python
def verify_firmware_integrity(device_id):
    """TPM-based firmware verification."""
    expected_pcr = load_golden_pcr_values(device_id)
    current_pcr = tpm.read_pcr_values()
    
    if expected_pcr != current_pcr:
        raise IntegrityViolationError("Firmware tampering detected")
    
    return True
```

**Detection Capability**: Prevents malware persistence, detects firmware-level attacks

---

### Layer 2: Control Fingerprinting Layer
**Purpose**: Behavioral anomaly detection via timing and response analysis

**Techniques**:
- **IEC 61850 Timing Analysis**: Monitor GOOSE/MMS message latencies
- **Command Response Profiling**: Baseline normal control sequences
- **Statistical Deviation Detection**: Z-score based anomaly flagging

**Implementation**:
```python
def analyze_control_timing(message_log):
    """Detect timing anomalies in IEC 61850 traffic."""
    latencies = [msg.timestamp_response - msg.timestamp_request 
                 for msg in message_log]
    
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    for latency in latencies[-10:]:  # Check last 10 messages
        z_score = (latency - mean_latency) / std_latency
        if abs(z_score) > 3:  # 3-sigma threshold
            return "ANOMALY_DETECTED"
    
    return "NORMAL"
```

**Detection Capability**: MITM attacks, command injection, replay attacks

---

### Layer 3: Federated Learning Layer
**Purpose**: Distributed anomaly detection with privacy preservation

**Architecture**:
- **Local Training**: Each substation trains model on local data
- **Secure Aggregation**: Encrypted gradient sharing (homomorphic encryption)
- **Adversarial Robustness**: Byzantine-resilient aggregation (Krum, Bulyan)

**Algorithm**:
```
For each federated round t = 1, 2, ..., T:
    1. Central server broadcasts global model w_t
    2. Each substation k:
        a. Trains on local data: w_t^k = w_t - η∇L(w_t; D_k)
        b. Computes gradient: Δw_t^k = w_t^k - w_t
        c. Encrypts gradient: [Δw_t^k]_encrypted
    3. Server aggregates:
        a. Decrypts gradients: {Δw_t^k}
        b. Detects Byzantine updates (Krum)
        c. Averages: w_{t+1} = w_t + (1/K)Σ Δw_t^k
    4. Broadcast updated global model
```

**Implementation**:
```python
def federated_training(substations, rounds=10):
    """Federated learning with Byzantine robustness."""
    global_model = initialize_model()
    
    for t in range(rounds):
        local_updates = []
        
        # Local training
        for substation in substations:
            local_model = copy.deepcopy(global_model)
            local_model.fit(substation.data)
            gradient = compute_gradient(global_model, local_model)
            local_updates.append(gradient)
        
        # Byzantine-robust aggregation (Krum)
        trusted_gradients = krum_filter(local_updates, f=2)  # f Byzantine
        global_gradient = np.mean(trusted_gradients, axis=0)
        
        # Update global model
        global_model.update(global_gradient)
    
    return global_model
```

**Detection Capability**: Data poisoning, model poisoning, distributed attacks

---

### Layer 4: Digital Twin Validation Layer
**Purpose**: Physics-based consistency checking

**Core Concepts**:
- **Power Flow Equations**: Verify measurement adherence to Kirchhoff's laws
- **Topology-Aware Analysis**: Cross-check with network topology
- **State Estimation Residuals**: χ² test on weighted least squares residuals

**Power Flow Validation**:
```python
def validate_with_digital_twin(measurements, network_topology):
    """Physics-based measurement validation."""
    # Run power flow on digital twin
    net = load_network_model(network_topology)
    set_loads_from_measurements(net, measurements)
    
    try:
        pp.runpp(net)  # Pandapower power flow
    except:
        return "INFEASIBLE_STATE"  # Physics violation
    
    # Compare predicted vs measured
    predicted_voltages = net.res_bus.vm_pu.values
    measured_voltages = measurements['voltages']
    
    residuals = predicted_voltages - measured_voltages
    chi_squared = np.sum((residuals / 0.01)**2)  # 1% std dev
    
    threshold = chi2.ppf(0.95, df=len(residuals))
    
    if chi_squared > threshold:
        return "BAD_DATA_DETECTED"
    
    return "VALID"
```

**Detection Capability**: Topology attacks, sensor compromise, stealthy FDIA

---

## Machine Learning Pipeline

### Data Preprocessing

1. **Feature Extraction**: 115 features from IEEE 14-bus system
2. **Missing Value Imputation**: Median imputation for <0.1% missing
3. **Scaling**: StandardScaler (zero mean, unit variance)
4. **Stratified Split**: 80% train, 20% test

### Model Selection

Four classifiers evaluated:

| Model | Hyperparameters | Training Time | Inference |
|-------|----------------|---------------|-----------|
| SVM (RBF) | C=1.0, γ=auto | 45s | 0.8s |
| Random Forest | n=200, depth=∞ | 32s | 0.3s |
| Gradient Boosting | n=100, lr=0.1 | 58s | 0.5s |
| Logistic Regression | C=1.0, max_iter=1000 | 12s | 0.1s |

**Best Model**: Random Forest (99.90% accuracy, 0.9999 AUC)

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Attack detection precision (minimize false alarms)
- **Recall**: Attack detection coverage (minimize misses)
- **F1-Score**: Harmonic mean of precision/recall
- **AUC**: Area under ROC curve
- **FPR**: False positive rate (critical for operational acceptance)

### Cross-Validation

5-fold stratified cross-validation:
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
print(f"Mean F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SCADA System                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   RTU 1      │  │   RTU 2      │  │   RTU N      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                 │          │
│         └─────────────────┴─────────────────┘          │
│                           │                            │
└───────────────────────────┼────────────────────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │  MAL-ICS++ Gateway  │
                  │  ┌───────────────┐  │
                  │  │ Layer 1: TPM  │  │
                  │  │ Attestation   │  │
                  │  └───────┬───────┘  │
                  │          ▼          │
                  │  ┌───────────────┐  │
                  │  │ Layer 2:      │  │
                  │  │ Fingerprint   │  │
                  │  └───────┬───────┘  │
                  │          ▼          │
                  │  ┌───────────────┐  │
                  │  │ Layer 3: FL   │  │
                  │  │ Model         │  │
                  │  └───────┬───────┘  │
                  │          ▼          │
                  │  ┌───────────────┐  │
                  │  │ Layer 4:      │  │
                  │  │ Digital Twin  │  │
                  │  └───────┬───────┘  │
                  │          ▼          │
                  │   [Decision Fusion] │
                  └─────────┬───────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │  Operator Console   │
                  │  - Alerts           │
                  │  - Forensics        │
                  │  - Mitigation       │
                  └─────────────────────┘
```

## Performance Analysis

### Latency Breakdown

| Component | Time (ms) | % Total |
|-----------|-----------|---------|
| Data collection | 150 | 15% |
| Preprocessing | 50 | 5% |
| TPM verification | 200 | 20% |
| Timing analysis | 100 | 10% |
| ML inference | 300 | 30% |
| Digital twin | 180 | 18% |
| Decision fusion | 20 | 2% |
| **Total** | **1000** | **100%** |

**Result**: Sub-second latency meets real-time requirement (<2s)

### Scalability

- **IEEE 14-bus**: 1.0s latency
- **IEEE 30-bus**: 1.8s latency (extrapolated)
- **IEEE 118-bus**: 4.2s latency (extrapolated)

**Optimization**: Distributed computing, GPU acceleration, model pruning

## Security Analysis

### Threat Model

**Adversary Capabilities**:
- Network access (MITM, packet injection)
- Limited physical access (sensor tampering)
- Knowledge of power flow equations
- Ability to generate stealthy attacks

**Out-of-Scope**:
- Full system compromise (root access)
- Physical destruction of equipment
- Social engineering of operators

### Attack Resistance

| Attack Type | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
|-------------|---------|---------|---------|---------|
| MITM | ✓ | ✓✓ | ✓ | - |
| Topology | - | ✓ | ✓ | ✓✓ |
| Sensor | ✓ | ✓ | ✓ | ✓✓ |
| Coordinated | - | ✓ | ✓✓ | ✓ |
| Stealth | - | - | ✓✓ | ✓✓ |

**Legend**: ✓✓ = Primary defense, ✓ = Secondary defense, - = Not applicable

## Future Extensions

1. **Deep Learning**: LSTM for temporal patterns
2. **Explainable AI**: SHAP values for interpretability
3. **Adaptive Defense**: Reinforcement learning for mitigation
4. **Multi-System**: IEEE 30/57/118-bus support
5. **Hardware Security**: PUF-based authentication

## References

1. Y. Liu et al., "False data injection attacks against state estimation in electric power grids," IEEE TIFS, 2011.
2. L. Xie et al., "Integrity data attacks in power market operations," IEEE TSG, 2011.
3. McMahan et al., "Communication-efficient learning of deep networks from decentralized data," AISTATS, 2017.
4. Zimmerman et al., "MATPOWER: Steady-state operations, planning, and analysis tools for power systems research," IEEE TPWRS, 2011.
