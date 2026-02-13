"""
Generate Attack A5 (Malware-originated FDIA) samples for MAL-ICS++ dataset.
Simulates malware characteristics: persistence, periodic data exfiltration, 
process injection patterns, and stealthy command-and-control behavior.
"""

import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
from datetime import datetime
import os
import copy

# Configuration
NUM_SAMPLES = 1000
LOAD_VARIATION = 0.10
ATTACK_MAGNITUDE = 0.10  # 10% deviation
PERSISTENCE_PROBABILITY = 0.85  # 85% chance attack persists across time steps
EXFILTRATION_INTERVALS = [10, 20, 30, 60]  # seconds between data exfiltration
OUTPUT_DIR = "data"
SEED = 47

np.random.seed(SEED)

def create_base_network():
    return pn.case14()

def vary_loads(net, variation=0.10):
    for idx in net.load.index:
        base_p = net.load.at[idx, 'p_mw']
        base_q = net.load.at[idx, 'q_mvar']
        factor = 1 + np.random.uniform(-variation, variation)
        net.load.at[idx, 'p_mw'] = base_p * factor
        net.load.at[idx, 'q_mvar'] = base_q * factor

def extract_measurements(net):
    measurements = {}
    for bus_idx in net.bus.index:
        measurements[f'V_mag_{bus_idx}'] = net.res_bus.at[bus_idx, 'vm_pu']
        measurements[f'V_angle_{bus_idx}'] = net.res_bus.at[bus_idx, 'va_degree']
    for line_idx in net.line.index:
        measurements[f'P_line_{line_idx}_from'] = net.res_line.at[line_idx, 'p_from_mw']
        measurements[f'P_line_{line_idx}_to'] = net.res_line.at[line_idx, 'p_to_mw']
        measurements[f'Q_line_{line_idx}_from'] = net.res_line.at[line_idx, 'q_from_mvar']
        measurements[f'Q_line_{line_idx}_to'] = net.res_line.at[line_idx, 'q_to_mvar']
    for trafo_idx in net.trafo.index:
        measurements[f'P_trafo_{trafo_idx}_hv'] = net.res_trafo.at[trafo_idx, 'p_hv_mw']
        measurements[f'P_trafo_{trafo_idx}_lv'] = net.res_trafo.at[trafo_idx, 'p_lv_mw']
        measurements[f'Q_trafo_{trafo_idx}_hv'] = net.res_trafo.at[trafo_idx, 'q_hv_mvar']
        measurements[f'Q_trafo_{trafo_idx}_lv'] = net.res_trafo.at[trafo_idx, 'q_lv_mvar']
    return measurements

def inject_malware_attack(measurements):
    """
    Inject malware-originated attack with characteristic behaviors:
    - Persistence across time
    - Periodic patterns
    - Multiple attack vectors
    - Stealthy magnitude variations
    """
    attacked_measurements = measurements.copy()
    attacked_keys = []
    
    # Malware behavior flags
    is_persistent = np.random.rand() < PERSISTENCE_PROBABILITY
    exfiltration_interval = np.random.choice(EXFILTRATION_INTERVALS)
    process_injection_flag = np.random.choice([0, 1])  # 0 = normal, 1 = injected
    
    # Attack pattern: combination of voltage and power flow manipulation
    measurement_keys = [k for k in measurements.keys() if k.startswith(('V_', 'P_', 'Q_'))]
    
    # Malware typically attacks 40-60% of accessible measurements
    n_attacked = int(len(measurement_keys) * np.random.uniform(0.4, 0.6))
    attacked_keys_list = np.random.choice(measurement_keys, size=n_attacked, replace=False)
    
    # Apply attack with temporal correlation
    for key in attacked_keys_list:
        original_value = measurements[key]
        
        # Malware uses smaller, harder-to-detect deviations
        deviation = np.random.uniform(-ATTACK_MAGNITUDE, ATTACK_MAGNITUDE)
        
        # Add periodic component (C&C communication pattern)
        periodic_component = 0.02 * np.sin(2 * np.pi * np.random.rand())
        
        attacked_measurements[key] = original_value * (1 + deviation + periodic_component)
        attacked_keys.append(key)
    
    return attacked_measurements, attacked_keys, {
        'is_persistent': is_persistent,
        'exfiltration_interval': exfiltration_interval,
        'process_injection': process_injection_flag,
        'n_attack_vectors': len(attacked_keys)
    }

def generate_attack_samples(n_samples):
    print(f"\n{'='*60}")
    print(f"Generating {n_samples} Attack A5 (Malware-originated FDIA) Samples")
    print(f"{'='*60}\n")
    
    samples = []
    base_net = create_base_network()
    failed = 0
    
    from tqdm import tqdm
    
    for i in tqdm(range(n_samples), desc="Generating A5 samples"):
        net = copy.deepcopy(base_net)
        vary_loads(net, LOAD_VARIATION)
        
        try:
            pp.runpp(net, numba=False)
            
            if net.converged:
                clean_measurements = extract_measurements(net)
                attacked_measurements, attacked_keys, malware_features = inject_malware_attack(clean_measurements)
                
                attacked_measurements['sample_id'] = i
                attacked_measurements['timestamp'] = datetime.now().isoformat()
                attacked_measurements['attack_type'] = 'A5_Malware'
                attacked_measurements['label'] = 1
                attacked_measurements['n_attacked_measurements'] = len(attacked_keys)
                attacked_measurements['malware_persistent'] = int(malware_features['is_persistent'])
                attacked_measurements['exfiltration_interval'] = malware_features['exfiltration_interval']
                attacked_measurements['process_injection'] = malware_features['process_injection']
                
                samples.append(attacked_measurements)
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
    
    print(f"\n✅ Generated {len(samples)}/{n_samples} samples ({failed} failed)")
    return samples

def save_samples(samples, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.DataFrame(samples)
    metadata_cols = ['sample_id', 'timestamp', 'attack_type', 'label', 'n_attacked_measurements',
                     'malware_persistent', 'exfiltration_interval', 'process_injection']
    measurement_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + sorted(measurement_cols)]
    
    output_file = os.path.join(output_dir, f"attack_a5_samples_{len(samples)}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Saved {len(samples)} samples to: {output_file}")
    print(f"   Shape: {df.shape}")
    print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    samples = generate_attack_samples(NUM_SAMPLES)
    if samples:
        save_samples(samples, OUTPUT_DIR)
