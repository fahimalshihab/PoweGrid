"""
Generate Attack A3 (Slow-drift FDIA) samples for MAL-ICS++ dataset.
Gradual drift attacks over time windows to evade temporal anomaly detection.
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
DRIFT_RATE = 0.002  # 0.2% per time step
TIME_WINDOW = 20  # 20 time steps to reach full attack
FINAL_MAGNITUDE = 0.15  # Final 15% deviation
OUTPUT_DIR = "data"
SEED = 45

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

def inject_slow_drift_attack(measurements, time_step):
    """
    Inject gradual drift attack that increases over time.
    Attack magnitude = (time_step / TIME_WINDOW) * FINAL_MAGNITUDE
    """
    attacked_measurements = measurements.copy()
    
    # Calculate current drift magnitude
    progress = min(time_step / TIME_WINDOW, 1.0)
    current_magnitude = progress * FINAL_MAGNITUDE
    
    # Select voltage measurements to attack
    voltage_keys = [k for k in measurements.keys() if k.startswith('V_mag_')]
    attacked_keys = np.random.choice(voltage_keys, size=len(voltage_keys)//2, replace=False)
    
    # Apply gradual drift
    for key in attacked_keys:
        original_value = measurements[key]
        attacked_measurements[key] = original_value * (1 + current_magnitude)
    
    return attacked_measurements, attacked_keys, current_magnitude

def generate_attack_samples(n_samples):
    print(f"\n{'='*60}")
    print(f"Generating {n_samples} Attack A3 (Slow-drift FDIA) Samples")
    print(f"{'='*60}\n")
    
    samples = []
    base_net = create_base_network()
    failed = 0
    
    from tqdm import tqdm
    
    for i in tqdm(range(n_samples), desc="Generating A3 samples"):
        net = copy.deepcopy(base_net)
        vary_loads(net, LOAD_VARIATION)
        
        # Random time step in drift progression
        time_step = np.random.randint(0, TIME_WINDOW + 10)
        
        try:
            pp.runpp(net, numba=False)
            
            if net.converged:
                clean_measurements = extract_measurements(net)
                attacked_measurements, attacked_keys, magnitude = inject_slow_drift_attack(
                    clean_measurements, time_step
                )
                
                attacked_measurements['sample_id'] = i
                attacked_measurements['timestamp'] = datetime.now().isoformat()
                attacked_measurements['attack_type'] = 'A3_SlowDrift'
                attacked_measurements['label'] = 1
                attacked_measurements['n_attacked_measurements'] = len(attacked_keys)
                attacked_measurements['drift_time_step'] = time_step
                attacked_measurements['drift_magnitude'] = magnitude
                
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
                     'drift_time_step', 'drift_magnitude']
    measurement_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + sorted(measurement_cols)]
    
    output_file = os.path.join(output_dir, f"attack_a3_samples_{len(samples)}.csv")
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
