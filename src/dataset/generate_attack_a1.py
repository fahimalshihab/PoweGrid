"""
Generate Attack A1 (Random FDIA) samples for MAL-ICS++ dataset.
Random magnitude attacks on voltage and power measurements.
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
ATTACK_MAGNITUDE_RANGE = (0.05, 0.20)  # 5-20% deviation
ATTACKED_MEASUREMENTS_RATIO = 0.30  # Attack 30% of measurements
OUTPUT_DIR = "data"
SEED = 43

np.random.seed(SEED)

def create_base_network():
    """Load IEEE 14-bus system."""
    return pn.case14()

def vary_loads(net, variation=0.10):
    """Apply random load variations to all loads."""
    for idx in net.load.index:
        base_p = net.load.at[idx, 'p_mw']
        base_q = net.load.at[idx, 'q_mvar']
        factor = 1 + np.random.uniform(-variation, variation)
        net.load.at[idx, 'p_mw'] = base_p * factor
        net.load.at[idx, 'q_mvar'] = base_q * factor

def extract_measurements(net):
    """Extract voltage and power measurements."""
    measurements = {}
    
    # Voltage measurements
    for bus_idx in net.bus.index:
        measurements[f'V_mag_{bus_idx}'] = net.res_bus.at[bus_idx, 'vm_pu']
        measurements[f'V_angle_{bus_idx}'] = net.res_bus.at[bus_idx, 'va_degree']
    
    # Line power flows
    for line_idx in net.line.index:
        measurements[f'P_line_{line_idx}_from'] = net.res_line.at[line_idx, 'p_from_mw']
        measurements[f'P_line_{line_idx}_to'] = net.res_line.at[line_idx, 'p_to_mw']
        measurements[f'Q_line_{line_idx}_from'] = net.res_line.at[line_idx, 'q_from_mvar']
        measurements[f'Q_line_{line_idx}_to'] = net.res_line.at[line_idx, 'q_to_mvar']
    
    # Transformer power flows
    for trafo_idx in net.trafo.index:
        measurements[f'P_trafo_{trafo_idx}_hv'] = net.res_trafo.at[trafo_idx, 'p_hv_mw']
        measurements[f'P_trafo_{trafo_idx}_lv'] = net.res_trafo.at[trafo_idx, 'p_lv_mw']
        measurements[f'Q_trafo_{trafo_idx}_hv'] = net.res_trafo.at[trafo_idx, 'q_hv_mvar']
        measurements[f'Q_trafo_{trafo_idx}_lv'] = net.res_trafo.at[trafo_idx, 'q_lv_mvar']
    
    return measurements

def inject_random_attack(measurements):
    """Inject random magnitude attacks on selected measurements."""
    attacked_measurements = measurements.copy()
    
    # Get measurement keys (exclude metadata)
    measurement_keys = [k for k in measurements.keys() if k.startswith(('V_', 'P_', 'Q_'))]
    
    # Randomly select measurements to attack
    n_attacked = int(len(measurement_keys) * ATTACKED_MEASUREMENTS_RATIO)
    attacked_keys = np.random.choice(measurement_keys, size=n_attacked, replace=False)
    
    # Apply random deviations
    for key in attacked_keys:
        original_value = measurements[key]
        attack_magnitude = np.random.uniform(*ATTACK_MAGNITUDE_RANGE)
        sign = np.random.choice([-1, 1])
        attacked_measurements[key] = original_value * (1 + sign * attack_magnitude)
    
    return attacked_measurements, attacked_keys

def generate_attack_samples(n_samples):
    """Generate attack A1 samples."""
    print(f"\n{'='*60}")
    print(f"Generating {n_samples} Attack A1 (Random FDIA) Samples")
    print(f"{'='*60}\n")
    
    samples = []
    base_net = create_base_network()
    failed = 0
    
    from tqdm import tqdm
    
    for i in tqdm(range(n_samples), desc="Generating A1 samples"):
        net = copy.deepcopy(base_net)
        vary_loads(net, LOAD_VARIATION)
        
        try:
            pp.runpp(net, numba=False)
            
            if net.converged:
                # Get clean measurements
                clean_measurements = extract_measurements(net)
                
                # Inject attack
                attacked_measurements, attacked_keys = inject_random_attack(clean_measurements)
                
                # Add metadata
                attacked_measurements['sample_id'] = i
                attacked_measurements['timestamp'] = datetime.now().isoformat()
                attacked_measurements['attack_type'] = 'A1_Random'
                attacked_measurements['label'] = 1  # 1 = attack
                attacked_measurements['n_attacked_measurements'] = len(attacked_keys)
                
                samples.append(attacked_measurements)
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
    
    print(f"\n✅ Generated {len(samples)}/{n_samples} samples ({failed} failed)")
    return samples

def save_samples(samples, output_dir):
    """Save samples to CSV."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.DataFrame(samples)
    
    # Reorder columns
    metadata_cols = ['sample_id', 'timestamp', 'attack_type', 'label', 'n_attacked_measurements']
    measurement_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + sorted(measurement_cols)]
    
    output_file = os.path.join(output_dir, f"attack_a1_samples_{len(samples)}.csv")
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
