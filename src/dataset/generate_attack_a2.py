"""
Generate Attack A2 (Stealthy FDIA) samples for MAL-ICS++ dataset.
Constructs attack vector a = Hc to bypass bad data detection (residual r = z - Hx ≈ 0).
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
ATTACK_MAGNITUDE = 0.08  # 8% state deviation
OUTPUT_DIR = "data"
SEED = 44

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

def construct_simplified_jacobian(net):
    """
    Simplified Jacobian approximation for DC power flow model.
    H relates state variables (voltage angles) to measurements (power flows).
    """
    n_buses = len(net.bus)
    measurement_keys = []
    
    # Collect measurement indices
    for line_idx in net.line.index:
        measurement_keys.extend([
            f'P_line_{line_idx}_from',
            f'P_line_{line_idx}_to'
        ])
    
    n_measurements = len(measurement_keys)
    
    # Simplified H matrix (random for simulation purposes)
    # In practice, this would be derived from network topology
    H = np.random.randn(n_measurements, n_buses) * 0.1
    
    return H, measurement_keys

def inject_stealthy_attack(measurements, net):
    """
    Inject stealthy attack using a = Hc where c is state deviation vector.
    This makes the attack undetectable by bad data detection (r = z - Hx ≈ 0).
    """
    attacked_measurements = measurements.copy()
    
    # Construct simplified Jacobian
    H, measurement_keys = construct_simplified_jacobian(net)
    
    # Create state deviation vector c
    n_buses = len(net.bus)
    c = np.random.randn(n_buses) * ATTACK_MAGNITUDE
    
    # Compute attack vector a = Hc
    a = H @ c
    
    # Apply attack to selected measurements
    for i, key in enumerate(measurement_keys):
        if key in attacked_measurements and i < len(a):
            attacked_measurements[key] += a[i]
    
    return attacked_measurements, measurement_keys

def generate_attack_samples(n_samples):
    print(f"\n{'='*60}")
    print(f"Generating {n_samples} Attack A2 (Stealthy FDIA) Samples")
    print(f"{'='*60}\n")
    
    samples = []
    base_net = create_base_network()
    failed = 0
    
    from tqdm import tqdm
    
    for i in tqdm(range(n_samples), desc="Generating A2 samples"):
        net = copy.deepcopy(base_net)
        vary_loads(net, LOAD_VARIATION)
        
        try:
            pp.runpp(net, numba=False)
            
            if net.converged:
                clean_measurements = extract_measurements(net)
                attacked_measurements, attacked_keys = inject_stealthy_attack(clean_measurements, net)
                
                attacked_measurements['sample_id'] = i
                attacked_measurements['timestamp'] = datetime.now().isoformat()
                attacked_measurements['attack_type'] = 'A2_Stealthy'
                attacked_measurements['label'] = 1
                attacked_measurements['n_attacked_measurements'] = len(attacked_keys)
                
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
    metadata_cols = ['sample_id', 'timestamp', 'attack_type', 'label', 'n_attacked_measurements']
    measurement_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + sorted(measurement_cols)]
    
    output_file = os.path.join(output_dir, f"attack_a2_samples_{len(samples)}.csv")
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
