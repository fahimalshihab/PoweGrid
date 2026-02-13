"""
Generate Attack A4 (Coordinated FDIA) samples for MAL-ICS++ dataset.
Multi-bus coordinated attacks targeting critical substations simultaneously.
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
ATTACK_MAGNITUDE = 0.12  # 12% deviation
CRITICAL_BUSES = [1, 2, 4, 5]  # Critical substations in IEEE 14-bus
OUTPUT_DIR = "data"
SEED = 46

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

def inject_coordinated_attack(measurements, net):
    """
    Inject coordinated attack on multiple critical buses simultaneously.
    Attacks all measurements associated with critical substations.
    """
    attacked_measurements = measurements.copy()
    attacked_keys = []
    
    # Attack voltage measurements at critical buses
    for bus_idx in CRITICAL_BUSES:
        v_mag_key = f'V_mag_{bus_idx}'
        v_angle_key = f'V_angle_{bus_idx}'
        
        if v_mag_key in attacked_measurements:
            attacked_measurements[v_mag_key] *= (1 + np.random.uniform(-ATTACK_MAGNITUDE, ATTACK_MAGNITUDE))
            attacked_keys.append(v_mag_key)
        
        if v_angle_key in attacked_measurements:
            attacked_measurements[v_angle_key] += np.random.uniform(-5, 5)  # Angle deviation in degrees
            attacked_keys.append(v_angle_key)
    
    # Attack power flows on lines connected to critical buses
    for line_idx in net.line.index:
        from_bus = net.line.at[line_idx, 'from_bus']
        to_bus = net.line.at[line_idx, 'to_bus']
        
        if from_bus in CRITICAL_BUSES or to_bus in CRITICAL_BUSES:
            for direction in ['from', 'to']:
                for power_type in ['P', 'Q']:
                    key = f'{power_type}_line_{line_idx}_{direction}'
                    if key in attacked_measurements:
                        attacked_measurements[key] *= (1 + np.random.uniform(-ATTACK_MAGNITUDE, ATTACK_MAGNITUDE))
                        attacked_keys.append(key)
    
    return attacked_measurements, attacked_keys

def generate_attack_samples(n_samples):
    print(f"\n{'='*60}")
    print(f"Generating {n_samples} Attack A4 (Coordinated FDIA) Samples")
    print(f"Critical buses: {CRITICAL_BUSES}")
    print(f"{'='*60}\n")
    
    samples = []
    base_net = create_base_network()
    failed = 0
    
    from tqdm import tqdm
    
    for i in tqdm(range(n_samples), desc="Generating A4 samples"):
        net = copy.deepcopy(base_net)
        vary_loads(net, LOAD_VARIATION)
        
        try:
            pp.runpp(net, numba=False)
            
            if net.converged:
                clean_measurements = extract_measurements(net)
                attacked_measurements, attacked_keys = inject_coordinated_attack(clean_measurements, net)
                
                attacked_measurements['sample_id'] = i
                attacked_measurements['timestamp'] = datetime.now().isoformat()
                attacked_measurements['attack_type'] = 'A4_Coordinated'
                attacked_measurements['label'] = 1
                attacked_measurements['n_attacked_measurements'] = len(attacked_keys)
                attacked_measurements['critical_buses'] = str(CRITICAL_BUSES)
                
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
    metadata_cols = ['sample_id', 'timestamp', 'attack_type', 'label', 'n_attacked_measurements', 'critical_buses']
    measurement_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + sorted(measurement_cols)]
    
    output_file = os.path.join(output_dir, f"attack_a4_samples_{len(samples)}.csv")
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
