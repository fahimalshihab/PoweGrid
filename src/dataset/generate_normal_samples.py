"""
Generate normal operation samples for MAL-ICS++ dataset.
Creates N samples with random load variations (±10%) from IEEE 14-bus baseline.
"""

import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Configuration
NUM_SAMPLES = 5000  # Full normal operation dataset
LOAD_VARIATION = 0.10  # ±10% load variation
OUTPUT_DIR = "data"
SEED = 42
BATCH_SIZE = 100  # Save progress every N samples

# Set random seed for reproducibility
np.random.seed(SEED)

def create_base_network():
    """Load IEEE 14-bus system."""
    return pn.case14()

def vary_loads(net, variation=0.10):
    """Apply random load variations to all loads."""
    for idx in net.load.index:
        base_p = net.load.at[idx, 'p_mw']
        base_q = net.load.at[idx, 'q_mvar']
        
        # Random variation factor (1 ± variation)
        factor = 1 + np.random.uniform(-variation, variation)
        
        net.load.at[idx, 'p_mw'] = base_p * factor
        net.load.at[idx, 'q_mvar'] = base_q * factor

def extract_measurements(net):
    """Extract voltage and power measurements (simulating PMU data)."""
    measurements = {}
    
    # Voltage measurements at all buses
    for bus_idx in net.bus.index:
        measurements[f'V_mag_{bus_idx}'] = net.res_bus.at[bus_idx, 'vm_pu']
        measurements[f'V_angle_{bus_idx}'] = net.res_bus.at[bus_idx, 'va_degree']
    
    # Power flow measurements on lines
    for line_idx in net.line.index:
        measurements[f'P_line_{line_idx}_from'] = net.res_line.at[line_idx, 'p_from_mw']
        measurements[f'P_line_{line_idx}_to'] = net.res_line.at[line_idx, 'p_to_mw']
        measurements[f'Q_line_{line_idx}_from'] = net.res_line.at[line_idx, 'q_from_mvar']
        measurements[f'Q_line_{line_idx}_to'] = net.res_line.at[line_idx, 'q_to_mvar']
    
    # Power flow measurements on transformers
    for trafo_idx in net.trafo.index:
        measurements[f'P_trafo_{trafo_idx}_hv'] = net.res_trafo.at[trafo_idx, 'p_hv_mw']
        measurements[f'P_trafo_{trafo_idx}_lv'] = net.res_trafo.at[trafo_idx, 'p_lv_mw']
        measurements[f'Q_trafo_{trafo_idx}_hv'] = net.res_trafo.at[trafo_idx, 'q_hv_mvar']
        measurements[f'Q_trafo_{trafo_idx}_lv'] = net.res_trafo.at[trafo_idx, 'q_lv_mvar']
    
    return measurements

def generate_normal_samples(n_samples, batch_size=100):
    """Generate n normal operation samples with batch saving."""
    print(f"\n{'='*60}")
    print(f"Generating {n_samples} Normal Operation Samples")
    print(f"{'='*60}\n")
    
    samples = []
    base_net = create_base_network()
    failed = 0
    
    import copy
    from tqdm import tqdm
    
    for i in tqdm(range(n_samples), desc="Generating samples"):
        # Create fresh network copy
        net = copy.deepcopy(base_net)
        
        # Apply random load variations
        vary_loads(net, LOAD_VARIATION)
        
        # Run power flow
        try:
            pp.runpp(net, numba=False)
            
            # Extract measurements
            measurements = extract_measurements(net)
            
            # Add metadata
            measurements['sample_id'] = i
            measurements['timestamp'] = datetime.now().isoformat()
            measurements['attack_type'] = 'Normal'
            measurements['label'] = 0  # 0 = normal
            
            # Check convergence
            if net.converged:
                samples.append(measurements)
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
    
    # Reorder columns: metadata first, then measurements
    metadata_cols = ['sample_id', 'timestamp', 'attack_type', 'label']
    measurement_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + sorted(measurement_cols)]
    
    # Save to CSV
    output_file = os.path.join(output_dir, f"normal_samples_{len(samples)}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Saved {len(samples)} samples to: {output_file}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print(f"{'='*60}\n")
    
    # Show summary statistics
    print("Summary Statistics (voltage magnitudes):")
    voltage_cols = [col for col in df.columns if col.startswith('V_mag_')]
    print(df[voltage_cols].describe().iloc[:3])
    print("\n...")

if __name__ == "__main__":
    # Generate samples
    samples = generate_normal_samples(NUM_SAMPLES)
    
    # Save to CSV
    if samples:
        save_samples(samples, OUTPUT_DIR)
    else:
        print("\n❌ No valid samples generated!")
