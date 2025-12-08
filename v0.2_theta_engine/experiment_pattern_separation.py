import numpy as np
import matplotlib.pyplot as plt
import math
from numba import cuda
import hippocampus_genes
import hippocampus_kernel

# ==========================================
# 🧪 Experiment 2: Pattern Separation (Super Sparse Mode)
# ==========================================

# Config
NUM_NEURONS = 10000
SIM_TIME = 500          
DT = 0.5
THETA_FREQ = 8.0

# Adjustment: Enforce strict sparse firing (Only highly activated neurons fire).
# Calculation: Max Input(1.0)*30 + MD(5)*1.5 - 35 = +10.0 (Fire)
# Calculation: Mid Input(0.5)*30 + MD(5)*1.5 - 35 = -5.0 (Silence) -> Noise suppression!
GAIN_LD = 30.0          
GAIN_MD = 5.0          
BIAS = -35.0            # Strong inhibitory bias

INPUT_OVERLAP = 0.90    # 90% overlap (Input images are mostly identical)

class PatternSeparationExperiment:
    def __init__(self):
        print("🧠 Initializing Pattern Separation Experiment (Super Sparse Mode)...")
        brain = hippocampus_genes.generate_hippocampal_network(NUM_NEURONS)
        self.params = brain['params']
        self.state = brain['state']
        
        self.d_a = cuda.to_device(self.params['a'])
        self.d_b = cuda.to_device(self.params['b'])
        self.d_c = cuda.to_device(self.params['c'])
        self.d_d = cuda.to_device(self.params['d'])
        
        self.v_init = self.state['v'].copy()
        self.u_init = self.state['u'].copy()
        
        self.d_v_in = cuda.to_device(self.v_init)
        self.d_v_out = cuda.device_array_like(self.d_v_in)
        self.d_u_in = cuda.to_device(self.u_init)
        self.d_u_out = cuda.device_array_like(self.d_u_in)
        
        self.d_input_MD = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_input_LD = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_spikes = cuda.device_array(NUM_NEURONS, dtype=np.int32)
        self.blocks, self.threads = hippocampus_kernel.get_block_grid_dim(NUM_NEURONS)

    def run_simulation(self, input_pattern_LD, phase_strength, phase_offset=0.0):
        self.d_v_in.copy_to_device(self.v_init)
        self.d_u_in.copy_to_device(self.u_init)
        
        # Apply Gain & Bias here
        # Pass adjusted values here instead of calculating inside the kernel
        pattern_with_bias = input_pattern_LD + BIAS
        self.d_input_LD.copy_to_device(pattern_with_bias)
        
        # MD Input (Constant baseline to be modulated)
        self.d_input_MD.copy_to_device(np.ones(NUM_NEURONS, dtype=np.float32) * GAIN_MD)
        
        total_spikes = np.zeros(NUM_NEURONS, dtype=np.int32)
        
        for t_step in range(int(SIM_TIME / DT)):
            current_time = t_step * DT
            # Peak Phase Start
            phase = 2 * math.pi * THETA_FREQ * (current_time / 1000.0) + phase_offset
            
            hippocampus_kernel.dg_process_kernel[self.blocks, self.threads](
                self.d_v_in, self.d_v_out, self.d_u_in, self.d_u_out,
                self.d_a, self.d_b, self.d_c, self.d_d,
                self.d_input_MD, self.d_input_LD,
                self.d_spikes,
                NUM_NEURONS, t_step, 
                phase, 
                phase_strength, 
                DT
            )
            
            spikes = self.d_spikes.copy_to_host()
            total_spikes += spikes
            
            self.d_v_in, self.d_v_out = self.d_v_out, self.d_v_in
            self.d_u_in, self.d_u_out = self.d_u_out, self.d_u_in
            
        return total_spikes

def calculate_similarity(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

def main():
    exp = PatternSeparationExperiment()
    
    print("\ngenerating Input Patterns (Overlap = 90%)...")
    np.random.seed(100)
    
    # Generate Pattern A & B
    base = np.random.rand(NUM_NEURONS).astype(np.float32)
    noise_A = np.random.rand(NUM_NEURONS).astype(np.float32) * 0.1
    noise_B = np.random.rand(NUM_NEURONS).astype(np.float32) * 0.1
    
    # Gain Apply
    pattern_A = (base + noise_A) * GAIN_LD
    pattern_B = (base + noise_B) * GAIN_LD
    
    input_sim = calculate_similarity(pattern_A, pattern_B)
    print(f"🔹 Input Correlation (A vs B): {input_sim:.4f}")
    
    # --- Condition 1: Theta OFF ---
    output_A_off = exp.run_simulation(pattern_A, phase_strength=0.0)
    output_B_off = exp.run_simulation(pattern_B, phase_strength=0.0)
    
    sparsity_A = np.count_nonzero(output_A_off) / NUM_NEURONS
    print(f"\n[Theta OFF]")
    print(f"   Sparsity: {sparsity_A*100:.1f}% (Target: 5-20%)")
    sim_off = calculate_similarity(output_A_off, output_B_off)
    print(f"   Output Correlation: {sim_off:.4f}")

    # --- Condition 2: Theta ON ---
    output_A_on = exp.run_simulation(pattern_A, phase_strength=0.5)
    output_B_on = exp.run_simulation(pattern_B, phase_strength=0.5)
    
    sparsity_A_on = np.count_nonzero(output_A_on) / NUM_NEURONS
    print(f"\n[Theta ON]")
    print(f"   Sparsity: {sparsity_A_on*100:.1f}%")
    sim_on = calculate_similarity(output_A_on, output_B_on)
    print(f"   Output Correlation: {sim_on:.4f}")
    
    print(f"\nDiff (Separation Gain): {sim_off - sim_on:.4f}")
    
    # Visualization
    labels = ['Input', 'Theta OFF', 'Theta ON']
    values = [input_sim, sim_off, sim_on]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['gray', 'blue', 'green'])
    plt.ylim(0, 1.1)
    plt.ylabel('Cosine Similarity')
    plt.title(f'Pattern Separation (Input Overlap: {input_sim:.3f})')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.axhline(y=input_sim, color='r', linestyle='--', label='Input Baseline')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()