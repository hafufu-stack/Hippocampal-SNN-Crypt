import numpy as np
import matplotlib.pyplot as plt
import math
from numba import cuda
import hippocampus_genes
import hippocampus_kernel

# ==========================================
# 🧪 Experiment 3: STDP Learning (Memory Formation)
# ==========================================
# Objective: Verify that the network can "learn" a specific pattern
#            by strengthening synaptic weights (LTP).

# Config
NUM_NEURONS = 10000
SIM_TIME_TEST = 200     # Short duration for testing
SIM_TIME_TRAIN = 1000   # Long duration for training
DT = 0.5
THETA_FREQ = 8.0

# Parameters
GAIN_LD = 20.0          
GAIN_MD = 10.0          
BIAS = -20.0            # Inhibitory bias
LEARNING_RATE = 0.05    # Learning rate for STDP

class LearningExperiment:
    def __init__(self):
        print("🧠 Initializing Learning Experiment...")
        brain = hippocampus_genes.generate_hippocampal_network(NUM_NEURONS)
        self.params = brain['params']
        self.state = brain['state']
        
        self.d_a = cuda.to_device(self.params['a'])
        self.d_b = cuda.to_device(self.params['b'])
        self.d_c = cuda.to_device(self.params['c'])
        self.d_d = cuda.to_device(self.params['d'])
        
        self.d_v_in = cuda.to_device(self.state['v'])
        self.d_v_out = cuda.device_array_like(self.d_v_in)
        self.d_u_in = cuda.to_device(self.state['u'])
        self.d_u_out = cuda.device_array_like(self.d_u_in)
        
        # Weights (Plastic)
        self.weights_host = np.ones(NUM_NEURONS, dtype=np.float32)
        self.d_weights_LD = cuda.to_device(self.weights_host)
        
        self.d_input_MD = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_input_LD = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_spikes = cuda.device_array(NUM_NEURONS, dtype=np.int32)
        
        self.blocks, self.threads = hippocampus_kernel.get_block_grid_dim(NUM_NEURONS)

    def run_phase(self, input_pattern, duration, is_training):
        # Reset neuron state for fair testing
        self.d_v_in.copy_to_device(self.state['v'])
        self.d_u_in.copy_to_device(self.state['u'])

        # Prepare inputs (Pass RAW positive signal)
        self.d_input_LD.copy_to_device(input_pattern)
        self.d_input_MD.copy_to_device(np.ones(NUM_NEURONS, dtype=np.float32) * GAIN_MD)
        
        lr = LEARNING_RATE if is_training else 0.0
        total_spikes = 0
        
        for t_step in range(int(duration / DT)):
            current_time = t_step * DT
            phase = 2 * math.pi * THETA_FREQ * (current_time / 1000.0)
            
            hippocampus_kernel.dg_process_kernel[self.blocks, self.threads](
                self.d_v_in, self.d_v_out, self.d_u_in, self.d_u_out,
                self.d_a, self.d_b, self.d_c, self.d_d,
                self.d_input_MD, self.d_input_LD,
                self.d_weights_LD, 
                self.d_spikes,
                NUM_NEURONS, t_step, 
                phase, 
                0.5, # Theta strength 
                lr,  # Learning Rate
                BIAS, 
                DT
            )
            
            spikes = self.d_spikes.copy_to_host()
            total_spikes += np.sum(spikes)
            
            self.d_v_in, self.d_v_out = self.d_v_out, self.d_v_in
            self.d_u_in, self.d_u_out = self.d_u_out, self.d_u_in
            
        return total_spikes

def main():
    exp = LearningExperiment()
    
    print("\ngenerating Patterns...")
    np.random.seed(999)
    
    # Generate Pattern A & B (Positive Signals)
    pattern_A = np.random.rand(NUM_NEURONS).astype(np.float32) * GAIN_LD
    pattern_B = np.random.rand(NUM_NEURONS).astype(np.float32) * GAIN_LD
    
    # 1. Pre-Test
    print("\n--- Phase 1: Pre-Test (Before Learning) ---")
    spikes_A_pre = exp.run_phase(pattern_A, SIM_TIME_TEST, is_training=False)
    print(f"   Response to A: {spikes_A_pre}")
    
    # 2. Training
    print(f"\n--- Phase 2: Training ({SIM_TIME_TRAIN}ms) ---")
    _ = exp.run_phase(pattern_A, SIM_TIME_TRAIN, is_training=True)
    
    weights = exp.d_weights_LD.copy_to_host()
    print(f"   Avg Weight: {np.mean(weights):.4f} (Max: {np.max(weights):.4f})")
    
    # 3. Post-Test
    print("\n--- Phase 3: Post-Test (After Learning) ---")
    spikes_A_post = exp.run_phase(pattern_A, SIM_TIME_TEST, is_training=False)
    print(f"   Response to A: {spikes_A_post}")
    
    # 4. Selectivity Test
    print("\n--- Phase 4: Selectivity Test (Pattern B) ---")
    spikes_B_post = exp.run_phase(pattern_B, SIM_TIME_TEST, is_training=False)
    print(f"   Response to B: {spikes_B_post}")
    
    # Visualization
    labels = ['A (Pre)', 'A (Post)', 'B (Post)']
    values = [spikes_A_pre, spikes_A_post, spikes_B_post]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['gray', 'green', 'blue'])
    plt.ylabel('Total Spikes (Population Response)')
    plt.title('Memory Formation via STDP')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f'{int(yval)}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    if spikes_A_post > spikes_A_pre:
        print("\n🏆 Success! Response increased after training.")
        print("   The network has successfully formed a memory of Pattern A.")
    else:
        print("\n❌ Learning failed. Check parameters.")

if __name__ == "__main__":
    main()