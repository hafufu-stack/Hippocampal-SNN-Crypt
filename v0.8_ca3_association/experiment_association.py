import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import hippocampus_genes
import hippocampus_kernel

# ==========================================
# 🧪 Experiment 8: Associative Memory (Pattern Completion)
# ==========================================
# Objective: Learn the association between Part A and Part B.
#            Then, trigger Part B by stimulating only Part A.

# Config
NUM_NEURONS = 2000      
CONNECTIONS = 100       
SIM_TIME_TRAIN = 1000   
SIM_TIME_TEST = 500     
DT = 0.5

# Parameters
GAIN_INPUT = 40.0
LEARNING_RATE = 0.1     # Increased for faster learning from zero

class AssociationExperiment:
    def __init__(self):
        print("🧠 Initializing CA3 Association Experiment...")
        brain = hippocampus_genes.generate_hippocampal_network(NUM_NEURONS)
        self.params = brain['params']
        self.state = brain['state']
        
        # Generate Recurrent Circuit
        print(f"🔗 Wiring {NUM_NEURONS} neurons ({CONNECTIONS} conns/neuron)...")
        indices, _ = hippocampus_genes.generate_recurrent_connections(NUM_NEURONS, CONNECTIONS)
        
        # ★修正: 初期ウェイトを「ほぼゼロ」にする
        # これにより、学習していないペアは反応しなくなる（選択性の向上）
        weights = np.random.uniform(0.0, 0.05, (NUM_NEURONS, CONNECTIONS)).astype(np.float32)
        
        # GPU Alloc
        self.d_a = cuda.to_device(self.params['a'])
        self.d_b = cuda.to_device(self.params['b'])
        self.d_c = cuda.to_device(self.params['c'])
        self.d_d = cuda.to_device(self.params['d'])
        
        self.d_v_in = cuda.to_device(self.state['v'])
        self.d_v_out = cuda.device_array_like(self.d_v_in)
        self.d_u_in = cuda.to_device(self.state['u'])
        self.d_u_out = cuda.device_array_like(self.d_u_in)
        
        self.d_rec_indices = cuda.to_device(indices)
        self.d_rec_weights = cuda.to_device(weights)
        
        self.d_input_ext = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_spikes = cuda.device_array(NUM_NEURONS, dtype=np.int32)
        self.d_prev_spikes = cuda.device_array(NUM_NEURONS, dtype=np.int32) 
        self.d_prev_spikes.copy_to_device(np.zeros(NUM_NEURONS, dtype=np.int32))

        self.blocks, self.threads = hippocampus_kernel.get_block_grid_dim(NUM_NEURONS)

    def run_phase(self, input_pattern, duration, is_training):
        self.d_input_ext.copy_to_device(input_pattern)
        
        spike_history_t = []
        spike_history_id = []
        
        lr = LEARNING_RATE if is_training else 0.0
        steps = int(duration / DT)
        
        for t_step in range(steps):
            time = t_step * DT
            
            hippocampus_kernel.ca3_process_kernel[self.blocks, self.threads](
                self.d_v_in, self.d_v_out, self.d_u_in, self.d_u_out,
                self.d_a, self.d_b, self.d_c, self.d_d,
                self.d_input_ext,
                self.d_prev_spikes, 
                self.d_rec_indices, self.d_rec_weights, 
                self.d_spikes,
                NUM_NEURONS, CONNECTIONS,
                lr, 
                DT
            )
            
            spikes = self.d_spikes.copy_to_host()
            idx = np.where(spikes > 0)[0]
            if len(idx) > 0:
                spike_history_t.extend([time] * len(idx))
                spike_history_id.extend(idx)
            
            self.d_prev_spikes.copy_to_device(spikes)
            self.d_v_in, self.d_v_out = self.d_v_out, self.d_v_in
            self.d_u_in, self.d_u_out = self.d_u_out, self.d_u_in

        return spike_history_t, spike_history_id

def main():
    exp = AssociationExperiment()
    
    # Pattern A: 0 - 500
    # Pattern B: 1000 - 1500
    pattern_full = np.zeros(NUM_NEURONS, dtype=np.float32)
    pattern_full[0:500] = GAIN_INPUT
    pattern_full[1000:1500] = GAIN_INPUT
    
    pattern_partial = np.zeros(NUM_NEURONS, dtype=np.float32)
    pattern_partial[0:500] = GAIN_INPUT 
    
    # 1. Training
    print("\n--- Phase 1: Training (Association A <-> B) ---")
    _, _ = exp.run_phase(pattern_full, SIM_TIME_TRAIN, is_training=True)
    
    # 2. Testing
    print("\n--- Phase 2: Testing (Recall B from A) ---")
    spikes_t, spikes_id = exp.run_phase(pattern_partial, SIM_TIME_TEST, is_training=False)
    
    print("\n📊 Analyzing Results...")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(spikes_t, spikes_id, s=1, color='blue', alpha=0.5)
    
    plt.axhspan(0, 500, color='green', alpha=0.1, label='Pattern A (Input ON)')
    plt.axhspan(1000, 1500, color='red', alpha=0.1, label='Pattern B (Input OFF)')
    
    plt.title('Associative Memory Recall (Selective)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Verification
    spikes_in_B = [i for i in spikes_id if 1000 <= i < 1500]
    # Check background (500-1000)
    spikes_background = [i for i in spikes_id if 500 <= i < 1000]
    
    print(f"   Spikes in B (Target): {len(spikes_in_B)}")
    print(f"   Spikes in Background (Noise): {len(spikes_background)}")
    
    if len(spikes_in_B) > len(spikes_background) * 5:
        print("\n🏆 Success! Strong selectivity. Background is quiet.")
    else:
        print("\n⚠️ Background is still noisy.")

if __name__ == "__main__":
    main()