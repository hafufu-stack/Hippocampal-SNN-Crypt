import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import hippocampus_genes
import hippocampus_kernel

# ==========================================
# 🧪 Experiment 7: CA3 Reverberation (Working Memory)
# ==========================================
# Objective: Demonstrate that activity persists even after the input is removed
#            due to recurrent connections (Attractor Dynamics).

# Config
NUM_NEURONS = 2000      
CONNECTIONS = 50        # Synapses per neuron (Sparse connectivity)
SIM_TIME = 500          
DT = 0.5

# Parameters
GAIN_INPUT = 50.0       # Strong initial stimulus

class ReverberationExperiment:
    def __init__(self):
        print("🧠 Initializing CA3 Recurrent Experiment...")
        brain = hippocampus_genes.generate_hippocampal_network(NUM_NEURONS)
        self.params = brain['params']
        self.state = brain['state']
        
        # Generate Recurrent Circuit
        print(f"🔗 Wiring {NUM_NEURONS} neurons (Sparse Recurrent)...")
        indices, weights = hippocampus_genes.generate_recurrent_connections(NUM_NEURONS, CONNECTIONS)
        
        # GPU Allocation
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

    def run(self):
        print("🚀 Running Simulation...")
        
        spike_history_t = []
        spike_history_id = []
        input_history = []
        
        steps = int(SIM_TIME / DT)
        
        for t_step in range(steps):
            time = t_step * DT
            
            # --- Input Control ---
            # Input is ON only for the first 100ms
            if time < 100:
                current_input = GAIN_INPUT
            else:
                current_input = 0.0 # Input OFF
                
            # Stimulate only a subset (First 200 neurons)
            inp = np.zeros(NUM_NEURONS, dtype=np.float32)
            if current_input > 0:
                inp[:200] = current_input
            
            self.d_input_ext.copy_to_device(inp)
            input_history.append(current_input)
            
            # Kernel Call
            hippocampus_kernel.ca3_process_kernel[self.blocks, self.threads](
                self.d_v_in, self.d_v_out, self.d_u_in, self.d_u_out,
                self.d_a, self.d_b, self.d_c, self.d_d,
                self.d_input_ext,
                self.d_prev_spikes, # Input from previous step
                self.d_rec_indices, self.d_rec_weights,
                self.d_spikes,      # Output
                NUM_NEURONS, CONNECTIONS,
                DT
            )
            
            # Record Spikes
            spikes = self.d_spikes.copy_to_host()
            idx = np.where(spikes > 0)[0]
            if len(idx) > 0:
                spike_history_t.extend([time] * len(idx))
                spike_history_id.extend(idx)
            
            # Feedback Loop: Update previous spikes
            self.d_prev_spikes.copy_to_device(spikes)
            
            # Swap Buffers
            self.d_v_in, self.d_v_out = self.d_v_out, self.d_v_in
            self.d_u_in, self.d_u_out = self.d_u_out, self.d_u_in

        return spike_history_t, spike_history_id, input_history

def main():
    exp = ReverberationExperiment()
    spikes_t, spikes_id, inputs = exp.run()
    
    print("\n📊 Analyzing Results...")
    
    plt.figure(figsize=(10, 8))
    
    # 1. Input Timeline
    plt.subplot(2, 1, 1)
    times = np.arange(0, SIM_TIME, DT)
    plt.plot(times, inputs, color='green', label='External Input')
    plt.title('External Input (Stops at 100ms)')
    plt.ylabel('Input Current')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Raster Plot
    plt.subplot(2, 1, 2)
    plt.scatter(spikes_t, spikes_id, s=1, color='blue', alpha=0.5)
    plt.axvline(x=100, color='red', linestyle='--', label='Input OFF')
    plt.title('CA3 Recurrent Activity (Reverberation)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Verification
    late_spikes = [t for t in spikes_t if t > 150]
    if len(late_spikes) > 100:
        print("\n🏆 Success! Activity persists after input stops (Reverberation).")
        print("   The network is maintaining the memory state via recurrent loops.")
    else:
        print("\n❌ Activity died out. Needs stronger weights or more connections.")

if __name__ == "__main__":
    main()