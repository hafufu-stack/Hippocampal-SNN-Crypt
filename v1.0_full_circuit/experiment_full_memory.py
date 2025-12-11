import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import hippocampus_genes
import hippocampus_kernels

# ==========================================
# 🧪 Experiment 10: Full Memory Circuit (v1.0)
# ==========================================
# Objective:
# 1. DG receives Input -> Fires -> Drives CA3 (Detonator).
# 2. CA3 fires -> Strengthens Recurrent weights (STDP).
# 3. Input OFF -> CA3 continues firing (Reverberation/Attractor).

# Config
N_DG = 10000
N_CA3 = 2000
SIM_TIME = 1000
DT = 0.5

# Connectivity & Learning (Tuned by User)
PROB_DG_CA3 = 0.005     # Mossy Fiber
PROB_CA3_CA3 = 0.03     # Recurrent (3% connection - Sparse enough to prevent epilepsy)
WEIGHT_MF = 50.0        # Strong driver
LEARNING_RATE = 0.5     # High rate for rapid one-shot learning

class FullMemoryCircuit:
    def __init__(self):
        print("🧠 Initializing Full Memory Circuit (DG + CA3)...")
        
        # 1. Init Populations
        self.dg = hippocampus_genes.generate_network_params(N_DG, "GC")
        self.ca3 = hippocampus_genes.generate_network_params(N_CA3, "CA3")
        
        # 2. Connections
        print("🔗 Wiring DG -> CA3 (Mossy Fibers)...")
        mf_ptr, mf_idx, mf_w = hippocampus_genes.generate_connections(N_DG, N_CA3, PROB_DG_CA3, WEIGHT_MF, False)
        self.d_mf_ptr = cuda.to_device(mf_ptr)
        self.d_mf_idx = cuda.to_device(mf_idx)
        self.d_mf_w = cuda.to_device(mf_w)
        
        print(f"🔗 Wiring CA3 -> CA3 (Recurrent, Prob={PROB_CA3_CA3})...")
        rec_ptr, rec_idx, rec_w = hippocampus_genes.generate_connections(N_CA3, N_CA3, PROB_CA3_CA3, 0.0, True)
        self.d_rec_ptr = cuda.to_device(rec_ptr)
        self.d_rec_idx = cuda.to_device(rec_idx)
        self.d_rec_w = cuda.to_device(rec_w) 
        
        # 3. GPU State
        self._alloc_gpu()
        
        # Dims
        self.dim_dg = hippocampus_kernels.get_dims(N_DG)
        self.dim_ca3 = hippocampus_kernels.get_dims(N_CA3)

    def _alloc_gpu(self):
        # DG
        self.d_dg_v = cuda.to_device(self.dg['state']['v'])
        self.d_dg_u = cuda.to_device(self.dg['state']['u'])
        self.d_dg_a = cuda.to_device(self.dg['params']['a'])
        self.d_dg_b = cuda.to_device(self.dg['params']['b'])
        self.d_dg_c = cuda.to_device(self.dg['params']['c'])
        self.d_dg_d = cuda.to_device(self.dg['params']['d'])
        self.d_dg_spikes = cuda.device_array(N_DG, dtype=np.int32)
        self.d_dg_input = cuda.device_array(N_DG, dtype=np.float32)
        
        # CA3
        self.d_ca3_v = cuda.to_device(self.ca3['state']['v'])
        self.d_ca3_u = cuda.to_device(self.ca3['state']['u'])
        self.d_ca3_a = cuda.to_device(self.ca3['params']['a'])
        self.d_ca3_b = cuda.to_device(self.ca3['params']['b'])
        self.d_ca3_c = cuda.to_device(self.ca3['params']['c'])
        self.d_ca3_d = cuda.to_device(self.ca3['params']['d'])
        self.d_ca3_spikes = cuda.device_array(N_CA3, dtype=np.int32)
        self.d_ca3_prev_spikes = cuda.device_array(N_CA3, dtype=np.int32)
        self.d_ca3_current_buffer = cuda.device_array(N_CA3, dtype=np.float32)

    def run(self):
        print("🚀 Running Full Simulation...")
        dg_history_t, dg_history_id = [], []
        ca3_history_t, ca3_history_id = [], []
        
        # Input Pattern (First 2000 neurons of DG)
        input_pattern = np.zeros(N_DG, dtype=np.float32)
        input_pattern[:2000] = 30.0 
        
        steps = int(SIM_TIME / DT)
        
        for t_step in range(steps):
            time = t_step * DT
            
            # --- 1. Clear Buffers ---
            hippocampus_kernels.clear_buffer_kernel[self.dim_ca3[0], self.dim_ca3[1]](self.d_ca3_current_buffer, N_CA3)
            
            # --- 2. DG Update ---
            # Input ON for 0-200ms, then OFF
            if time < 200:
                self.d_dg_input.copy_to_device(input_pattern)
            else:
                self.d_dg_input.copy_to_device(np.zeros(N_DG, dtype=np.float32))
            
            hippocampus_kernels.dg_update_kernel[self.dim_dg[0], self.dim_dg[1]](
                self.d_dg_v, self.d_dg_u, self.d_dg_a, self.d_dg_b, self.d_dg_c, self.d_dg_d,
                self.d_dg_input, self.d_dg_spikes, DT, N_DG
            )
            
            # --- 3. Transmission (DG -> CA3) ---
            hippocampus_kernels.synapse_transmission_kernel[self.dim_dg[0], self.dim_dg[1]](
                self.d_dg_spikes, self.d_ca3_current_buffer,
                self.d_mf_ptr, self.d_mf_idx, self.d_mf_w, N_DG
            )
            
            # --- 4. Recurrent Transmission (CA3 -> CA3) ---
            hippocampus_kernels.synapse_transmission_kernel[self.dim_ca3[0], self.dim_ca3[1]](
                self.d_ca3_prev_spikes, self.d_ca3_current_buffer,
                self.d_rec_ptr, self.d_rec_idx, self.d_rec_w, N_CA3
            )
            
            # --- 5. CA3 Update ---
            hippocampus_kernels.ca3_update_stdp_kernel[self.dim_ca3[0], self.dim_ca3[1]](
                self.d_ca3_v, self.d_ca3_u, self.d_ca3_a, self.d_ca3_b, self.d_ca3_c, self.d_ca3_d,
                self.d_ca3_current_buffer,
                self.d_ca3_prev_spikes,
                self.d_rec_ptr, self.d_rec_idx, self.d_rec_w,
                self.d_ca3_spikes,
                LEARNING_RATE, DT, N_CA3
            )
            
            # --- 6. Recurrent STDP Update ---
            hippocampus_kernels.ca3_stdp_kernel[self.dim_ca3[0], self.dim_ca3[1]](
                self.d_ca3_prev_spikes, self.d_ca3_spikes,
                self.d_rec_ptr, self.d_rec_idx, self.d_rec_w,
                LEARNING_RATE, N_CA3
            )
            
            # --- 7. Recording & Feedback ---
            self.d_ca3_prev_spikes.copy_to_device(self.d_ca3_spikes)
            
            if t_step % 20 == 0: 
                dg_s = self.d_dg_spikes.copy_to_host()
                ca3_s = self.d_ca3_spikes.copy_to_host()
                
                dg_ids = np.where(dg_s > 0)[0]
                if len(dg_ids) > 0:
                    dg_history_t.extend([time] * len(dg_ids))
                    dg_history_id.extend(dg_ids)
                    
                ca3_ids = np.where(ca3_s > 0)[0]
                if len(ca3_ids) > 0:
                    ca3_history_t.extend([time] * len(ca3_ids))
                    ca3_history_id.extend(ca3_ids)
            
            if t_step % 100 == 0:
                print(f"   t={time}ms...", end='\r')

        return dg_history_t, dg_history_id, ca3_history_t, ca3_history_id

def main():
    model = FullMemoryCircuit()
    dg_t, dg_id, ca3_t, ca3_id = model.run()
    
    print("\n📊 Analyzing Results...")
    
    plt.figure(figsize=(12, 10))
    
    # 1. DG
    plt.subplot(2, 1, 1)
    plt.scatter(dg_t, dg_id, s=1, color='blue', alpha=0.5)
    plt.axvline(x=200, color='red', linestyle='--', label='Input OFF')
    plt.title('DG (Input Layer)')
    plt.ylabel('Neuron ID')
    plt.legend()
    plt.xlim(0, SIM_TIME)
    
    # 2. CA3
    plt.subplot(2, 1, 2)
    plt.scatter(ca3_t, ca3_id, s=1, color='green', alpha=0.5)
    plt.axvline(x=200, color='red', linestyle='--', label='Input OFF')
    plt.title('CA3 (Memory Layer)')
    plt.ylabel('Neuron ID')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.xlim(0, SIM_TIME)
    
    plt.tight_layout()
    plt.show()
    
    # Verification
    late_activity = [t for t in ca3_t if t > 300]
    if len(late_activity) > 100:
        print("\n🏆 v1.0 SUCCESS! Memory formation and maintenance confirmed.")
        print("   Stable Attractor State achieved.")
    else:
        print("\n⚠️ Memory faded.")

if __name__ == "__main__":
    main()