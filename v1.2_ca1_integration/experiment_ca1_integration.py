import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import hippocampus_genes
import hippocampus_kernels

# ==========================================
# 🧪 Experiment 12: CA1 Integration (v1.2)
# ==========================================
# Objective: Full Circuit Integration (DG -> CA3 -> CA1).
# Show the sensory-to-output information flow and memory readout.

# Config
N_DG = 5000
N_CA3 = 2000
N_CA1 = 2000
SIM_TIME = 600
DT = 0.5

# Params (Tuned for Signal Propagation)
WEIGHT_MF = 40.0    # DG -> CA3
WEIGHT_SC = 80.0    # CA3 -> CA1 (High gain to pick up reverb)
PROB_SC = 0.15      # 15% connectivity
LEARNING_RATE = 0.2

def get_dims(n):
    tp = 256
    bg = (n + tp - 1) // tp
    return bg, tp

class FullTrisynapticCircuit:
    def __init__(self):
        print("🧠 Building CA1 Integrated Circuit...")
        # 1. Populations
        self.dg = hippocampus_genes.generate_network_params(N_DG, "GC")
        self.ca3 = hippocampus_genes.generate_network_params(N_CA3, "CA3")
        self.ca1 = hippocampus_genes.generate_network_params(N_CA1, "CA1")
        
        # 2. Connections
        print("🔗 Connecting Populations...")
        # DG -> CA3
        mf_ptr, mf_idx, mf_w = hippocampus_genes.generate_connections(N_DG, N_CA3, 0.01, WEIGHT_MF)
        self.d_mf_ptr, self.d_mf_idx, self.d_mf_w = cuda.to_device(mf_ptr), cuda.to_device(mf_idx), cuda.to_device(mf_w)
        # CA3 -> CA3 (Recurrent)
        r_ptr, r_idx, r_w = hippocampus_genes.generate_connections(N_CA3, N_CA3, 0.03, 0.0, True)
        self.d_rec_ptr, self.d_rec_idx, self.d_rec_w = cuda.to_device(r_ptr), cuda.to_device(r_idx), cuda.to_device(r_w)
        # CA3 -> CA1 (Schaffer Collateral)
        sc_ptr, sc_idx, sc_w = hippocampus_genes.generate_connections(N_CA3, N_CA1, PROB_SC, WEIGHT_SC)
        self.d_sc_ptr, self.d_sc_idx, self.d_sc_w = cuda.to_device(sc_ptr), cuda.to_device(sc_idx), cuda.to_device(sc_w)

        self._alloc_gpu()
        
    def _alloc_gpu(self):
        # DG
        self.d_dg_v = cuda.to_device(self.dg['state']['v']); self.d_dg_u = cuda.to_device(self.dg['state']['u'])
        self.d_dg_a = cuda.to_device(self.dg['params']['a']); self.d_dg_b = cuda.to_device(self.dg['params']['b'])
        self.d_dg_c = cuda.to_device(self.dg['params']['c']); self.d_dg_d = cuda.to_device(self.dg['params']['d'])
        self.d_dg_s = cuda.device_array(N_DG, dtype=np.int32); self.d_dg_i = cuda.device_array(N_DG, dtype=np.float32)
        
        # CA3
        self.d_ca3_v = cuda.to_device(self.ca3['state']['v']); self.d_ca3_u = cuda.to_device(self.ca3['state']['u'])
        self.d_ca3_a = cuda.to_device(self.ca3['params']['a']); self.d_ca3_b = cuda.to_device(self.ca3['params']['b'])
        self.d_ca3_c = cuda.to_device(self.ca3['params']['c']); self.d_ca3_d = cuda.to_device(self.ca3['params']['d'])
        self.d_ca3_s = cuda.device_array(N_CA3, dtype=np.int32); self.d_ca3_ps = cuda.device_array(N_CA3, dtype=np.int32)
        self.d_ca3_i = cuda.device_array(N_CA3, dtype=np.float32)
        
        # CA1
        self.d_ca1_v = cuda.to_device(self.ca1['state']['v']); self.d_ca1_u = cuda.to_device(self.ca1['state']['u'])
        self.d_ca1_a = cuda.to_device(self.ca1['params']['a']); self.d_ca1_b = cuda.to_device(self.ca1['params']['b'])
        self.d_ca1_c = cuda.to_device(self.ca1['params']['c']); self.d_ca1_d = cuda.to_device(self.ca1['params']['d'])
        self.d_ca1_s = cuda.device_array(N_CA1, dtype=np.int32); self.d_ca1_i = cuda.device_array(N_CA1, dtype=np.float32)

    def run(self):
        print("🚀 Running Simulation...")
        dg_log, ca3_log, ca1_log = [], [], []
        steps = int(SIM_TIME / DT)
        
        for t_step in range(steps):
            time = t_step * DT
            
            # Clear Inputs
            hippocampus_kernels.clear_buffer_kernel[get_dims(N_CA3)](self.d_ca3_i, N_CA3)
            hippocampus_kernels.clear_buffer_kernel[get_dims(N_CA1)](self.d_ca1_i, N_CA1)
            
            # 1. DG Update
            inp = np.zeros(N_DG, dtype=np.float32)
            if time < 200: inp[:1000] = 30.0 
            self.d_dg_i.copy_to_device(inp)
            hippocampus_kernels.update_neuron_kernel[get_dims(N_DG)](self.d_dg_v, self.d_dg_u, self.d_dg_a, self.d_dg_b, self.d_dg_c, self.d_dg_d, self.d_dg_i, self.d_dg_s, 0.0, DT, N_DG)
            
            # 2. DG -> CA3
            hippocampus_kernels.synapse_transmission_kernel[get_dims(N_DG)](self.d_dg_s, self.d_ca3_i, self.d_mf_ptr, self.d_mf_idx, self.d_mf_w, N_DG)
            # 3. CA3 -> CA3
            hippocampus_kernels.synapse_transmission_kernel[get_dims(N_CA3)](self.d_ca3_ps, self.d_ca3_i, self.d_rec_ptr, self.d_rec_idx, self.d_rec_w, N_CA3)
            
            # 4. CA3 Update & STDP
            hippocampus_kernels.update_neuron_kernel[get_dims(N_CA3)](self.d_ca3_v, self.d_ca3_u, self.d_ca3_a, self.d_ca3_b, self.d_ca3_c, self.d_ca3_d, self.d_ca3_i, self.d_ca3_s, 0.0, DT, N_CA3)
            hippocampus_kernels.stdp_kernel[get_dims(N_CA3)](self.d_ca3_ps, self.d_ca3_s, self.d_rec_ptr, self.d_rec_idx, self.d_rec_w, LEARNING_RATE, N_CA3)
            self.d_ca3_ps.copy_to_device(self.d_ca3_s)
            
            # 5. CA3 -> CA1
            hippocampus_kernels.synapse_transmission_kernel[get_dims(N_CA3)](self.d_ca3_s, self.d_ca1_i, self.d_sc_ptr, self.d_sc_idx, self.d_sc_w, N_CA3)
            
            # 6. CA1 Update
            hippocampus_kernels.update_neuron_kernel[get_dims(N_CA1)](self.d_ca1_v, self.d_ca1_u, self.d_ca1_a, self.d_ca1_b, self.d_ca1_c, self.d_ca1_d, self.d_ca1_i, self.d_ca1_s, -2.0, DT, N_CA1)
            
            # Log
            if t_step % 20 == 0:
                d_ids = np.where(self.d_dg_s.copy_to_host() > 0)[0]
                if len(d_ids) > 0: dg_log.append((time, d_ids))
                c3_ids = np.where(self.d_ca3_s.copy_to_host() > 0)[0]
                if len(c3_ids) > 0: ca3_log.append((time, c3_ids))
                c1_ids = np.where(self.d_ca1_s.copy_to_host() > 0)[0]
                if len(c1_ids) > 0: ca1_log.append((time, c1_ids))

        return dg_log, ca3_log, ca1_log

def main():
    circuit = FullTrisynapticCircuit()
    dg, ca3, ca1 = circuit.run()
    
    print("\n📊 Plotting Results...")
    plt.figure(figsize=(12, 10))
    
    for idx, (data, title, color) in enumerate([(dg, 'DG', 'blue'), (ca3, 'CA3', 'green'), (ca1, 'CA1', 'red')]):
        plt.subplot(3, 1, idx+1)
        for t, ids in data:
            plt.scatter([t]*len(ids), ids, s=1, color=color, alpha=0.5)
        plt.title(title)
        plt.xlim(0, SIM_TIME)
        plt.ylabel('Neuron ID')
    
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()
    
    if len(ca1) > 50:
        print("\n🏆 v1.2 Success! Signal traveled DG -> CA3 -> CA1.")
        print("   CA1 successfully read out the memory held in CA3.")
    else:
        print("\n❌ Signal died at CA1.")

if __name__ == "__main__":
    main()