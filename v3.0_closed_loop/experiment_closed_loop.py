import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import hippocampus_genes
import hippocampus_kernels

# ==========================================
# 🧪 Experiment 15: Closed-loop Sequence Replay (v3.0 Sedated)
# ==========================================

# Config
N_EC = 2000 
N_DG = 5000
N_CA3 = 2000
N_CA1 = 2000
SIM_TIME = 1200 
DT = 0.5

# Connectivity (Very weak to prevent explosion)
WEIGHT_MF = 12.0    
WEIGHT_SC = 1.0     # Weak readout
WEIGHT_FB = 0.5     # Weak feedback
WEIGHT_EC_CA1 = 0.5 

# Learning Rate
LEARNING_RATE = 0.01 # Slow and steady learning

# Global Inhibition
BIAS_STRONG = -16.0  # Strong brake

def get_dims(n):
    tp = 256
    bg = (n + tp - 1) // tp
    return bg, tp

class ClosedLoopCircuit:
    def __init__(self):
        print("🧠 Initializing Closed-Loop Circuit (Sedated Mode)...")
        
        self.ec = hippocampus_genes.generate_network_params(N_EC, "EC")
        self.dg = hippocampus_genes.generate_network_params(N_DG, "GC")
        self.ca3 = hippocampus_genes.generate_network_params(N_CA3, "CA3")
        self.ca1 = hippocampus_genes.generate_network_params(N_CA1, "CA1")
        
        print("🔗 Wiring Feedforward & Feedback Loops...")
        # Feedforward
        ec_dg_p, ec_dg_i, ec_dg_w = hippocampus_genes.generate_connections(N_EC, N_DG, 0.05, 30.0)
        self.d_ec_dg = (cuda.to_device(ec_dg_p), cuda.to_device(ec_dg_i), cuda.to_device(ec_dg_w))
        
        dg_ca3_p, dg_ca3_i, dg_ca3_w = hippocampus_genes.generate_connections(N_DG, N_CA3, 0.01, WEIGHT_MF)
        self.d_dg_ca3 = (cuda.to_device(dg_ca3_p), cuda.to_device(dg_ca3_i), cuda.to_device(dg_ca3_w))
        
        r_p, r_i, r_w = hippocampus_genes.generate_connections(N_CA3, N_CA3, 0.05, 0.0, True)
        self.d_rec = (cuda.to_device(r_p), cuda.to_device(r_i), cuda.to_device(r_w))
        
        c3_c1_p, c3_c1_i, c3_c1_w = hippocampus_genes.generate_connections(N_CA3, N_CA1, 0.1, WEIGHT_SC, True)
        self.d_ca3_ca1 = (cuda.to_device(c3_c1_p), cuda.to_device(c3_c1_i), cuda.to_device(c3_c1_w))
        
        ec_c1_p, ec_c1_i, ec_c1_w = hippocampus_genes.generate_connections(N_EC, N_CA1, 0.1, WEIGHT_EC_CA1)
        self.d_ec_ca1 = (cuda.to_device(ec_c1_p), cuda.to_device(ec_c1_i), cuda.to_device(ec_c1_w))
        
        # Feedback Loop
        c1_ec_p, c1_ec_i, c1_ec_w = hippocampus_genes.generate_connections(N_CA1, N_EC, 0.1, 0.0, True)
        self.d_ca1_ec = (cuda.to_device(c1_ec_p), cuda.to_device(c1_ec_i), cuda.to_device(c1_ec_w))

        self._alloc_gpu()
        self.dims = {
            'ec': get_dims(N_EC), 'dg': get_dims(N_DG),
            'ca3': get_dims(N_CA3), 'ca1': get_dims(N_CA1)
        }

    def _alloc_gpu(self):
        def create(n): return (
            cuda.device_array(n, dtype=np.float32), cuda.device_array(n, dtype=np.float32),
            cuda.device_array(n, dtype=np.float32), cuda.device_array(n, dtype=np.int32)
        )
        def params(p): return (
            cuda.to_device(p['params']['a']), cuda.to_device(p['params']['b']),
            cuda.to_device(p['params']['c']), cuda.to_device(p['params']['d'])
        )
        
        self.d_ec_v, self.d_ec_u, self.d_ec_i, self.d_ec_s = create(N_EC)
        self.d_dg_v, self.d_dg_u, self.d_dg_i, self.d_dg_s = create(N_DG)
        self.d_ca3_v, self.d_ca3_u, self.d_ca3_i, self.d_ca3_s = create(N_CA3)
        self.d_ca1_v, self.d_ca1_u, self.d_ca1_i, self.d_ca1_s = create(N_CA1)
        
        self.d_ca3_ps = cuda.to_device(np.zeros(N_CA3, dtype=np.int32))
        
        self.p_ec = params(self.ec)
        self.p_dg = params(self.dg)
        self.p_ca3 = params(self.ca3)
        self.p_ca1 = params(self.ca1)

    def run(self):
        print("🚀 Running Sequence Simulation...")
        
        ec_log, ca1_log = [], []
        
        np.random.seed(42)
        input_stream = np.zeros((3, N_EC), dtype=np.float32)
        input_stream[0, 0:500] = 30.0    # A
        input_stream[1, 500:1000] = 30.0 # B
        input_stream[2, 1000:1500] = 30.0# C
        
        steps = int(SIM_TIME / DT)
        
        for t_step in range(steps):
            time = t_step * DT
            
            external_input = np.zeros(N_EC, dtype=np.float32)
            is_training = False
            
            if time < 300:
                external_input = input_stream[0]
                is_training = True
            elif time < 600:
                external_input = input_stream[1]
                is_training = True
            elif time < 900:
                external_input = input_stream[2]
                is_training = True
            elif time < 950:
                # Cue A
                external_input = input_stream[0]
                is_training = False
            else:
                external_input = np.zeros(N_EC, dtype=np.float32)
                is_training = False
            
            lr = LEARNING_RATE if is_training else 0.0

            # 1. EC Update
            self.d_ec_i.copy_to_device(external_input)
            
            # Feedback CA1 -> EC
            hippocampus_kernels.synapse_transmission_kernel[self.dims['ca1'][0], self.dims['ca1'][1]](self.d_ca1_s, self.d_ec_i, *self.d_ca1_ec, N_CA1)
            
            # ★修正: Biasを -15.0 にして、簡単には発火させない
            hippocampus_kernels.update_neuron_kernel[self.dims['ec'][0], self.dims['ec'][1]](
                self.d_ec_v, self.d_ec_u, *self.p_ec, self.d_ec_i, self.d_ec_s, BIAS_STRONG, DT, N_EC
            )
            
            # STDP for Feedback
            hippocampus_kernels.stdp_kernel[self.dims['ca1'][0], self.dims['ca1'][1]](self.d_ca1_s, self.d_ec_s, *self.d_ca1_ec, lr, N_CA1)

            # Clear buffers
            hippocampus_kernels.clear_buffer_kernel[self.dims['dg'][0], self.dims['dg'][1]](self.d_dg_i, N_DG)
            hippocampus_kernels.clear_buffer_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_i, N_CA3)
            hippocampus_kernels.clear_buffer_kernel[self.dims['ca1'][0], self.dims['ca1'][1]](self.d_ca1_i, N_CA1)
            
            # Propagate
            hippocampus_kernels.synapse_transmission_kernel[self.dims['ec'][0], self.dims['ec'][1]](self.d_ec_s, self.d_dg_i, *self.d_ec_dg, N_EC)
            hippocampus_kernels.synapse_transmission_kernel[self.dims['ec'][0], self.dims['ec'][1]](self.d_ec_s, self.d_ca1_i, *self.d_ec_ca1, N_EC)
            
            hippocampus_kernels.update_neuron_kernel[self.dims['dg'][0], self.dims['dg'][1]](self.d_dg_v, self.d_dg_u, *self.p_dg, self.d_dg_i, self.d_dg_s, 0.0, DT, N_DG)
            
            hippocampus_kernels.synapse_transmission_kernel[self.dims['dg'][0], self.dims['dg'][1]](self.d_dg_s, self.d_ca3_i, *self.d_dg_ca3, N_DG)
            hippocampus_kernels.synapse_transmission_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_ps, self.d_ca3_i, *self.d_rec, N_CA3)
            
            hippocampus_kernels.update_neuron_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_v, self.d_ca3_u, *self.p_ca3, self.d_ca3_i, self.d_ca3_s, 0.0, DT, N_CA3)
            
            hippocampus_kernels.stdp_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_ps, self.d_ca3_s, *self.d_rec, lr, N_CA3)
            hippocampus_kernels.copy_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_s, self.d_ca3_ps, N_CA3)
            
            hippocampus_kernels.synapse_transmission_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_s, self.d_ca1_i, *self.d_ca3_ca1, N_CA3)
            
            # CA1 Update
            # ★修正: Biasを -15.0 に
            hippocampus_kernels.update_neuron_kernel[self.dims['ca1'][0], self.dims['ca1'][1]](self.d_ca1_v, self.d_ca1_u, *self.p_ca1, self.d_ca1_i, self.d_ca1_s, BIAS_STRONG, DT, N_CA1)
            
            hippocampus_kernels.stdp_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_s, self.d_ca1_s, *self.d_ca3_ca1, lr, N_CA3)
            
            if t_step % 20 == 0:
                ec_log.append((time, np.where(self.d_ec_s.copy_to_host() > 0)[0]))
                ca1_log.append((time, np.where(self.d_ca1_s.copy_to_host() > 0)[0]))
                if t_step % 500 == 0: print(f"   t={time:.1f}ms...", end='\r')

        return ec_log, ca1_log

def main():
    circuit = ClosedLoopCircuit()
    ec_data, ca1_data = circuit.run()
    
    print("\n📊 Analyzing Sequence Replay...")
    plt.figure(figsize=(10, 8))
    
    # EC Activity
    plt.subplot(2, 1, 1)
    for t, ids in ec_data:
        plt.scatter([t]*len(ids), ids, s=1, color='orange', alpha=0.5)
    plt.title('EC Activity (Input & Feedback)')
    plt.ylabel('Neuron ID')
    plt.axvline(x=900, color='black', linestyle='-', linewidth=2, label='Training End')
    plt.axvline(x=950, color='red', linestyle='--', label='Cue A End')
    
    plt.axhspan(0, 500, alpha=0.1, color='red', label='A')
    plt.axhspan(500, 1000, alpha=0.1, color='green', label='B')
    plt.axhspan(1000, 1500, alpha=0.1, color='blue', label='C')
    plt.legend(loc='upper right')
    plt.xlim(0, SIM_TIME)
    
    # CA1 Activity
    plt.subplot(2, 1, 2)
    for t, ids in ca1_data:
        plt.scatter([t]*len(ids), ids, s=1, color='red', alpha=0.5)
    plt.title('CA1 Output')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.xlim(0, SIM_TIME)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Check after 950ms.")
    print("   Look for spontaneous activity moving A -> B -> C.")

if __name__ == "__main__":
    main()