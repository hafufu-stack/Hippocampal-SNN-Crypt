import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import hippocampus_genes
import hippocampus_kernels

# ==========================================
# 🧪 Experiment 14: Novelty Detection & One-Shot Learning (v2.1 Re-Tuned 2)
# ==========================================

# Config
N_EC = 2000 
N_DG = 5000
N_CA3 = 2000
N_CA1 = 2000
SIM_TIME = 900 
DT = 0.2

# Connectivity (Boost Memory Path)
WEIGHT_MF = 20.0    
WEIGHT_SC = 2.0     # ★修正: 0.5 -> 2.0 (記憶の影響力を強める)
WEIGHT_EC_CA1 = 1.0 
PROB_SC = 0.1

# Learning Rates
LR_BASE = 0.01      # ★修正: 0.0001 -> 0.01 (Phase1でも基礎学習させる)
LR_DOPAMINE = 0.1   # Boost

# Bias
CA1_BIAS = -1.0     # ★修正: -2.0 -> -1.0 (発火しやすくして差を見えるように)

def get_dims(n):
    tp = 256
    bg = (n + tp - 1) // tp
    return bg, tp

class NoveltyCircuit:
    def __init__(self):
        print("🧠 Initializing Novelty Circuit...")
        
        self.ec = hippocampus_genes.generate_network_params(N_EC, "EC")
        self.dg = hippocampus_genes.generate_network_params(N_DG, "GC")
        self.ca3 = hippocampus_genes.generate_network_params(N_CA3, "CA3")
        self.ca1 = hippocampus_genes.generate_network_params(N_CA1, "CA1")
        
        print("🔗 Wiring Circuits...")
        ec_dg_p, ec_dg_i, ec_dg_w = hippocampus_genes.generate_connections(N_EC, N_DG, 0.05, 30.0)
        self.d_ec_dg = (cuda.to_device(ec_dg_p), cuda.to_device(ec_dg_i), cuda.to_device(ec_dg_w))
        
        dg_ca3_p, dg_ca3_i, dg_ca3_w = hippocampus_genes.generate_connections(N_DG, N_CA3, 0.01, WEIGHT_MF)
        self.d_dg_ca3 = (cuda.to_device(dg_ca3_p), cuda.to_device(dg_ca3_i), cuda.to_device(dg_ca3_w))
        
        r_p, r_i, r_w = hippocampus_genes.generate_connections(N_CA3, N_CA3, 0.05, 0.0, True)
        self.d_rec = (cuda.to_device(r_p), cuda.to_device(r_i), cuda.to_device(r_w))
        
        # CA3 -> CA1 (Start weak but not zero)
        c3_c1_p, c3_c1_i, c3_c1_w = hippocampus_genes.generate_connections(N_CA3, N_CA1, 0.1, WEIGHT_SC, True)
        self.d_ca3_ca1 = (cuda.to_device(c3_c1_p), cuda.to_device(c3_c1_i), cuda.to_device(c3_c1_w))
        
        # EC -> CA1
        ec_c1_p, ec_c1_i, ec_c1_w = hippocampus_genes.generate_connections(N_EC, N_CA1, 0.1, WEIGHT_EC_CA1)
        self.d_ec_ca1 = (cuda.to_device(ec_c1_p), cuda.to_device(ec_c1_i), cuda.to_device(ec_c1_w))

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
        
        self.d_ec_s = cuda.to_device(np.zeros(N_EC, dtype=np.int32))
        self.d_dg_v, self.d_dg_u, self.d_dg_i, self.d_dg_s = create(N_DG)
        self.d_ca3_v, self.d_ca3_u, self.d_ca3_i, self.d_ca3_s = create(N_CA3)
        self.d_ca1_v, self.d_ca1_u, self.d_ca1_i, self.d_ca1_s = create(N_CA1)
        
        self.d_ca3_ps = cuda.to_device(np.zeros(N_CA3, dtype=np.int32))
        
        self.p_dg = params(self.dg)
        self.p_ca3 = params(self.ca3)
        self.p_ca1 = params(self.ca1)

    def run(self):
        print("🚀 Running Novelty Simulation...")
        
        ca1_activity = []
        dopamine_levels = []
        phases = []
        
        np.random.seed(99)
        pattern_A = (np.random.rand(N_EC) < 0.1).astype(np.int32)
        pattern_B = (np.random.rand(N_EC) < 0.1).astype(np.int32)
        
        current_lr = LR_BASE
        baseline_firing = None
        
        steps = int(SIM_TIME / DT)
        
        for t_step in range(steps):
            time = t_step * DT
            
            if time < 300:
                input_pattern = pattern_A
                phase_label = 0
            elif time < 600:
                input_pattern = pattern_B
                phase_label = 1
            else:
                input_pattern = pattern_B
                phase_label = 2
            
            # --- Novelty Detection Logic ---
            if t_step % 50 == 0: 
                ca1_firing = np.sum(self.d_ca1_s.copy_to_host())
                
                # Update baseline during Phase 1
                if time < 250:
                    if baseline_firing is None:
                        baseline_firing = ca1_firing
                    else:
                        baseline_firing = 0.9 * baseline_firing + 0.1 * ca1_firing
                
                # Logic: Relative Drop check
                # If firing drops below 70% of baseline -> Novelty
                if baseline_firing is not None and ca1_firing < baseline_firing * 0.7:
                    current_lr = LR_DOPAMINE 
                else:
                    current_lr = LR_BASE
                
                if t_step % 500 == 0:
                     print(f"   t={time:.1f}ms | CA1={ca1_firing} | Base={baseline_firing:.1f} | LR={current_lr}")

            self.d_ec_s.copy_to_device(input_pattern)
            
            hippocampus_kernels.clear_buffer_kernel[self.dims['dg'][0], self.dims['dg'][1]](self.d_dg_i, N_DG)
            hippocampus_kernels.clear_buffer_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_i, N_CA3)
            hippocampus_kernels.clear_buffer_kernel[self.dims['ca1'][0], self.dims['ca1'][1]](self.d_ca1_i, N_CA1)
            
            hippocampus_kernels.synapse_transmission_kernel[self.dims['ec'][0], self.dims['ec'][1]](self.d_ec_s, self.d_dg_i, *self.d_ec_dg, N_EC)
            hippocampus_kernels.synapse_transmission_kernel[self.dims['ec'][0], self.dims['ec'][1]](self.d_ec_s, self.d_ca1_i, *self.d_ec_ca1, N_EC)
            
            hippocampus_kernels.update_neuron_kernel[self.dims['dg'][0], self.dims['dg'][1]](self.d_dg_v, self.d_dg_u, *self.p_dg, self.d_dg_i, self.d_dg_s, 0.0, DT, N_DG)
            
            hippocampus_kernels.synapse_transmission_kernel[self.dims['dg'][0], self.dims['dg'][1]](self.d_dg_s, self.d_ca3_i, *self.d_dg_ca3, N_DG)
            hippocampus_kernels.synapse_transmission_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_ps, self.d_ca3_i, *self.d_rec, N_CA3)
            
            hippocampus_kernels.update_neuron_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_v, self.d_ca3_u, *self.p_ca3, self.d_ca3_i, self.d_ca3_s, 0.0, DT, N_CA3)
            
            hippocampus_kernels.stdp_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_ps, self.d_ca3_s, *self.d_rec, current_lr, N_CA3)
            hippocampus_kernels.copy_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_s, self.d_ca3_ps, N_CA3)
            
            hippocampus_kernels.synapse_transmission_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_s, self.d_ca1_i, *self.d_ca3_ca1, N_CA3)
            
            hippocampus_kernels.update_neuron_kernel[self.dims['ca1'][0], self.dims['ca1'][1]](self.d_ca1_v, self.d_ca1_u, *self.p_ca1, self.d_ca1_i, self.d_ca1_s, CA1_BIAS, DT, N_CA1)
            
            # CA3->CA1 STDP
            hippocampus_kernels.stdp_kernel[self.dims['ca3'][0], self.dims['ca3'][1]](self.d_ca3_s, self.d_ca1_s, *self.d_ca3_ca1, current_lr, N_CA3)
            
            if t_step % 20 == 0:
                c1 = np.sum(self.d_ca1_s.copy_to_host())
                ca1_activity.append(c1)
                dopamine_levels.append(current_lr)
                phases.append(phase_label)
                
        return ca1_activity, dopamine_levels, phases

def main():
    model = NoveltyCircuit()
    act, dopa, phases = model.run()
    
    print("\n📊 Analyzing Novelty Detection...")
    
    time_axis = np.arange(len(act)) * 20 * DT
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('CA1 Activity (Recognition)', color='blue')
    ax1.plot(time_axis, act, color='blue', label='CA1 Output')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, max(act)*1.2)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Dopamine (Learning Rate)', color='orange')
    ax2.plot(time_axis, dopa, color='orange', linestyle='--', label='Dopamine')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(0, 0.2) # Scale for better visibility
    
    plt.axvspan(0, 300, color='green', alpha=0.1, label='Phase 1: A (Known)')
    plt.axvspan(300, 600, color='red', alpha=0.1, label='Phase 2: B (Novelty!)')
    plt.axvspan(600, 900, color='blue', alpha=0.1, label='Phase 3: B (Learned)')
    
    plt.title('Novelty-Induced One-Shot Learning')
    plt.show()
    
    avg_p2 = np.mean(act[len(act)//3 : 2*len(act)//3])
    avg_p3 = np.mean(act[2*len(act)//3 :])
    
    print(f"   Phase 2 (Novelty) Avg Activity: {avg_p2:.1f}")
    print(f"   Phase 3 (Re-entry) Avg Activity: {avg_p3:.1f}")
    
    # 成功判定: Phase3がPhase2より有意に高いか
    if avg_p3 > avg_p2 * 1.2:
        print("\n🏆 Success! Novelty triggered Dopamine, and Pattern B was learned instantly.")
    else:
        print("\n❌ Learning failed. Dopamine effect was too weak.")

if __name__ == "__main__":
    main()