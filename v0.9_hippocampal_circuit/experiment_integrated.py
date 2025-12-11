import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import hippocampus_genes
import hippocampus_kernels

# ==========================================
# 🧪 Experiment 9: Integrated Circuit (DG -> CA3)
# ==========================================
# Objective: Verify signal transmission from DG to CA3 via "Mossy Fibers".
#            Show that DG acts as a sparse driver for CA3.

# Config
N_DG = 10000            # DG Granule Cells
N_CA3 = 2000            # CA3 Pyramidal Cells
SIM_TIME = 500
DT = 0.5

# Connectivity Parameters
# Mossy Fibers: Very sparse connection, but very strong ("Detonator")
PROB_DG_CA3 = 0.005     # Each DG connects to ~0.5% of CA3 (very sparse)
WEIGHT_MF = 30.0        # Strong weight!

# Input Parameters
GAIN_INPUT = 20.0       # Input to DG

class IntegratedCircuit:
    def __init__(self):
        print("🧠 Initializing Integrated Circuit (DG-CA3)...")
        
        # 1. Initialize Populations
        self.dg = hippocampus_genes.generate_network_params(N_DG, "GC")
        self.ca3 = hippocampus_genes.generate_network_params(N_CA3, "CA3")
        
        # 2. Build Connections (Mossy Fibers: DG -> CA3)
        print("🔗 Wiring Mossy Fibers (DG -> CA3)...")
        ptrs, idxs, wgts = hippocampus_genes.generate_connections(N_DG, N_CA3, PROB_DG_CA3, WEIGHT_MF)
        self.mf_pointers = cuda.to_device(ptrs)
        self.mf_indices = cuda.to_device(idxs)
        self.mf_weights = cuda.to_device(wgts)
        
        # 3. GPU Allocations
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
        self.d_ca3_input = cuda.device_array(N_CA3, dtype=np.float32) # Accumulated input from DG
        
        # Kernel Dims
        self.dim_dg = hippocampus_kernels.get_dims(N_DG)
        self.dim_ca3 = hippocampus_kernels.get_dims(N_CA3)
        self.dim_syn = hippocampus_kernels.get_dims(N_DG) # Pre-synaptic parallelization

    def run(self):
        print("🚀 Running Simulation...")
        
        dg_raster_t, dg_raster_id = [], []
        ca3_raster_t, ca3_raster_id = [], []
        
        steps = int(SIM_TIME / DT)
        
        for t_step in range(steps):
            time = t_step * DT
            
            # --- 1. Reset CA3 Input Buffer ---
            hippocampus_kernels.clear_buffer_kernel[self.dim_ca3[0], self.dim_ca3[1]](
                self.d_ca3_input, N_CA3
            )
            
            # --- 2. DG Dynamics ---
            # Input to DG (First 100ms only, simple pulse)
            if time < 100:
                # Random input to DG
                if t_step % 10 == 0: # Change input noise occasionally
                    inp = np.random.rand(N_DG).astype(np.float32) * GAIN_INPUT
                    self.d_dg_input.copy_to_device(inp)
            else:
                # Silence
                self.d_dg_input.copy_to_device(np.zeros(N_DG, dtype=np.float32))
            
            hippocampus_kernels.izhikevich_update_kernel[self.dim_dg[0], self.dim_dg[1]](
                self.d_dg_v, self.d_dg_u, self.d_dg_a, self.d_dg_b, self.d_dg_c, self.d_dg_d,
                self.d_dg_input, self.d_dg_spikes, DT, N_DG
            )
            
            # --- 3. Transmission (DG -> CA3) ---
            # Mossy Fiber: Propagate spikes from DG to CA3 input buffer
            hippocampus_kernels.synapse_transmission_kernel[self.dim_syn[0], self.dim_syn[1]](
                self.d_dg_spikes,
                self.d_ca3_input, # Destination buffer
                self.mf_pointers, self.mf_indices, self.mf_weights,
                N_DG
            )
            
            # --- 4. CA3 Dynamics ---
            # CA3 receives input ONLY from DG (via d_ca3_input buffer)
            hippocampus_kernels.izhikevich_update_kernel[self.dim_ca3[0], self.dim_ca3[1]](
                self.d_ca3_v, self.d_ca3_u, self.d_ca3_a, self.d_ca3_b, self.d_ca3_c, self.d_ca3_d,
                self.d_ca3_input, self.d_ca3_spikes, DT, N_CA3
            )
            
            # --- 5. Recording ---
            # Copy spikes to host (slow, but fine for demo)
            dg_s = self.d_dg_spikes.copy_to_host()
            ca3_s = self.d_ca3_spikes.copy_to_host()
            
            dg_ids = np.where(dg_s > 0)[0]
            if len(dg_ids) > 0:
                dg_raster_t.extend([time] * len(dg_ids))
                dg_raster_id.extend(dg_ids)
                
            ca3_ids = np.where(ca3_s > 0)[0]
            if len(ca3_ids) > 0:
                ca3_raster_t.extend([time] * len(ca3_ids))
                ca3_raster_id.extend(ca3_ids)
                
            if t_step % 100 == 0:
                print(f"   t={time}ms...", end='\r')

        return dg_raster_t, dg_raster_id, ca3_raster_t, ca3_raster_id

def main():
    model = IntegratedCircuit()
    dg_t, dg_id, ca3_t, ca3_id = model.run()
    
    print("\n📊 Analyzing Results...")
    
    plt.figure(figsize=(12, 10))
    
    # DG Raster
    plt.subplot(2, 1, 1)
    plt.scatter(dg_t, dg_id, s=1, color='blue', alpha=0.5)
    plt.axvline(x=100, color='red', linestyle='--')
    plt.title('DG Activity (Driven by External Input)')
    plt.ylabel('DG Neuron ID')
    plt.xlim(0, SIM_TIME)
    
    # CA3 Raster
    plt.subplot(2, 1, 2)
    plt.scatter(ca3_t, ca3_id, s=1, color='green', alpha=0.5)
    plt.axvline(x=100, color='red', linestyle='--')
    plt.title('CA3 Activity (Driven by DG Mossy Fibers)')
    plt.ylabel('CA3 Neuron ID')
    plt.xlabel('Time (ms)')
    plt.xlim(0, SIM_TIME)
    
    plt.tight_layout()
    plt.show()
    
    # Check if CA3 fired
    if len(ca3_t) > 100:
        print("\n🏆 Success! Signal transmitted from DG to CA3.")
        print("   CA3 fired solely due to Mossy Fiber inputs.")
    else:
        print("\n❌ Signal lost. CA3 did not fire. Check connection weights.")

if __name__ == "__main__":
    main()