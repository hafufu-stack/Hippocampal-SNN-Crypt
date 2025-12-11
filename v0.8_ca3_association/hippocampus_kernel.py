import numpy as np
from numba import cuda
import math

# ==========================================
# 🧠 Hippocampal CA3 Kernel (Recurrent STDP)
# ==========================================

@cuda.jit
def ca3_process_kernel(
    v_in, v_out, u_in, u_out,
    a, b, c, d,
    input_External,       
    prev_spikes,          # Spikes from t-1 (Feedback)
    rec_indices,          # Connection Map
    rec_weights,          # ★ Mutable Weights (Plasticity)
    spike_out,
    total_neurons,
    connections_per_neuron,
    learning_rate,        # ★ STDP Learning Rate
    dt,
):
    """
    CUDA Kernel for CA3 with Recurrent Plasticity.
    If pre-synaptic neuron (prev_spikes) and post-synaptic neuron (me) 
    fire together, the weight increases (Hebbian Learning).
    """
    tid = cuda.grid(1)

    if tid < total_neurons:
        # 1. Load State
        local_v = v_in[tid]
        local_u = u_in[tid]

        # 2. Calculate Recurrent Input & Apply STDP
        recurrent_current = 0.0
        did_i_spike_last_step = (local_v >= 30.0) # Check if I am firing NOW (simplification)
        
        # In this kernel design, v_in is from t-1. 
        # But we check spike condition at the END of this step.
        # Ideally, Hebbian: Pre (t-1) active AND Post (t) active -> Strengthen.
        
        # Let's calculate current first
        for i in range(connections_per_neuron):
            pre_id = rec_indices[tid, i]
            
            # Check if partner spiked
            if prev_spikes[pre_id] == 1:
                weight = rec_weights[tid, i]
                recurrent_current += weight

        # 3. Total Current
        I = input_External[tid] + recurrent_current + 10.0 

        # 4. Izhikevich Update
        dv = 0.04 * local_v * local_v + 5.0 * local_v + 140.0 - local_u + I
        du = a[tid] * (b[tid] * local_v - local_u)

        local_v += dv * dt
        local_u += du * dt

        # 5. Spike & Learning
        if local_v >= 30.0:
            local_v = c[tid]
            local_u += d[tid]
            spike_out[tid] = 1
            
            # ★ Recurrent STDP Logic ★
            # I fired! Let's strengthen connections from partners who helped me (spiked previously).
            if learning_rate > 0.0:
                for i in range(connections_per_neuron):
                    pre_id = rec_indices[tid, i]
                    
                    # If Pre-synaptic neuron was active, strengthen the link
                    if prev_spikes[pre_id] == 1:
                        # Weight update
                        rec_weights[tid, i] += learning_rate
                        
                        # Cap weight (Max 5.0)
                        if rec_weights[tid, i] > 5.0:
                            rec_weights[tid, i] = 5.0
        else:
            spike_out[tid] = 0

        # Clipping
        if local_v > 30.0: local_v = 30.0
        if local_v < -90.0: local_v = -90.0

        v_out[tid] = local_v
        u_out[tid] = local_u

def get_block_grid_dim(total_elements):
    threads_per_block = 256
    blocks_per_grid = (total_elements + (threads_per_block - 1)) // threads_per_block
    return blocks_per_grid, threads_per_block