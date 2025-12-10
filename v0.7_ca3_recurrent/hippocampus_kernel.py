import numpy as np
from numba import cuda
import math

# ==========================================
# 🧠 Hippocampal CA3 Kernel (Recurrent Network)
# ==========================================

@cuda.jit
def ca3_process_kernel(
    v_in, v_out, u_in, u_out,
    a, b, c, d,
    input_External,       # Input from outside (e.g., DG or Sensory)
    prev_spikes,          # Feedback: Spikes from the previous time step
    rec_indices,          # Connectivity Map [N, Conns]
    rec_weights,          # Recurrent Weights [N, Conns]
    spike_out,
    total_neurons,
    connections_per_neuron,
    dt,
):
    """
    CUDA Kernel for CA3 Recurrent Network.
    Calculates membrane potential based on External Input AND Internal Recurrent Input.
    """
    tid = cuda.grid(1)

    if tid < total_neurons:
        # 1. Load State
        local_v = v_in[tid]
        local_u = u_in[tid]

        # 2. Calculate Recurrent Input (Feedback from network)
        recurrent_current = 0.0
        
        # Iterate over all presynaptic partners
        for i in range(connections_per_neuron):
            pre_id = rec_indices[tid, i]
            
            # Check if the partner spiked in the last step
            if prev_spikes[pre_id] == 1:
                weight = rec_weights[tid, i]
                recurrent_current += weight

        # 3. Total Current
        # I = External + Recurrent + Bias
        I = input_External[tid] + recurrent_current + 10.0 

        # 4. Izhikevich Update
        dv = 0.04 * local_v * local_v + 5.0 * local_v + 140.0 - local_u + I
        du = a[tid] * (b[tid] * local_v - local_u)

        local_v += dv * dt
        local_u += du * dt

        # 5. Spike Mechanism
        if local_v >= 30.0:
            local_v = c[tid]
            local_u += d[tid]
            spike_out[tid] = 1
        else:
            spike_out[tid] = 0

        # Stability Clipping
        if local_v > 30.0: local_v = 30.0
        if local_v < -90.0: local_v = -90.0

        v_out[tid] = local_v
        u_out[tid] = local_u

def get_block_grid_dim(total_elements):
    threads_per_block = 256
    blocks_per_grid = (total_elements + (threads_per_block - 1)) // threads_per_block
    return blocks_per_grid, threads_per_block