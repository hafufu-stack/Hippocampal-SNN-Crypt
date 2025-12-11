import numpy as np
from numba import cuda

# ==========================================
# 🧠 Hippocampal Integrated Kernels
# ==========================================

@cuda.jit
def izhikevich_update_kernel(v, u, a, b, c, d, I_input, spike_out, dt, total_neurons):
    """
    Standard Izhikevich Update Logic.
    Shared by both DG and CA3.
    """
    tid = cuda.grid(1)
    if tid < total_neurons:
        local_v = v[tid]
        local_u = u[tid]
        
        # Update
        dv = 0.04 * local_v * local_v + 5.0 * local_v + 140.0 - local_u + I_input[tid]
        du = a[tid] * (b[tid] * local_v - local_u)
        
        local_v += dv * dt
        local_u += du * dt
        
        # Spike
        if local_v >= 30.0:
            local_v = c[tid]
            local_u += d[tid]
            spike_out[tid] = 1
        else:
            spike_out[tid] = 0
            
        # Clip
        if local_v > 30.0: local_v = 30.0
        if local_v < -90.0: local_v = -90.0
        
        v[tid] = local_v
        u[tid] = local_u

@cuda.jit
def synapse_transmission_kernel(
    pre_spikes,         # Spikes from Pre-population (e.g., DG)
    post_current_buffer,# Input buffer of Post-population (e.g., CA3)
    syn_pointers,       # CSR format pointers
    syn_indices,        # CSR format indices
    syn_weights,        # Weights
    n_pre_neurons
):
    """
    Propagates spikes from Pre to Post.
    Adds current to the Post-neuron's input buffer.
    """
    tid = cuda.grid(1)
    
    if tid < n_pre_neurons:
        # If I (Pre-neuron) spiked...
        if pre_spikes[tid] == 1:
            # Send signal to all my targets
            start_idx = syn_pointers[tid]
            end_idx = syn_pointers[tid + 1]
            
            for k in range(start_idx, end_idx):
                target_id = syn_indices[k]
                weight = syn_weights[k]
                
                # Atomic Add is necessary because multiple Pre-neurons 
                # might target the same Post-neuron simultaneously.
                cuda.atomic.add(post_current_buffer, target_id, weight)

@cuda.jit
def clear_buffer_kernel(buffer, size):
    tid = cuda.grid(1)
    if tid < size:
        buffer[tid] = 0.0

def get_dims(n):
    tp = 256
    bg = (n + tp - 1) // tp
    return bg, tp