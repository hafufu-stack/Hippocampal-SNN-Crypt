import numpy as np
from numba import cuda

# ==========================================
# 🧠 Hippocampal Integrated Kernels (v1.0)
# ==========================================

@cuda.jit
def clear_buffer_kernel(buffer, size):
    tid = cuda.grid(1)
    if tid < size:
        buffer[tid] = 0.0

# --- 1. DG Dynamics ---
@cuda.jit
def dg_update_kernel(v, u, a, b, c, d, I_input, spike_out, dt, total_neurons):
    tid = cuda.grid(1)
    if tid < total_neurons:
        local_v = v[tid]
        local_u = u[tid]
        
        dv = 0.04 * local_v * local_v + 5.0 * local_v + 140.0 - local_u + I_input[tid]
        du = a[tid] * (b[tid] * local_v - local_u)
        
        local_v += dv * dt
        local_u += du * dt
        
        if local_v >= 30.0:
            local_v = c[tid]
            local_u += d[tid]
            spike_out[tid] = 1
        else:
            spike_out[tid] = 0
            
        if local_v > 30.0: local_v = 30.0
        if local_v < -90.0: local_v = -90.0
        
        v[tid] = local_v
        u[tid] = local_u

# --- 2. Transmission ---
@cuda.jit
def synapse_transmission_kernel(
    pre_spikes, post_current_buffer, 
    syn_pointers, syn_indices, syn_weights, n_pre_neurons
):
    tid = cuda.grid(1)
    if tid < n_pre_neurons:
        if pre_spikes[tid] == 1:
            start = syn_pointers[tid]
            end = syn_pointers[tid + 1]
            for k in range(start, end):
                target = syn_indices[k]
                w = syn_weights[k]
                cuda.atomic.add(post_current_buffer, target, w)

# --- 3. CA3 Dynamics (Recurrent + STDP) ---
@cuda.jit
def ca3_update_stdp_kernel(
    v, u, a, b, c, d,
    input_from_dg,      # Input buffer (Includes Recurrent)
    prev_spikes,        
    rec_pointers, rec_indices, rec_weights, 
    spike_out,
    learning_rate,      
    dt, total_neurons
):
    """
    Handles CA3 dynamics.
    User Tuning: No bias/brake applied to allow reverberation.
    """
    tid = cuda.grid(1)
    if tid < total_neurons:
        local_v = v[tid]
        local_u = u[tid]
        
        # Total Current
        # Tuning: Direct input, no inhibitory bias
        I = input_from_dg[tid] 
        
        # Izhikevich Update
        dv = 0.04 * local_v * local_v + 5.0 * local_v + 140.0 - local_u + I
        du = a[tid] * (b[tid] * local_v - local_u)
        local_v += dv * dt
        local_u += du * dt
        
        # Spike Check
        spiked = 0
        if local_v >= 30.0:
            local_v = c[tid]
            local_u += d[tid]
            spiked = 1
        
        if local_v > 30.0: local_v = 30.0
        if local_v < -90.0: local_v = -90.0
        
        v[tid] = local_v
        u[tid] = local_u
        spike_out[tid] = spiked

@cuda.jit
def ca3_stdp_kernel(
    pre_spikes, post_spikes,
    syn_pointers, syn_indices, syn_weights,
    learning_rate, n_pre_neurons
):
    """
    Updates weights based on activity (Hebbian Learning).
    """
    tid = cuda.grid(1)
    if tid < n_pre_neurons:
        if pre_spikes[tid] == 1:
            start = syn_pointers[tid]
            end = syn_pointers[tid + 1]
            
            for k in range(start, end):
                target_id = syn_indices[k]
                if post_spikes[target_id] == 1:
                    # LTP
                    w = syn_weights[k]
                    w += learning_rate
                    if w > 5.0: w = 5.0 
                    syn_weights[k] = w

def get_dims(n):
    tp = 256
    bg = (n + tp - 1) // tp
    return bg, tp