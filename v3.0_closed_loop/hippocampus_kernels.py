import numpy as np
from numba import cuda

# ==========================================
# 🧠 Hippocampal Integrated Kernels (v2.1)
# ==========================================

@cuda.jit
def clear_buffer_kernel(buffer, size):
    tid = cuda.grid(1)
    if tid < size: buffer[tid] = 0.0

@cuda.jit
def copy_kernel(src, dst, n):
    """
    Safe device-to-device copy kernel.
    Copies src array to dst array.
    """
    tid = cuda.grid(1)
    if tid < n:
        dst[tid] = src[tid]

@cuda.jit
def update_neuron_kernel(v, u, a, b, c, d, I_input, spike_out, bias, dt, total_neurons):
    tid = cuda.grid(1)
    if tid < total_neurons:
        local_v = v[tid]
        local_u = u[tid]
        
        I = I_input[tid] + bias
        
        dv = 0.04 * local_v * local_v + 5.0 * local_v + 140.0 - local_u + I
        du = a[tid] * (b[tid] * local_v - local_u)
        
        local_v += dv * dt
        local_u += du * dt
        
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

@cuda.jit
def stdp_kernel(
    pre_spikes, post_spikes, 
    syn_pointers, syn_indices, syn_weights, 
    learning_rate, n_pre_neurons
):
    """
    STDP (LTP only) with soft saturation.
    w += lr * (w_max - w)
    """
    tid = cuda.grid(1)
    if tid < n_pre_neurons:
        if pre_spikes[tid] == 1:
            start = syn_pointers[tid]
            end = syn_pointers[tid + 1]
            for k in range(start, end):
                target = syn_indices[k]
                if post_spikes[target] == 1:
                    w = syn_weights[k]
                    # Soft update: approach 5.0 asymptotically
                    w += learning_rate * (5.0 - w)
                    if w > 5.0: w = 5.0 
                    syn_weights[k] = w

def get_dims(n):
    tp = 256
    bg = (n + tp - 1) // tp
    return bg, tp