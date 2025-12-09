import numpy as np
from numba import cuda
import math

# ==========================================
# 🧠 Hippocampal DG Kernel v0.3 (STDP Learning)
# ==========================================

@cuda.jit
def dg_process_kernel(
    v_in, v_out, u_in, u_out,
    a, b, c, d,
    input_MD, input_LD,
    weights_LD,           # Synaptic Weights (Plastic)
    spike_out,
    total_neurons,
    time_step,
    theta_phase,
    phase_lock_strength,
    learning_rate,
    bias,                 # Bias passed separately for correct STDP calculation
    dt,
):
    """
    CUDA Kernel for DG Granule Cells with STDP Learning.
    
    Features:
    - Theta Phase Modulation (Gating)
    - STDP (Spike-Timing-Dependent Plasticity): 
      Weights increase if the neuron fires in response to strong input (LTP).
    """

    tid = cuda.grid(1)

    if tid < total_neurons:
        # 1. Load State
        local_v = v_in[tid]
        local_u = u_in[tid]
        local_w = weights_LD[tid]

        # 2. Get Inputs
        i_md = input_MD[tid]
        i_ld = input_LD[tid] # Pure positive signal from host

        # --- Theta Modulation ---
        modulation = 1.0 + phase_lock_strength * math.cos(theta_phase)

        # --- Calculate Current ---
        # I = (Signal * Weight) + (Context * Mod) + Bias
        current_LD = i_ld * local_w
        I = current_LD + (i_md * modulation) + bias

        # 3. Izhikevich Update
        dv = 0.04 * local_v * local_v + 5.0 * local_v + 140.0 - local_u + I
        du = a[tid] * (b[tid] * local_v - local_u)

        local_v += dv * dt
        local_u += du * dt

        # 4. Spike & Learning Mechanism
        if local_v >= 30.0:
            local_v = c[tid]
            local_u += d[tid]
            spike_out[tid] = 1
            
            # --- STDP (LTP: Long-Term Potentiation) ---
            # If input contributed to the spike, strengthen the weight.
            if i_ld > 0.1: 
                local_w += learning_rate * i_ld
                
                # Weight Capping (Max 2.0)
                if local_w > 2.0: local_w = 2.0
                
        else:
            spike_out[tid] = 0
            # LTD (Long-Term Depression) is currently disabled to prioritize learning speed.
            # To enable forgetting: uncomment the decay logic below.
            # if learning_rate > 0.0:
            #    local_w *= 0.99995

        # Stability Clipping
        if local_v > 30.0: local_v = 30.0
        if local_v < -90.0: local_v = -90.0

        # 5. Save State
        v_out[tid] = local_v
        u_out[tid] = local_u
        weights_LD[tid] = local_w # Write back updated weight

def get_block_grid_dim(total_elements):
    threads_per_block = 256
    blocks_per_grid = (total_elements + (threads_per_block - 1)) // threads_per_block
    return blocks_per_grid, threads_per_block