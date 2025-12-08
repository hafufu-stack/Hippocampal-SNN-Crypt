import numpy as np
from numba import cuda
import math

# ==========================================
# 🧠 Hippocampal DG Kernel (MD/LD Interaction)
# ==========================================


@cuda.jit
def dg_process_kernel(
    v_in,
    v_out,
    u_in,
    u_out,
    a,
    b,
    c,
    d,
    input_MD,
    input_LD,  # Two independent input sources
    spike_out,
    total_neurons,
    time_step,
    theta_phase,  # Current Theta phase (0.0 ~ 2pi)
    phase_lock_strength,  # Strength of phase-based gating
    dt,
):
    """
    CUDA Kernel for Dentate Gyrus (DG) Granule Cells.
    Implements the interaction between Medial Dendrite (MD) context inputs
    and Lateral Dendrite (LD) sensory inputs modulated by Theta rhythm.
    """

    tid = cuda.grid(1)

    if tid < total_neurons:
        # 1. Load State
        local_v = v_in[tid]
        local_u = u_in[tid]

        # 2. Get Inputs
        # MD: Medial Dendrite (Context/Place) -> Modulated by Theta Rhythm
        # LD: Lateral Dendrite (Sensory/Item) -> Strong driver signal

        i_md = input_MD[tid]
        i_ld = input_LD[tid]

        # --- ★ THE HYPOTHESIS IMPLEMENTATION ---
        # Hypothesis: MD input strengthens at specific Theta phases (Phase Precession/Locking).
        # When phase is optimal (cos > 0), MD opens the gate, facilitating LD signal transmission.

        # Simple Modulation:
        # modulation = 1.0 + phase_lock_strength * cos(theta_phase + neuron_preferred_phase)
        # For simplicity in v0.2, assume all neurons prefer peak theta (0).

        modulation = 1.0 + 0.5 * math.cos(theta_phase)

        # Total Current I calculation
        # MD acts as a modulation signal (Gating), LD acts as the driving signal.
        # I = I_LD + (I_MD * modulation)
        # Alternatively, non-linear interaction: I = I_LD * (1 + I_MD * modulation)

        # Implementation: "Addition + Modulation"
        I = i_ld + (i_md * modulation)

        # 3. Izhikevich Update (Numerical Integration)
        dv = 0.04 * local_v * local_v + 5.0 * local_v + 140.0 - local_u + I
        du = a[tid] * (b[tid] * local_v - local_u)

        local_v += dv * dt
        local_u += du * dt

        # 4. Spike Mechanism
        if local_v >= 30.0:
            local_v = c[tid]
            local_u += d[tid]
            spike_out[tid] = 1  # Spike event
        else:
            spike_out[tid] = 0

        # Stability Clipping
        if local_v > 30.0:
            local_v = 30.0
        if local_v < -90.0:
            local_v = -90.0

        v_out[tid] = local_v
        u_out[tid] = local_u


def get_block_grid_dim(total_elements):
    threads_per_block = 256
    blocks_per_grid = (total_elements + (threads_per_block - 1)) // threads_per_block
    return blocks_per_grid, threads_per_block
