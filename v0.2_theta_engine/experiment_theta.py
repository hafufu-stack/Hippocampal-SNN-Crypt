import numpy as np
import matplotlib.pyplot as plt
import math
from numba import cuda
import hippocampus_genes
import hippocampus_kernel

# ==========================================
# 🧪 Experiment 1: Theta Phase Modulation
# ==========================================
# Hypothesis: LD input (Item) fires strongly only when
#             MD input (Context) phase is optimal.

# Configuration
NUM_NEURONS = 10000  # 10,000 Neurons
SIM_TIME = 1000  # 1000ms (1 second)
DT = 0.5  # 0.5ms time step
THETA_FREQ = 8.0  # 8Hz (Typical hippocampal theta frequency)

# Input Settings
GAIN_LD = 15.0  # LD Input Gain (Image/Item)
GAIN_MD = 10.0  # MD Input Gain (Place/Context)
BIAS = 0.0


class HippocampalThetaExperiment:
    def __init__(self):
        print("🧠 Initializing Hippocampal Theta Experiment...")

        # 1. Generate Neurons
        # Initialize as Granule Cells (GC)
        brain = hippocampus_genes.generate_hippocampal_network(NUM_NEURONS)
        self.params = brain["params"]
        self.state = brain["state"]

        # 2. Allocate GPU Memory
        self.d_a = cuda.to_device(self.params["a"])
        self.d_b = cuda.to_device(self.params["b"])
        self.d_c = cuda.to_device(self.params["c"])
        self.d_d = cuda.to_device(self.params["d"])

        self.d_v_in = cuda.to_device(self.state["v"])
        self.d_v_out = cuda.device_array_like(self.d_v_in)
        self.d_u_in = cuda.to_device(self.state["u"])
        self.d_u_out = cuda.device_array_like(self.d_u_in)

        # Inputs (MD & LD)
        self.d_input_MD = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_input_LD = cuda.device_array(NUM_NEURONS, dtype=np.float32)

        # Output Spikes (for 1 step)
        self.d_spikes = cuda.device_array(NUM_NEURONS, dtype=np.int32)

        # Kernel Config
        self.blocks, self.threads = hippocampus_kernel.get_block_grid_dim(NUM_NEURONS)

    def run(self):
        print(f"🚀 Running Simulation ({SIM_TIME}ms @ {THETA_FREQ}Hz Theta)...")

        # --- Prepare Inputs ---
        # LD: Random Pattern (Imagine a static image) - Constant input
        # MD: Constant Baseline (Modulated by Theta inside kernel)
        np.random.seed(42)
        ld_host = np.random.rand(NUM_NEURONS).astype(np.float32) * GAIN_LD
        md_host = np.ones(NUM_NEURONS, dtype=np.float32) * GAIN_MD

        self.d_input_LD.copy_to_device(ld_host)
        self.d_input_MD.copy_to_device(md_host)

        # Data Recording
        spike_history_t = []
        spike_history_id = []
        theta_phases = []
        total_spikes_per_step = []

        # --- Simulation Loop ---
        for t_step in range(int(SIM_TIME / DT)):
            current_time = t_step * DT

            # Calculate Theta Phase (0 ~ 2pi)
            # Phase = 2 * pi * f * t
            phase = 2 * math.pi * THETA_FREQ * (current_time / 1000.0)

            # Kernel Call
            hippocampus_kernel.dg_process_kernel[self.blocks, self.threads](
                self.d_v_in,
                self.d_v_out,
                self.d_u_in,
                self.d_u_out,
                self.d_a,
                self.d_b,
                self.d_c,
                self.d_d,
                self.d_input_MD,
                self.d_input_LD,
                self.d_spikes,
                NUM_NEURONS,
                t_step,
                phase,  # Pass current theta phase
                0.5,  # Phase lock strength
                DT,
            )

            # Record Data (To Host)
            spikes = self.d_spikes.copy_to_host()
            n_spikes = np.sum(spikes)

            if n_spikes > 0:
                ids = np.where(spikes > 0)[0]
                # Downsample for plotting if too many spikes
                if len(ids) > 100:
                    ids = np.random.choice(ids, 100, replace=False)

                spike_history_t.extend([current_time] * len(ids))
                spike_history_id.extend(ids)

            total_spikes_per_step.append(n_spikes)
            theta_phases.append(math.cos(phase))  # Store cosine for visualization

            # Swap Buffers
            self.d_v_in, self.d_v_out = self.d_v_out, self.d_v_in
            self.d_u_in, self.d_u_out = self.d_u_out, self.d_u_in

            if t_step % 100 == 0:
                print(f"   t={current_time}ms...", end="\r")

        return spike_history_t, spike_history_id, total_spikes_per_step, theta_phases


def main():
    exp = HippocampalThetaExperiment()
    spikes_t, spikes_id, spike_counts, theta_wave = exp.run()

    print("\n📊 Analyzing Results...")

    # Time axis
    times = np.arange(0, SIM_TIME, DT)

    # Plotting
    plt.figure(figsize=(12, 10))

    # 1. Theta Wave (The input rhythm)
    plt.subplot(3, 1, 1)
    plt.plot(times, theta_wave, color="green", label="MD Theta Phase (cos)")
    plt.title("MD Input Modulation (Theta Rhythm 8Hz)")
    plt.ylabel("Modulation")
    plt.xlim(0, 500)  # Show first 500ms
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # 2. Raster Plot (Individual Spikes)
    plt.subplot(3, 1, 2)
    plt.scatter(spikes_t, spikes_id, s=1, color="black", alpha=0.5)
    plt.title("Neuron Spikes (LD Input is CONSTANT)")
    plt.ylabel("Neuron ID")
    plt.xlim(0, 500)
    plt.grid(True, alpha=0.3)

    # 3. Population Firing Rate (PSTH)
    plt.subplot(3, 1, 3)
    # Smooth the spike counts
    window = 10
    smoothed_rate = np.convolve(spike_counts, np.ones(window) / window, mode="same")

    plt.plot(times, smoothed_rate, color="blue", label="Population Firing Rate")
    plt.title("Result: Firing Rate follows Theta Phase!")
    plt.xlabel("Time (ms)")
    plt.ylabel("Spikes / Step")
    plt.xlim(0, 500)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("✅ Experiment Done.")
    print(
        "   Observation: If the Blue line (Firing Rate) matches the Green line (Theta),"
    )
    print("   your hypothesis 'Phase controls information flow' is PROVEN.")


if __name__ == "__main__":
    main()
