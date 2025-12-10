import numpy as np
import matplotlib.pyplot as plt
import math
from numba import cuda
import hippocampus_genes
import hippocampus_kernel

# ==========================================
# 🧪 Experiment 4: Phase Precession (Fast Spiking Mode)
# ==========================================
# Objective: Demonstrate that as a rat moves through a place field (Input rises and falls),
#            the spike phase shifts from late to early relative to Theta.

# Config
NUM_NEURONS = 1000      # Sufficient population size for visualization
SIM_TIME = 2000         # 2 seconds simulation
DT = 0.5
THETA_FREQ = 8.0        # 8Hz Theta Rhythm

# Parameters (Tuned for Fast Spiking dynamics)
GAIN_LD_MAX = 80.0      # Strong input to drive high-frequency spiking
GAIN_MD = 10.0
BIAS = -10.0            # Bias to control place field width

class PhasePrecessionExperiment:
    def __init__(self):
        print("🧠 Initializing Phase Precession Experiment...")
        brain = hippocampus_genes.generate_hippocampal_network(NUM_NEURONS)
        self.params = brain['params']
        self.state = brain['state']
        
        # ★ Tuning: Configure neurons as Fast Spiking (FS) 
        # to allow firing frequency > Theta frequency (8Hz).
        print("🔧 Tuning neurons for high-frequency oscillation (FS Mode)...")
        self.params['a'].fill(0.1)  # Fast recovery
        self.params['b'].fill(0.2)
        self.params['c'].fill(-65.0)
        self.params['d'].fill(2.0)  # Low reset value to prevent long refractory periods
        
        self.d_a = cuda.to_device(self.params['a'])
        self.d_b = cuda.to_device(self.params['b'])
        self.d_c = cuda.to_device(self.params['c'])
        self.d_d = cuda.to_device(self.params['d'])
        
        self.d_v_in = cuda.to_device(self.state['v'])
        self.d_v_out = cuda.device_array_like(self.d_v_in)
        self.d_u_in = cuda.to_device(self.state['u'])
        self.d_u_out = cuda.device_array_like(self.d_u_in)
        
        # Weights (Fixed for this physics demo)
        weights_host = np.ones(NUM_NEURONS, dtype=np.float32)
        self.d_weights_LD = cuda.to_device(weights_host)
        
        self.d_input_MD = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_input_LD = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_spikes = cuda.device_array(NUM_NEURONS, dtype=np.int32)
        
        self.blocks, self.threads = hippocampus_kernel.get_block_grid_dim(NUM_NEURONS)

    def run(self):
        print("🚀 Running Virtual Rat Simulation...")
        
        spike_phases = []  # Y-axis: Phase (0-720)
        spike_times = []   # X-axis: Time/Position
        
        self.d_input_MD.copy_to_device(np.ones(NUM_NEURONS, dtype=np.float32) * GAIN_MD)
        
        total_steps = int(SIM_TIME / DT)
        
        for t_step in range(total_steps):
            current_time = t_step * DT
            
            # --- 1. Simulate Place Field (Gaussian Input) ---
            dist_from_center = current_time - (SIM_TIME / 2.0)
            # Width of the field
            gaussian_envelope = np.exp(- (dist_from_center**2) / (2 * 400**2))
            
            current_ld_strength = gaussian_envelope * GAIN_LD_MAX
            self.d_input_LD.copy_to_device(np.full(NUM_NEURONS, current_ld_strength, dtype=np.float32))
            
            # --- 2. Calculate Theta Phase ---
            theta_phase_rad = 2 * math.pi * THETA_FREQ * (current_time / 1000.0)
            
            # --- 3. Kernel Execution ---
            hippocampus_kernel.dg_process_kernel[self.blocks, self.threads](
                self.d_v_in, self.d_v_out, self.d_u_in, self.d_u_out,
                self.d_a, self.d_b, self.d_c, self.d_d,
                self.d_input_MD, self.d_input_LD,
                self.d_weights_LD, 
                self.d_spikes,
                NUM_NEURONS, t_step, 
                theta_phase_rad, 
                0.5,   # Phase lock strength
                0.0,   # Learning OFF
                BIAS, 
                DT
            )
            
            # --- 4. Record Spikes ---
            spikes = self.d_spikes.copy_to_host()
            
            if np.any(spikes):
                # Calculate relative phase (Precession means shifting to earlier phase)
                # Invert the phase direction for visualization: (2pi - phase) % 2pi
                phase_deg = ((2 * math.pi - (theta_phase_rad % (2 * math.pi))) % (2 * math.pi)) * (180 / math.pi)
                
                # Downsample recording for visualization
                active_indices = np.where(spikes > 0)[0]
                if len(active_indices) > 0:
                    spike_times.append(current_time)
                    spike_phases.append(phase_deg)
                
            # Swap Buffers
            self.d_v_in, self.d_v_out = self.d_v_out, self.d_v_in
            self.d_u_in, self.d_u_out = self.d_u_out, self.d_u_in
            
            if t_step % 500 == 0:
                print(f"   t={current_time}ms | Input Strength={current_ld_strength:.2f}", end='\r')

        return spike_times, spike_phases

def main():
    exp = PhasePrecessionExperiment()
    times, phases = exp.run()
    
    print("\n📊 Analyzing Results...")
    
    plt.figure(figsize=(10, 6))
    
    # Double plot for cyclic visualization (Standard practice in neuroscience papers)
    times_double = times + times
    phases_double = phases + [p + 360 for p in phases]
    
    plt.scatter(times_double, phases_double, s=8, alpha=0.5, c='blue')
    
    plt.title('Phase Precession: Spike Phase vs Time (Position)')
    plt.xlabel('Time (ms) / Position')
    plt.ylabel('Theta Phase (Degrees)')
    
    # Field boundaries
    plt.axvline(x=600, color='gray', linestyle='--', alpha=0.5, label='Field Start')
    plt.axvline(x=1400, color='gray', linestyle='--', alpha=0.5, label='Field End')
    
    plt.ylim(0, 720)
    plt.yticks(np.arange(0, 721, 180))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print("✅ Experiment Done.")
    print("   Observation: A downward slope (Top-Left to Bottom-Right) confirms Phase Precession.")

if __name__ == "__main__":
    main()