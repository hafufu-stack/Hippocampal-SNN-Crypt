import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from numba import cuda
from sklearn.linear_model import Ridge 
import hippocampus_genes
import hippocampus_kernel

# ==========================================
# 🧪 Experiment 6: Pattern Completion (Recall)
# ==========================================
# Objective: Can the network reconstruct a clean image from a noisy input
#            using the memory traces formed via STDP?

# Config
IMG_SIZE = 100
NUM_NEURONS = IMG_SIZE * IMG_SIZE
SIM_TIME_TRAIN = 2000   # STDP Learning Duration
SIM_TIME_READOUT = 500  # Readout Training Duration
SIM_TIME_TEST = 500     # Recall Test Duration
DT = 0.5
THETA_FREQ = 8.0

# Parameters
GAIN_LD = 30.0
GAIN_MD = 10.0
BIAS = -25.0
LEARNING_RATE = 0.1

class RecallExperiment:
    def __init__(self):
        print("🧠 Initializing Recall Experiment...")
        brain = hippocampus_genes.generate_hippocampal_network(NUM_NEURONS)
        self.params = brain['params']
        self.state = brain['state']
        
        self.d_a = cuda.to_device(self.params['a'])
        self.d_b = cuda.to_device(self.params['b'])
        self.d_c = cuda.to_device(self.params['c'])
        self.d_d = cuda.to_device(self.params['d'])
        
        self.v_init = self.state['v'].copy()
        self.u_init = self.state['u'].copy()
        
        self.d_v_in = cuda.to_device(self.v_init)
        self.d_v_out = cuda.device_array_like(self.d_v_in)
        self.d_u_in = cuda.to_device(self.u_init)
        self.d_u_out = cuda.device_array_like(self.d_u_in)
        
        # Weights (Initialize randomly)
        np.random.seed(42)
        weights_host = np.random.uniform(0.5, 1.0, NUM_NEURONS).astype(np.float32)
        self.d_weights_LD = cuda.to_device(weights_host)
        
        self.d_input_MD = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_input_LD = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_spikes = cuda.device_array(NUM_NEURONS, dtype=np.int32)
        
        self.blocks, self.threads = hippocampus_kernel.get_block_grid_dim(NUM_NEURONS)
        
        # Readout Model (Ridge Regression)
        self.readout = Ridge(alpha=1.0)

    def load_image(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError("Image not found")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0

    def create_noisy_image(self, img_flat, noise_level=0.5):
        # Add Gaussian noise to test recall robustness
        noise = np.random.normal(0, noise_level, img_flat.shape).astype(np.float32)
        noisy = img_flat + noise
        return np.clip(noisy, 0.0, 1.0)

    def run_simulation(self, input_pattern, duration, mode):
        """
        mode: 'stdp' (Learn weights), 'readout' (Train decoder), 'test' (Recall)
        """
        # Reset neuron state
        self.d_v_in.copy_to_device(self.v_init)
        self.d_u_in.copy_to_device(self.u_init)

        # Inputs
        pattern_scaled = input_pattern * GAIN_LD
        self.d_input_LD.copy_to_device(pattern_scaled)
        self.d_input_MD.copy_to_device(np.ones(NUM_NEURONS, dtype=np.float32) * GAIN_MD)
        
        # Learning Rate Control
        lr = LEARNING_RATE if mode == 'stdp' else 0.0
        
        total_spikes = np.zeros(NUM_NEURONS, dtype=np.float32)
        
        for t_step in range(int(duration / DT)):
            current_time = t_step * DT
            phase = 2 * math.pi * THETA_FREQ * (current_time / 1000.0)
            
            hippocampus_kernel.dg_process_kernel[self.blocks, self.threads](
                self.d_v_in, self.d_v_out, self.d_u_in, self.d_u_out,
                self.d_a, self.d_b, self.d_c, self.d_d,
                self.d_input_MD, self.d_input_LD,
                self.d_weights_LD, 
                self.d_spikes,
                NUM_NEURONS, t_step, 
                phase, 
                0.5, # Theta ON
                lr,  
                BIAS, 
                DT
            )
            
            # Sampling spikes for readout
            if t_step % 10 == 0: 
                spikes = self.d_spikes.copy_to_host()
                total_spikes += spikes
            
            self.d_v_in, self.d_v_out = self.d_v_out, self.d_v_in
            self.d_u_in, self.d_u_out = self.d_u_out, self.d_u_in
            
        return total_spikes

def main():
    exp = RecallExperiment()
    
    # 1. Load Data
    try:
        img_clean = exp.load_image("demo_hiragana_input.png").flatten()
    except:
        print("Please place 'demo_hiragana_input.png' in the folder.")
        return
        
    img_noisy = exp.create_noisy_image(img_clean, noise_level=0.4)

    # 2. Phase 1: STDP Learning (Form Memory)
    print("\n--- Phase 1: Learning Memory (STDP) ---")
    print("   The network is memorizing the clean image...")
    _ = exp.run_simulation(img_clean, SIM_TIME_TRAIN, mode='stdp')
    
    # 3. Phase 2: Train Readout (Decoder)
    print("\n--- Phase 2: Training Readout Decoder ---")
    print("   Teaching the decoder: 'This spike pattern means THIS image'.")
    spikes_clean = exp.run_simulation(img_clean, SIM_TIME_READOUT, mode='readout')
    
    # Train Linear Regression: Spikes -> Pixel Values
    exp.readout.fit(spikes_clean.reshape(1, -1), img_clean.reshape(1, -1))

    # 4. Phase 3: Recall Test (Noisy Input)
    print("\n--- Phase 3: Recall Test (Input: Noisy Image) ---")
    print("   Can the network clean up the noise?")
    spikes_noisy = exp.run_simulation(img_noisy, SIM_TIME_TEST, mode='test')
    
    # Decode the spikes
    img_reconstructed = exp.readout.predict(spikes_noisy.reshape(1, -1)).flatten()
    img_reconstructed = np.clip(img_reconstructed, 0.0, 1.0) 

    # Calculate Improvement
    mse_input = np.mean((img_noisy - img_clean)**2)
    mse_output = np.mean((img_reconstructed - img_clean)**2)
    improvement = (mse_input - mse_output) / mse_input * 100
    
    print(f"   Input Error (MSE): {mse_input:.4f}")
    print(f"   Output Error (MSE): {mse_output:.4f}")
    print(f"   Noise Reduction: {improvement:.1f}%")

    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_clean.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title("Original Memory")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_noisy.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title("Noisy Input")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_reconstructed.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(f"Reconstructed (Recall)\nNoise Reduced: {improvement:.1f}%")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    if mse_output < mse_input:
        print("\n🏆 Success! The network reconstructed the clean image from memory.")
    else:
        print("\n🤔 Failed to improve image quality.")

if __name__ == "__main__":
    main()