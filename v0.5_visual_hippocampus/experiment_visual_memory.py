import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from numba import cuda
import hippocampus_genes
import hippocampus_kernel

# ==========================================
# 🧪 Experiment 5: Visual Memory Integration
# ==========================================
# Objective: Feed real image data into the DG network and 
#            visualize the formed "Memory Trace" (Synaptic Weights).

# Config
IMG_SIZE = 100          # Resize input image to 100x100
NUM_NEURONS = IMG_SIZE * IMG_SIZE # 10,000 Neurons
SIM_TIME_TRAIN = 2000   # Training duration
SIM_TIME_TEST = 500     # Testing duration
DT = 0.5
THETA_FREQ = 8.0

# Parameters
GAIN_LD = 30.0          # Strong input for the image pixels
GAIN_MD = 10.0          # Context signal
BIAS = -25.0            # Sparse coding bias
LEARNING_RATE = 0.1     # Fast learning for demo

class VisualMemoryExperiment:
    def __init__(self):
        print("🧠 Initializing Visual Memory Experiment...")
        brain = hippocampus_genes.generate_hippocampal_network(NUM_NEURONS)
        self.params = brain['params']
        self.state = brain['state']
        
        self.d_a = cuda.to_device(self.params['a'])
        self.d_b = cuda.to_device(self.params['b'])
        self.d_c = cuda.to_device(self.params['c'])
        self.d_d = cuda.to_device(self.params['d'])
        
        # State Buffers
        self.v_init = self.state['v'].copy()
        self.u_init = self.state['u'].copy()
        
        self.d_v_in = cuda.to_device(self.v_init)
        self.d_v_out = cuda.device_array_like(self.d_v_in)
        self.d_u_in = cuda.to_device(self.u_init)
        self.d_u_out = cuda.device_array_like(self.d_u_in)
        
        # Weights (Initialize to random low noise to see pattern emergence)
        # Random noise between 0.5 and 1.0
        np.random.seed(42)
        weights_host = np.random.uniform(0.5, 1.0, NUM_NEURONS).astype(np.float32)
        self.d_weights_LD = cuda.to_device(weights_host)
        
        self.d_input_MD = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_input_LD = cuda.device_array(NUM_NEURONS, dtype=np.float32)
        self.d_spikes = cuda.device_array(NUM_NEURONS, dtype=np.int32)
        
        self.blocks, self.threads = hippocampus_kernel.get_block_grid_dim(NUM_NEURONS)

    def load_image(self, filepath):
        print(f"📂 Loading image: {filepath}")
        # Load as grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("Image file not found!")
        
        # Resize to match neuron count (100x100)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize to 0.0 - 1.0 and Flatten
        # Invert colors if needed (White text on black background is preferred for SNN)
        # Assuming the input 'demo_hiragana_input.png' is black text on white or similar.
        # Let's ensure high values = high input current.
        
        img_norm = img.astype(np.float32) / 255.0
        
        # If the image is "Black Text on White", we might want to invert it 
        # so the text becomes "High Input".
        # Let's visualize it later to check. For now, assume bright pixels = input.
        
        return img_norm.flatten()

    def create_noisy_image(self, original_flat, noise_level=0.3):
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, original_flat.shape).astype(np.float32)
        noisy = original_flat + noise
        return np.clip(noisy, 0.0, 1.0)

    def run_phase(self, input_pattern, duration, is_training):
        # Reset state
        self.d_v_in.copy_to_device(self.v_init)
        self.d_u_in.copy_to_device(self.u_init)

        # Prepare inputs
        pattern_scaled = input_pattern * GAIN_LD
        self.d_input_LD.copy_to_device(pattern_scaled)
        self.d_input_MD.copy_to_device(np.ones(NUM_NEURONS, dtype=np.float32) * GAIN_MD)
        
        lr = LEARNING_RATE if is_training else 0.0
        total_spikes = 0
        
        steps = int(duration / DT)
        
        for t_step in range(steps):
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
            
            # Simple sync copy for spike counting (slow but fine for demo)
            if t_step % 100 == 0:
                spikes = self.d_spikes.copy_to_host()
                total_spikes += np.sum(spikes)
            
            self.d_v_in, self.d_v_out = self.d_v_out, self.d_v_in
            self.d_u_in, self.d_u_out = self.d_u_out, self.d_u_in
            
        return total_spikes

def main():
    exp = VisualMemoryExperiment()
    
    # 1. Load Image
    # Try to load the demo image. 
    # NOTE: Ensure 'demo_hiragana_input.png' is in the same folder!
    try:
        img_flat = exp.load_image("demo_hiragana_input.png")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please place 'demo_hiragana_input.png' in the v0.5 folder.")
        return

    # Create a noisy version for testing recall
    img_noisy = exp.create_noisy_image(img_flat)

    # 2. Pre-Test (Clean Image)
    print("\n--- Phase 1: Pre-Test (Before Learning) ---")
    spikes_pre = exp.run_phase(img_flat, SIM_TIME_TEST, is_training=False)
    print(f"   Response to Image: {spikes_pre} spikes (Baseline)")
    
    # Capture weights before learning
    weights_pre = exp.d_weights_LD.copy_to_host()

    # 3. Training
    print(f"\n--- Phase 2: Training ({SIM_TIME_TRAIN}ms) ---")
    print("   The network is watching the image and strengthening synapses...")
    _ = exp.run_phase(img_flat, SIM_TIME_TRAIN, is_training=True)
    
    # 4. Post-Test (Noisy Image - Pattern Completion Test)
    print("\n--- Phase 3: Recall Test (Noisy Image) ---")
    spikes_post_noisy = exp.run_phase(img_noisy, SIM_TIME_TEST, is_training=False)
    print(f"   Response to Noisy Image: {spikes_post_noisy} spikes")
    
    # Capture weights after learning
    weights_post = exp.d_weights_LD.copy_to_host()
    
    print("\n📊 Visualizing Memory Trace...")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # A. Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(img_flat.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title("Input Image (Visual Stimulus)")
    plt.axis('off')
    
    # B. Weights Before Learning
    plt.subplot(1, 3, 2)
    plt.imshow(weights_pre.reshape(IMG_SIZE, IMG_SIZE), cmap='hot', vmin=0, vmax=2.0)
    plt.title("Synaptic Weights (Before)")
    plt.axis('off')
    
    # C. Weights After Learning (Memory Engram)
    plt.subplot(1, 3, 3)
    plt.imshow(weights_post.reshape(IMG_SIZE, IMG_SIZE), cmap='hot', vmin=0, vmax=2.0)
    plt.title("Synaptic Weights (After)\nMemory Engram")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    if spikes_post_noisy > spikes_pre:
        print("\n🏆 Success! The network formed a visual memory.")
        print("   Look at the 'After' heatmap. Can you see the image imprinted in the weights?")

if __name__ == "__main__":
    main()