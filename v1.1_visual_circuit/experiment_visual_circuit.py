import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import cuda
from sklearn.linear_model import Ridge
import hippocampus_genes
import hippocampus_kernels

# ==========================================
# 🧪 Experiment 11: Visual Circuit (v1.1)
# ==========================================
# Objective:
# 1. Input Image -> DG -> CA3.
# 2. CA3 learns and holds the pattern (Reverberation).
# 3. Reconstruct the image from CA3 activity AFTER input is gone.

# Config
IMG_SIZE = 100
N_DG = IMG_SIZE * IMG_SIZE # 10000 neurons (1:1 with image)
N_CA3 = 5000               # Compressed representation
SIM_TIME = 600
DT = 0.5

# Connectivity
PROB_DG_CA3 = 0.005     
PROB_CA3_CA3 = 0.05     
WEIGHT_MF = 50.0        
LEARNING_RATE = 0.2     

class VisualCircuit:
    def __init__(self):
        print("🧠 Initializing Visual Circuit...")
        
        self.dg = hippocampus_genes.generate_network_params(N_DG, "GC")
        self.ca3 = hippocampus_genes.generate_network_params(N_CA3, "CA3")
        
        # Connections
        print("🔗 Wiring DG -> CA3...")
        mf_ptr, mf_idx, mf_w = hippocampus_genes.generate_connections(N_DG, N_CA3, PROB_DG_CA3, WEIGHT_MF, False)
        self.d_mf_ptr = cuda.to_device(mf_ptr)
        self.d_mf_idx = cuda.to_device(mf_idx)
        self.d_mf_w = cuda.to_device(mf_w)
        
        print("🔗 Wiring CA3 -> CA3...")
        rec_ptr, rec_idx, rec_w = hippocampus_genes.generate_connections(N_CA3, N_CA3, PROB_CA3_CA3, 0.0, True)
        self.d_rec_ptr = cuda.to_device(rec_ptr)
        self.d_rec_idx = cuda.to_device(rec_idx)
        self.d_rec_w = cuda.to_device(rec_w)
        
        self._alloc_gpu()
        self.dim_dg = hippocampus_kernels.get_dims(N_DG)
        self.dim_ca3 = hippocampus_kernels.get_dims(N_CA3)
        
        # Readout (Decoder): CA3 Activity -> Image Pixel
        self.decoder = Ridge(alpha=1.0)

    def _alloc_gpu(self):
        # DG (Allocations omitted for brevity, same as v1.0)
        self.d_dg_v = cuda.to_device(self.dg['state']['v'])
        self.d_dg_u = cuda.to_device(self.dg['state']['u'])
        self.d_dg_a = cuda.to_device(self.dg['params']['a'])
        self.d_dg_b = cuda.to_device(self.dg['params']['b'])
        self.d_dg_c = cuda.to_device(self.dg['params']['c'])
        self.d_dg_d = cuda.to_device(self.dg['params']['d'])
        self.d_dg_spikes = cuda.device_array(N_DG, dtype=np.int32)
        self.d_dg_input = cuda.device_array(N_DG, dtype=np.float32)
        
        # CA3
        self.d_ca3_v = cuda.to_device(self.ca3['state']['v'])
        self.d_ca3_u = cuda.to_device(self.ca3['state']['u'])
        self.d_ca3_a = cuda.to_device(self.ca3['params']['a'])
        self.d_ca3_b = cuda.to_device(self.ca3['params']['b'])
        self.d_ca3_c = cuda.to_device(self.ca3['params']['c'])
        self.d_ca3_d = cuda.to_device(self.ca3['params']['d'])
        self.d_ca3_spikes = cuda.device_array(N_CA3, dtype=np.int32)
        self.d_ca3_prev_spikes = cuda.device_array(N_CA3, dtype=np.int32)
        self.d_ca3_current_buffer = cuda.device_array(N_CA3, dtype=np.float32)

    def load_image(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError("Image not found")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Normalize: 0.0 - 30.0 (Strong Input)
        return (img.astype(np.float32) / 255.0) * 30.0

    def run(self, input_image):
        print("🚀 Running Simulation...")
        
        # Data collectors
        ca3_activity_ON = []  # Activity during input (for training decoder)
        ca3_activity_OFF = [] # Activity after input (for testing recall)
        
        steps = int(SIM_TIME / DT)
        
        for t_step in range(steps):
            time = t_step * DT
            
            # 1. Clear Buffer
            hippocampus_kernels.clear_buffer_kernel[self.dim_ca3[0], self.dim_ca3[1]](self.d_ca3_current_buffer, N_CA3)
            
            # 2. DG Update (Input ON 0-200ms)
            if time < 200:
                self.d_dg_input.copy_to_device(input_image.flatten())
            else:
                self.d_dg_input.copy_to_device(np.zeros(N_DG, dtype=np.float32))
                
            hippocampus_kernels.dg_update_kernel[self.dim_dg[0], self.dim_dg[1]](
                self.d_dg_v, self.d_dg_u, self.d_dg_a, self.d_dg_b, self.d_dg_c, self.d_dg_d,
                self.d_dg_input, self.d_dg_spikes, DT, N_DG
            )
            
            # 3. DG -> CA3
            hippocampus_kernels.synapse_transmission_kernel[self.dim_dg[0], self.dim_dg[1]](
                self.d_dg_spikes, self.d_ca3_current_buffer,
                self.d_mf_ptr, self.d_mf_idx, self.d_mf_w, N_DG
            )
            
            # 4. CA3 Recurrent
            hippocampus_kernels.synapse_transmission_kernel[self.dim_ca3[0], self.dim_ca3[1]](
                self.d_ca3_prev_spikes, self.d_ca3_current_buffer,
                self.d_rec_ptr, self.d_rec_idx, self.d_rec_w, N_CA3
            )
            
            # 5. CA3 Update & STDP
            hippocampus_kernels.ca3_update_stdp_kernel[self.dim_ca3[0], self.dim_ca3[1]](
                self.d_ca3_v, self.d_ca3_u, self.d_ca3_a, self.d_ca3_b, self.d_ca3_c, self.d_ca3_d,
                self.d_ca3_current_buffer,
                self.d_ca3_prev_spikes,
                self.d_rec_ptr, self.d_rec_idx, self.d_rec_w,
                self.d_ca3_spikes,
                LEARNING_RATE, DT, N_CA3
            )
            
            hippocampus_kernels.ca3_stdp_kernel[self.dim_ca3[0], self.dim_ca3[1]](
                self.d_ca3_prev_spikes, self.d_ca3_spikes,
                self.d_rec_ptr, self.d_rec_idx, self.d_rec_w,
                LEARNING_RATE, N_CA3
            )
            
            self.d_ca3_prev_spikes.copy_to_device(self.d_ca3_spikes)
            
            # 6. Collect Data
            if t_step % 10 == 0:
                spikes = self.d_ca3_spikes.copy_to_host()
                if time < 200:
                    ca3_activity_ON.append(spikes)
                elif time > 300: # Wait for steady state
                    ca3_activity_OFF.append(spikes)
            
            if t_step % 100 == 0:
                print(f"   t={time}ms...", end='\r')

        return np.array(ca3_activity_ON), np.array(ca3_activity_OFF)

def main():
    model = VisualCircuit()
    
    # Load Image
    try:
        img_original = model.load_image("demo_hiragana_input.png")
    except:
        print("Place 'demo_hiragana_input.png' in the folder!")
        return

    # Run Simulation
    # ON: 0-200ms (Learning & Encoding)
    # OFF: 300-600ms (Reverberation & Recall)
    activity_on, activity_off = model.run(img_original)
    
    print("\n🧠 Decoding Thoughts...")
    
    # 1. Train Decoder (Mind Reading)
    # "When CA3 fires like THIS, the image was THAT."
    # X: CA3 Activity (during input), Y: Original Image
    # Average activity over time to get a stable firing rate vector
    X_train = np.mean(activity_on, axis=0).reshape(1, -1)
    Y_train = img_original.flatten().reshape(1, -1)
    
    model.decoder.fit(X_train, Y_train)
    
    # 2. Decode Memory (The Mind's Eye)
    # "Input is gone. Based on current CA3 reverberation, what is the brain seeing?"
    X_test = np.mean(activity_off, axis=0).reshape(1, -1)
    img_reconstructed = model.decoder.predict(X_test).reshape(IMG_SIZE, IMG_SIZE)
    
    # ★修正: オートコントラスト（一番明るい場所を「白(1.0)」に強制変換する）
    max_val = np.max(img_reconstructed)
    if max_val > 0:
        img_reconstructed = img_reconstructed / max_val
        
    img_reconstructed = np.clip(img_reconstructed, 0.0, 1.0)
    
    # Visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title("Visual Input (0-200ms)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_reconstructed, cmap='gray')
    plt.title("Reconstructed Memory (300ms+)\n(Input Removed)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Experiment Done.")
    print("   If the right image looks like 'あ', the brain is holding the visual memory!")

if __name__ == "__main__":
    main()