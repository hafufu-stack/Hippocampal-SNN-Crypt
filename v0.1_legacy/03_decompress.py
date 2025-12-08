import numpy as np
import os
import pickle
from PIL import Image

# ==========================================
# Configuration
# ==========================================
OUTPUT_DIR = "output_data"
PARAM_DIR = "snn_params"


def decompress_image():
    # 1. Load SNN Key (Weights)
    key_path = os.path.join(PARAM_DIR, "snn_key.pkl")
    if not os.path.exists(key_path):
        print("❌ SNN Key not found.")
        return

    with open(key_path, "rb") as f:
        params = pickle.load(f)
    weights = params["weights"]

    # 2. Load Compressed Spike Data
    data_path = os.path.join(OUTPUT_DIR, "compressed.dat")
    if not os.path.exists(data_path):
        print("❌ Compressed data not found. Please run '02_compress.py' first.")
        return

    with open(data_path, "rb") as f:
        spikes = pickle.load(f)

    indices = spikes["indices"]
    activities = spikes["activities"]
    original_shape = spikes["original_shape"]

    print(f"🔓 Data loaded. Spikes: {len(indices)}")
    print("   Reconstructing memory from neural activity...")

    # 3. Decoding (Reconstruction)
    # Reconstruct = Σ (Activity_i * Weight_i)
    # Superposition of the receptive fields of fired neurons
    reconstructed_vector = np.zeros(weights.shape[1])

    for i, neuron_idx in enumerate(indices):
        w = weights[neuron_idx]
        reconstructed_vector += w * activities[i]

    # 4. Post-processing
    # Rescale to 0-255 image format
    reconstructed_vector = reconstructed_vector - np.min(reconstructed_vector)
    reconstructed_vector = reconstructed_vector / np.max(reconstructed_vector)
    reconstructed_vector = reconstructed_vector * 255

    reconstructed_img_array = reconstructed_vector.reshape(original_shape).astype(
        np.uint8
    )

    # Save Image
    save_path = os.path.join(OUTPUT_DIR, "restored_image.png")
    img = Image.fromarray(reconstructed_img_array)
    img.save(save_path)

    print(f"✨ Restoration Complete!")
    print(f"🖼️ Image saved to: {save_path}")


if __name__ == "__main__":
    decompress_image()
