import numpy as np
import os
import pickle
from PIL import Image

# ==========================================
# Configuration
# ==========================================
INPUT_DIR = "input_data"
OUTPUT_DIR = "output_data"
PARAM_DIR = "snn_params"

# Number of active neurons allowed to fire per image.
# This determines the sparsity level.
MAX_SPIKES = 500


def compress_image():
    # 1. Load SNN Key (Weights)
    key_path = os.path.join(PARAM_DIR, "snn_key.pkl")
    if not os.path.exists(key_path):
        print("❌ SNN Key not found. Please run '01_generate_snn.py' first.")
        return

    with open(key_path, "rb") as f:
        params = pickle.load(f)

    weights = params["weights"]  # Shape: (Num_Neurons, Input_Size)
    input_shape = params["input_shape"]

    # Find the first image in the input directory
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    files = [
        f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not files:
        print(f"❌ No images found in '{INPUT_DIR}'. Please add an image.")
        return

    target_file = files[0]
    print(f"📥 Processing image: {target_file}")

    # 2. Preprocessing
    img_path = os.path.join(INPUT_DIR, target_file)
    img = Image.open(img_path).convert("L")  # Convert to Grayscale
    img_resized = img.resize(input_shape)

    # Normalize to 0-1 and zero-center
    img_data = np.array(img_resized).flatten() / 255.0
    img_data = img_data - np.mean(img_data)
    img_data = img_data / (np.linalg.norm(img_data) + 1e-8)

    # 3. Encoding (Spike Generation)
    # Calculate membrane potentials: Potential = Input • Weights
    potentials = np.dot(weights, img_data)

    # Winners-Take-All mechanism
    # Select the top N neurons with the highest response
    fired_indices = np.argsort(potentials)[-MAX_SPIKES:]

    # Record the indices and their activities (Sparse Representation)
    spikes = {
        "indices": fired_indices,
        "activities": potentials[fired_indices],
        "original_shape": input_shape,
    }

    # 4. Save Compressed Data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "compressed.dat")
    with open(save_path, "wb") as f:
        pickle.dump(spikes, f)

    print(f"⚡ Compression Complete!")
    print(f"   Original Dimensions: {len(img_data)}")
    print(f"   Spikes (Compressed): {len(fired_indices)}")
    print(f"💾 Saved to: {save_path}")


if __name__ == "__main__":
    compress_image()
