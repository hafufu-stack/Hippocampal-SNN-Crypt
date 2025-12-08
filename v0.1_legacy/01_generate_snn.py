import numpy as np
import os
import pickle

# ==========================================
# Configuration
# ==========================================
# Number of neurons (The complexity of the "Key").
# Higher values improve reconstruction quality but increase file size.
NUM_NEURONS = 10000
# Input image shape (Resized for processing)
INPUT_SHAPE = (32, 32)
# Directory to save the SNN parameters (The Key)
PARAM_DIR = "snn_params"


def generate_snn():
    """Generates random receptive fields (weights) for the SNN and saves them."""
    print(
        f"⚡ SNN Generation Started: {NUM_NEURONS} neurons, Input shape {INPUT_SHAPE}"
    )

    os.makedirs(PARAM_DIR, exist_ok=True)

    # 1. Initialize Weights
    # Randomly generate the receptive fields (synaptic weights).
    # This acts as the "Secret Key" for encryption.
    # Shape: (Num_Neurons, Input_Flattened_Size)
    input_size = INPUT_SHAPE[0] * INPUT_SHAPE[1]

    # Using Gaussian distribution for initialization
    weights = np.random.randn(NUM_NEURONS, input_size)

    # 2. Normalize Weights
    # Normalize the vector length to 1 for fair comparison of responsiveness.
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    weights = weights / (norms + 1e-8)  # Avoid division by zero

    # 3. Save Parameters
    params = {
        "weights": weights,
        "input_shape": INPUT_SHAPE,
        "num_neurons": NUM_NEURONS,
    }

    save_path = os.path.join(PARAM_DIR, "snn_key.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(params, f)

    print(f"💾 SNN Key saved to: {save_path}")
    print("Ready to encode data into spike patterns!")


if __name__ == "__main__":
    generate_snn()
