import numpy as np

# ==========================================
# 🧬 Neuron Parameters
# ==========================================
NEURON_TYPES = {
    "GC":  {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0}, # DG Granule Cell
    "CA3": {"a": 0.02, "b": 0.2,  "c": -55.0, "d": 4.0}, # CA3 Pyramidal Cell
}

def generate_network_params(n_neurons, neuron_type="GC"):
    """
    Generate Izhikevich parameters for a population.
    """
    params = NEURON_TYPES[neuron_type]
    
    a = np.full(n_neurons, params["a"], dtype=np.float32)
    b = np.full(n_neurons, params["b"], dtype=np.float32)
    c = np.full(n_neurons, params["c"], dtype=np.float32)
    d = np.full(n_neurons, params["d"], dtype=np.float32)
    
    # Add biological heterogeneity
    r = np.random.rand(n_neurons)
    c += 10.0 * r**2
    d -= 4.0 * r**2

    v = -65.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v

    return {
        "params": {"a": a, "b": b, "c": c, "d": d},
        "state": {"v": v, "u": u}
    }

def generate_connections(n_pre, n_post, connection_prob, weight_val, random_weight=False):
    """
    Generate sparse connections (CSR-like format).
    Used for both DG->CA3 and CA3->CA3.
    """
    pointers = [0]
    all_indices = []
    
    for i in range(n_pre):
        # Determine number of targets based on probability
        n_targets = int(n_post * connection_prob)
        if n_targets < 1: n_targets = 1
        
        # Select random targets
        targets = np.random.choice(n_post, n_targets, replace=False)
        all_indices.extend(targets)
        pointers.append(len(all_indices))
        
    all_indices = np.array(all_indices, dtype=np.int32)
    pointers = np.array(pointers, dtype=np.int32)
    
    # Weights generation
    if random_weight:
        # For CA3 Recurrent: Start weak, learn later
        weights = np.random.uniform(0.0, 0.1, len(all_indices)).astype(np.float32)
    else:
        # For Mossy Fiber: Strong fixed weights
        weights = np.full(len(all_indices), weight_val, dtype=np.float32)
        weights *= np.random.uniform(0.9, 1.1, len(weights)) # Slight noise
    
    return pointers, all_indices, weights