import numpy as np

# ==========================================
# 🧬 Neuron Parameters
# ==========================================
# GC: Granule Cell (DG) - Sparse firing, strong filter
# CA3: Pyramidal Cell - Bursting, recurrent connections
NEURON_TYPES = {
    "GC":  {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0}, 
    "CA3": {"a": 0.02, "b": 0.2,  "c": -55.0, "d": 4.0}, 
}

def generate_network_params(n_neurons, neuron_type="GC"):
    """
    Generate parameters for a specific population (GC or CA3).
    """
    params = NEURON_TYPES[neuron_type]
    
    a = np.full(n_neurons, params["a"], dtype=np.float32)
    b = np.full(n_neurons, params["b"], dtype=np.float32)
    c = np.full(n_neurons, params["c"], dtype=np.float32)
    d = np.full(n_neurons, params["d"], dtype=np.float32)
    
    # Add heterogeneity
    r = np.random.rand(n_neurons)
    c += 10.0 * r**2
    d -= 4.0 * r**2

    v = -65.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v

    return {
        "params": {"a": a, "b": b, "c": c, "d": d},
        "state": {"v": v, "u": u}
    }

def generate_connections(n_pre, n_post, connection_prob, weight_val):
    """
    Generate sparse connections between two populations.
    Used for:
    1. DG -> CA3 (Mossy Fiber): Very sparse, very strong.
    2. CA3 -> CA3 (Recurrent): Sparse, moderate strength.
    
    Returns:
        indices: [Total_Synapses] (Post-synaptic ID for each connection)
        pointers: [n_pre + 1] (CSR-like format: start index for each pre-neuron)
        weights: [Total_Synapses]
    """
    # Simple adjacency list format flattened for GPU
    # "For each Pre-neuron, who does it connect to?"
    
    connections = []
    
    for i in range(n_pre):
        # Determine targets for neuron i
        # Randomly select targets based on probability
        # For simplicity/speed in Python loop, we select fixed count or probability
        n_targets = int(n_post * connection_prob)
        if n_targets < 1: n_targets = 1
        
        targets = np.random.choice(n_post, n_targets, replace=False)
        connections.append(targets)
        
    # Flatten for GPU
    # pointers[i] tells where the connections for neuron i start in the indices array
    pointers = [0]
    all_indices = []
    
    for targets in connections:
        all_indices.extend(targets)
        pointers.append(len(all_indices))
        
    all_indices = np.array(all_indices, dtype=np.int32)
    pointers = np.array(pointers, dtype=np.int32)
    weights = np.full(len(all_indices), weight_val, dtype=np.float32)
    
    # Add noise to weights
    weights *= np.random.uniform(0.8, 1.2, len(weights))
    
    return pointers, all_indices, weights