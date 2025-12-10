import numpy as np

# Hippocampal Neuron Parameters
# CA3 Pyramidal Cell (Bursting type)
NEURON_TYPES = {
    "GC":  {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0}, 
    "CA3": {"a": 0.02, "b": 0.2,  "c": -55.0, "d": 4.0}, # Lower threshold for bursting
}

def generate_hippocampal_network(n_neurons, genome=None):
    """
    Generates neuron parameters.
    For v0.7, we initialize them as CA3 Pyramidal Cells.
    """
    if genome is None:
        genome = {"syn_strength_rec": 0.5}

    # Use CA3 Parameters for this experiment
    a = np.full(n_neurons, NEURON_TYPES["CA3"]["a"], dtype=np.float32)
    b = np.full(n_neurons, NEURON_TYPES["CA3"]["b"], dtype=np.float32)
    c = np.full(n_neurons, NEURON_TYPES["CA3"]["c"], dtype=np.float32)
    d = np.full(n_neurons, NEURON_TYPES["CA3"]["d"], dtype=np.float32)
    
    # Add heterogeneity
    r = np.random.rand(n_neurons)
    c += 10.0 * r**2
    d -= 4.0 * r**2

    v = -65.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v

    return {
        "params": {"a": a, "b": b, "c": c, "d": d},
        "state": {"v": v, "u": u},
        "genome": genome,
    }

def generate_recurrent_connections(n_neurons, connections_per_neuron=50):
    """
    Generate sparse recurrent connections for CA3.
    Each neuron receives input from 'connections_per_neuron' other random neurons.
    
    Returns:
        pre_indices: [n_neurons, connections] (Who connects to me?)
        weights:     [n_neurons, connections] (How strong?)
    """
    pre_indices = np.zeros((n_neurons, connections_per_neuron), dtype=np.int32)
    weights = np.zeros((n_neurons, connections_per_neuron), dtype=np.float32)
    
    for i in range(n_neurons):
        # Pick random partners
        partners = np.random.choice(n_neurons, connections_per_neuron, replace=False)
        pre_indices[i, :] = partners
        # Random initial weights
        weights[i, :] = np.random.uniform(0.5, 1.5, connections_per_neuron)
        
    return pre_indices, weights