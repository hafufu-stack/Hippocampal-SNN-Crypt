import numpy as np

# Hippocampal Neuron Parameters (Adapted from Izhikevich & Neural-Phase-Crypt)
# GC: Granule Cell (Dentate Gyrus) - Typically behaves like RS/Bursting
NEURON_TYPES = {
    "GC": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},  # Similar to RS
    "MC": {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},  # Mossy Cell (FS-like)
    "Inh": {"a": 0.05, "b": 0.25, "c": -65.0, "d": 2.0},  # Inhibitory Interneuron
}


def generate_hippocampal_network(n_neurons, genome=None):
    """
    Generates neuron parameters for the DG network.
    Currently initializes a homogenous population of Granule Cells (GC).
    """
    if genome is None:
        # Default config if no genome provided
        genome = {"syn_strength_MD": 0.1, "syn_strength_LD": 0.1}

    # For v0.1/v0.2, we assume all neurons are Granule Cells (GC) for simplicity.
    # Diversity can be added here later.

    a = np.full(n_neurons, NEURON_TYPES["GC"]["a"], dtype=np.float32)
    b = np.full(n_neurons, NEURON_TYPES["GC"]["b"], dtype=np.float32)
    c = np.full(n_neurons, NEURON_TYPES["GC"]["c"], dtype=np.float32)
    d = np.full(n_neurons, NEURON_TYPES["GC"]["d"], dtype=np.float32)

    # Add heterogeneity (Individual differences) to prevent artificial synchronization
    r = np.random.rand(n_neurons)
    c += 15.0 * r**2
    d -= 6.0 * r**2

    v = -65.0 * np.ones(n_neurons, dtype=np.float32)
    u = b * v

    return {
        "params": {"a": a, "b": b, "c": c, "d": d},
        "state": {"v": v, "u": u},
        "genome": genome,
    }
