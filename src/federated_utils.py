import numpy as np

def federated_aggregate(node_weights):
    avg_weights = []
    for weights_list in zip(*node_weights):
        avg_weights.append(np.mean(weights_list, axis=0))
    return avg_weights