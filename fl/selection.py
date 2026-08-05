import numpy as np

def sample_clients(num_clients: int, frac: float, rng: np.random.Generator):
    k = max(1, int(round(num_clients * frac)))
    idxs = rng.choice(np.arange(num_clients), size=k, replace=False).tolist()
    return idxs
