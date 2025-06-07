import torch

def get_sample_weights(populacje, wieki):
    weight_map = {
        (2, 3): 1.3,
        (2, 4): 1.5,
        (2, 5): 1.6,
        (2, 6): 1.8,
        (2, 7): 2.0,
        (2, 8): 2.0,
        (2, 9): 2.0,
        (2, 10): 2.0
    }
    weights = [weight_map.get((int(p), int(w)), 1.0) for p, w in zip(populacje, wieki)]
    return torch.tensor(weights, dtype=torch.float32)
