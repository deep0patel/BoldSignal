import numpy as np


def align_modalities(features: dict) -> dict:
    """
    Trim all modality arrays to the same minimum time length.
    features: dict of {name: (T, D) array}
    returns: same dict, all trimmed to min T
    """
    min_len = min(arr.shape[0] for arr in features.values())
    return {k: v[:min_len] for k, v in features.items()}
