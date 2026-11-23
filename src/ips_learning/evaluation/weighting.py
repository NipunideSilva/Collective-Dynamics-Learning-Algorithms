# Optional weighted norms and KDE-based support weights

#KDE is define in the statistics module we pass it and use it here to compute normalized support weights. 
# This is a bit of a hack but it avoids circular imports and keeps the code organized.
import numpy as np


class WeightNormalizer:
    """
    Simple normalization helper for externally prepared weights.
    """

    @staticmethod
    def normalize(weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        total = np.sum(weights)
        if total <= 0:
            raise ValueError("Weights must sum to a positive value.")
        return weights / total