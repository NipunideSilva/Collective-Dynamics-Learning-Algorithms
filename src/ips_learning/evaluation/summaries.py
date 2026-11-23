# Aggregation across trajectories or repeated trials

import numpy as np


class SummaryAggregator:
    """
    Aggregate scalar metrics across repeated runs.
    """

    @staticmethod
    def mean_std(values) -> dict:
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }