# estimates localization domains from FeatureDataset

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LocalizedDomain:
    r_min: float
    r_max: float
    env_min: np.ndarray   # shape (D,)
    env_max: np.ndarray   # shape (D,)
    environment_variable: str


class DomainEstimator:
    """
    Estimate observed support from a FeatureDataset.

    Second-order:
        radial domain from R
        environment domain from V

    First-order:
        radial domain from R
        environment domain from X
    """

    def estimate_from_features(self, features) -> LocalizedDomain:
        system_order = features.metadata.get("system_order")

        if system_order not in (1, 2):
            raise ValueError(
                f"features.metadata['system_order'] must be 1 or 2, got {system_order}."
            )

        R = np.asarray(features.R, dtype=float)   # (M, L, N, N, 1)

        if R.ndim != 5: 
            raise ValueError(f"Expected R shape (M, L, N, N, 1), got {R.shape}.")

        r_values = np.abs(R)
        r_min = float(np.min(r_values))
        r_max = float(np.max(r_values))

        if system_order == 2:
            if features.V is None:
                raise ValueError("Second-order features must contain V.")
            env_data = np.asarray(features.V, dtype=float)
            environment_variable = "velocity"

        else:
            env_data = np.asarray(features.X, dtype=float)
            environment_variable = "position"

        if env_data.ndim != 4:
            raise ValueError(
                f"Expected environment data shape (M, L, N, D), got {env_data.shape}."
            )

        axes = (0, 1, 2)
        env_min = np.min(env_data, axis=axes).astype(float)
        env_max = np.max(env_data, axis=axes).astype(float)

        return LocalizedDomain(
            r_min=r_min,
            r_max=r_max,
            env_min=env_min,
            env_max=env_max,
            environment_variable=environment_variable,
        )