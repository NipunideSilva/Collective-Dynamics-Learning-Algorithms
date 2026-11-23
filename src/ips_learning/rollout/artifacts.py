from dataclasses import dataclass
import numpy as np


@dataclass
class LearnedTrajectoryArtifact:
    """
    Output of Stage 2.5.

    Stores learned rollout outputs only.
    It does not know anything about fitting or evaluation.

    First-order:
        X_hat : (M, L, N, D)
        V_hat : None

    Second-order:
        X_hat : (M, L, N, D)
        V_hat : (M, L, N, D)

    Attributes
    ----------
    t : np.ndarray
        Time grid, shape (L,)

    X_hat : np.ndarray
        Learned positions, shape (M, L, N, D)

    U0 : np.ndarray
        Initial conditions used for rollout, shape (M, state_dim)

    metadata : dict
        Rollout metadata

    V_hat : np.ndarray or None
        Learned velocities for second-order systems.
        None for first-order systems.
    """
    t: np.ndarray
    X_hat: np.ndarray
    U0: np.ndarray
    metadata: dict
    V_hat: np.ndarray | None = None