from __future__ import annotations

import numpy as np

from ips_learning.systems.base import FirstOrderSystemBase

from ips_learning.bases.specs import (
    ParametricFunctionSpec,
    ParametricFBasisSpec,
)


class OpinionDynamicsSystem(FirstOrderSystemBase):
    """
    First-order Opinion Dynamics system written in IPS-compatible form.

    State format
    ------------
    Flattened state vector of shape (N * D,):

        X.flatten()

    where

        X : (N, D)

    Dynamics
    --------
        dX_i/dt =  (1/N) sum_j phi(X_j - X_i) (X_j - X_i)

    For D = 1 and

        phi(r) = K sin(r) / r,

    the interaction term becomes

        K/N sum_j sin(X_j - X_i).

    The supervised target is

        LHS = dX/dt.
    """

    def __init__(
        self,
        N: int,
        D: int = 1,
    ) -> None:
        super().__init__(N=N, D=D)

        if D != 1:
            raise ValueError("Opinion DynamicsSystem currently supports D=1 only.")

        self.system_name = "opinion"
        self.interaction_type = "energy"

        self.confidence_radius = 1.0
        self.transition_radius = 1.0 / np.sqrt(2)

        self.kernel_inner_value = 1.0
        self.kernel_outer_value = 0.1


    def f_env(self, X: np.ndarray) -> np.ndarray:
        """
        First-order environmental force.
        Opinion dynamics has no environmental force,
        so f(X) ≡ 0.
        """
        X = np.asarray(X, dtype=float)
        return np.zeros_like(X)

    def kernel(self, R: np.ndarray) -> np.ndarray:
        """
        Opinion Dynamics interaction kernel.

        Uses intercting distnce R = |X_j - X_i|.

            phi(R) = 1 if 0 <= R < 1/sqrt(2),
            phi(R) = 0.1 if 1/sqrt(2) <= R < 1,
            phi(R) = 0 if R >= 1.   
       
        """
        R = np.asarray(R, dtype=float)

        K_val = np.zeros_like(R)
        K_val[(R >= 0) & (R < self.transition_radius)] = self.kernel_inner_value

        K_val[(R >= self.transition_radius) &(R < self.confidence_radius)] = self.kernel_outer_value

        return K_val


    def interaction_force(self, X: np.ndarray) -> np.ndarray:
        """
        Collective Opinion Dynamics interaction:

            F_i = (1/N) sum_j phi(X_j - X_i) (X_j - X_i).
        """
        R_D, R = self.pairwise_position_differences(X)
        K_val = self.kernel(R)

        idx = np.arange(self.N)
        K_val[idx, idx, :] = 0.0

        return (1.0 / self.N) * np.sum(K_val * R_D, axis=1)

    def lhs(self, X: np.ndarray) -> np.ndarray:
        """
        Regression target:

            LHS = dX/dt = f_env(X) + interaction_force(X).
        """
        return self.f_env(X) + self.interaction_force(X)

    def metadata(self) -> dict:
        data = super().metadata()
        data.update(
            {
                "kernel_type": "bounded_confidence",
                "confidence_radius": 1.0,
                "transition_radius": 1.0 / np.sqrt(2.0),
                "kernel_inner_value": 1.0,
                "kernel_outer_value": 0.1,
                "has_environment_force": False,
            }
        )
        return data
    


