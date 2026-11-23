from __future__ import annotations

import numpy as np

from ips_learning.systems.base import SecondOrderSystemBase


class CuckerSmaleSystem(SecondOrderSystemBase):
    """
    Pure second-order Cucker-Smale system.

        dX_i/dt = V_i
        dV_i/dt = (1/N) sum_j phi_A(r_ij)(V_j - V_i)

    with

        phi_A(r) = 1 / (1 + r^2)^beta.
    """

    def __init__(
        self,
        N: int,
        D: int = 2,
        beta: float = 0.1,
    ) -> None:
        super().__init__(N=N, D=D)

        self.system_name = "cs"
        self.interaction_type = "alignment"
        self.beta = float(beta)

    def f_env(self, V: np.ndarray) -> np.ndarray:
        return np.zeros_like(V, dtype=float)

    def kernel(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        return 1.0 / ((1.0 + R**2) ** self.beta)
    
    def alignment_kernel(self, R: np.ndarray) -> np.ndarray:
        return self.kernel(R)


    def energy_kernel(self, R: np.ndarray) -> np.ndarray:
        return np.zeros_like(R, dtype=float)

    def interaction_force(self, X: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
        if V is None:
            raise ValueError("CuckerSmaleSystem requires V.")

        R_D, R = self.pairwise_position_differences(X)
        VR_D = self.pairwise_velocity_differences(V)

        K = self.kernel(R)

        idx = np.arange(self.N)
        K[idx, idx, :] = 0.0

        return (1.0 / self.N) * np.sum(K * VR_D, axis=1)

    def metadata(self) -> dict:
        data = super().metadata()
        data.update(
            {
                "beta": self.beta,
            }
        )
        return data