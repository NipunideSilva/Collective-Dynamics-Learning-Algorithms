from __future__ import annotations

import numpy as np

from ips_learning.systems.base import SecondOrderSystemBase


class SPPCSSystem(SecondOrderSystemBase):
    """
    SPP + Cucker-Smale second-order model.

        dX_i/dt = V_i

        dV_i/dt = f(V_i)
                  + (1/N) sum_j phi_E(r_ij)(X_j - X_i)
                  + (1/N) sum_j phi_A(r_ij)(V_j - V_i)

    where

        f(V_i) = (a - b ||V_i||^2) V_i.
    """

    def __init__(
        self,
        N: int,
        D: int = 2,
        cr: float = 0.6,
        lr: float = 0.5,
        ca: float = 1.0,
        la: float = 1.0,
        a: float = 1.0,
        b: float = 0.5,
        beta: float = 1.0,
        eps_r: float = 1e-1,
    ) -> None:
        super().__init__(N=N, D=D)

        self.system_name = "sppcs"
        self.interaction_type = "energy_alignment"

        self.cr = float(cr)
        self.lr = float(lr)
        self.ca = float(ca)
        self.la = float(la)
        self.a = float(a)
        self.b = float(b)
        self.beta = float(beta)
        self.eps_r = float(eps_r)

    def f_env(self, V: np.ndarray) -> np.ndarray:
        V = np.asarray(V, dtype=float)
        v2 = np.linalg.norm(V, axis=-1, keepdims=True) ** 2
        return (self.a - self.b * v2) * V

    def energy_kernel(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        R_safe = np.maximum(R, self.eps_r)

        return (
            (-self.cr / self.lr) * np.exp(-R_safe / self.lr)
            + (self.ca / self.la) * np.exp(-R_safe / self.la)
        ) * (self.N / R_safe)

    def alignment_kernel(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        return 1.0 / ((1.0 + R**2) ** self.beta)

    def kernel(self, R: np.ndarray) -> np.ndarray:
        """
        Backward-compatible default kernel.

        For energy-only pipeline calls, return phi_E.
        Prefer energy_kernel() and alignment_kernel() for model-selection plots.
        """
        return self.energy_kernel(R)

    def interaction_force(self, X: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
        if V is None:
            raise ValueError("SPPCSSystem requires V.")

        R_D, R = self.pairwise_position_differences(X)
        VR_D = self.pairwise_velocity_differences(V)

        K_E = self.energy_kernel(R)
        K_A = self.alignment_kernel(R)

        idx = np.arange(self.N)
        K_E[idx, idx, :] = 0.0
        K_A[idx, idx, :] = 0.0

        force_E = (1.0 / self.N) * np.sum(K_E * R_D, axis=1)
        force_A = (1.0 / self.N) * np.sum(K_A * VR_D, axis=1)

        return force_E + force_A

    def metadata(self) -> dict:
        data = super().metadata()
        data.update(
            {
                "cr": self.cr,
                "lr": self.lr,
                "ca": self.ca,
                "la": self.la,
                "a": self.a,
                "b": self.b,
                "beta": self.beta,
                "eps_r": self.eps_r,
                "true_energy_kernel": "spp",
                "true_alignment_kernel": "cucker_smale",
            }
        )
        return data