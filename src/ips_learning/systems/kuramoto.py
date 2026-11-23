from __future__ import annotations

import numpy as np

from ips_learning.systems.base import FirstOrderSystemBase

from ips_learning.bases.specs import (
    ParametricFunctionSpec,
    ParametricFBasisSpec,
)


class KuramotoSystem(FirstOrderSystemBase):
    """
    First-order Kuramoto-type system written in IPS-compatible form.

    State format
    ------------
    Flattened state vector of shape (N * D,):

        X.flatten()

    where

        X : (N, D)

    Dynamics
    --------
        dX_i/dt = f_env(X_i)
                  + (1/N) sum_j phi(X_j - X_i) (X_j - X_i)

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
        K: float = 2.0,
        eps_r: float = 1e-12,
        wrap_angles: bool = True,
    ) -> None:
        super().__init__(N=N, D=D)

        if D != 1:
            raise ValueError("KuramotoSystem currently supports D=1 only.")

        self.system_name = "kuramoto"
        self.interaction_type = "energy"

        self.K = float(K)
        self.eps_r = float(eps_r)
        self.wrap_angles = bool(wrap_angles)

    def f_env(self, X: np.ndarray) -> np.ndarray:
        """
        First-order non-collective term f(X).

        This follows your uploaded Kuramoto implementation:

            f_env(X) = X / 100.
        """
        X = np.asarray(X, dtype=float)

        if X.shape[-1] != self.D:
            raise ValueError(f"Expected last dimension {self.D}, got {X.shape}.")

        return X / 100.0

    def kernel(self, R_D: np.ndarray) -> np.ndarray:
        """
        Kuramoto interaction kernel.

        Uses the signed phase difference R_D = X_j - X_i.

            phi(R_D) = K sin(R_D) / R_D

        so that

            phi(R_D) * R_D = K sin(R_D).
        """
        R_D = np.asarray(R_D, dtype=float)

        K_val = np.empty_like(R_D, dtype=float)
        mask = np.abs(R_D) > self.eps_r

        K_val[mask] = self.K * np.sin(R_D[mask]) / R_D[mask]
        K_val[~mask] = self.K #sin(t)/t ->1 as t->0

        return K_val

    def pairwise_phase_differences(self, X: np.ndarray) -> np.ndarray:
        """
        R_D[i,j,:] = X_j - X_i.

        If wrap_angles=True, differences are mapped to (-pi, pi].
        This is usually better for Kuramoto phase dynamics.
        """
        X = np.asarray(X, dtype=float)

        if X.shape != (self.N, self.D):
            raise ValueError(f"Expected X shape {(self.N, self.D)}, got {X.shape}.")

        R_D = X[np.newaxis, :, :] - X[:, np.newaxis, :]

        if self.wrap_angles:
            R_D = (R_D + np.pi) % (2.0 * np.pi) - np.pi

        return R_D

    def interaction_force(self, X: np.ndarray) -> np.ndarray:
        """
        Collective Kuramoto interaction:

            F_i = (1/N) sum_j phi(X_j - X_i) (X_j - X_i).
        """
        R_D = self.pairwise_phase_differences(X)
        K_val = self.kernel(R_D)

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
                "K": self.K,
                "eps_r": self.eps_r,
                "wrap_angles": self.wrap_angles,
            }
        )
        return data
    
def kuramoto_basis_z1(Z):
    return Z[:, 0]

def parametric_f_basis() -> ParametricFBasisSpec:
    """
    Parametric drift basis for forced Kuramoto.

        True form:
            f_env(X) = X / 100

        Learned model:
            f(X) = a * X
        """
    return ParametricFBasisSpec(
        terms=(
            ParametricFunctionSpec(
                name="x",
                function=kuramoto_basis_z1,
            ),
        )
    )