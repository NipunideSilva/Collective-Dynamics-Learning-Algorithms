from __future__ import annotations

import numpy as np

from ips_learning.systems.base import SecondOrderSystemBase
from ips_learning.bases.specs import (
    ParametricFunctionSpec,
    ParametricFBasisSpec,
)


class SPPSystem(SecondOrderSystemBase):
    """
        True second-order Self-Propelled Particle (SPP) system in D dimensions.

        State format
        ------------
        Flattened state vector of shape (2 * N * D,):
            [X.flatten(), V.flatten()]

        where
            X : (N, D) positions
            V : (N, D) velocities

        Dynamics
        --------
            dX/dt = V
            dV/dt = f_env(V) + (1/N) * sum_j kernel(r_ij) * (x_j - x_i)

        with
            r_ij = ||x_j - x_i||.

        dX/dt = V
        dV/dt = LHS(X, V)

        For SPP, the supervised target is:
            LHS = dV/dt = ddot{X}
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
        eps_r: float = 1e-1,
    ) -> None:
        """
        Parameters
        ----------
        N : int
            Number of agents.
        D : int, default=2
            Spatial dimension.
        cr, lr, ca, la : float
            Interaction-kernel parameters.
        a, b : float
            Environmental/self-propulsion parameters.
        eps_r : float, default=1e-1
            Minimum radius used to avoid division by zero or near-singularity.
        """
        super().__init__(N=N, D=D)
        self.system_name = "spp"
        self.interaction_type = "energy"

        self.cr = float(cr)
        self.lr = float(lr)
        self.ca = float(ca)
        self.la = float(la)
        self.a = float(a)
        self.b = float(b)
        self.eps_r = float(eps_r)

        self.state_dim = 2 * self.N * self.D

    
   
    def f_env(self, V: np.ndarray) -> np.ndarray:
        """
        Environmental/self-propulsion force.
        f_env(V) = (a - b ||V||^2) V

        Parameters
        ----------
        V : np.ndarray of shape (N, D)
            Agent velocities.

        Returns
        -------
        f : np.ndarray of shape (N, D)
            Environmental force evaluated at each velocity.
        """
        V = np.asarray(V, dtype=float)

        if V.shape[-1] != (self.D):
            raise ValueError(f"Expected V shape {(-1, self.D)}, got {V.shape}.")

        v2 = np.linalg.norm(V, axis=-1, keepdims=True) ** 2
        return (self.a - self.b * v2) * V

    def kernel(self, R: np.ndarray) -> np.ndarray:
        """
        Interaction kernel φ(R).

        Supported input shapes
        ----------------------
        scalar
        (K,) - radial grid / plotting inputs
        (K,1)

        Returns matching shape.
        """
        R = np.asarray(R, dtype=float)

        original_shape = R.shape

        R_safe = np.maximum(R, self.eps_r)

        K = (
            (-self.cr / self.lr) * np.exp(-R_safe / self.lr)
            + (self.ca / self.la) * np.exp(-R_safe / self.la)
        ) * (self.N / R_safe)

        return K.reshape(original_shape)
    
    def interaction_force(self, X: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
        """
        Compute collective interaction force.

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Agent positions.

        Returns
        -------
        F_int : np.ndarray of shape (N, D)
            Collective interaction force for each agent.
        """
        if X.shape != (self.N, self.D):
            raise ValueError(f"Expected X shape {(self.N, self.D)}, got {X.shape}.")

        # R_D[i, j, :] = x_j - x_i
        R_D, R = self.pairwise_position_differences(X)
        K = self.kernel(R)

        idx = np.arange(self.N)
        K[idx, idx, :] = 0.0

        F_int = (1.0 / self.N) * np.sum(K * R_D, axis=1)
        return F_int
    
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
                "eps_r": self.eps_r,
            }
        )
        return data
    
def spp_basis_v1(Z):
    return Z[:, 0]


def spp_basis_v2(Z):
    return Z[:, 1]

def spp_basis_v2v1(Z):
    return np.sum(Z**2, axis=1) * Z[:, 0]

def spp_basis_v2v2(Z):
    return np.sum(Z**2, axis=1) * Z[:, 1]

#chage this to l1 norm
def spp_basis_v1v1(Z):
    return np.sqrt(np.sum(Z**2, axis=1)) * Z[:, 0]

def spp_basis_v1v2(Z):
    return np.sqrt(np.sum(Z**2, axis=1)) * Z[:, 1]


def parametric_f_basis() -> ParametricFBasisSpec:
    """
    Parametric basis for SPP self-propulsion term.

    True form:
        f(v) = alpha v - beta ||v||^2 v

    Scalar basis terms:
        v_1
        v_2
        ||v||^2 v_1
        ||v||^2 v_2
    """
    return ParametricFBasisSpec(
        terms=(
            ParametricFunctionSpec(
                name="v_1",
                function=spp_basis_v1,
            ),
            ParametricFunctionSpec(
                name="v_2",
                function=spp_basis_v2,
            ),
            ParametricFunctionSpec(
                name="||v||^2 v_1",
                function=spp_basis_v2v1,
            ),
            ParametricFunctionSpec(
                name="||v||^2 v_2",
                function=spp_basis_v2v2,
            ),
        )
    )

def parametric_f_basis2() -> ParametricFBasisSpec:
    """
    Parametric basis for SPP self-propulsion term.

    True form:
        f(v) = alpha v - beta ||v||^2 v

    Scalar basis terms:
        v_1
        v_2
        ||v|| v_1
        ||v|| v_2
    """
    return ParametricFBasisSpec(
        terms=(
            ParametricFunctionSpec(
                name="v_1",
                function=spp_basis_v1,
            ),
            ParametricFunctionSpec(
                name="v_2",
                function=spp_basis_v2,
            ),
            ParametricFunctionSpec(
                name="||v|| v_1",
                function=spp_basis_v1v1,
            ),
            ParametricFunctionSpec(
                name="||v|| v_2",
                function=spp_basis_v1v2,
            ),
        )
    )


    
    
# Minimal validation order:
# system = SPPSystem(...)
# X0, V0 = system.split_state(u0)
# acc = system.lhs(X0, V0)
# rhs = system.rhs(0.0, u0)
# t, X, V, LHS = system.simulate(u0, t_eval)