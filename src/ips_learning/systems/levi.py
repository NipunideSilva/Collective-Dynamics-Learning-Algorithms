from __future__ import annotations

import numpy as np

from ips_learning.systems.base import SecondOrderSystemBase

from ips_learning.bases.specs import (
    ParametricFunctionSpec,
    ParametricFBasisSpec,
)

class LeviSystem(SecondOrderSystemBase):
    """
    True second-order Levi / phototaxis-type alignment system in D dimensions.

    State format
    ------------
    Flattened state vector of shape (2 * N * D,):

        [X.flatten(), V.flatten()]

    where

        X : (N, D) positions
        V : (N, D) velocities

    Dynamics
    --------
        dX_i/dt = V_i

        dV_i/dt = f_env(V_i)
                  + (1/N) sum_j phi(||X_j - X_i||) (V_j - V_i)

    with

        f_env(V_i) = I0 * (u * e_l - V_i)

        phi(r) = lamda / (1 + r^2)^beta

    This is an alignment-type second-order system.  
    """
     
    def __init__(
        self,
        N: int,
        D: int = 2,
        u: float = 0.1,
        el: np.ndarray | tuple[float, float] = (-1 / np.sqrt(2), 1 / np.sqrt(2)),
        I0: float = 1.0,
        beta : float = 0.1,
        lamda: float = 1.0,
        eps_r: float = 1e-12,
    ) -> None:
        """
        Parameters
        ----------
        N : int
            Number of agents.
        D : int, default=2
            Spatial dimension.
        beta, lamda : float
            Interaction-kernel parameters.
        u, el, I0, lamda : float
            el is  unit direction of the light source, 
            u is the self-propulsion speed/terminal speed of the particles, 
            I0 is  constant intensity of the light source
            Environmental/self-propulsion parameters.
        eps_r : float, default=1e-12
            Minimum radius used to avoid division by zero or near-singularity.
        """
        
        super().__init__(N=N, D=D)
        self.system_name = "levi"
        self.interaction_type = "alignment"

        self.u = float(u)
        self.el = np.asarray(el, dtype=float).reshape(self.D)
        self.I0 = float(I0)
        self.beta = float(beta)
        self.lamda = float(lamda)
        self.eps_r = float(eps_r)

        self.state_dim = 2 * self.N * self.D

    
    def f_env(self, V: np.ndarray) -> np.ndarray:
        """
        Environmental/self-propulsion force.
        f_env(V) = I0 (U * el - V)

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
            raise ValueError(f"Expected V shape {self.D}, got {V.shape}.")

        return self.I0 * (self.u * self.el - V)

    def kernel(self, R: np.ndarray) -> np.ndarray:
        """
        Interaction kernel φ(R).
        True Levi interaction kernel.

            phi(r) = lamda / (1 + r^2)^beta

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

        K = self.lamda/ ((1 + R_safe**2)**self.beta)

        return K.reshape(original_shape)
    
    def alignment_kernel(self, R: np.ndarray) -> np.ndarray:
        return self.kernel(R)


    def energy_kernel(self, R: np.ndarray) -> np.ndarray:
        return np.zeros_like(R, dtype=float)
    
    
    def interaction_force(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Alignment interaction force.

            F_i = (1/N) sum_j phi(||X_j - X_i||) (V_j - V_i)

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Agent positions.
        V : np.ndarray of shape (N, D)

        Returns
        -------
        F_int : np.ndarray of shape (N, D)
            Collective interaction force for each agent.
        """
        _, R = self.pairwise_position_differences(X)
        VR_D = self.pairwise_velocity_differences(V)

        K = self.kernel(R)

        idx = np.arange(self.N)
        K[idx, idx, :] = 0.0


        F_int = (1.0 / self.N) * np.sum(K * VR_D, axis=1)
        return F_int
    def metadata(self) -> dict:
        data = super().metadata()
        data.update(
            {
                "u": self.u,
                "el": self.el.tolist(),
                "I0": self.I0,
                "beta": self.beta,
                "lamda": self.lamda,
                "eps_r": self.eps_r,
            }
        )
        return data
    

def levi_basis_one(Z):
    return np.ones(Z.shape[0])


def levi_basis_v1(Z):
    return Z[:, 0]


def levi_basis_v2(Z):
    return Z[:, 1]


def parametric_f_basis() -> ParametricFBasisSpec:
    """
    Parametric basis for Levi phototaxis environment term.

    True form:
        f(v) = I0 (U_el - v)

    This is affine in velocity:

        f(v) = c + A v

    Scalar basis terms:
        1, v_1, v_2

    Learned coefficients are assigned componentwise.
    """
    return ParametricFBasisSpec(
        terms=(
            ParametricFunctionSpec("1", levi_basis_one),
            ParametricFunctionSpec("v_1", levi_basis_v1),
            ParametricFunctionSpec("v_2", levi_basis_v2),
        )
    )
