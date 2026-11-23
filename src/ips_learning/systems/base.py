from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp


InteractionType = Literal["energy", "alignment", "energy_alignment"]


class SecondOrderSystemBase(ABC):
    """
    Base class for second-order interacting particle systems.

    State layout:
        state = [X.flatten(), V.flatten()]

    Dynamics:
        dX/dt = V
        dV/dt = f_env(V) + interaction_force(X, V)
    """

    system_name: str = "second_order"
    interaction_type: InteractionType = "energy"

    def __init__(self, N: int, D: int = 2) -> None:
        self.N = int(N)
        self.D = int(D)
        self.state_dim = 2 * self.N * self.D

    def split_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=float)

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"Expected state shape {(self.state_dim,)}, got {state.shape}."
            )

        split = self.N * self.D
        X = state[:split].reshape(self.N, self.D)
        V = state[split:].reshape(self.N, self.D)
        return X, V

    def join_state(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        V = np.asarray(V, dtype=float)

        if X.shape != (self.N, self.D):
            raise ValueError(f"Expected X shape {(self.N, self.D)}, got {X.shape}.")
        if V.shape != (self.N, self.D):
            raise ValueError(f"Expected V shape {(self.N, self.D)}, got {V.shape}.")

        return np.concatenate([X.reshape(-1), V.reshape(-1)])

    def pairwise_position_differences(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)

        if X.shape != (self.N, self.D):
            raise ValueError(f"Expected X shape {(self.N, self.D)}, got {X.shape}.")

        R_D = X[np.newaxis, :, :] - X[:, np.newaxis, :]
        R = np.linalg.norm(R_D, axis=-1, keepdims=True)
        return R_D, R

    def pairwise_velocity_differences(self, V: np.ndarray) -> np.ndarray:
        V = np.asarray(V, dtype=float)

        if V.shape != (self.N, self.D):
            raise ValueError(f"Expected V shape {(self.N, self.D)}, got {V.shape}.")

        return V[np.newaxis, :, :] - V[:, np.newaxis, :]

    @abstractmethod
    def f_env(self, V: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def kernel(self, R: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def interaction_force(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def lhs(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        if X.shape != (self.N, self.D):
            raise ValueError(f"Expected X shape {(self.N, self.D)}, got {X.shape}.")
        if V.shape != (self.N, self.D):
            raise ValueError(f"Expected V shape {(self.N, self.D)}, got {V.shape}.")

        env = self.f_env(V) 
        interaction = self.interaction_force(X, V)
        acceleration = env + interaction

        if not np.all(np.isfinite(acceleration)):
            raise FloatingPointError(
                "Learned acceleration contains NaN or infinity."
            )

        max_acceleration = np.max(np.abs(acceleration))

        if max_acceleration > 1e8:
            raise FloatingPointError(
                "Learned acceleration became excessively large: "
                f"max |a| = {max_acceleration:.3e}."
            )
        return acceleration

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        X, V = self.split_state(state)
        dXdt = V
        dVdt = self.lhs(X, V)
        return self.join_state(dXdt, dVdt)

    def simulate(
        self,
        u0: np.ndarray,
        t_eval: np.ndarray,
        method: str = "Radau",
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u0 = np.asarray(u0, dtype=float)
        t_eval = np.asarray(t_eval, dtype=float)

        if u0.shape != (self.state_dim,):
            raise ValueError(f"Expected u0 shape {(self.state_dim,)}, got {u0.shape}.")
        if t_eval.ndim != 1 or t_eval.size < 2:
            raise ValueError("t_eval must be a 1D array with at least two time points.")

        sol = solve_ivp(
            fun=self.rhs,
            t_span=(t_eval[0], t_eval[-1]),
            y0=u0,
            t_eval=t_eval,
            method=method,
            dense_output=False,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solve failed: {sol.message}")

        Y = sol.y.T
        split = self.N * self.D

        X = Y[:, :split].reshape(-1, self.N, self.D)
        V = Y[:, split:].reshape(-1, self.N, self.D)

        LHS = np.empty_like(V)
        for l in range(X.shape[0]):
            LHS[l] = self.lhs(X[l], V[l])

        return sol.t, X, V, LHS

    def metadata(self) -> dict:
        return {
            "system_name": self.system_name,
            "system_order": 2,
            "interaction_type": self.interaction_type,
            "N": self.N,
            "D": self.D,
            "state_dim": self.state_dim,
        }
    

class FirstOrderSystemBase(ABC):
    """
    Base class for first-order interacting particle systems.

    State layout:
        state = X.flatten()

    Dynamics:
        dX/dt = f_env(X) + interaction_force(X)
    """

    system_name: str = "first_order"
    interaction_type: InteractionType = "energy"

    def __init__(self, N: int, D: int = 1) -> None:
        self.N = int(N)
        self.D = int(D)
        self.state_dim = self.N * self.D

    def split_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"Expected state shape {(self.state_dim,)}, got {state.shape}."
            )

        return state.reshape(self.N, self.D)

    def join_state(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if X.shape != (self.N, self.D):
            raise ValueError(f"Expected X shape {(self.N, self.D)}, got {X.shape}.")

        return X.reshape(-1)

    def pairwise_position_differences(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)

        if X.shape != (self.N, self.D):
            raise ValueError(f"Expected X shape {(self.N, self.D)}, got {X.shape}.")

        R_D = X[np.newaxis, :, :] - X[:, np.newaxis, :]
        R = np.linalg.norm(R_D, axis=-1, keepdims=True)
        return R_D, R

    @abstractmethod
    def f_env(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def kernel(self, R_D: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def interaction_force(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def lhs(self, X: np.ndarray) -> np.ndarray:
        if X.shape != (self.N, self.D):
            raise ValueError(f"Expected X shape {(self.N, self.D)}, got {X.shape}.")

        return self.f_env(X) + self.interaction_force(X)

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        X = self.split_state(state)
        dXdt = self.lhs(X)
        return self.join_state(dXdt)

    def simulate(
        self,
        u0: np.ndarray,
        t_eval: np.ndarray,
        method: str = "BDF",
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, None, np.ndarray]:
        u0 = np.asarray(u0, dtype=float)
        t_eval = np.asarray(t_eval, dtype=float)

        if u0.shape != (self.state_dim,):
            raise ValueError(f"Expected u0 shape {(self.state_dim,)}, got {u0.shape}.")

        sol = solve_ivp(
            fun=self.rhs,
            t_span=(t_eval[0], t_eval[-1]),
            y0=u0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solve failed: {sol.message}")

        X = sol.y.T.reshape(-1, self.N, self.D)

        LHS = np.empty_like(X)
        for l in range(X.shape[0]):
            LHS[l] = self.lhs(X[l])

        return sol.t, X, None, LHS

    def metadata(self) -> dict:
        return {
            "system_name": self.system_name,
            "system_order": 1,
            "interaction_type": self.interaction_type,
            "N": self.N,
            "D": self.D,
            "state_dim": self.state_dim,
        }