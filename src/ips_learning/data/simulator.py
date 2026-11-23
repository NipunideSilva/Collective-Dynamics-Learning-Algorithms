from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class TrajectoryBatch:
    """
    Container for one or many simulated trajectories.

    Attributes
    ----------
    t : np.ndarray of shape (L,)
        Common time grid.
    X : np.ndarray of shape (M, L, N, D)
        Simulated positions.
    V : np.ndarray of shape (M, L, N, D)
        Simulated velocities.
    LHS : np.ndarray of shape (M, L, N, D)
        Exact supervised targets for the system.
        For SPP: LHS = ddot{X}.
    U0 : np.ndarray of shape (M, state_dim)
        Initial conditions used to generate the trajectories.


    Batched simulated trajectories.

    First-order:
        X   : (M, L, N, D)
        V   : None
        LHS : dX/dt

    Second-order:
        X   : (M, L, N, D)
        V   : (M, L, N, D)
        LHS : dV/dt = ddot X
    """
    t: np.ndarray
    X: np.ndarray
    V: np.ndarray | None
    LHS: np.ndarray
    U0: np.ndarray


class Simulator:
    """
    Generic simulator wrapper for interacting particle systems.

    The system object is expected to implement:
        simulate(u0, t_eval, ...) -> (t, X, V, LHS)

    where:
        u0 : shape (state_dim,)
        t_eval : shape (L,)
        X : shape (L, N, D)
        V : shape (L, N, D)
        LHS : shape (L, N, D)
    """

    def __init__(self, system) -> None:
        """
        Parameters
        ----------
        system : object
            A dynamical system object with a method:
                simulate(u0, t_eval, ...) -> (t, X, V)
        """
        if not hasattr(system, "simulate"):
            raise TypeError("system must implement a simulate(u0, t_eval, ...) method.")

        self.system = system

    def simulate_one(
        self,
        u0: np.ndarray,
        t_eval: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray|None, np.ndarray]:
        """
        Simulate one trajectory.

        Parameters
        ----------
        u0 : np.ndarray of shape (state_dim,)
            One flattened initial condition.
        t_eval : np.ndarray of shape (L,)
            Time grid.
        **kwargs
            Additional keyword arguments forwarded to system.simulate.

        Returns
        -------
        t : np.ndarray of shape (L,)
            Time grid returned by the system solver.
        X : np.ndarray of shape (L, N, D)
            Simulated positions.
        V : np.ndarray of shape (L, N, D) or None
            Velocity trajectories for second-order systems.
            None for first-order systems
            Simulated velocities.
        LHS : np.ndarray of shape (L, N, D)
            Exact supervised targets for the system.
        """
        u0 = np.asarray(u0, dtype=float)
        t_eval = np.asarray(t_eval, dtype=float)

        if u0.ndim != 1:
            raise ValueError(f"u0 must be a 1D array, got shape {u0.shape}.")
        if t_eval.ndim != 1 or t_eval.size < 2:
            raise ValueError("t_eval must be a 1D array with at least two time points.")

        t, X, V, LHS = self.system.simulate(u0=u0, t_eval=t_eval, **kwargs)

        t = np.asarray(t, dtype=float)
        X = np.asarray(X, dtype=float)
        V = None if V is None else np.asarray(V, dtype=float)
        LHS = np.asarray(LHS, dtype=float)
        if t.shape != t_eval.shape:
            raise ValueError(
                f"Expected returned t shape {t_eval.shape}, got {t.shape}."
            )
        if X.ndim != 3:
            raise ValueError(f"Expected X to have shape (L, N, D), got {X.shape}.")
        if LHS.ndim != 3:
            raise ValueError(f"Expected LHS to have shape (L, N, D), got {LHS.shape}.")
        if X.shape != LHS.shape:
            raise ValueError(
                f"X, LHS must have the same shape, got "
                f"X={X.shape}, LHS={LHS.shape}."
            )
        if X.shape[0] != t_eval.size:
            raise ValueError(
                f"Expected trajectory length {t_eval.size}, got {X.shape[0]}."
            )
        if V is not None:
            if V.ndim != 3:
                raise ValueError(f"Expected V to have shape (L, N, D), got {V.shape}.")
            if V.shape != X.shape:
                raise ValueError(
                    f"V must have same shape as X, got V={V.shape}, X={X.shape}."
                )

        return t, X, V, LHS
    
    def simulate_many(
        self,
        U0: np.ndarray,
        t_eval: np.ndarray,
        show_progress: bool = True,
        **kwargs,
    ) -> TrajectoryBatch:
        """
        Simulate a batch of trajectories using a common time grid.

        Parameters
        ----------
        U0 : np.ndarray of shape (M, state_dim)
            Batch of flattened initial conditions.
        t_eval : np.ndarray of shape (L,)
            Common time grid for all trajectories.
        show_progress : bool, default=True
            Whether to print simple progress messages.
        **kwargs
            Additional keyword arguments forwarded to system.simulate.

        Returns
        -------
        TrajectoryBatch
            Batched trajectories with shapes:
                t  : (L,)
                X  : (M, L, N, D)
                V  : (M, L, N, D)
                LHS: (M, L, N, D)
                U0 : (M, state_dim)
        """
        U0 = np.asarray(U0, dtype=float)
        t_eval = np.asarray(t_eval, dtype=float)

        if U0.ndim != 2:
            raise ValueError(f"U0 must be a 2D array of shape (M, state_dim), got {U0.shape}.")
        if t_eval.ndim != 1 or t_eval.size < 2:
            raise ValueError("t_eval must be a 1D array with at least two time points.")

        M = U0.shape[0]

        X_list = []
        V_list = []
        has_velocity = None
        LHS_list = []
        t_ref = None

        for m in range(M):
            if show_progress and ((m + 1) % max(1, M // 10) == 0 or m == M - 1):
                print(f"Simulating trajectory {m + 1}/{M}")

            t, X, V, LHS = self.simulate_one(U0[m], t_eval=t_eval, **kwargs)

            if t_ref is None:
                t_ref = t
            else:
                if not np.allclose(t, t_ref):
                    raise ValueError("All trajectories must share the same returned time grid.")


            if has_velocity is None:
                has_velocity = V is not None
            else:
                if has_velocity != (V is not None):
                    raise ValueError("All trajectories must consistently return V or None.")

            X_list.append(X)
            LHS_list.append(LHS)

            if V is not None:
                V_list.append(V)

        X_batch = np.stack(X_list, axis=0)  # (M, L, N, D)
        if has_velocity:
            V_batch = np.stack(V_list, axis=0)  # (M, L, N, D)
        else:
            V_batch = None 
        LHS_batch = np.stack(LHS_list, axis=0)  # (M, L, N, D)

        return TrajectoryBatch(
            t=t_ref,
            X=X_batch,
            V=V_batch,
            LHS=LHS_batch,
            U0=U0,
        )
