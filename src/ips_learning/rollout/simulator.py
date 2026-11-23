#Numerically integrates the learned model
#This class integrates the learned system and produces trajectory artifacts.

import numpy as np
from scipy.integrate import solve_ivp

from ips_learning.rollout.artifacts import LearnedTrajectoryArtifact


class LearnedSystemRollout:
    """
    Rollout engine for learned dynamical systems.

    First-order:
        state = X.flatten()
        output = X_hat, V_hat=None

    Second-order:
        state = [X.flatten(), V.flatten()]
        output = X_hat, V_hat
    """

    def __init__(self, learned_system):
        self.learned_system = learned_system

    def simulate_one(
        self,
        u0: np.ndarray,
        t_eval: np.ndarray,
        method: str = "Radau",
        rtol: float = 1e-3,
        atol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Simulate one learned trajectory.

        Returns
        -------
        t : np.ndarray of shape (L,)
        X_hat : np.ndarray of shape (L, N, D)
        V_hat : np.ndarray of shape (L, N, D) or None
        """
        u0 = np.asarray(u0, dtype=float)
        t_eval = np.asarray(t_eval, dtype=float)

        system_order = getattr(self.learned_system, "system_order", None)

        if system_order not in (1, 2):
            raise ValueError(
                f"learned_system.system_order must be 1 or 2, got {system_order}."
            )

        if u0.shape != (self.learned_system.state_dim,):
            raise ValueError(
                f"Expected u0 shape {(self.learned_system.state_dim,)}, got {u0.shape}."
            )

        if t_eval.ndim != 1 or t_eval.size < 2:
            raise ValueError("t_eval must be a 1D array with at least two time points.")

        sol = solve_ivp(
            fun=self.learned_system.rhs,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=u0,
            t_eval=t_eval,
            method=method,
            dense_output=False,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"Learned-system rollout failed: {sol.message}")

        Y = sol.y.T
        N = self.learned_system.N
        D = self.learned_system.D

        if system_order == 1:
            X_hat = Y.reshape(-1, N, D)
            V_hat = None
            return sol.t, X_hat, V_hat

        split = N * D
        X_hat = Y[:, :split].reshape(-1, N, D)
        V_hat = Y[:, split:].reshape(-1, N, D)

        return sol.t, X_hat, V_hat

    def simulate_many(
        self,
        U0: np.ndarray,
        t_eval: np.ndarray,
        method: str = "Radau",
        rtol: float = 1e-3,
        atol: float = 1e-6,
        show_progress: bool = True,
        metadata: dict | None = None,
    ) -> LearnedTrajectoryArtifact:
        """
        Simulate a batch of learned trajectories.

        Parameters
        ----------
        U0 : np.ndarray of shape (M, state_dim)
        t_eval : np.ndarray of shape (L,)

        Returns
        -------
        LearnedTrajectoryArtifact
        """
        U0 = np.asarray(U0, dtype=float)
        t_eval = np.asarray(t_eval, dtype=float)

        system_order = getattr(self.learned_system, "system_order", None)

        if system_order not in (1, 2):
            raise ValueError(
                f"learned_system.system_order must be 1 or 2, got {system_order}."
            )

        if U0.ndim != 2:
            raise ValueError(f"Expected U0 shape (M, state_dim), got {U0.shape}.")

        if U0.shape[1] != self.learned_system.state_dim:
            raise ValueError(
                f"Expected U0.shape[1]={self.learned_system.state_dim}, "
                f"got {U0.shape[1]}."
            )

        M = U0.shape[0]

        X_list = []
        V_list = [] if system_order == 2 else None
        t_ref = None

        for m in range(M):
            if show_progress and ((m + 1) % max(1, M // 10) == 0 or m == M - 1):
                print(f"Rolling out learned trajectory {m + 1}/{M}")

            t, X_hat, V_hat = self.simulate_one(
                u0=U0[m],
                t_eval=t_eval,
                method=method,
                rtol=rtol,
                atol=atol,
            )

            if t_ref is None:
                t_ref = t
            else:
                if not np.allclose(t, t_ref):
                    raise ValueError(
                        "All learned trajectories must share the same returned time grid."
                    )

            X_list.append(X_hat)

            if system_order == 2:
                if V_hat is None:
                    raise ValueError("Second-order rollout returned V_hat=None.")
                V_list.append(V_hat)

        X_hat_batch = np.stack(X_list, axis=0)

        if system_order == 2:
            V_hat_batch = np.stack(V_list, axis=0)
        else:
            V_hat_batch = None

        artifact_metadata = {} if metadata is None else dict(metadata)
        artifact_metadata.update(
            {
                "stage": "2.5",
                "artifact_type": "rollout",
                "system_order": system_order,
                "solver_method": method,
                "solver_rtol": rtol,
                "solver_atol": atol,
                "n_rollouts": int(M),
                "N": int(self.learned_system.N),
                "D": int(self.learned_system.D),
                "state_dim": int(self.learned_system.state_dim),
                "interaction_type": getattr(
                    self.learned_system,
                    "interaction_type",
                    None,
                ),
            }
        )

        return LearnedTrajectoryArtifact(
            t=t_ref,
            X_hat=X_hat_batch,
            V_hat=V_hat_batch,
            U0=U0,
            metadata=artifact_metadata,
        )