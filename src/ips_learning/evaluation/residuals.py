# Residual error on observed states
# Residual error compares the true left-hand side LHS to the learned right-hand side evaluated on the observed states.

import numpy as np


class ResidualEvaluator:
    """
    Compute residual error on observed states.

    First-order:
        residual = LHS - learned_model.lhs(X)
        LHS = dot X

    Second-order:
        residual = LHS - learned_model.acceleration(X, V)
        LHS = ddot X
    """

    def evaluate(self, features, learned_model) -> dict:
        X = np.asarray(features.X, dtype=float)
        LHS = np.asarray(features.LHS, dtype=float)

        system_order = features.metadata.get("system_order")

        if system_order not in (1, 2):
            raise ValueError(
                f"features.metadata['system_order'] must be 1 or 2, got {system_order}."
            )

        if X.shape != LHS.shape:
            raise ValueError(
                f"X and LHS must have same shape, got X={X.shape}, LHS={LHS.shape}."
            )

        if system_order == 2:
            if features.V is None:
                raise ValueError("Second-order residual evaluation requires features.V.")
            V = np.asarray(features.V, dtype=float)

            if V.shape != X.shape:
                raise ValueError(
                    f"V and X must have same shape, got V={V.shape}, X={X.shape}."
                )
        else:
            V = None

        M, L, N, D = X.shape

        sq_sum = 0.0
        scale_sum = 0.0
        per_traj = []

        for m in range(M):
            sq_m = 0.0
            scale_m = 0.0

            for l in range(L):
                if system_order == 1:
                    lhs_hat = learned_model.lhs(X[m, l])
                else:
                    lhs_hat = learned_model.acceleration(X[m, l], V[m, l])

                diff = LHS[m, l] - lhs_hat

                sq_m += float(np.sum(diff ** 2))
                scale_m += float(np.sum(LHS[m, l] ** 2))

            per_traj.append(np.sqrt(sq_m / (L * N)))
            sq_sum += sq_m
            scale_sum += scale_m

        e_res = np.sqrt(sq_sum / (M * L * N))
        relative_res = np.sqrt(sq_sum / scale_sum) if scale_sum > 0 else np.nan

        return {
            "E_res": float(e_res),
            "E_res_relative": float(relative_res),
            "per_trajectory_E_res": np.array(per_traj, dtype=float),
            "system_order": int(system_order),
            "n_trajectories": int(M),
            "n_time_points": int(L),
            "N": int(N),
            "D": int(D),
        }