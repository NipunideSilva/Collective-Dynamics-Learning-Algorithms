from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ips_learning.statistics.density import build_empirical_error_density

#. run  rollouts ------------------------------------------------------
def model_selection_rollout_fn(
    *,
    learned_model,
    dataset,
    split: str,
    candidate,
    config,
):
    """
    Roll out learned model from each initial condition in dataset.

    Saves:
        artifacts/model_selection_rollouts/{system}/{profile}/{dataset}/{candidate}/{split}/rollout.npz
    """
    out_dir = (
        Path("artifacts/model_selection_rollouts")
        / config.system_name
        / config.profile_name
        / config.dataset_name
        / candidate.name
        / split
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "rollout.npz"

    t = np.asarray(dataset.t, dtype=float)
    X_true = np.asarray(dataset.X, dtype=float)
    V_true = None if getattr(dataset, "V", None) is None else np.asarray(dataset.V, dtype=float)

    M, L, N, D = X_true.shape

    X_hat = np.zeros_like(X_true)
    V_hat = None if V_true is None else np.zeros_like(V_true)

    # Uses existing learned model RHS with scipy solve_ivp.
    from scipy.integrate import solve_ivp

    for m in range(M):
        if learned_model.system_order == 1:
            y0 = learned_model.join_state(X_true[m, 0])
        else:
            y0 = learned_model.join_state(X_true[m, 0], V_true[m, 0])

        sol = solve_ivp(
            fun=learned_model.rhs,
            t_span=(float(t[0]), float(t[-1])),
            y0=y0,
            t_eval=t,
            method="Radau",
            rtol=1e-6,
            atol=1e-6,
        )

        if not sol.success:
            raise RuntimeError(
                f"Rollout failed for candidate={candidate.name}, "
                f"split={split}, trajectory={m}: {sol.message}"
            )

        if learned_model.system_order == 1:
            X_hat[m] = sol.y.T.reshape(L, N, D)
        else:
            state = sol.y.T
            split_idx = N * D
            X_hat[m] = state[:, :split_idx].reshape(L, N, D)
            V_hat[m] = state[:, split_idx:].reshape(L, N, D)

    if V_hat is None:
        np.savez_compressed(
            out_path,
            t=t,
            X_true=X_true,
            X_hat=X_hat,
        )
    else:
        np.savez_compressed(
            out_path,
            t=t,
            X_true=X_true,
            V_true=V_true,
            X_hat=X_hat,
            V_hat=V_hat,
        )

    return out_path


# evaluation---------------------------------------------------------

def _trajectory_linf_stats(
    X_true: np.ndarray,
    X_hat: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> dict[str, float]:
    """
    Compute trajectory L-infinity error statistics over a time interval.

    X_true, X_hat have shape:
        (M, L, N, D)

    For each trajectory m:

        e_m(t_l) = sqrt( mean_i ||X_true[m,l,i] - X_hat[m,l,i]||^2 )

        Linf_m = max_l e_m(t_l)

    Return:
        mean_m Linf_m
        std_m Linf_m
    """
    diff = X_true[:, start_idx:end_idx] - X_hat[:, start_idx:end_idx]
    # shape: (M, L_sub, N, D)

    diff2 = np.sum(diff**2, axis=-1)
    # shape: (M, L_sub, N)

    e_t = np.sqrt(np.mean(diff2, axis=-1))
    # shape: (M, L_sub)

    linf_per_traj = np.max(e_t, axis=1)
    # shape: (M,)

    RMSE_traj = np.sqrt(np.mean(diff2))

    return {
        "RMSE": float(RMSE_traj),
        "Linf_mean": float(np.mean(linf_per_traj)),
        "Linf_std": float(np.std(linf_per_traj, ddof=1))
        if linf_per_traj.shape[0] > 1
        else 0.0,
    }
#-----------------------AIC/BIC Likelyhood valuation------------------------------------------------------------
def _trajectory_reconstruction_information(
        X_true: np.ndarray,
        X_hat: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> dict[str, float | int]:
        """
        Compute quantities required for Gaussian AIC/BIC evaluation.

        Parameters
        ----------
        X_true, X_hat
            Position trajectories with shape (M, L, N, D).

        start_idx, end_idx
            Time-index interval used for training reconstruction.

        Returns
        -------
        dict
            trajectory_reconstruction_rss
                Sum of squared scalar position-coordinate errors.

            trajectory_reconstruction_n_obs
                Number of scalar position observations entering the RSS:

                    n = M * L_interval * N * D.
        """
        X_true = np.asarray(X_true, dtype=float)
        X_hat = np.asarray(X_hat, dtype=float)

        if X_true.shape != X_hat.shape:
            raise ValueError(
                "X_true and X_hat must have the same shape, "
                f"got {X_true.shape} and {X_hat.shape}."
            )

        X_true_interval = X_true[:, start_idx:end_idx]
        X_hat_interval = X_hat[:, start_idx:end_idx]

        if X_true_interval.size == 0:
            raise ValueError(
                "The reconstruction interval contains no observations."
            )

        residual = X_true_interval - X_hat_interval

        rss = float(np.sum(residual**2))
        n_observations = int(residual.size)

        return {
            "trajectory_reconstruction_rss": rss,
            "trajectory_reconstruction_n_obs": n_observations,
        }


def _find_time_index(t: np.ndarray, value: float | None) -> int:
    if value is None:
        return len(t)

    return int(np.searchsorted(t, value, side="right"))


def model_selection_evaluate_fn(
    *,
    learned_model,
    dataset,
    rollout_artifact,
    split: str,
    candidate,
    config,
):
    """
    Compute real validation/test trajectory metrics from saved rollout.
    """
    if rollout_artifact is None:
        raise ValueError("rollout_artifact is required for trajectory evaluation.")

    data = np.load(rollout_artifact)

    t = np.asarray(data["t"], dtype=float)
    X_true = np.asarray(data["X_true"], dtype=float)
    X_hat = np.asarray(data["X_hat"], dtype=float)

    fit_end_time = getattr(config, "fit_end_time", None)
    fit_end_idx = _find_time_index(t, fit_end_time)

    #------------------trajectory rollouts---------------------------------------
    #[0,T]
    E_traj_reconstruction = _trajectory_linf_stats(
        X_true=X_true,
        X_hat=X_hat,
        start_idx=0,
        end_idx=fit_end_idx,
    )
    #[0, Tf]
    E_traj_full = _trajectory_linf_stats(
        X_true=X_true,
        X_hat=X_hat,
        start_idx=0,
        end_idx=len(t),
    )
    #[T, Tf]
    if fit_end_idx >= len(t):
        E_traj_prediction = E_traj_full
    else:
        E_traj_prediction = _trajectory_linf_stats(
            X_true=X_true,
            X_hat=X_hat,
            start_idx=fit_end_idx,
            end_idx=len(t),
        )

    E_res = _residual_rmse(
        learned_model=learned_model,
        dataset=dataset,
        start_idx=0,
        end_idx=fit_end_idx,
    )
#----------------------------framework compatibility ---------------------
    X_scale = float(
    np.sqrt(np.mean(np.sum(X_true[:, :fit_end_idx]**2, axis=-1)))
    )

    LHS = np.asarray(dataset.LHS, dtype=float)
    LHS_scale = float(
        np.sqrt(np.mean(np.sum(LHS[:, :fit_end_idx]**2, axis=-1)))
    )

    eps = 1e-12

    E_recon_rel = (
        E_traj_reconstruction["RMSE"] / (X_scale + eps)
    )

    E_res_rel = (
        E_res / (LHS_scale + eps)
    )
#------------------------------------------------------------------------------
    metrics = {
        "status": "ok",
        "failed": False,
        "candidate": candidate.name,
        "split": split,
    }

    if split == "train":
        information_criteria_data = _trajectory_reconstruction_information(
            X_true=X_true,
            X_hat=X_hat,
            start_idx=0,
            end_idx=fit_end_idx,
        )
        metrics.update(
            {
                # train-fit reconstruction: [t0,T]
                "E_traj_reconstruction": E_traj_reconstruction["Linf_mean"],
                "E_traj_reconstruction_std": E_traj_reconstruction["Linf_std"],
                "E_traj_reconstruction_rel": E_recon_rel,

                #  # Quantities used by Gaussian AIC/BIC
                # "trajectory_reconstruction_rss": (
                #     information_criteria_data[
                #         "trajectory_reconstruction_rss"
                #     ]
                # ),
                # "trajectory_reconstruction_n_obs": (
                #     information_criteria_data[
                #         "trajectory_reconstruction_n_obs"
                #     ]
                # ),

                # train-IC prediction: [T,Tf]
                "E_traj_prediction": E_traj_prediction["Linf_mean"],
                "E_traj_prediction_std": E_traj_prediction["Linf_std"],

                # # train full rollout: [t0,Tf]
                # "E_traj_full": E_traj_full["Linf_mean"],
                # "E_traj_full_std": E_traj_full["Linf_std"],

                # equation consistency on train data
                "E_res": E_res,
                "E_res_rel": E_res_rel,
                 # Quantities used by Gaussian AIC/BIC
                **information_criteria_data,
            }
        )
    elif split == "validation":
        metrics.update(
            {
                # primary model-selection metric on unseen ICs
                "E_traj_full": E_traj_full["Linf_mean"],
                "E_traj_full_std": E_traj_full["Linf_std"],
            }
        )

    elif split == "test":
        metrics.update(
            {
                # final reporting only
                "E_traj_reconstruction": E_traj_reconstruction["Linf_mean"],
                "E_traj_reconstruction_std": E_traj_reconstruction["Linf_std"],
                "E_traj_prediction": E_traj_prediction["Linf_mean"],
                "E_traj_prediction_std": E_traj_prediction["Linf_std"],
                "E_traj_full": E_traj_full["Linf_mean"],
                "E_traj_full_std": E_traj_full["Linf_std"],
            }
        )

    # else:
    #     metrics.update(
    #         {
    #             "E_traj_reconstruction": E_traj_reconstruction["Linf_mean"],
    #             "E_traj_reconstruction_std": E_traj_reconstruction["Linf_std"],
    #             "E_traj_prediction": E_traj_prediction["Linf_mean"],
    #             "E_traj_prediction_std": E_traj_prediction["Linf_std"],
    #             "E_traj_full": E_traj_full["Linf_mean"],
    #             "E_traj_full_std": E_traj_full["Linf_std"],
    #             "E_res": E_res,
    #             "E_res_rel": E_res_rel,
    #             "E_traj_reconstruction_rel": E_recon_rel,
    #         }
    #     )

    return metrics

    



#----kernel and f  diagnostics----------------------------

def _weighted_l2_norm(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)

    if weights is None:
        return float(np.sqrt(np.mean(values**2)))

    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights / np.sum(weights)

    return float(np.sqrt(np.sum(weights * values**2)))


def model_selection_kernel_diagnostics_fn(
    *,
    fit_result,
    learned_model,
    candidate,
    config,
    features=None,
):
    if features is None:
        raise ValueError("features must be provided for weighted diagnostics.")

    empirical_density = build_empirical_error_density(
        features=features,
        localization_metadata=fit_result.basis_bundle.localization_metadata,
    )

    r_grid = empirical_density.r_grid
    r_eval = r_grid.reshape(-1, 1)

    kernel_weights = np.maximum(
        np.asarray(empirical_density.kernel_weights, dtype=float).reshape(-1),
        0.0,
    )

    if np.sum(kernel_weights) <= 0.0:
        kernel_weights = None

    if candidate.interaction_type == "energy_alignment":
        phi_E = learned_model.learned_energy_kernel(r_eval)
        phi_A = learned_model.learned_alignment_kernel(r_eval)

    elif candidate.interaction_type == "energy":
        phi_E = learned_model.learned_kernel(r_eval)
        phi_A = np.zeros_like(phi_E)

    elif candidate.interaction_type == "alignment":
        phi_A = learned_model.learned_kernel(r_eval)
        phi_E = np.zeros_like(phi_A)

    else:
        phi_E = np.zeros((r_eval.shape[0], 1))
        phi_A = np.zeros_like(phi_E)

    # --------------------------------------------------
    # Environment-force norm diagnostic
    # --------------------------------------------------
    z1_grid = empirical_density.z1_grid
    z2_grid = empirical_density.z2_grid

    if z2_grid is None:
        z_grid = z1_grid.reshape(-1, 1)
    else:
        Z1, Z2 = np.meshgrid(z1_grid, z2_grid, indexing="ij")
        z_grid = np.column_stack([Z1.ravel(), Z2.ravel()])

    f_hat = np.asarray(
        learned_model.learned_environment(z_grid),
        dtype=float,
    )

    f_mag2 = np.sum(f_hat**2, axis=-1).reshape(-1)

    env_weights = np.maximum(
        np.asarray(
            empirical_density.environment_weights,
            dtype=float,
        ).reshape(-1),
        0.0,
    )

    if np.sum(env_weights) <= 0.0:
        f_norm = float(np.sqrt(np.mean(f_mag2)))
    else:
        env_weights = env_weights / np.sum(env_weights)
        f_norm = float(np.sqrt(np.sum(env_weights * f_mag2)))

    return {
        "r_grid": r_grid.reshape(-1),
        "phi_E": np.asarray(phi_E).reshape(-1),
        "phi_A": np.asarray(phi_A).reshape(-1),
        "phi_E_norm": _weighted_l2_norm(phi_E, weights=kernel_weights),
        "phi_A_norm": _weighted_l2_norm(phi_A, weights=kernel_weights),
        "f_norm": f_norm,
        "condition_number": float(fit_result.diagnostics.condition_number),
    }


# helper to compute residual directly from the dataset and learned model------

def _residual_rmse(
        *,
        learned_model,
        dataset,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> float:
        """
        Compute residual RMSE over the specified time interval.
        """
            
        X = np.asarray(dataset.X, dtype=float)
        LHS = np.asarray(dataset.LHS, dtype=float)

        V = None
        if getattr(dataset, "V", None) is not None:
            V = np.asarray(dataset.V, dtype=float)

        if learned_model.system_order == 2 and V is None:
            raise ValueError("Second-order residual evaluation requires dataset.V.")

        M, L, N, D = X.shape
        if end_idx is None:
            end_idx = L

        if not 0 <= start_idx < end_idx <= L:
            raise ValueError(
                "Invalid residual-evaluation interval: "
                f"start_idx={start_idx}, end_idx={end_idx}, L={L}."
            )
        
        residual = np.zeros_like(
            LHS[:, start_idx:end_idx]
        )

        for m in range(M):
            for local_idx, ell in enumerate(
                range(start_idx, end_idx)
            ):
                if learned_model.system_order == 1:
                    rhs = learned_model.lhs(
                        X[m, ell]
                    )
                else:
                    rhs = learned_model.acceleration(
                        X[m, ell],
                        V[m, ell],
                    )

                residual[m, local_idx] = (
                    LHS[m, ell] - rhs
                )

        return float(
            np.sqrt(
                np.mean(
                    np.sum(
                        residual**2,
                        axis=-1,
                    )
                )
            )
        )