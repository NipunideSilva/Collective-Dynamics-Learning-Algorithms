import numpy as np


class KernelErrorEvaluator:
    """
    Compare true and learned interaction kernels.

    Supports:
    - grid-based evaluation via r_grid
    - sample-based evaluation via r_samples: monte carlo
    """

    def evaluate(
        self,
        true_kernel,
        learned_kernel,
        r_grid: np.ndarray | None = None,
        r_samples: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        include_radial_measure: bool = True, # This is false for kuramoto or adapt to kuramoto
    ) -> dict:
        if r_samples is not None:
            # Did not amend run_evalution.py to pass r_samples, but keep this option here for future use.
            r_eval = np.asarray(r_samples, dtype=float).reshape(-1, 1)#MOnte Carlo
            mode = "samples"
        elif r_grid is not None:
            r_eval = np.asarray(r_grid, dtype=float).reshape(-1, 1)
            mode = "grid"
        else:
            raise ValueError("Provide either r_grid or r_samples.")

        phi_true = np.asarray(true_kernel(r_eval)).reshape(-1)
        phi_hat = np.asarray(learned_kernel(r_eval)).reshape(-1)

        diff2 = (phi_true - phi_hat) ** 2
        true2 = phi_true ** 2
       

        if include_radial_measure:
            diff2 = diff2 * (r_eval[:, 0] ** 2)
            true2 = true2 * (r_eval[:, 0] ** 2)

        if weights is None:
            e = np.sqrt(np.mean(diff2))
            w = None
            denom = np.sqrt(np.mean(true2))
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != diff2.shape[0]:
                raise ValueError("weights must have same length as evaluation points.")
            w = w / np.sum(w)
            e = np.sqrt(np.sum(w * diff2))
            denom = np.sqrt(np.sum(w * true2))

        eps = 1e-14
        e_relative = e / max(denom, eps)

        return {
            "E_phi": float(e),
            "E_phi_relative": float(e_relative),
            "mode": mode,
            "r_eval": r_eval[:, 0],
            "phi_true": phi_true,
            "phi_hat": phi_hat,
            "weights": w,
        }


class EnvironmentErrorEvaluator:
    """
    Compare true and learned environment/drift functions.

    Second-order:
        compares f(V)

    First-order:
        compares f(X)

    Supports:
    - sample-based evaluation via Z_samples
    - 1D grid via z1_grid
    - 2D grid via z1_grid, z2_grid

    Backward-compatible aliases:
    - V_samples -> Z_samples
    - v1_grid, v2_grid -> z1_grid, z2_grid
    """

    def evaluate(
        self,
        true_env,
        learned_env,
        z1_grid: np.ndarray | None = None,
        z2_grid: np.ndarray | None = None,
        Z_samples: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        # backward-compatible aliases
        v1_grid: np.ndarray | None = None,
        v2_grid: np.ndarray | None = None,
        V_samples: np.ndarray | None = None,
    ) -> dict:
        if Z_samples is None and V_samples is not None:
            Z_samples = V_samples

        if z1_grid is None and v1_grid is not None:
            z1_grid = v1_grid

        if z2_grid is None and v2_grid is not None:
            z2_grid = v2_grid

        if Z_samples is not None:
            Z_eval = np.asarray(Z_samples, dtype=float)

            if Z_eval.ndim != 2:
                raise ValueError("Z_samples must have shape (n_samples, D).")

            mode = "samples"
            grid_shape = None
            z1_grid = None
            z2_grid = None

        else:
            if z1_grid is None:
                raise ValueError(
                    "Provide either Z_samples or at least z1_grid."
                )

            z1_grid = np.asarray(z1_grid, dtype=float).reshape(-1)

            if z2_grid is None:
                # 1D environment grid
                Z_eval = z1_grid.reshape(-1, 1)
                mode = "grid_1d"
                grid_shape = z1_grid.shape
            else:
                # 2D environment grid
                z2_grid = np.asarray(z2_grid, dtype=float).reshape(-1)
                Z1, Z2 = np.meshgrid(z1_grid, z2_grid, indexing="ij")
                Z_eval = np.column_stack([Z1.ravel(), Z2.ravel()])
                mode = "grid_2d"
                grid_shape = Z1.shape

        f_true = np.asarray(true_env(Z_eval), dtype=float)
        f_hat = np.asarray(learned_env(Z_eval), dtype=float)

        if f_true.shape != f_hat.shape:
            raise ValueError(
                f"true_env and learned_env outputs must match, got "
                f"{f_true.shape} and {f_hat.shape}."
            )

        if f_true.ndim != 2:
            raise ValueError(
                f"Environment outputs must have shape (n_eval, D), got {f_true.shape}."
            )

        output_dim = f_true.shape[1]

        diff2 = np.sum((f_true - f_hat) ** 2, axis=-1)
        true2 = np.sum(f_true ** 2, axis=-1)

        if weights is None:
            e = np.sqrt(np.mean(diff2))
            w = None
            denom = np.sqrt(np.mean(true2))
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != diff2.shape[0]:
                raise ValueError("weights must have same length as evaluation points.")
            w = w / np.sum(w)
            e = np.sqrt(np.sum(w * diff2))
            denom = np.sqrt(np.sum(w * true2))

        eps = 1e-14
        e_relative = e / max(denom, eps)

        result = {
            "E_env": float(e),
            "E_env_relative": float(e_relative),
            "mode": mode,
            "Z_eval": Z_eval,
            "f_true": f_true,
            "f_hat": f_hat,
            "weights": w,

            # backward-compatible alias
            "V_eval": Z_eval,
        }

        if mode == "grid_1d":
            result.update(
                {
                    "z1_grid": z1_grid,
                    "f_true_grid": f_true.reshape(grid_shape + (output_dim,)),
                    "f_hat_grid": f_hat.reshape(grid_shape + (output_dim,)),
                    "weights_grid": None if w is None else w.reshape(grid_shape),

                    # backward-compatible alias
                    "v1_grid": z1_grid,
                }
            )

        if mode == "grid_2d":
            result.update(
                {
                    "z1_grid": z1_grid,
                    "z2_grid": z2_grid,
                    "f_true_grid": f_true.reshape(grid_shape + (output_dim,)),
                    "f_hat_grid": f_hat.reshape(grid_shape + (output_dim,)),
                    "weights_grid": None if w is None else w.reshape(grid_shape),

                    # backward-compatible aliases
                    "v1_grid": z1_grid,
                    "v2_grid": z2_grid,
                }
            )

        return result