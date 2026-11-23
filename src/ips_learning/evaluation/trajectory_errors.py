# Pointwise, reconstruction, and prediction errors
# pointwise trajectory error 
# reconstruction error on training horizon
# prediction error on extended or unseen-IC horizon


import numpy as np


class TrajectoryErrorEvaluator:
    """
    Compute pointwise, reconstruction, and prediction trajectory errors.
    """

    @staticmethod
    def pointwise_error_curve(X_true: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X_true, X_hat : np.ndarray of shape (L, N, D)

        Returns
        -------
        e_t : np.ndarray of shape (L,)
            e(t_l) = sqrt( (1/N) sum_i ||x_i - xhat_i||^2 )
        """
        diff2 = np.sum((X_true - X_hat) ** 2, axis=-1)  # (L, N)
        return np.sqrt(np.mean(diff2, axis=-1)) #(L,)
    

    @staticmethod
    def summarize_horizon(e_t: np.ndarray) -> dict:
        e_t = np.asarray(e_t, dtype=float)

        return {
            "Linf": float(np.max(e_t)),
            "L2_mean": float(np.sqrt(np.mean(e_t ** 2))),# didn't use
            "time_curve": e_t,
        }
    
    def evaluate_batch(
        self,
        t_true: np.ndarray,
        X_true_batch: np.ndarray,
        t_hat: np.ndarray,
        X_hat_batch: np.ndarray,
        split_index: int | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        t_true : (L,)
        X_true_batch : (M, L, N, D)
        t_hat : (L,)
        X_hat_batch : (M, L, N, D)
        split_index : int or None
            If provided, defines:
                reconstruction interval = [:split_index+1]
                prediction interval = [split_index:]
            If None, whole curve is summarized.

        Returns
        -------
        dict
        """
        if not np.allclose(t_true, t_hat):
            raise ValueError("True and learned time grids must match.")

        M = X_true_batch.shape[0]
        all_curves = []

        recon_vals = []
        pred_vals = []

        for m in range(M):
            e_t = self.pointwise_error_curve(X_true_batch[m], X_hat_batch[m])
            all_curves.append(e_t)

            if split_index is None:
                recon = self.summarize_horizon(e_t)
                recon_vals.append(recon["Linf"])
            else:
                recon = self.summarize_horizon(e_t[:split_index + 1])
                pred = self.summarize_horizon(e_t[split_index:])

                recon_vals.append(recon["Linf"])
                pred_vals.append(pred["Linf"])

        results = {
            "pointwise_error_curves": np.array(all_curves, dtype=float),   # (M, L)
            "mean_pointwise_curve": np.mean(np.array(all_curves), axis=0),
            "std_pointwise_curve": np.std(np.array(all_curves), axis=0),
            "reconstruction_Linf_mean": float(np.mean(recon_vals)),
            "reconstruction_Linf_std": float(np.std(recon_vals)),
        }

        if split_index is not None:
            results["prediction_Linf_mean"] = float(np.mean(pred_vals))
            results["prediction_Linf_std"] = float(np.std(pred_vals))

        return results