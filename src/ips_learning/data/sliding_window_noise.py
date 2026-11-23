from __future__ import annotations

import numpy as np
from ips_learning.data.dataset_repository import RawTrajectoryDataset
from scipy.ndimage import uniform_filter1d

class SlidingWindowNoise:
    """
    Apply controlled noise to trajectory datasets.

    Supported noise types:
        additive_gaussian:
            A_noisy = A + zeta * std(A) * eta,
            eta ~ N(0, 1)

        multiplicative_uniform:
            A_noisy = A * (1 + epsilon),
            epsilon ~ Uniform[-zeta, zeta]
    """

    def __init__(
        self,
        noise_level: float,
        seed: int = 0,
        noise_type: str = "additive_gaussian",
    ) -> None:
        if noise_type not in {"additive_gaussian", "multiplicative_uniform"}:
            raise ValueError(
                "noise_type must be 'additive_gaussian' or "
                f"'multiplicative_uniform', got {noise_type!r}."
            )

        self.noise_level = float(noise_level)
        self.rng = np.random.default_rng(seed)
        self.noise_type = noise_type

    def perturb(self, dataset: RawTrajectoryDataset) -> RawTrajectoryDataset:
        """
    Add noise to X, then smooth X along the time axis using a moving average.

    Works for both:
        first-order systems:  V is None, LHS = dX/dt
        second-order systems: V exists, LHS = dV/dt
    """
        zeta = self.noise_level

        if self.noise_type == "additive_gaussian":

            def noisy(A):
                scale = np.std(A)
                return A + zeta * scale * self.rng.normal(size=A.shape)

            noise_model = "additive_gaussian_scaled_by_global_std"

        else:

            def noisy(A):
                eps = self.rng.uniform(
                    -zeta,
                    zeta,
                    size=A.shape,
                )
                return A * (1.0 + eps)

            noise_model = "multiplicative_uniform_relative"
            # --------------------------------------------------
    # 1. Add observation noise to positions/states.
    # --------------------------------------------------

        WINDOW_SIZES = {}
        # --- Moving Average Filters Added ---
        # 1. Inject noise into X
        X_noisy = noisy(dataset.X)

        # Define your smoothing window size (must be an odd integer, e.g., 5, 7, 11)
        WINDOW_SIZE = self.estimate_dataset_window(X_noisy, max_window=11)
        WINDOW_SIZES["X"] = WINDOW_SIZE

        # 2. Smooth X_noisy BEFORE taking any gradients to suppress noise amplification
        X_smoothed = uniform_filter1d(
                        X_noisy,
                        size=WINDOW_SIZE,
                        axis=1,
                        mode="nearest",
                    )
        if dataset.V is None:
            # --------------------------------------------------
            # First-order system:
            #   observed state = X
            #   LHS = dX/dt
            #
            # Example: Kuramoto.
            # --------------------------------------------------
            V_smoothed = None

            LHS_raw = np.gradient(
                X_smoothed,
                dataset.t,
                axis=1,
                edge_order=2,
            )
            WINDOW_SIZE = self.estimate_dataset_window(LHS_raw, max_window=11)
            WINDOW_SIZES["LHS"] = WINDOW_SIZE

            LHS_smoothed = uniform_filter1d(
                        LHS_raw,
                        size=WINDOW_SIZE,
                        axis=1,
                        mode="nearest",
                    )

        else:
            # --------------------------------------------------
            # Second-order system:
            #   observed state = (X, V)
            #   LHS = dV/dt
            #
            # We estimate V from smoothed X, then estimate acceleration
            # from smoothed V.
            # --------------------------------------------------
            V_from_X = np.gradient(
                X_smoothed,
                dataset.t,
                axis=1,
                edge_order=2,
            )
            WINDOW_SIZE = self.estimate_dataset_window(V_from_X, max_window=11)
            WINDOW_SIZES["V"] = WINDOW_SIZE

            V_smoothed = uniform_filter1d(
                        V_from_X,
                        size=WINDOW_SIZE,
                        axis=1,
                        mode="nearest",
                    )

            LHS_raw = np.gradient(
                V_smoothed,
                dataset.t,
                axis=1,
                edge_order=2,
            )
            WINDOW_SIZE = self.estimate_dataset_window(LHS_raw, max_window=11)
            WINDOW_SIZES["LHS"] = WINDOW_SIZE

            LHS_smoothed = uniform_filter1d(
                        LHS_raw,
                        size=WINDOW_SIZE,
                        axis=1,
                        mode="nearest",
                    )

        smoothed_dataset = RawTrajectoryDataset(
                    t=dataset.t,
                    X=X_smoothed,
                    V=V_smoothed,
                    LHS=LHS_smoothed,
                    U0=dataset.U0,
                    metadata={
                        **dataset.metadata,
                        "noise_model": noise_model,
                        "noise_type": self.noise_type,
                        "noise_level": float(zeta),
                        "smoothing_windows": WINDOW_SIZES,
                    },
                )           


        return  smoothed_dataset

    
    def estimate_dataset_window(
        self,
        X_noisy: np.ndarray,
        max_window: int = 11,
    ) -> int:
        """
        Estimate one smoothing window for the whole dataset.

        The estimate is based on a representative subset of scalar time series
        X[m, :, n, d]. This avoids choosing a different smoothing strength for
        every trajectory/agent/component.
        """
        M, L, N, D = X_noisy.shape

        candidate_windows = [w for w in range(3, max_window + 1, 2)]
        scores = []

        # Use a small representative subset to keep this cheap.
        m_ids = range(min(M, 5))
        n_ids = range(min(N, 5))
        d_ids = range(D)

        for window in candidate_windows:
            window_scores = []

            for m in m_ids:
                for n in n_ids:
                    for d in d_ids:
                        signal = X_noisy[m, :, n, d]

                        smoothed = uniform_filter1d(
                                signal,
                                size=window,
                                mode="nearest",
                            )
                        residual = signal - smoothed

                        if residual.size < 3 or np.std(residual) < 1e-14:
                            continue

                        corr = np.corrcoef(residual[:-1], residual[1:])[0, 1]

                        if np.isfinite(corr):
                            window_scores.append(abs(corr))

            if window_scores:
                scores.append(float(np.mean(window_scores)))
            else:
                scores.append(np.inf)

        best_idx = int(np.argmin(scores))
        return candidate_windows[best_idx]