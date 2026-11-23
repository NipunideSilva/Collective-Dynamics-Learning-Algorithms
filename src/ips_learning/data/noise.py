from __future__ import annotations

import numpy as np
from ips_learning.data.dataset_repository import RawTrajectoryDataset


class AdditiveGaussianNoise:
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

        X_noisy = noisy(dataset.X)
        V_noisy =  None if dataset.V is None else noisy(dataset.V)
        LHS_noisy = noisy(dataset.LHS)
        # LHS_noisy = dataset.LHS  # Don't add noise to LHS since it's used for learning

        return RawTrajectoryDataset(
            t=dataset.t,
            X=X_noisy,
            V=V_noisy,
            LHS=LHS_noisy,
            U0=dataset.U0,
            metadata={
                **dataset.metadata,
                "noise_model": noise_model,
                "noise_type": self.noise_type,
                "noise_level": float(zeta),
            },
        )