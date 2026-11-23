from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.neighbors import KernelDensity


DEFAULT_KERNEL_GRID_SIZE = 300
DEFAULT_ENVIRONMENT_GRID_SIZE = 120


@dataclass
class ObservedSupport1D:
    """
    Finite-sample observed support in 1D.

    Attributes
    ----------
    samples : np.ndarray of shape (n, 1)
    min_value : float
    max_value : float
    """
    samples: np.ndarray
    min_value: float
    max_value: float


@dataclass
class ObservedSupport2D:
    """
    Finite-sample observed support in 2D.

    Attributes
    ----------
    samples : np.ndarray of shape (n, 2)
    min_value : np.ndarray of shape (2,)
    max_value : np.ndarray of shape (2,)
    """
    samples: np.ndarray
    min_value: np.ndarray
    max_value: np.ndarray


@dataclass
class DensityEstimate1D:
    """
    Reusable 1D KDE estimate.
    """
    samples: np.ndarray
    bandwidth: float
    kernel: str
    kde: KernelDensity

    def evaluate(self, grid: np.ndarray) -> np.ndarray:
        grid = np.asarray(grid, dtype=float).reshape(-1, 1)
        return np.exp(self.kde.score_samples(grid))


@dataclass
class DensityEstimate2D:
    """
    Reusable 2D KDE estimate.
    """
    samples: np.ndarray
    bandwidth: float
    kernel: str
    kde: KernelDensity

    def evaluate_on_grid(self, x1_grid: np.ndarray, x2_grid: np.ndarray) -> np.ndarray:
        x1_grid = np.asarray(x1_grid, dtype=float).reshape(-1)
        x2_grid = np.asarray(x2_grid, dtype=float).reshape(-1)

        X1, X2 = np.meshgrid(x1_grid, x2_grid, indexing="ij")
        pts = np.column_stack([X1.ravel(), X2.ravel()])
        pdf = np.exp(self.kde.score_samples(pts)).reshape(X1.shape)
        return pdf


class SampleExtractor:
    @staticmethod
    def pairwise_scalar_samples(R):
        R = np.asarray(R, dtype=float)

        if R.ndim != 5:
            raise ValueError(f"Expected R shape (M,L,N,N,q), got {R.shape}.")

        if R.shape[-1] != 1:
            raise ValueError("Pairwise scalar density requires last dimension 1.")

        _, _, N, _, _ = R.shape
        iu, ju = np.triu_indices(N, k=1)

        samples = np.abs(R[:, :, iu, ju, 0]).reshape(-1, 1)
        return samples

    @staticmethod
    def state_samples(Z):
        Z = np.asarray(Z, dtype=float)

        if Z.ndim != 4:
            raise ValueError(f"Expected Z shape (M,L,N,D), got {Z.shape}.")

        return Z.reshape(-1, Z.shape[-1])
    

class DensityEstimator1D:
    @staticmethod
    def silverman_bandwidth(samples):
        x = np.asarray(samples, dtype=float).reshape(-1)
        n = x.size
        std = np.std(x, ddof=1)

        if n < 2 or std == 0:
            return 1.0

        return 1.06 * std * n ** (-1.0 / 5.0)

    def fit(self, samples, bandwidth=None, kernel="gaussian"):
        samples = np.asarray(samples, dtype=float).reshape(-1, 1)

        if bandwidth is None:
            bandwidth = self.silverman_bandwidth(samples)

        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        kde.fit(samples)

        return DensityEstimate1D(
            samples=samples,
            bandwidth=float(bandwidth),
            kernel=kernel,
            kde=kde,
        )
    
class DensityEstimator2D:
    @staticmethod
    def default_bandwidth(samples):
        samples = np.asarray(samples, dtype=float)

        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("samples must have shape (n, 2).")

        std_mean = np.std(samples, axis=0, ddof=1).mean()

        if std_mean == 0:
            return 1.0

        return 0.2 * std_mean

    def fit(self, samples, bandwidth=None, kernel="gaussian"):
        samples = np.asarray(samples, dtype=float)

        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("DensityEstimator2D requires samples shape (n,2).")

        if bandwidth is None:
            bandwidth = self.default_bandwidth(samples)

        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        kde.fit(samples)

        return DensityEstimate2D(
            samples=samples,
            bandwidth=float(bandwidth),
            kernel=kernel,
            kde=kde,
        )

class KernelDensityEstimator(DensityEstimator1D):
    def fit_from_R(self, R, bandwidth=None, kernel="gaussian"):
        samples = SampleExtractor.pairwise_scalar_samples(R)
        return self.fit(samples, bandwidth=bandwidth, kernel=kernel)



class EnvironmentDensityEstimator:
    def fit_from_Z(self, Z, bandwidth=None, kernel="gaussian"):
        samples = SampleExtractor.state_samples(Z)
        D = samples.shape[1]

        if D == 1:
            return DensityEstimator1D().fit(
                samples,
                bandwidth=bandwidth,
                kernel=kernel,
            )

        if D == 2:
            return DensityEstimator2D().fit(
                samples,
                bandwidth=bandwidth,
                kernel=kernel,
            )

        raise NotImplementedError(f"Only D=1 or D=2 supported, got D={D}.")
    






@dataclass
class EmpiricalErrorDensity:
    """
    Empirical density grids for weighted function-recovery errors.

    Densities are estimated directly from the training features currently
    used by the learning pipeline. Returned density values are left
    unnormalized; downstream error evaluators normalize the weights.
    """

    r_grid: np.ndarray
    kernel_weights: np.ndarray
    z1_grid: np.ndarray
    z2_grid: np.ndarray | None
    environment_weights: np.ndarray


def build_empirical_error_density(
    features,
    localization_metadata: dict,
    *,
    kernel_grid_size: int = DEFAULT_KERNEL_GRID_SIZE,
    environment_grid_size: int = DEFAULT_ENVIRONMENT_GRID_SIZE,
    bandwidth=None,
    kernel: str = "gaussian",
) -> EmpiricalErrorDensity:
    """
    Construct empirical error-weighting densities from current training features.

    Pairwise interaction weights are estimated from features.R.

    Environmental-force weights are estimated from:
    - features.X for first-order systems,
    - features.V for second-order systems.
    """
    if kernel_grid_size < 2:
        raise ValueError("kernel_grid_size must be at least 2.")

    if environment_grid_size < 2:
        raise ValueError("environment_grid_size must be at least 2.")

    loc = localization_metadata

    # --------------------------------------------------
    # Pairwise-distance empirical density
    # --------------------------------------------------
    r_min = float(loc["r_min"])
    r_max = float(loc["r_max"])

    r_grid = np.linspace(
        r_min,
        r_max,
        kernel_grid_size,
    )

    kernel_density = KernelDensityEstimator().fit_from_R(
        features.R,
        bandwidth=bandwidth,
        kernel=kernel,
    )

    kernel_weights = kernel_density.evaluate(r_grid)

    # --------------------------------------------------
    # Environmental-state empirical density
    # --------------------------------------------------
    env_min_raw = loc.get("env_min", loc.get("v_min"))
    env_max_raw = loc.get("env_max", loc.get("v_max"))

    if env_min_raw is None or env_max_raw is None:
        raise KeyError(
            "localization_metadata must contain env_min/env_max "
            "or v_min/v_max."
        )

    env_min = np.asarray(env_min_raw, dtype=float).reshape(-1)
    env_max = np.asarray(env_max_raw, dtype=float).reshape(-1)

    if env_min.shape != env_max.shape:
        raise ValueError("Environment bounds must have matching shapes.")

    system_order = features.metadata.get("system_order")

    if system_order == 1:
        Z = np.asarray(features.X, dtype=float)

    elif system_order == 2:
        if features.V is None:
            raise ValueError(
                "Second-order empirical environment density requires features.V."
            )
        Z = np.asarray(features.V, dtype=float)

    else:
        raise ValueError(
            "features.metadata['system_order'] must be 1 or 2, "
            f"got {system_order!r}."
        )

    state_dimension = Z.shape[-1]

    if env_min.size != state_dimension:
        raise ValueError(
            "Environment bounds do not match feature dimension: "
            f"{env_min.size} != {state_dimension}."
        )

    environment_density = EnvironmentDensityEstimator().fit_from_Z(
        Z,
        bandwidth=bandwidth,
        kernel=kernel,
    )

    z1_grid = np.linspace(
        env_min[0],
        env_max[0],
        environment_grid_size,
    )

    if state_dimension == 1:
        z2_grid = None
        environment_weights = environment_density.evaluate(z1_grid)

    elif state_dimension == 2:
        z2_grid = np.linspace(
            env_min[1],
            env_max[1],
            environment_grid_size,
        )

        environment_weights = environment_density.evaluate_on_grid(
            z1_grid,
            z2_grid,
        ).reshape(-1)

    else:
        raise NotImplementedError(
            "Empirical error density currently supports D=1 or D=2, "
            f"got D={state_dimension}."
        )

    return EmpiricalErrorDensity(
        r_grid=np.asarray(r_grid, dtype=float).reshape(-1),
        kernel_weights=np.asarray(
            kernel_weights,
            dtype=float,
        ).reshape(-1),
        z1_grid=np.asarray(z1_grid, dtype=float).reshape(-1),
        z2_grid=(
            None
            if z2_grid is None
            else np.asarray(z2_grid, dtype=float).reshape(-1)
        ),
        environment_weights=np.asarray(
            environment_weights,
            dtype=float,
        ).reshape(-1),
    )
