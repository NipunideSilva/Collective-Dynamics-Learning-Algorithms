from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline


@dataclass
class EnvironmentBasis:
    family: str
    degree: int
    knots: list[np.ndarray]
    domain_min: np.ndarray
    domain_max: np.ndarray
    n_basis_per_dim: int
    evaluators: list

    def evaluate_per_dim(self, Z: np.ndarray) -> list[np.ndarray]:
        """
        Parameters
        ----------
        Z : np.ndarray of shape (..., D)

        Second-order:
            Z = V

        First-order:
            Z = X

        Returns
        -------
        list[np.ndarray]
            One array per dimension, each of shape (..., n_basis_per_dim).
        """
        Z = np.asarray(Z, dtype=float)

        if Z.shape[-1] != len(self.evaluators):
            raise ValueError(
                f"Expected last dimension {len(self.evaluators)}, got {Z.shape[-1]}."
            )

        outs = []
        for d in range(Z.shape[-1]):
            outs.append(np.asarray(self.evaluators[d](Z[..., d]), dtype=float))

        return outs


class BSplineEnvironmentBasisBuilder:
    def build(
        self,
        degree: int,
        n_basis_per_dim: int,
        domain_min: np.ndarray,
        domain_max: np.ndarray,
    ) -> EnvironmentBasis:
        domain_min = np.asarray(domain_min, dtype=float)
        domain_max = np.asarray(domain_max, dtype=float)

        if domain_min.shape != domain_max.shape:
            raise ValueError(
                f"domain_min and domain_max must have same shape, got "
                f"{domain_min.shape} and {domain_max.shape}."
            )

        evaluators = []
        knots = []

        for d in range(domain_min.shape[0]):
            kd = self._setup_knots(
                domain_min[d],
                domain_max[d],
                degree,
                n_basis_per_dim,
            )
            spline = BSpline(
                t=kd,
                c=np.eye(n_basis_per_dim),
                k=degree,
            )
            evaluators.append(spline)
            knots.append(kd)

        return EnvironmentBasis(
            family="bspline",
            degree=degree,
            n_basis_per_dim=n_basis_per_dim,
            domain_min=domain_min,
            domain_max=domain_max,
            evaluators=evaluators,
            knots=knots,
        )

    @staticmethod
    def _setup_knots(
        x_min: float,
        x_max: float,
        deg: int,
        n: int,
    ) -> np.ndarray:
        if n <= deg:
            raise ValueError(
                f"n_basis_per_dim={n} must be greater than degree={deg}."
            )

        if not np.isfinite(x_min) or not np.isfinite(x_max):
            raise ValueError("Domain bounds must be finite.")

        if x_max <= x_min:
            pad = 1e-6 if x_min == 0.0 else 1e-6 * abs(x_min)
            x_min = x_min - pad
            x_max = x_max + pad

        x_interior = np.linspace(x_min, x_max, n - deg + 1)

        return np.concatenate(
            [
                [x_interior[0]] * deg,
                x_interior,
                [x_interior[-1]] * deg,
            ]
        )


# Backward compatibility for old second-order code
BSplineVelocityBasisBuilder = BSplineEnvironmentBasisBuilder