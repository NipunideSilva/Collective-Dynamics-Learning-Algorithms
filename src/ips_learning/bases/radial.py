#radial basis builders and evaluators

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline


@dataclass
class RadialBasis:
    family: str
    degree: int
    n_basis: int
    r_min: float
    r_max: float
    evaluator: callable
    knots: np.ndarray | None = None

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        r : np.ndarray
            Shape (...,) or (..., 1)

        Returns
        -------
        np.ndarray
            Shape (..., n_basis)
        """
        r = np.asarray(r, dtype=float)
        if r.ndim > 0 and r.shape[-1] == 1:
            out = self.evaluator(r[..., 0])
        else:
            out = self.evaluator(r)
        return np.asarray(out, dtype=float)


class BSplineRadialBasisBuilder:
    def build(self, degree: int, n_basis: int, r_min: float, r_max: float) -> RadialBasis:
        knots = self._setup_knots(r_min, r_max, degree, n_basis)
        spline = BSpline(t=knots, c=np.eye(n_basis), k=degree,)

        return RadialBasis(
            family="bspline",
            degree=degree,
            n_basis=n_basis,
            r_min=r_min,
            r_max=r_max,
            evaluator=spline,
            knots=knots,
        )

    @staticmethod
    def _setup_knots(x_min: float, x_max: float, deg: int, n: int) -> np.ndarray:
        x_interior = np.linspace(x_min, x_max, n - deg + 1)
        all_knots = np.concatenate([
            [x_interior[0]] * deg,
            x_interior,
            [x_interior[-1]] * deg,
        ])
        return all_knots