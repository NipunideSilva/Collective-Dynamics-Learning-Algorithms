from dataclasses import dataclass
from typing import Literal
from typing import Callable
#Only store basis choices.

@dataclass(frozen=True)
class RadialBasisSpec:
    family: Literal["bspline", "polynomial"] = "bspline"
    degree: int = 1
    n_basis: int = 10
    threshold_quantile: float | None = None
    min_radius_clip: float | None = None


@dataclass
class EnvironmentBasisSpec:
    family: Literal["bspline"] = "bspline"
    degree: int =1
    n_basis_per_dim: int =10

@dataclass(frozen=True)
class BasisConfig:
    radial: RadialBasisSpec
    environment: EnvironmentBasisSpec

    energy: RadialBasisSpec | None = None
    alignment: RadialBasisSpec | None = None

    @property
    def velocity(self) -> EnvironmentBasisSpec:
        """
        Backward compatibility alias for older second-order code.
        # remove once fixed
        """
        return self.environment
    

@dataclass(frozen=True)
class ParametricFunctionSpec:
    name: str
    function: Callable


@dataclass(frozen=True)
class ParametricFBasisSpec:
    terms: tuple[ParametricFunctionSpec, ...]

    @property
    def n_terms(self) -> int:
        return len(self.terms)