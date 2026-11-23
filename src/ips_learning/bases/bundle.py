# packages the localized bases together.

from dataclasses import dataclass
from ips_learning.bases.specs import ParametricFBasisSpec


@dataclass
class BasisBundle:
    radial_basis: object | None = None

    # Used for second-order energy + alignment models.
    # If these are None, downstream code may fall back to radial_basis.
    energy_basis: object | None = None
    alignment_basis: object | None = None

    environment_basis: object | None = None
    parametric_f_basis: ParametricFBasisSpec | None = None
    localization_metadata: dict | None = None