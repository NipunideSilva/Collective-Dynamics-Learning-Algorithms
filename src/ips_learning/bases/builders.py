from ips_learning.bases.specs import BasisConfig, ParametricFBasisSpec
from ips_learning.bases.domains import DomainEstimator
from ips_learning.bases.radial import BSplineRadialBasisBuilder
from ips_learning.bases.environment import BSplineEnvironmentBasisBuilder
from ips_learning.bases.bundle import BasisBundle
import numpy as np

class BasisBuilder:
    """
    Public Stage 2.2 interface:
    FeatureDataset -> BasisBundle

    Second-order:
        environmental variable = V

    First-order:
        environmental variable = X
    """

    def __init__(self, config: BasisConfig):
        self.config = config
        self.domain_estimator = DomainEstimator()
        self.radial_builder = BSplineRadialBasisBuilder()
        self.environment_builder = BSplineEnvironmentBasisBuilder()

    def build(
            self,
            features,
            parametric_f_basis: ParametricFBasisSpec | None = None,
        ) -> BasisBundle:


        system_order = features.metadata.get("system_order")

        if system_order not in (1, 2):
            raise ValueError(
                f"features.metadata['system_order'] must be 1 or 2, got {system_order}."
            )
        
        domain = self.domain_estimator.estimate_from_features(features)

        radial_basis = self.radial_builder.build(
            degree=self.config.radial.degree,
            n_basis=self.config.radial.n_basis,
            r_min=domain.r_min,
            r_max=domain.r_max,
        )

        energy_spec = self.config.energy or self.config.radial
        alignment_spec = self.config.alignment or self.config.radial

        energy_basis = self.radial_builder.build(
            degree=energy_spec.degree,
            n_basis=energy_spec.n_basis,
            r_min=domain.r_min,
            r_max=domain.r_max,
        )

        alignment_basis = self.radial_builder.build(
            degree=alignment_spec.degree,
            n_basis=alignment_spec.n_basis,
            r_min=domain.r_min,
            r_max=domain.r_max,
        )

        environment_basis = self.environment_builder.build(
            degree=self.config.environment.degree,
            n_basis_per_dim=self.config.environment.n_basis_per_dim,
            domain_min=domain.env_min,
            domain_max=domain.env_max,
        )

        if parametric_f_basis is None:
            parametric_f_terms = []
            n_parametric_f_terms = 0
        else:
            parametric_f_terms = [term.name for term in parametric_f_basis.terms]
            n_parametric_f_terms = len(parametric_f_basis.terms)

        localization_metadata = {
            "system_name": features.metadata.get("system_name"),
            "system_order": system_order,
            "interaction_type": features.metadata.get("interaction_type"),
            "basis_stage": "2.2",

            # What variable the environmental basis was built on
            "environment_variable": domain.environment_variable,

            # Radial domain
            "r_min": domain.r_min,
            "r_max": domain.r_max,

            # Generalized environmental domain
            "env_min": domain.env_min.tolist(),
            "env_max": domain.env_max.tolist(),

            # Basis config
            "radial_family": self.config.radial.family,
            "environment_family": self.config.environment.family,
            "radial_degree": self.config.radial.degree,
            "environment_degree": self.config.environment.degree,
            "n_radial_basis": self.config.radial.n_basis,
            "n_energy_basis": self.config.radial.n_basis,
            "n_alignment_basis": self.config.radial.n_basis,
            "energy_basis_source": "radial_basis",
            "alignment_basis_source": "radial_basis",
            "n_environment_basis_per_dim": self.config.environment.n_basis_per_dim,
            # Knots
            "radial_knots": (
                radial_basis.knots.tolist()
                if radial_basis.knots is not None
                else None
            ),
            "environment_knots": [k.tolist() for k in environment_basis.knots],
            # parametric /non parametric
            "parametric_f_terms": parametric_f_terms,
            "n_parametric_f_terms": n_parametric_f_terms,
            
        }

        return BasisBundle(
            radial_basis=radial_basis,

            # Model-selection full second-order layout.
            # These are logically separate basis slots even if they share
            # the same radial basis object initially.
            energy_basis=energy_basis,
            alignment_basis=alignment_basis,

            environment_basis=environment_basis,
            parametric_f_basis=parametric_f_basis,
            localization_metadata=localization_metadata,
        )
