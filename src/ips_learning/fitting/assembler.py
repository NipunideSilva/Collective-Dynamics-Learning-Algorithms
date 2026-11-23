# aggregates trajectory-wise blocks into normal equations.

import numpy as np
from joblib import Parallel, delayed

from ips_learning.fitting.design_blocks import DesignBlockBuilder
from ips_learning.fitting.fit_result import FitResult
from ips_learning.fitting.solver import LinearSystemSolver


class NormalEquationAssembler:
    """
    Assemble Stage 2.3 normal equations from trajectory-wise design blocks.
    """

    def __init__(
        self,
        parallel_jobs: int = 1,
        interaction_type: str = "energy",
        algorithm: str = "nonparametric_f",
        include_environment: bool = True,
    ):
        if algorithm not in {"nonparametric_f", "parametric_f"}:
            raise ValueError(
                "algorithm must be 'nonparametric_f' or 'parametric_f', "
                f"got {algorithm!r}."
            )

        self.parallel_jobs = parallel_jobs
        self.interaction_type = interaction_type
        self.algorithm = algorithm
        self.include_environment = include_environment

        self.block_builder = DesignBlockBuilder(
            interaction_type=interaction_type,
            algorithm=algorithm,
            include_environment=include_environment,
        )

    def fit(
        self,
        features,
        basis_bundle,
        solver: LinearSystemSolver,
        *,
        system_name: str | None = None,
        ) -> FitResult:
         
        """
            Stage 2.3 main entry point for first- and second-order systems.

            First-order:
                dot X = f(X) + f_phi(X)

            Second-order:
                ddot X = f(V) + f_phi(X)
        """
        feature_metadata = getattr(features, "metadata", {})
        system_name = system_name or feature_metadata.get("system_name", "unknown")

        system_order = feature_metadata.get("system_order")
        if system_order not in (1, 2):
            raise ValueError(
                f"features.metadata['system_order'] must be 1 or 2, got {system_order}."
            )

        
        if self.algorithm == "parametric_f":
            if basis_bundle.parametric_f_basis is None:
                raise ValueError(
                    "algorithm='parametric_f' requires basis_bundle.parametric_f_basis."
                )

            parametric_f_terms = [
                term.name for term in basis_bundle.parametric_f_basis.terms
            ]
        else:
            parametric_f_terms = []
        
        # prevents accidentally fitting Levi with an energy block or SPP with an alignment block.
        feature_interaction_type = feature_metadata.get("interaction_type")

        is_model_selection = bool(feature_metadata.get("model_selection_mode", False))

        if feature_interaction_type is not None and not is_model_selection:
            if feature_interaction_type != self.interaction_type:
                raise ValueError(
                    "Feature interaction_type does not match assembler interaction_type: "
                    f"features={feature_interaction_type!r}, "
                    f"assembler={self.interaction_type!r}."
                )
        if system_order == 1 and self.interaction_type in {"alignment", "energy_alignment"}:
            raise ValueError("First-order fitting only supports energy interaction.")

        X = np.asarray(features.X)
        LHS = np.asarray(features.LHS)
        R = np.asarray(features.R)
        R_D = np.asarray(features.R_D)

        if system_order == 2:
            if features.V is None:
                raise ValueError("Second-order fitting requires features.V.")
            V = np.asarray(features.V)
        else:
            V = None

        VR_D = getattr(features, "VR_D", None)
        if VR_D is not None:
            VR_D = np.asarray(VR_D)

        M = X.shape[0]

        if self.parallel_jobs > 1:
            results = Parallel(n_jobs=self.parallel_jobs)(
                delayed(self._build_one_normal_equation)(
                    X[m],
                    V[m] if V is not None else None,
                    LHS[m],
                    R[m],
                    R_D[m],
                    VR_D[m] if VR_D is not None else None,
                    basis_bundle,
                    system_order,
                )
                for m in range(M)
            )
        else:
            results = [
                self._build_one_normal_equation(
                    X[m],
                    V[m] if V is not None else None,
                    LHS[m],
                    R[m],
                    R_D[m],
                    VR_D[m] if VR_D is not None else None,
                    basis_bundle,
                    system_order,
                )
                for m in range(M)
            ]

        A = None
        b = None

        for A_m, b_m in results:
            if A is None:
                A = A_m.copy()
                b = b_m.copy()
            else:
                A += A_m
                b += b_m

        A /= float(M)
        b /= float(M)

        theta, diagnostics = solver.solve(A, b)

        if self.interaction_type == "energy_alignment":
            energy_basis = getattr(basis_bundle, "energy_basis", None)
            alignment_basis = getattr(basis_bundle, "alignment_basis", None)

            if energy_basis is None:
                energy_basis = basis_bundle.radial_basis
            if alignment_basis is None:
                alignment_basis = basis_bundle.radial_basis

            n_energy = energy_basis.n_basis
            n_alignment = alignment_basis.n_basis
            n_radial = n_energy + n_alignment
            n_interaction = n_radial
        else:
            n_single = basis_bundle.radial_basis.n_basis
            n_energy = n_single if self.interaction_type == "energy" else 0
            n_alignment = n_single if self.interaction_type == "alignment" else 0
            n_radial = n_single
            n_interaction = n_single

        if self.include_environment:
            n_env_total = A.shape[0] - n_interaction
        else:
            n_env_total = 0


        algorithm = self.algorithm
        environment_variable = (
            "position" if system_order == 1 else "velocity"
        )

        model_family = (
            f"first_order_{algorithm}"
            if system_order == 1
            else f"second_order_{algorithm}"
        )

        if system_order == 1:
            if self.include_environment:
                model_form = "dot_X = f(X) + f_phi_E(X)"
            else:
                model_form = "dot_X = f_phi_E(X)"

        elif self.interaction_type == "energy_alignment":
            if self.include_environment:
                model_form = (
                    "ddot_X = f(V) + f_phi_E(X) + f_phi_A(X,V)"
                )
            else:
                model_form = (
                    "ddot_X = f_phi_E(X) + f_phi_A(X,V)"
                )

        elif self.interaction_type == "alignment":
            if self.include_environment:
                model_form = "ddot_X = f(V) + f_phi_A(X,V)"
            else:
                model_form = "ddot_X = f_phi_A(X,V)"

        else:  # energy
            if self.include_environment:
                model_form = "ddot_X = f(V) + f_phi_E(X)"
            else:
                model_form = "ddot_X = f_phi_E(X)"

        metadata = {
            "stage": "2.3",
            "model_family": model_family,
            "model_form": model_form,
            "system_name": system_name,
            "system_order": system_order,
            "interaction_type": self.interaction_type,
            "include_environment": bool(self.include_environment),
            "environment_variable": environment_variable,
            "algorithm": algorithm,
            "phi_representation": "nonparametric",
            "f_representation": (
                "nonparametric" if self.algorithm == "nonparametric_f"
                else "parametric"
            ),
            "n_trajectories": int(M),
            "n_time_points": int(X.shape[1]),
            "N": int(X.shape[2]),
            "D": int(X.shape[3]),
            "n_radial_basis": int(n_radial),
            "n_energy_basis": int(n_energy),
            "n_alignment_basis": int(n_alignment),
            "n_interaction_basis": int(n_interaction),
            "theta_layout": (
                ["theta_phi_E", "theta_phi_A", "theta_f"]
                if self.interaction_type == "energy_alignment"
                else ["theta_phi", "theta_f"]
            ),

            "feature_metadata": feature_metadata,
            # backward-compatible key
            "n_environment": int(n_env_total),

            "localization_metadata": basis_bundle.localization_metadata,
        }

        if algorithm == "parametric_f":
            if basis_bundle.parametric_f_basis is None:
                raise ValueError(
                    "algorithm='parametric_f' requires basis_bundle.parametric_f_basis."
                )

            metadata.update(
                {
                    "parametric_f_terms": parametric_f_terms,
                    "n_parametric_f_terms": basis_bundle.parametric_f_basis.n_terms,
                    "theta_f_order": "componentwise",
                }
            )

        return FitResult(
            theta=theta,
            n_radial=n_radial,
            n_energy =n_energy,
            n_alignment = n_alignment,
            n_environment=n_env_total,
            basis_bundle=basis_bundle,
            A=A,
            b=b,
            diagnostics=diagnostics,
            metadata=metadata,
        )
    

    def _build_one_normal_equation(
        self,
        X_m,
        V_m,
        LHS_m,
        R_m,
        R_D_m,
        VR_D_m,
        basis_bundle,
        system_order: int,
    ):
        """
        Build one trajectory-wise normal-equation contribution.

        Returns
        -------
        A_m : np.ndarray
        b_m : np.ndarray
        """
        Psi_m, y_m = self.block_builder.build_for_one_trajectory(
            X=X_m,
            V=V_m,
            LHS=LHS_m,
            R=R_m,
            R_D=R_D_m,
            VR_D=VR_D_m,
            basis_bundle=basis_bundle,
            system_order=system_order,
        )

        L = X_m.shape[0]
        N = X_m.shape[1]
        scale = 1.0 / (L * N)

        A_m = scale * (Psi_m.T @ Psi_m)
        b_m = scale * (Psi_m.T @ y_m)

        return A_m, b_m
