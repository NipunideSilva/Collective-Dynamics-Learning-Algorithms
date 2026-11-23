from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pickle


class NormalSystemRepository:
    """
    Save/load Stage 2.3 normal-equation artifacts.

    Stores:
    - A
    - b
    - theta
    - block matrices
    - diagnostics
    - metadata

    theta_phi_E / theta_phi_A / theta_f for full second-order models
    theta_phi / theta_f for single-interaction models
    """

    def __init__(self, root_dir: str | Path = "artifacts/fits") -> None:
        self.root_dir = Path(root_dir)

    def case_dir(
        self,
        *,
        system_name: str,
        profile_name: str,
        algorithm: str = "nonparametric_f",
        fit_case_name: str | None = None,
        M: int | None = None,
        L: int | None = None,
    ) -> Path:
        """
        Resolve artifact directory for a fitted system.

        Two supported modes:

        Named case:
            fit_case_name="dataset_train_fit_trial_001"

        Sweep case:
            M=20, L=101
        """
        if algorithm not in {"nonparametric_f", "parametric_f"}:
            raise ValueError(
                "algorithm must be 'nonparametric_f' or 'parametric_f', "
                f"got {algorithm!r}."
            )

        # algorithm = metadata.get("algorithm", "nonparametric_f")
        base = self.root_dir / system_name / profile_name / algorithm

        if fit_case_name is not None:
            path = base / fit_case_name
            path.mkdir(parents=True, exist_ok=True)
            return path

        if M is not None and L is not None:
            path = base / f"M_{M:03d}_L_{L:03d}"
            path.mkdir(parents=True, exist_ok=True) 
            return path

        raise ValueError(
            "Provide either fit_case_name OR both M and L."
        )

    def save(
        self,
        *,
        system_name: str,
        profile_name: str,
        A: np.ndarray,
        b: np.ndarray,
        theta: np.ndarray,
        diagnostics: dict,
        metadata: dict,
        fit_case_name: str | None = None,
        M: int | None = None,
        L: int | None = None,
        overwrite: bool = True,

        n_environment: int | None = None,

        n_radial: int | None = None,

        n_energy: int | None = None,
        n_alignment: int | None = None,
    ) -> Path:
        
        interaction_type = metadata.get("interaction_type", "energy")

        out_dir = self.case_dir(
                system_name=system_name,
                profile_name=profile_name,
                algorithm=metadata.get("algorithm", "nonparametric_f"),
                fit_case_name=fit_case_name,
                M=M,
                L=L,
            )
    

        normal_path = out_dir / "normal_system.npz"
        coef_path = out_dir / "coefficients.npz"
        diag_path = out_dir / "diagnostics.json"
        meta_path = out_dir / "metadata.json"

        if normal_path.exists() and not overwrite:
            raise FileExistsError(normal_path)

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        theta = np.asarray(theta, dtype=float)


        
        if interaction_type == "energy_alignment":
            if n_energy is None or n_alignment is None:
                raise ValueError(
                    "energy_alignment save requires n_energy and n_alignment."
                )

            n_energy = int(n_energy)
            n_alignment = int(n_alignment)
            n_interaction = n_energy + n_alignment
            n_radial = n_interaction

        else:
            if n_radial is None:
                raise ValueError("save requires n_radial for single-interaction fits.")

            n_radial = int(n_radial)
            n_interaction = n_radial
            n_energy = n_radial if interaction_type == "energy" else 0
            n_alignment = n_radial if interaction_type == "alignment" else 0

        if n_environment is None:
            n_environment = theta.shape[0] - n_interaction

        n_environment = int(n_environment)

        expected = n_interaction + n_environment
        if theta.shape[0] != expected:
            raise ValueError(
                f"theta has length {theta.shape[0]}, expected {expected} "
                f"= n_interaction({n_interaction}) + n_environment({n_environment})."
            )

        if A.shape != (theta.shape[0], theta.shape[0]):
            raise ValueError(
                f"A shape {A.shape} is inconsistent with theta length {theta.shape[0]}."
            )

        if interaction_type == "energy_alignment":

            theta_phi_E = theta[:n_energy]
            theta_phi_A = theta[n_energy:n_interaction]
            theta_f = theta[n_interaction:]

            A_phi_E = A[:n_energy, :n_energy]
            A_phi_A = A[n_energy:n_interaction, n_energy:n_interaction]
            A_f = A[n_interaction:, n_interaction:]

            A_phi_E_phi_A = A[:n_energy, n_energy:n_interaction]
            A_phi_A_phi_E = A[n_energy:n_interaction, :n_energy]
            A_phi_E_f = A[:n_energy, n_interaction:]
            A_f_phi_E = A[n_interaction:, :n_energy]
            A_phi_A_f = A[n_energy:n_interaction, n_interaction:]
            A_f_phi_A = A[n_interaction:, n_energy:n_interaction]

            np.savez_compressed(
                normal_path,
                A=A,
                b=b,
                A_phi_E=A_phi_E,
                A_phi_A=A_phi_A,
                A_f=A_f,
                A_phi_E_phi_A=A_phi_E_phi_A,
                A_phi_A_phi_E=A_phi_A_phi_E,
                A_phi_E_f=A_phi_E_f,
                A_f_phi_E=A_f_phi_E,
                A_phi_A_f=A_phi_A_f,
                A_f_phi_A=A_f_phi_A,
            )

            np.savez_compressed(
                coef_path,
                theta=theta,
                theta_phi_E=theta_phi_E,
                theta_phi_A=theta_phi_A,
                theta_f=theta_f,
            )

        else:
            n_interaction = n_radial

            theta_phi = theta[:n_radial]
            theta_f = theta[n_radial:]

            A_phi = A[:n_radial, :n_radial]
            A_f = A[n_radial:, n_radial:]
            A_phi_f = A[:n_radial, n_radial:]
            A_f_phi = A[n_radial:, :n_radial]

            np.savez_compressed(
                normal_path,
                A=A,
                b=b,
                A_phi=A_phi,
                A_f=A_f,
                A_phi_f=A_phi_f,
                A_f_phi=A_f_phi,    
            )
            np.savez_compressed(
                coef_path,
                theta=theta,
                theta_phi=theta_phi,
                theta_f=theta_f,
            )

        # Not necessary but added  for completeness and to avoid confusion about what diagnostics and metadata are associated with the fit.
        metadata = dict(metadata)
        metadata["n_radial"] = int(n_radial)
        metadata["n_interaction"] = int(n_interaction)
        metadata["n_energy"] = int(n_energy)
        metadata["n_alignment"] = int(n_alignment)
        metadata["n_environment"] = int(n_environment)
        metadata["theta_layout"] = (
            ["theta_phi_E", "theta_phi_A", "theta_f"]
            if interaction_type == "energy_alignment"
            else ["theta_phi", "theta_f"]
        )


        metadata.setdefault(
            "environment_variable",
            "velocity" if metadata.get("system_order") == 2 else "position",
        )
        metadata["artifact_type"] = "fit"

        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return out_dir

    def load(
        self,
        *,
        system_name: str,
        profile_name: str,
        fit_case_name: str | None = None,
        M: int | None = None,
        L: int | None = None,
        algorithm: str = "nonparametric_f",
    ) -> dict:
        in_dir = self.case_dir(
                system_name=system_name,
                profile_name=profile_name,
                algorithm=algorithm,
                fit_case_name=fit_case_name,
                M=M,
                L=L,
            )

        normal = np.load(in_dir / "normal_system.npz")
        coefs = np.load(in_dir / "coefficients.npz")

        with open(in_dir / "diagnostics.json", "r", encoding="utf-8") as f:
            diagnostics = json.load(f)

        with open(in_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if "n_environment" not in metadata and "n_velocity" in metadata:
            metadata["n_environment"] = metadata["n_velocity"]

        if "environment_variable" not in metadata:
            metadata["environment_variable"] = (
                "velocity" if metadata.get("system_order") == 2 else "position"
            )

        result = {
            "A": normal["A"],
            "b": normal["b"],
            "theta": coefs["theta"],
            "theta_f": coefs["theta_f"],
            "diagnostics": diagnostics,
            "metadata": metadata,
        }

        interaction_type = metadata.get("interaction_type", "energy")

        if interaction_type == "energy_alignment":
            result.update(
                {
                    "A_phi_E": normal["A_phi_E"],
                    "A_phi_A": normal["A_phi_A"],
                    "A_f": normal["A_f"],
                    "theta_phi_E": coefs["theta_phi_E"],
                    "theta_phi_A": coefs["theta_phi_A"],
                }
            )

            for key in [
                "A_phi_E_phi_A",
                "A_phi_A_phi_E",
                "A_phi_E_f",
                "A_f_phi_E",
                "A_phi_A_f",
                "A_f_phi_A",
            ]:
                if key in normal:
                    result[key] = normal[key]

        else:
            result.update(
                {
                    "A_phi": normal["A_phi"],
                    "A_f": normal["A_f"],
                    "A_phi_f": normal["A_phi_f"],
                    "A_f_phi": normal["A_f_phi"],
                    "theta_phi": coefs["theta_phi"],
                }
            )

        return result
    
    @staticmethod
    def _diagnostics_to_dict(diag) -> dict:
        out = {}
        for key in [
            "method",
            "condition_number",
            "rank",
            "min_singular",
            "max_singular",
            "ridge_lambda",
            "svd_tol",
        ]:
            if hasattr(diag, key):
                val = getattr(diag, key)
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                out[key] = val

        if hasattr(diag, "singular_values"):
            sv = getattr(diag, "singular_values")
            out["singular_values"] = np.asarray(sv).tolist()

        return out
    


    #-----------------------------------------------------------------------------------
    # want to do this  with .npz + .json because it  safer for long-term artifact-first workflows. but I am lazy 
    # and want to be able to dump the whole fit result in one go,
    #  so adding these helper methods for now. can remove later if we decide we don't need them.
    # highly doubt that.
    def save_fit_result(
        self,
        *,
        system_name: str,
        profile_name: str,
        fit_result,
        fit_case_name: str | None = None,
        M: int | None = None,
        L: int | None = None,
        algorithm: str | None = None,
        overwrite: bool = True,
    ) -> Path:
        
        algorithm = algorithm or fit_result.metadata.get(
        "algorithm",
        "nonparametric_f",
    )
        out_dir = self.case_dir(
                system_name=system_name,
                profile_name=profile_name,
                algorithm=algorithm,
                fit_case_name=fit_case_name,
                M=M,
                L=L,
            )
        fit_path = out_dir / "fit_result.pkl"

        if fit_path.exists() and not overwrite:
            raise FileExistsError(fit_path)

        with open(fit_path, "wb") as f:
            pickle.dump(fit_result, f)

        return fit_path


    def load_fit_result(
        self,
        *,
        system_name: str,
        profile_name: str,
        fit_case_name: str | None = None,
        M: int | None = None,       
        L: int | None = None,
        algorithm: str = "nonparametric_f",
        ):
        in_dir = self.case_dir(
                system_name=system_name,
                profile_name=profile_name,
                algorithm=algorithm,
                fit_case_name=fit_case_name,
                M=M,
                L=L,
            )
        fit_path = in_dir / "fit_result.pkl"

        if not fit_path.exists():
            raise FileNotFoundError(fit_path)

        with open(fit_path, "rb") as f:
            return pickle.load(f)
        

# repo.save(
#     system_name="kuramoto",
#     profile_name="baseline",
#     fit_case_name="dataset_train_fit_trial_000",
#     A=fit_result.A,
#     b=fit_result.b,
#     theta=fit_result.theta,
#     n_radial=fit_result.n_radial,
#     n_environment=fit_result.n_environment,
#     diagnostics=repo._diagnostics_to_dict(fit_result.diagnostics),
#     metadata=fit_result.metadata,
# )