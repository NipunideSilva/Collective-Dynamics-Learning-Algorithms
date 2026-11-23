from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from ips_learning.bases.bundle import BasisBundle
from ips_learning.bases.radial import RadialBasis
from ips_learning.bases.environment import EnvironmentBasis


class BasisRepository:
    """
    Save/load basis artifacts.

    Directory layouts
    -----------------
    Profile-aware layout:
        artifacts/bases/<system_name>/<profile_name>/<split>/<basis_name>/

    """

    def __init__(self, root_dir: str | Path = "artifacts/bases") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def basis_dir(
        self,
        system_name: str,
        split: str,
        basis_name: str,
        profile_name: str | None = None,
    ) -> Path:
        if profile_name is None:
            path = self.root_dir / system_name / split / basis_name
        else:
            path = self.root_dir / system_name / profile_name / split / basis_name

        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        basis_bundle: BasisBundle,
        system_name: str,
        split: str,
        basis_name: str,
        overwrite: bool = True,
        profile_name: str | None = None,
    ) -> Path:
        out_dir = self.basis_dir(
            system_name=system_name,
            profile_name=profile_name,
            split=split,
            basis_name=basis_name,
        )
        json_path = out_dir / "basis_metadata.json"

        if json_path.exists() and not overwrite:
            raise FileExistsError(json_path)

        metadata = dict(basis_bundle.localization_metadata)

        metadata["system_name"] = system_name
        metadata["split"] = split
        metadata["basis_name"] = basis_name

        if profile_name is not None:
            metadata["profile_name"] = profile_name

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        return out_dir

    def load(
        self,
        system_name: str,
        split: str,
        basis_name: str,
        profile_name: str | None = None,
    ) -> BasisBundle:
        if profile_name is None:
            in_dir = self.root_dir / system_name / split / basis_name
        else:
            in_dir = (
                self.root_dir
                / system_name
                / profile_name
                / split
                / basis_name
            )

        json_path = in_dir / "basis_metadata.json"

        if not json_path.exists():
            raise FileNotFoundError(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        radial_basis = self._rebuild_radial_basis(metadata)
        environment_basis = self._rebuild_environment_basis(metadata)

        return BasisBundle(
            radial_basis=radial_basis,
            environment_basis=environment_basis,
            localization_metadata=metadata,
        )

    def _rebuild_radial_basis(self, metadata: dict) -> RadialBasis:
        family = metadata["radial_family"]
        degree = int(metadata["radial_degree"])
        n_basis = int(metadata["n_radial_basis"])
        r_min = float(metadata["r_min"])
        r_max = float(metadata["r_max"])
        knots = metadata["radial_knots"]

        if family != "bspline":
            raise NotImplementedError(f"Unsupported radial family: {family}")

        knots = np.asarray(knots, dtype=float)
        spline = BSpline(t=knots, c=np.eye(n_basis), k=degree)

        return RadialBasis(
            family=family,
            degree=degree,
            n_basis=n_basis,
            r_min=r_min,
            r_max=r_max,
            evaluator=spline,
            knots=knots,
        )

    def _rebuild_environment_basis(self, metadata: dict) -> EnvironmentBasis:
        family = metadata.get("environment_family", metadata.get("velocity_family"))
        degree = int(metadata.get("environment_degree", metadata.get("velocity_degree")))
        n_basis_per_dim = int(
            metadata.get(
                "n_environment_basis_per_dim",
                metadata.get("n_velocity_basis_per_dim"),
            )
        )

        domain_min = np.asarray(
            metadata.get("env_min", metadata.get("v_min")),
            dtype=float,
        )
        domain_max = np.asarray(
            metadata.get("env_max", metadata.get("v_max")),
            dtype=float,
        )

        knots_list = metadata.get(
            "environment_knots",
            metadata.get("velocity_knots"),
        )

        if family != "bspline":
            raise NotImplementedError(f"Unsupported environment family: {family}")

        if knots_list is None:
            raise KeyError("Missing environment_knots or velocity_knots in metadata.")

        evaluators = []
        knots = []

        for kd in knots_list:
            kd = np.asarray(kd, dtype=float)
            spline = BSpline(t=kd, c=np.eye(n_basis_per_dim), k=degree)
            evaluators.append(spline)
            knots.append(kd)

        return EnvironmentBasis(
            family=family,
            degree=degree,
            n_basis_per_dim=n_basis_per_dim,
            domain_min=domain_min,
            domain_max=domain_max,
            evaluators=evaluators,
            knots=knots,
        )
