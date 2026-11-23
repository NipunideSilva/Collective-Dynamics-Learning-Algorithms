from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ips_learning.features.feature_builder import FeatureDataset


class FeatureRepository:

    """
    Save/load processed FeatureDataset artifacts.

    Directory layouts
    -----------------
    Profile-aware layout:
        artifacts/features/<system_name>/<profile_name>/<split>/<feature_name>/

    """


    def __init__(self, root_dir: str | Path = "artifacts/features"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def feature_dir(
        self,
        system_name: str,
        split: str,
        feature_name: str,
        profile_name: str | None = None,
    ) -> Path:
        if profile_name is None:
            path = self.root_dir / system_name / split / feature_name
        else:
            path = self.root_dir / system_name / profile_name / split / feature_name

        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        features: FeatureDataset,
        system_name: str,
        split: str,
        feature_name: str,
        overwrite: bool = True,
        profile_name: str | None = None,
    ) -> Path:
        out_dir = self.feature_dir(
            system_name=system_name,
            profile_name=profile_name,
            split=split,
            feature_name=feature_name,
        )

        npz_path = out_dir / "features.npz"
        json_path = out_dir / "metadata.json"

        if npz_path.exists() and not overwrite:
            raise FileExistsError(npz_path)

        arrays = {
            "t": features.t,
            "X": features.X,
            "LHS": features.LHS,
            "R": features.R,
            "R_D": features.R_D,
        }

        if features.V is not None:
            arrays["V"] = features.V

        if features.VR_D is not None:
            arrays["VR_D"] = features.VR_D

        np.savez_compressed(npz_path, **arrays)

        metadata = dict(features.metadata)

        metadata["system_name"] = system_name
        metadata["split"] = split
        metadata["feature_name"] = feature_name

        if profile_name is not None:
            metadata["profile_name"] = profile_name

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        return out_dir

    def load(
        self,
        system_name: str,
        split: str,
        feature_name: str,
        profile_name: str | None = None,
    ) -> FeatureDataset:
        if profile_name is None:
            in_dir = self.root_dir / system_name / split / feature_name
        else:
            in_dir = (
                self.root_dir
                / system_name
                / profile_name
                / split
                / feature_name
            )

        data = np.load(in_dir / "features.npz", allow_pickle=True)

        with open(in_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        V = data["V"] if "V" in data.files else None
        VR_D = data["VR_D"] if "VR_D" in data.files else None

        return FeatureDataset(
            t=data["t"],
            X=data["X"],
            V=V,
            LHS=data["LHS"],
            R=data["R"],
            R_D=data["R_D"],
            VR_D=VR_D,
            metadata=metadata,
        )