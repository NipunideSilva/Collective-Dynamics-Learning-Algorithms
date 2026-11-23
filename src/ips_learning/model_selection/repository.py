from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ModelSelectionRepository:
    """
    Save/load model-selection candidate artifacts.

    This repository stores only model-selection-level artifacts:
    - candidate metrics
    - candidate metadata
    - model-selection summaries
    - selected model record
    - website-ready tables

    Expensive fit_result and rollout artifacts should be saved by their
    own repositories and referenced here by path.
    """

    def __init__(self, root_dir: str | Path = "artifacts/model_selection") -> None:
        self.root_dir = Path(root_dir)

    def experiment_dir(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        create: bool = True,
    ) -> Path:
        path = self.root_dir / system_name / profile_name / dataset_name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def candidate_dir(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str,
        create: bool = True,
    ) -> Path:
        path = (
            self.experiment_dir(
                system_name=system_name,
                profile_name=profile_name,
                dataset_name=dataset_name,
                create=create,
            )
            / "candidates"
            / candidate_name
        )
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def summaries_dir(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        create: bool = True,
    ) -> Path:
        path = (
            self.experiment_dir(
                system_name=system_name,
                profile_name=profile_name,
                dataset_name=dataset_name,
                create=create,
            )
            / "summaries"
        )
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def figures_dir(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        create: bool = True,
    ) -> Path:
        path = (
            self.experiment_dir(
                system_name=system_name,
                profile_name=profile_name,
                dataset_name=dataset_name,
                create=create,
            )
            / "figures"
        )
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save_candidate_metadata(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str,
        metadata: dict[str, Any],
    ) -> Path:
        out_dir = self.candidate_dir(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
            candidate_name=candidate_name,
        )

        path = out_dir / "candidate_metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._json_safe(metadata),
                f,
                indent=2,
                allow_nan=False,
            )

        return path

    def load_candidate_metadata(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str,
    ) -> dict[str, Any]:
        path = (
            self.candidate_dir(
                system_name=system_name,
                profile_name=profile_name,
                dataset_name=dataset_name,
                candidate_name=candidate_name,
                create=False,
            )
            / "candidate_metadata.json"
        )

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_candidate_metrics(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str,
        metrics: dict[str, Any],
        split: str = "validation",
    ) -> Path:
        out_dir = self.candidate_dir(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
            candidate_name=candidate_name,
        )

        path = out_dir / f"{split}_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._json_safe(metrics),
                f,
                indent=2,
                allow_nan=False,
            )

        return path

    def load_candidate_metrics(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str,
        split: str = "validation",
    ) -> dict[str, Any]:
        path = (
            self.candidate_dir(
                system_name=system_name,
                profile_name=profile_name,
                dataset_name=dataset_name,
                candidate_name=candidate_name,
                create=False,
            )
            / f"{split}_metrics.json"
        )

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_candidate_references(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str,
        references: dict[str, str],
    ) -> Path:
        """
        Save pointers to expensive artifacts produced elsewhere.

        Example:
            {
                "fit_result": "artifacts/fits/...",
                "validation_rollout": "artifacts/rollouts/...",
                "test_rollout": "artifacts/rollouts/..."
            }
        """
        out_dir = self.candidate_dir(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
            candidate_name=candidate_name,
        )

        path = out_dir / "artifact_references.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._json_safe(references),
                f,
                indent=2,
                allow_nan=False,
            )

        return path

    def load_candidate_references(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str,
    ) -> dict[str, str]:
        path = (
            self.candidate_dir(
                system_name=system_name,
                profile_name=profile_name,
                dataset_name=dataset_name,
                candidate_name=candidate_name,
                create=False,
            )
            / "artifact_references.json"
        )

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_kernel_diagnostics(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str,
        diagnostics: dict[str, np.ndarray | float],
    ) -> Path:
        out_dir = self.candidate_dir(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
            candidate_name=candidate_name,
        )

        path = out_dir / "kernel_diagnostics.npz"
        np.savez_compressed(path, **diagnostics)
        return path

    def load_kernel_diagnostics(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str,
    ) -> dict[str, Any]:
        path = (
            self.candidate_dir(
                system_name=system_name,
                profile_name=profile_name,
                dataset_name=dataset_name,
                candidate_name=candidate_name,
                create=False,
            )
            / "kernel_diagnostics.npz"
        )

        data = np.load(path)
        return {key: data[key] for key in data.files}

    def save_summary_table(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        rows: list[dict[str, Any]],
        file_stem: str = "model_selection_summary",
    ) -> tuple[Path, Path]:
        out_dir = self.summaries_dir(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
        )

        df = pd.DataFrame(rows)

        csv_path = out_dir / f"{file_stem}.csv"
        tex_path = out_dir / f"{file_stem}.tex"

        df.to_csv(csv_path, index=False)
        df.to_latex(tex_path, index=False)

        return csv_path, tex_path

    def load_summary_table(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        file_stem: str = "model_selection_summary",
    ) -> pd.DataFrame:
        path = (
            self.summaries_dir(
                system_name=system_name,
                profile_name=profile_name,
                dataset_name=dataset_name,
                create=False,
            )
            / f"{file_stem}.csv"
        )

        return pd.read_csv(path)

    def save_selected_model(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        selected: dict[str, Any],
    ) -> Path:
        out_dir = self.summaries_dir(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
        )

        path = out_dir / "selected_model.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._json_safe(selected),
                f,
                indent=2,
                allow_nan=False,
            )

        return path

    def load_selected_model(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
    ) -> dict[str, Any]:
        path = (
            self.summaries_dir(
                system_name=system_name,
                profile_name=profile_name,
                dataset_name=dataset_name,
                create=False,
            )
            / "selected_model.json"
        )

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    @staticmethod
    def _json_safe(obj):
        import math
        import numpy as np
        import pandas as pd

        if obj is None:
            return None

        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj

        if isinstance(obj, (np.floating,)):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return None
            return val

        if isinstance(obj, (np.integer,)):
            return int(obj)

        if isinstance(obj, dict):
            return {
                str(key): ModelSelectionRepository._json_safe(value)
                for key, value in obj.items()
            }

        if isinstance(obj, list):
            return [
                ModelSelectionRepository._json_safe(value)
                for value in obj
            ]

        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass

        return obj