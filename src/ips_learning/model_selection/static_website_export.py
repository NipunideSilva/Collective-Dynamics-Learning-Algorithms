from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


class StaticModelSelectionWebsiteExporter:
    """
    Export model-selection results into a static GitHub Pages structure.

    Output:
        docs/assets/model_selection/{system_name}/
            manifest.json
            selected_model.json
            summary_table.csv
            repeated_trials.csv
            figures/
    """

    def __init__(
        self,
        *,
        artifact_root: str | Path = "artifacts",
        output_root: str | Path = "docs/assets/model_selection",
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.output_root = Path(output_root)

    def export_system(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
    ) -> Path:
        src_root = (
            self.artifact_root
            / "model_selection"
            / system_name
            / profile_name
            / dataset_name
        )

        if not src_root.exists():
            raise FileNotFoundError(src_root)

        out_dir = self.output_root / system_name
        out_dir.mkdir(parents=True, exist_ok=True)

        summaries_dir = src_root / "summaries"
        figures_dir = src_root / "figures"

        selected_src = summaries_dir / "selected_model.json"
        summary_src = summaries_dir / "model_selection_summary.csv"

        selected_dst = out_dir / "selected_model.json"
        summary_dst = out_dir / "summary_table.csv"

        self._copy_required(selected_src, selected_dst)
        self._copy_required(summary_src, summary_dst)

        if figures_dir.exists():
            self._copy_tree(figures_dir, out_dir / "figures")

        repeated_summary_names = [
            "selection_frequency",
            "order_frequency",
            "interaction_frequency",
            "environment_frequency",
            "selected_metric_summary",
            "candidate_metric_summary",
            "failure_summary",
        ]

        for name in repeated_summary_names:
            src = (
                self.artifact_root
                / "model_selection"
                / system_name
                / profile_name
                / "repeated_trials"
                / "summaries"
                / f"{name}.csv"
            )

            dst = out_dir / f"{name}.csv"

            if src.exists():
                shutil.copy2(src, dst)

        manifest = self._build_manifest(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
            out_dir=out_dir,
        )

        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                self._json_safe(manifest),
                f,
                indent=2,
                allow_nan=False,
            )

        return out_dir

    def export_all(
        self,
        systems: list[dict[str, str]],
    ) -> dict[str, Path]:
        outputs = {}

        for item in systems:
            out_dir = self.export_system(
                system_name=item["system_name"],
                profile_name=item["profile_name"],
                dataset_name=item["dataset_name"],
            )
            outputs[item["system_name"]] = out_dir

        index_manifest = {
            "systems": [
                {
                    "system_name": name,
                    "manifest": f"{name}/manifest.json",
                }
                for name in outputs
            ]
        }

        self.output_root.mkdir(parents=True, exist_ok=True)

        with open(self.output_root / "index.json", "w", encoding="utf-8") as f:
           json.dump(
                self._json_safe(index_manifest),
                f,
                indent=2,
                allow_nan=False,
            )
        return outputs

    def _build_manifest(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        out_dir: Path,
    ) -> dict[str, Any]:
        selected_model = self._load_json_if_exists(out_dir / "selected_model.json")

        figures = []
        fig_dir = out_dir / "figures"

        if fig_dir.exists():
            for path in sorted(fig_dir.rglob("*")):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
                    figures.append(str(path.relative_to(out_dir)))

        summary_records = []
        summary_path = out_dir / "summary_table.csv"

        if summary_path.exists():
            summary_records = pd.read_csv(summary_path).to_dict(orient="records")

        repeated_trial_tables = {}

        for name in [
            "selection_frequency",
            "order_frequency",
            "interaction_frequency",
            "environment_frequency",
            "selected_metric_summary",
            "candidate_metric_summary",
            "failure_summary",
        ]:
            path = out_dir / f"{name}.csv"

            if path.exists():
                repeated_trial_tables[name] = (
                    pd.read_csv(path)
                    .to_dict(orient="records")
                )

        return {
            "system_name": system_name,
            "profile_name": profile_name,
            "dataset_name": dataset_name,
            "selected_model": selected_model,
            "summary_table": "summary_table.csv",
            "repeated_trial_tables": {
                                        name: f"{name}.csv"
                                        for name in repeated_trial_tables
                                    },
            "summary_records": summary_records,
            "repeated_trial_records":repeated_trial_tables,
            "figures": figures,
        }

    @staticmethod
    def _copy_required(src: Path, dst: Path) -> None:
        if not src.exists():
            raise FileNotFoundError(src)

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    @staticmethod
    def _copy_tree(src: Path, dst: Path) -> None:
        if dst.exists():
            shutil.rmtree(dst)

        shutil.copytree(src, dst)

    @staticmethod
    def _load_json_if_exists(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}

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
                str(key): StaticModelSelectionWebsiteExporter._json_safe(value)
                for key, value in obj.items()
            }

        if isinstance(obj, list):
            return [
                StaticModelSelectionWebsiteExporter._json_safe(value)
                for value in obj
            ]

        if pd.isna(obj):
            return None

        return obj