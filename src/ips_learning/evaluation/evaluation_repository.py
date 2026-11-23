from __future__ import annotations

from pathlib import Path
import pickle
import json
import numpy as np


class EvaluationRepository:
    """
    Save/load run one baseline evaluation bundle.
    """

    def __init__(self, root_dir: str | Path = "artifacts/evaluation") -> None:
        self.root_dir = Path(root_dir)

    def case_dir(
        self,
        system_name: str,
        profile_name: str,
        M: int,
        L: int,
    ) -> Path:
        path = (
            self.root_dir
            / system_name
            / profile_name
            / f"M_{M:03d}_L_{L:03d}"
            / "evaluation"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        *,
        system_name: str,
        profile_name: str,
        M: int,
        L: int,
        report_recon,
        metrics_extended_train,
        metrics_test,
        overwrite: bool = True,
    ) -> Path:
        out_dir = self.case_dir(system_name, profile_name, M, L)
        path = out_dir / "evaluation_bundle.pkl"

        if path.exists() and not overwrite:
            raise FileExistsError(path)

        bundle = {
            "report_recon": report_recon,
            "metrics_extended_train": metrics_extended_train,
            "metrics_test": metrics_test,
        }

        with open(path, "wb") as f:
            pickle.dump(bundle, f)

        # unnecessary duplication of data, but allows for quick access to scalars without loading the full bundle
        summary = {}

        array_payload = {}

        for group_name, group_metrics in {
            "report_recon": report_recon,
            "metrics_extended_train": metrics_extended_train,
            "metrics_test": metrics_test,
        }.items():
            if hasattr(group_metrics, "__dict__"):
                group_metrics = group_metrics.__dict__

            if isinstance(group_metrics, dict):
                scalars, arrays = self._split_metric_dict(group_metrics)
                summary[group_name] = scalars
                for k, v in arrays.items():
                    array_payload[f"{group_name}__{k}"] = v
            else:
                summary[group_name] = str(group_metrics)

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        if array_payload:
            np.savez_compressed(out_dir / "arrays.npz", **array_payload)

        return path

    def load(
        self,
        *,
        system_name: str,
        profile_name: str,
        M: int,
        L: int,
    ) -> dict:
        in_dir = self.case_dir(system_name, profile_name, M, L)
        path = in_dir / "evaluation_bundle.pkl"

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "rb") as f:
            return pickle.load(f)

    def exists(
        self,
        *,
        system_name: str,
        profile_name: str,
        M: int,
        L: int,
    ) -> bool:
        path = (
            self.root_dir
            / system_name
            / profile_name
            / f"M_{M:03d}_L_{L:03d}"
            / "evaluation"
            / "evaluation_bundle.pkl"
        )
        return path.exists()
    
    def _split_metric_dict(self, metrics: dict) -> tuple[dict, dict]:
        scalars = {}
        arrays = {}

        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            elif np.isscalar(value) or isinstance(value, (str, int, float, bool)) or value is None:
                scalars[key] = value
            else:
                try:
                    arr = np.asarray(value)
                    if arr.ndim > 0:
                        arrays[key] = arr
                    else:
                        scalars[key] = arr.item()
                except Exception:
                    scalars[key] = str(value)

        return scalars, arrays
