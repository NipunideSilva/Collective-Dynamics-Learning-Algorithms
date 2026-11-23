import json
from pathlib import Path
import numpy as np


class ArtifactExporter:
    """
    Save configuration and summary outputs.
    """

    @staticmethod
    def save_json(path: str | Path, obj: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

    @staticmethod
    def save_npz(path: str | Path, **arrays) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)