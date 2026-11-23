from pathlib import Path
import json
import joblib


class DensityRepository:
    def __init__(self, root="artifacts/densities"):
        self.root = Path(root)

    def save(self,
        dist_est,
        vel_est,
        system_name,
        split,
        name,
        metadata,
        profile_name: str | None = None,
             ):

        if profile_name is None:
            path = self.root / system_name / split / name
        else:
            path = (
                self.root
                / system_name
                / profile_name
                / split
                / name
            )

        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(dist_est, path / "pairwise_density.joblib")
        joblib.dump(vel_est, path / "velocity_density.joblib")

        metadata = dict(metadata)
        metadata["system_name"] = system_name
        metadata["split"] = split
        metadata["density_name"] = name

        if profile_name is not None:
            metadata["profile_name"] = profile_name

        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        return path

    def load(self, system_name, split, name, profile_name: str | None = None):
        if profile_name is None:
            path = self.root / system_name / split / name
        else:
            path = (
                self.root
                / system_name
                / profile_name
                / split
                / name
            )

        dist = joblib.load(path / "pairwise_density.joblib")
        vel = joblib.load(path / "velocity_density.joblib")

        return dist, vel
