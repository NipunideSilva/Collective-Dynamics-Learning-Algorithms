#saves rollout artifacts to disk
# A rollout means simulating or executing a sequence of actions through time using a policy or model.

from pathlib import Path
import json
import numpy as np
from ips_learning.rollout.artifacts import LearnedTrajectoryArtifact


class RolloutRepository:
    """
    Save and load learned trajectory artifacts.
    """

    def __init__(self, root_dir: str | Path = "artifacts/rollouts") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _artifact_arrays(artifact: LearnedTrajectoryArtifact) -> dict:
        arrays = {
            "t": artifact.t,
            "X_hat": artifact.X_hat,
            "U0": artifact.U0,
        }

        if artifact.V_hat is not None:
            arrays["V_hat"] = artifact.V_hat

        return arrays
    
    def case_dir(
        self,
        *,
        system_name: str,
        profile_name: str,
        case_name: str,
        scenario: str,
        rollout_name: str,
        algorithm: str = "nonparametric_f",
    ) -> Path:
        
        if algorithm not in {"nonparametric_f", "parametric_f"}:
            raise ValueError(
                "algorithm must be 'nonparametric_f' or 'parametric_f', "
                f"got {algorithm!r}."
            )
        path = (
            self.root_dir
            / system_name
            / profile_name
            / algorithm
            / case_name
            / scenario
            / rollout_name
        )
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def save_case(
        self,
        *,
        artifact: LearnedTrajectoryArtifact,
        system_name: str,
        profile_name: str,
        case_name: str,
        rollout_name: str,
        algorithm: str = "nonparametric_f",
        overwrite: bool = False,
        scenario: str | None = None,
    ) -> Path:
        
        if scenario is None:
            scenario = artifact.metadata.get("scenario", "unknown")

        save_dir = self.case_dir(
            system_name=system_name,
            profile_name=profile_name,
            algorithm=algorithm,
            case_name=case_name,
            scenario=scenario,
            rollout_name=rollout_name,
        )

        npz_path = save_dir / "learned_trajectories.npz"
        json_path = save_dir / "metadata.json"

        if not overwrite and (npz_path.exists() or json_path.exists()):
            raise FileExistsError(
                f"Rollout already exists at {save_dir}. Use overwrite=True to replace it."
            )

        np.savez_compressed(
            npz_path,
            **self._artifact_arrays(artifact),
        )

        metadata = dict(artifact.metadata)
        metadata.setdefault("algorithm", algorithm)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        return save_dir
    
    
    def load_case(
        self,
        *,
        system_name: str,
        profile_name: str,
        case_name: str,
        scenario: str,
        rollout_name: str,
        algorithm: str = "nonparametric_f",
    ) -> LearnedTrajectoryArtifact:
        """
        Load a saved rollout artifact using the generalized case-based layout.

        Path:
            artifacts/rollouts/{system}/{profile}/{algorithm}/{case}/{scenario}/{rollout}/
        """
        load_dir = self.case_dir(
            system_name=system_name,
            profile_name=profile_name,
            algorithm=algorithm,
            case_name=case_name,
            scenario=scenario,
            rollout_name=rollout_name,
        )

        npz_path = load_dir / "learned_trajectories.npz"
        json_path = load_dir / "metadata.json"

        if not npz_path.exists():
            raise FileNotFoundError(f"Missing rollout data file: {npz_path}")

        if not json_path.exists():
            raise FileNotFoundError(f"Missing rollout metadata file: {json_path}")

        data = np.load(npz_path)

        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        V_hat = data["V_hat"] if "V_hat" in data.files else None

        return LearnedTrajectoryArtifact(
            t=data["t"],
            X_hat=data["X_hat"],
            V_hat=V_hat,
            U0=data["U0"],
            metadata=metadata,
        )
    
    def exists_case(
        self,
        *,
        system_name: str,
        profile_name: str,
        case_name: str,
        scenario: str,
        rollout_name: str,
        algorithm: str = "nonparametric_f",
    ) -> bool:
        """
        Check whether a generalized case-based rollout artifact exists.
        artifacts/rollouts/{system}/{profile}/{algorithm}/{case}/{scenario}/{rollout}/
        """
        load_dir = self.case_dir(
            system_name=system_name,
            profile_name=profile_name,
            algorithm=algorithm,
            case_name=case_name,
            scenario=scenario,
            rollout_name=rollout_name,
        )

        return (
            (load_dir / "learned_trajectories.npz").exists()
            and (load_dir / "metadata.json").exists()
        )
            