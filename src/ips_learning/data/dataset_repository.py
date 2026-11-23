from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

@dataclass
class RawTrajectoryDataset:
    """
    Canonical raw trajectory dataset.

    Attributes
    ----------
    t : np.ndarray of shape (L,)
        Common time grid.
    X : np.ndarray of shape (M, L, N, D)
        Positions.
    V : np.ndarray of shape (M, L, N, D)
        Velocities.
    LHS : np.ndarray of shape (M, L, N, D)
        Exact supervised targets.
    U0 : np.ndarray of shape (M, state_dim)
        Initial conditions used to generate the trajectories.
    metadata : dict
        Human-readable metadata describing the dataset.
    """
    t: np.ndarray
    X: np.ndarray
    LHS: np.ndarray
    U0: np.ndarray
    metadata: dict
    V: np.ndarray | None = None

class DatasetRepository:
    """
    Repository for saving and loading raw trajectory datasets.

    Directory layouts
    -----------------
    Profile-aware layout:
        artifacts/datasets/<system_name>/<profile_name>/<split>/<dataset_name>/

    Legacy layout:
        artifacts/datasets/<system_name>/<split>/<dataset_name>/

    The legacy layout is retained for backward compatibility when
    profile_name is not provided.
    """

    def __init__(self, root_dir: str | Path = "artifacts/datasets") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def dataset_dir(
        self,
        system_name: str,
        split: str,
        dataset_name: str,
        profile_name: str | None = None,
    ) -> Path:
        """
        Return the directory path for a dataset.

        Parameters
        ----------
        system_name : str
            Name of the system, e.g. "spp".
        split : str
            Dataset split, e.g. "train", "test_interp", "test_extrap".
        dataset_name : str
            User-defined dataset name.
        profile_name : str or None
           Optional dataset-generation profile. When provided, artifacts are
            stored under an additional profile-specific directory.

        Returns
        -------
        Path
            Dataset directory.
        """
        if profile_name is None:
            path = self.root_dir / system_name / split / dataset_name
        else:
            path = (
                self.root_dir
                / system_name
                / profile_name
                / split
                / dataset_name
            )

        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        dataset: RawTrajectoryDataset,
        system_name: str,
        split: str,
        dataset_name: str,
        overwrite: bool = False,
        profile_name: str | None = None,
    ) -> Path:
        """
        Save a raw trajectory dataset.

        Parameters
        ----------
        dataset : RawTrajectoryDataset
            Dataset to save.
        system_name : str
            Name of the system.
        split : str
            Dataset split.
        dataset_name : str
            Dataset identifier.
        overwrite : bool, default=False
            Whether to overwrite an existing dataset.
        profile_name : str or None
            Optional dataset-generation profile. When provided, artifacts are
            stored under an additional profile-specific directory.

        Returns
        -------
        Path
            Path to the saved dataset directory.
        """
        self._validate(dataset)

        save_dir = self.dataset_dir(
            system_name=system_name,
            profile_name=profile_name,
            split=split,
            dataset_name=dataset_name,
        )

        npz_path = save_dir / "trajectories.npz"
        json_path = save_dir / "metadata.json"

        if not overwrite and (npz_path.exists() or json_path.exists()):
            raise FileExistsError(
                f"Dataset already exists at {save_dir}. "
                "Use overwrite=True if you want to replace it."
            )

        arrays = {
            "t": dataset.t,
            "X": dataset.X,
            "LHS": dataset.LHS,
            "U0": dataset.U0,
        }

        if dataset.V is not None:
            arrays["V"] = dataset.V

        np.savez_compressed(npz_path, **arrays)

        metadata = dict(dataset.metadata)
        metadata["system_name"] = system_name
        metadata["split"] = split
        metadata["dataset_name"] = dataset_name

        if profile_name is not None:
            metadata["profile_name"] = profile_name

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        return save_dir

    def load(
        self,
        system_name: str,
        split: str,
        dataset_name: str,
        profile_name: str | None = None
    ) -> RawTrajectoryDataset:
        """
        Load a raw trajectory dataset.

        Parameters
        ----------
        system_name : str
            Name of the system.
        split : str
            Dataset split.
        dataset_name : str
            Dataset identifier.
        profile_name : str or None
            Optional dataset-generation profile. When provided, artifacts are
            loaded from an additional profile-specific directory.

        Returns
        -------
        RawTrajectoryDataset
            Loaded dataset.
        """
        if profile_name is None:
            load_dir = self.root_dir / system_name / split / dataset_name
        else:
            load_dir = (
                self.root_dir
                / system_name
                / profile_name
                / split
                / dataset_name
            )

        npz_path = load_dir / "trajectories.npz"
        json_path = load_dir / "metadata.json"

        if not npz_path.exists():
            raise FileNotFoundError(f"Missing trajectories file: {npz_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {json_path}")

        arrays = np.load(npz_path)
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        dataset = RawTrajectoryDataset(
            t=arrays["t"],
            X=arrays["X"],
            V=arrays["V"] if "V" in arrays.files else None,
            LHS=arrays["LHS"],
            U0=arrays["U0"],
            metadata=metadata,
        )


        self._validate(dataset)
        return dataset

    def list_datasets(
        self,
        system_name: str | None = None,
        split: str | None = None,
        profile_name: str | None = None,
    ) -> list[Path]:
        """
        List saved dataset directories.

        Parameters
        ----------
        system_name : str or None
            If provided, restrict listing to one system.
        split : str or None
            If provided, restrict listing to one split.
        profile_name : str or None
            If provided, restrict listing to one dataset profile.
            Requires system_name.

        Returns
        -------
        list[Path]
            Sorted list of dataset directories.
        """
        if profile_name is not None and system_name is None:
            raise ValueError(
                "system_name must be provided when profile_name is specified."
            )

        base = self.root_dir

        if system_name is not None:
            base = base / system_name

        if profile_name is not None:
            base = base / profile_name

        if split is not None:
            base = base / split

        if not base.exists():
            return []

        dirs = []
        for candidate in base.rglob("*"):
            if candidate.is_dir():
                if (
                    (candidate / "trajectories.npz").exists()
                    and (candidate / "metadata.json").exists()
                ):
                    dirs.append(candidate)

        return sorted(dirs)

    @staticmethod
    def _validate(dataset: RawTrajectoryDataset) -> None:
        """
        Validate dataset shapes and consistency.

        Expected shapes
        ---------------
        t  : (L,)
        X  : (M, L, N, D)
        V  : (M, L, N, D)
        LHS: (M, L, N, D)
        U0 : (M, state_dim)
        """
        t = np.asarray(dataset.t)
        X = np.asarray(dataset.X)
        V = None if dataset.V is None else np.asarray(dataset.V)
        LHS = np.asarray(dataset.LHS)
        U0 = np.asarray(dataset.U0)

        if not isinstance(dataset.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        system_order = dataset.metadata.get("system_order")

        if system_order == 1:
            if V is not None:
                raise ValueError("First-order dataset should not contain V.")

        if system_order == 2:
            if V is None:
                raise ValueError("Second-order dataset must contain V.")

        if t.ndim != 1 or t.size < 2:
            raise ValueError("t must be a 1D array with at least two time points.")
        if X.ndim != 4:
            raise ValueError(f"X must have shape (M, L, N, D), got {X.shape}.")

        if  X.shape != LHS.shape:
            raise ValueError(
                f"X, and LHS must have the same shape, got "
                f"X={X.shape}, LHS={LHS.shape}."
            )

        if V is not None and V.shape != dataset.X.shape:
            raise ValueError(
                f"V and X must have same shape. Got V={V.shape}, X={X.shape}."
            )
        if X.shape[1] != t.size:
            raise ValueError(
                f"Time dimension of X/V/LHS ({X.shape[1]}) must match len(t) ({t.size})."
            )
        if U0.ndim != 2:
            raise ValueError(f"U0 must have shape (M, state_dim), got {U0.shape}.")
        if U0.shape[0] != X.shape[0]:
            raise ValueError(
                f"Number of initial conditions ({U0.shape[0]}) must match "
                f"number of trajectories ({X.shape[0]})."
            )






# #Minimal order:
# system = SPPSystem(N=10, D=2)
# ic_gen = InitialConditionGenerator(N=10, D=2)
# sim = Simulator(system)

# batch_ic = ic_gen.clumps(M=4, seed=123)
# traj_batch = sim.simulate_many(batch_ic.U0, t_eval=t_eval)

# dataset = RawTrajectoryDataset(
#     t=traj_batch.t,
#     X=traj_batch.X,
#     V=traj_batch.V,
#     LHS=traj_batch.LHS,
#     U0=traj_batch.U0,
#     metadata=metadata,
# )

# repo = DatasetRepository()
# repo.save(dataset, system_name="spp", split="train", dataset_name="dataset_001")
# loaded = repo.load("spp", "train", "dataset_001")
