# step1
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


from ips_learning.data.dataset_repository import RawTrajectoryDataset
@dataclass
class FeatureDataset:
    """
    Learning-ready dataset for interacting particle systems.

    Attributes
    ----------
    t   : (L,)
    X   : (M, L, N, D)
    V   : (M, L, N, D)
    LHS : (M, L, N, D)

    R   : (M, L, N, N, 1)
    R_D : (M, L, N, N, D)
    VR_D: (M, L, N, N, D)

    metadata : dict
    """
    t: np.ndarray
    X: np.ndarray
    LHS: np.ndarray
    R: np.ndarray
    R_D: np.ndarray
    metadata: dict
    V: np.ndarray | None = None
    VR_D: np.ndarray | None = None

class FeatureBuilder:
    """
    Build geometric features for
        first-order interacting particle systems.
        second-order interacting particle systems.

    Input:
        RawTrajectoryDataset with X, LHS and or V.

    Output:
        FeatureDataset containing pairwise position differences,
        pairwise distances, and pairwise velocity differences.
    """

    def build(self, raw_dataset) -> FeatureDataset:
        """
        Convert RawTrajectoryDataset -> FeatureDataset
        """

        system_order = raw_dataset.metadata.get("system_order")

        if system_order not in (1, 2):
            raise ValueError(
                f"metadata['system_order'] must be 1 or 2, got {system_order}."
            )

        t = raw_dataset.t
        X = raw_dataset.X
        LHS = raw_dataset.LHS

        source_metadata = dict(raw_dataset.metadata)

        R_D = self._compute_relative_positions(X)
        R = self._compute_distances(R_D)

        if system_order == 2: #switch 1 and 2 later since 2nd order is main
            V = raw_dataset.V

            if V is None:
                raise ValueError("Second-order dataset must contain V.")

            self._validate_raw_shapes(t, X, V, LHS)
            VR_D = self._compute_velocity_differences(V)


            metadata = {
                "source_dataset": source_metadata,
                "feature_stage": "2.1",
                "system_name": source_metadata.get("system_name"),
                "system_order": source_metadata.get("system_order", 2),
                "interaction_type": source_metadata.get("interaction_type"),
                "description": "Second-order feature dataset with pairwise geometry.",
            }

            return FeatureDataset(
                t=t,
                X=X,
                V=V,
                LHS=LHS,
                R=R,
                R_D=R_D,
                VR_D=VR_D,
                metadata=metadata,
            )
        #system_order == 1
        self._validate_raw_shapes(t, X, None, LHS)
        metadata = {
            "source_dataset": source_metadata,
            "feature_stage": "2.1",
            "system_name": source_metadata.get("system_name"),
            "system_order": source_metadata.get("system_order", 1),
            "interaction_type": source_metadata.get("interaction_type"),
            "description": "First-order feature dataset with pairwise geometry.",
        }

        return FeatureDataset(
            t=t,
            X=X,
            V=None,
            LHS=LHS,
            R=R,
            R_D=R_D,
            VR_D=None,
            metadata=metadata,
        )

    def build_for_candidate(self, dataset, candidate):
        """
        Build feature data using the state representation required by a
        model-selection candidate.

        The raw dataset remains unchanged. This method only changes the
        candidate-specific interpretation of the data.

        First-order candidate:
            X, LHS = dX/dt

        Second-order candidate:
            X, V = dX/dt, LHS = d2X/dt2
        """
        view = make_maximal_derivative_view(dataset)

        candidate_dataset = dataset_for_candidate_from_view(
            view=view,
            candidate=candidate,
        )

        features = self.build(candidate_dataset)

        features.metadata["model_selection_mode"] = True
        features.metadata["candidate_name"] = candidate.name
        features.metadata["candidate_system_order"] = candidate.system_order
        features.metadata["candidate_interaction_type"] = candidate.interaction_type
        features.metadata["candidate_include_environment"] = getattr(
            candidate,
            "include_environment",
            True,
        )
        features.metadata["generating_system_order"] = dataset.metadata.get(
            "system_order"
        )

        return features


    @staticmethod
    def _validate_raw_shapes(
        t: np.ndarray,
        X: np.ndarray,
        V: np.ndarray | None,
        LHS: np.ndarray,
    ) -> None:
        if t.ndim != 1 or t.size < 2:
            raise ValueError("t must be a 1D array with at least two time points.")

        if X.ndim != 4:
            raise ValueError(f"X must have shape (M, L, N, D), got {X.shape}.")

        if V is not None and V.ndim != 4:
            raise ValueError(f"V must have shape (M, L, N, D), got {V.shape}.")

        if LHS.ndim != 4:
            raise ValueError(f"LHS must have shape (M, L, N, D), got {LHS.shape}.")

        if X.shape != LHS.shape:
            raise ValueError(
                f"X and LHS must match in shape, got "
                f"X={X.shape}, LHS={LHS.shape}."
            )

        if V is not None and X.shape != V.shape:
            raise ValueError(
                f"X and V must match in shape, got "
                f"X={X.shape}, V={V.shape}."
            )

        if X.shape[1] != t.size:
            raise ValueError(
                f"Time dimension {X.shape[1]} must match len(t)={t.size}."
            )

    # Compute pairwise relative positions.
    def _compute_relative_positions(self, X: np.ndarray) -> np.ndarray:
        """
        Compute R_D[i,j] = x_j - x_i

        Input:
            X: (M, L, N, D)

        Output:
            R_D: (M, L, N, N, D)
        """
        return X[:, :, np.newaxis, :, :] - X[:, :, :, np.newaxis, :]

    # Compute pairwise distances.
    def _compute_distances(self, R_D: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances.

        Input:
            R_D: (M, L, N, N, D)

        Output:
            R: (M, L, N, N, 1)
        """
        return np.linalg.norm(R_D, axis=-1, keepdims=True)

    # Compute pairwise velocity differences.
    def _compute_velocity_differences(self, V):
        """
        Compute VR_D[i,j] = v_j - v_i

        Input:
            V: (M, L, N, D)

        Output:
            VR_D: (M, L, N, N, D)
        """
        return V[:, :, np.newaxis, :, :] - V[:, :, :, np.newaxis, :]


def make_maximal_derivative_view(dataset):
    t = np.asarray(dataset.t, dtype=float)
    X = np.asarray(dataset.X, dtype=float)

    if dataset.V is not None and dataset.LHS is not None:
        V = np.asarray(dataset.V, dtype=float)
        A = np.asarray(dataset.LHS, dtype=float)
    else:
        V = np.asarray(dataset.LHS, dtype=float)
        A = np.gradient(V, t, axis=1)

    return {
        "t": t,
        "X": X,
        "V": V,
        "A": A,
        "U0": dataset.U0,
        "metadata": dict(dataset.metadata),
    }


def dataset_for_candidate_from_view(view, candidate):
    metadata = dict(view["metadata"])

    if candidate.system_order == 1:
        metadata["system_order"] = 1
        metadata["lhs_name"] = "velocity"
        metadata["lhs_symbol"] = "\\dot{X}"

        return RawTrajectoryDataset(
            t=view["t"],
            X=view["X"],
            V=None,
            LHS=view["V"],
            U0=view["U0"],
            metadata=metadata,
        )

    if candidate.system_order == 2:
        metadata["system_order"] = 2
        metadata["lhs_name"] = "acceleration"
        metadata["lhs_symbol"] = "\\ddot{X}"

        return RawTrajectoryDataset(
            t=view["t"],
            X=view["X"],
            V=view["V"],
            LHS=view["A"],
            U0=view["U0"],
            metadata=metadata,
        )

    raise ValueError(f"Unsupported candidate system_order={candidate.system_order}.")
