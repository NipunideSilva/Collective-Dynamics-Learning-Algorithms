from __future__ import annotations

import numpy as np

from ips_learning.data.dataset_repository import RawTrajectoryDataset


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
    raise ValueError(
        f"Unsupported candidate system_order={candidate.system_order}."
    )