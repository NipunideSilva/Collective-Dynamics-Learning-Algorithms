from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ips_learning.data.initial_conditions import InitialConditionGenerator
from ips_learning.data.simulator import Simulator
from ips_learning.data.dataset_repository import RawTrajectoryDataset, DatasetRepository


def build_first_order_metadata(
    *,
    system,
    ic_generator_name: str,
    ic_params: dict,
    seed: int,
    M: int,
    L: int,
    t_eval: np.ndarray,
    solver_method: str,
    solver_rtol: float,
    solver_atol: float,
    description: str,
    profile_name: str | None = None,
    split_config: dict | None = None,
) -> dict:
    metadata = {
        "description": description,
        "system_order": 1,
        "N": system.N,
        "D": system.D,
        "M": M,
        "L": L,
        "t0": float(t_eval[0]),
        "T": float(t_eval[-1]),
        "state_dim": system.state_dim,
        "seed": seed,
        "ic_generator": ic_generator_name,
        "ic_params": ic_params,
        "lhs_name": "velocity",
        "lhs_symbol": "\\dot{X}",
        "solver_method": solver_method,
        "solver_rtol": solver_rtol,
        "solver_atol": solver_atol,
    }

    if hasattr(system, "metadata"):
        metadata.update(system.metadata())

    if profile_name is not None:
        metadata["profile_name"] = profile_name

    if split_config is not None:
        metadata["split_config"] = split_config

    return metadata


def sample_first_order_initial_conditions(
    *,
    M: int,
    N: int,
    D: int,
    seed: int,
    low: float = 0.0,
    high: float = 2.0 * np.pi,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(M, N * D))


def run_one_first_order_dataset(
    *,
    system_name: str,
    system_factory: Callable,
    split: str,
    dataset_name: str,
    N: int,
    D: int,
    M: int,
    L: int,
    t0: float,
    T: float,
    seed: int,
    ic_generator_name: str,
    ic_params: dict,
    solver_method: str = "BDF",
    solver_rtol: float = 1e-6,
    solver_atol: float = 1e-6,
    overwrite: bool = True,
    system_params: dict | None = None,
    profile_name: str | None = None,
    split_config: dict | None = None,
    description: str | None = None,
    root_dir: str | Path = "artifacts/datasets",
) -> Path:
    system_params = {} if system_params is None else dict(system_params)

    system = system_factory(N=N, D=D, **system_params)

    t_eval = np.linspace(t0, T, L)

    U0 = sample_first_order_initial_conditions(
        M=M,
        N=N,
        D=D,
        seed=seed,
        low=float(ic_params.get("low", 0.0)),
        high=float(ic_params.get("high", 2.0 * np.pi)),
    )

    sim = Simulator(system)
    traj_batch = sim.simulate_many(
        U0,
        t_eval=t_eval,
        method=solver_method,
        rtol=solver_rtol,
        atol=solver_atol,
    )

    if traj_batch.V is not None:
        raise ValueError("First-order dataset generation expected V=None.")

    metadata = build_first_order_metadata(
        system=system,
        ic_generator_name=ic_generator_name,
        ic_params=ic_params,
        seed=seed,
        M=M,
        L=L,
        t_eval=t_eval,
        solver_method=solver_method,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        description=description or f"{system_name} {split} first-order dataset",
        profile_name=profile_name,
        split_config=split_config,
    )

    dataset = RawTrajectoryDataset(
        t=traj_batch.t,
        X=traj_batch.X,
        V=None,
        LHS=traj_batch.LHS,
        U0=traj_batch.U0,
        metadata=metadata,
    )

    repo = DatasetRepository(root_dir=root_dir)
    return repo.save(
        dataset,
        system_name=system_name,
        profile_name=profile_name,
        split=split,
        dataset_name=dataset_name,
        overwrite=overwrite,
    )