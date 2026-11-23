from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np

from ips_learning.data.initial_conditions import InitialConditionGenerator
from ips_learning.data.simulator import Simulator
from ips_learning.data.dataset_repository import RawTrajectoryDataset, DatasetRepository


@dataclass(frozen=True)
class SplitSpec:
    split: str
    dataset_name: str
    M: int
    L: int
    T: float
    seed: int


@dataclass(frozen=True)
class TrialSplitSpec:
    split: str
    dataset_name_template: str
    num_trials: int
    M: int
    L: int
    T: float
    seed_start: int


@dataclass(frozen=True)
class DatasetCollectionProfile:
    profile_name: str
    system_name: str
    system_factory: Callable
    common_config: dict
    train: TrialSplitSpec
    test: SplitSpec
    kde: SplitSpec
    validation: SplitSpec


def generate_initial_conditions(
    *,
    ic_generator: InitialConditionGenerator,
    ic_generator_name: str,
    M: int,
    seed: int,
    ic_params: dict,
):
    if not hasattr(ic_generator, ic_generator_name):
        raise ValueError(f"Unknown initial condition generator: {ic_generator_name}")

    method = getattr(ic_generator, ic_generator_name)
    return method(M=M, seed=seed, **ic_params)


def build_second_order_metadata(
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
        "system_order": 2,
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
        "lhs_name": "acceleration",
        "lhs_symbol": "\\ddot{X}",
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


def run_one_second_order_dataset(
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
    solver_method: str = "Radau",
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
    ic_gen = InitialConditionGenerator(N=N, D=D)

    t_eval = np.linspace(t0, T, L)

    ic_batch = generate_initial_conditions(
        ic_generator=ic_gen,
        ic_generator_name=ic_generator_name,
        M=M,
        seed=seed,
        ic_params=ic_params,
    )

    sim = Simulator(system)
    traj_batch = sim.simulate_many(
        ic_batch.U0,
        t_eval=t_eval,
        method=solver_method,
        rtol=solver_rtol,
        atol=solver_atol,
    )

    metadata = build_second_order_metadata(
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
        description=description or f"{system_name} {split} dataset",
        profile_name=profile_name,
        split_config=split_config,
    )

    dataset = RawTrajectoryDataset(
        t=traj_batch.t,
        X=traj_batch.X,
        V=traj_batch.V,
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