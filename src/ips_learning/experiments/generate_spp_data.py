from __future__ import annotations

import numpy as np

from ips_learning.systems.spp import SPPSystem
from ips_learning.experiments.generate_second_order_data import (
    SplitSpec,
    TrialSplitSpec,
    DatasetCollectionProfile,
    run_one_second_order_dataset,
)
# These are wrappers around the more general second-order data generation code,
#  which specify SPP-specific defaults and parameters. 
# See generate_second_order_data.py for the more general code.

# ==========================================================
# SPP profile builder
# ==========================================================

def build_spp_dataset_profile(profile: str = "baseline", dataset_prefix: str | None = None,) -> DatasetCollectionProfile:
    """
    Build a named SPP dataset-generation profile.
    'baseline' is the default profile, which generates clumps

    This is the correct place for SPP-specific defaults, including:
    - system parameters
    - IC family
    - solver settings
    - split-specific M, L, T, seeds, and names
    """
    prefix = "" if dataset_prefix is None else f"{dataset_prefix}_"

    if profile == "baseline":
        common = dict(
            system_order=2,
            N=10,
            D=2,
            t0=0.0,
            ic_generator_name="clumps",
            ic_params={
                "v_eq": np.sqrt(1.0 / 0.5),
                "L": 2.0,
            },
            solver_method="Radau",
            solver_rtol=1e-6,
            solver_atol=1e-6,
            overwrite=True,
            system_params={
                "cr": 0.6,
                "lr": 0.5,
                "ca": 1.0,
                "la": 1.0,
                "a": 1.0,
                "b": 0.5,
                "eps_r": 1e-1,
            },
        )
        return DatasetCollectionProfile(
            system_name="spp",
            system_factory=SPPSystem,
            profile_name=profile,
            common_config=common,
            train=TrialSplitSpec(
                split="train",
                dataset_name_template= f"{prefix}dataset_train_trial_{{trial:03d}}",
                num_trials=10,
                M=20,
                L=201,
                T=2.0,
                seed_start=101,
            ),
            test=SplitSpec(
                split="test",
                dataset_name=f"{prefix}dataset_test",
                M=10,
                L=201,
                T=2.0,
                seed=202,
            ),
            kde=SplitSpec(
                split="kde",
                dataset_name=f"{prefix}dataset_kde",
                M=2000,
                L=51,
                T=0.5,
                seed=303,
            ),
            validation=SplitSpec(
                split="validation",
                dataset_name=f"{prefix}dataset_validation",
                M=10,
                L=201,
                T=2.0,
                seed=404,
            ),
        )
    elif profile == "spp_velocity_grid_clumps":
        common = dict(
            system_order=2,
            N=10,
            D=2,
            t0=0.0,
            ic_generator_name="spp_velocity_grid_clumps",
            ic_params={
                "v_eq": np.sqrt(1.0 / 0.5),
                "L_values":(2.0, 4.0, 6.0, 8.0),
                "center_range":(-4.0, 4.0),
                "v_box_scale":  2.0,
                "jitter": 0.05,
            },
            solver_method="Radau",
            solver_rtol=1e-6,
            solver_atol=1e-6,
            overwrite=True,
            system_params={
                "cr": 0.6,
                "lr": 0.5,
                "ca": 1.0,
                "la": 1.0,
                "a": 1.0,
                "b": 0.5,
                "eps_r": 1e-1,
            },
        )

        return DatasetCollectionProfile(
            system_name="spp",
            system_factory=SPPSystem,
            profile_name=profile,
            common_config=common,
            train=TrialSplitSpec(
                split="train",
                dataset_name_template=f"{prefix}dataset_train_trial_{{trial:03d}}",
                num_trials=10,
                M=50,
                L=201,
                T=2.0,
                seed_start=101,
            ),
            test=SplitSpec(
                split="test",
                dataset_name=f"{prefix}dataset_test",
                M=10,
                L=201,
                T=2.0,
                seed=202,
            ),
            kde=SplitSpec(
                split="kde",
                dataset_name=f"{prefix}dataset_kde",
                M=2000,
                L=51,
                T=0.5,
                seed=303,
            ),
            validation=SplitSpec(
                split="validation",
                dataset_name=f"{prefix}dataset_validation",
                M=10,
                L=201,
                T=2.0,
                seed=404,
            ),
        )
    elif profile == "spp_swarm":
        common = dict(
            system_order=2,
            N=10,
            D=2,
            t0=0.0,
            ic_generator_name="swarm",
            ic_params={
                "v_eq": np.sqrt(1.6 / 0.5),
                "L": 5.0,
            },
            solver_method="Radau",
            solver_rtol=1e-6,
            solver_atol=1e-6,
            overwrite=True,
            system_params={
                "cr": 1.0,
                "lr": 0.5,
                "ca": 0.5,
                "la": 2.0,
                "a": 1.6,
                "b": 0.5,
                "eps_r": 1e-1,
            },
        )
        return DatasetCollectionProfile(
            system_name="spp",
            system_factory=SPPSystem,
            profile_name=profile,
            common_config=common,
            train=TrialSplitSpec(
                split="train",
                dataset_name_template=f"{prefix}dataset_train_trial_{{trial:03d}}",
                num_trials=10,
                M=50,
                L=1001,
                T=10,
                seed_start=101,
            ),
            test=SplitSpec(
                split="test",
                dataset_name=f"{prefix}dataset_test",
                M=10,
                L=1001,
                T=10.0,
                seed=202,
            ),
            kde=SplitSpec(
                split="kde",
                dataset_name=f"{prefix}dataset_kde",
                M=2000,
                L=401,
                T=4,
                seed=303,
            ),
            validation=SplitSpec(
                split="validation",
                dataset_name=f"{prefix}dataset_validation",
                M=10,
                L=201,
                T=2.0,
                seed=404,
            ),
        )
    elif profile == "spp_swarm_spread":
        common = dict(
            system_order=2,
            N=10,
            D=2,
            t0=0.0,
            ic_generator_name="spp_velocity_grid_clumps",
            ic_params={
                "v_eq": np.sqrt(1.6 / 0.5),
                "L_values":( 4.0, 6.0),
                "center_range":(-4.0, 4.0),
                "v_box_scale":  2.0,
                "jitter": 0.05,
            },
            solver_method="Radau",
            solver_rtol=1e-6,
            solver_atol=1e-6,
            overwrite=True,
            system_params={
                "cr": 1.0,
                "lr": 0.5,
                "ca": 0.5,
                "la": 2.0,
                "a": 1.6,
                "b": 0.5,
                "eps_r": 1e-1,
            },
        )
        return DatasetCollectionProfile(
            system_name="spp",
            system_factory=SPPSystem,
            profile_name=profile,
            common_config=common,
            train=TrialSplitSpec(
                split="train",
                dataset_name_template=f"{prefix}dataset_train_trial_{{trial:03d}}",
                num_trials=10,
                M=50,
                L=1001,
                T=10,
                seed_start=101,
            ),
            test=SplitSpec(
                split="test",
                dataset_name=f"{prefix}dataset_test",
                M=10,
                L=1001,
                T=10.0,
                seed=202,
            ),
            kde=SplitSpec(
                split="kde",
                dataset_name=f"{prefix}dataset_kde",
                M=2000,
                L=401,
                T=4,
                seed=303,
            ),
            validation=SplitSpec(
                split="validation",
                dataset_name=f"{prefix}dataset_validation",
                M=10,
                L=201,
                T=2.0,
                seed=404,
            ),
        )
    
    raise ValueError(f"Unknown SPP dataset profile: {profile}")


def run_one_spp_dataset(**kwargs):
    return run_one_second_order_dataset(
        system_name="spp",
        system_factory=SPPSystem,
        **kwargs,
    )