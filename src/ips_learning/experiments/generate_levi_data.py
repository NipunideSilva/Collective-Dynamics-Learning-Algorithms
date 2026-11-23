from __future__ import annotations

import numpy as np

from ips_learning.systems.levi import LeviSystem
from ips_learning.experiments.generate_second_order_data import (
    SplitSpec,
    TrialSplitSpec,
    DatasetCollectionProfile,
    run_one_second_order_dataset,
)

# These are wrappers around the more general second-order data generation code,
#  which specify Levi-specific defaults and parameters. 
# See generate_second_order_data.py for the more general code.

def build_levi_dataset_profile(profile: str = "profile1") -> DatasetCollectionProfile:
    """
    Build Levi dataset-generation profile using the legacy Archive/levi parameters.

    Legacy profile1:
        u      = 0.1
        el     = (-1/sqrt(2), 1/sqrt(2))
        I0     = 1
        lamda  = 1
        beta   = 0.1
        N      = 10

        M      = 20
        t0     = 0.25
        T_fit  = 0.75
        dt     = 0.01
        L_fit  = 51
        Trials = 10
    """
    if profile == "profile1":

        common = dict(
            system_order=2,
            N=10,
            D=2,
            t0=0.25,
            ic_generator_name="levi_uniform",
            ic_params={
                "x_low": 0.0,
                "x_high": 2.0,
                "v_low": 0.0,
                "v_high": 1.0,
            },
            solver_method="Radau",
            solver_rtol=1e-6,
            solver_atol=1e-6,
            overwrite=True,
            system_params={
                "u": 0.1,
                "el": (-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
                "I0": 1.0,
                "lamda": 1.0,
                "beta": 0.1,
                "eps_r": 1e-12,
            },
        )

        return DatasetCollectionProfile(
            system_name="levi",
            system_factory=LeviSystem,
            profile_name=profile,
            common_config=common,
            train=TrialSplitSpec(
                split="train",
                dataset_name_template="dataset_train_trial_{trial:03d}",
                num_trials=10,
                M=20,
                L=201,
                T=2.25,
                seed_start=1101,
            ),
            test=SplitSpec(
                split="test",
                dataset_name="dataset_test",
                M=10,
                L=201,
                T=2.25,
                seed=2202,
            ),
            kde=SplitSpec(
                split="kde",
                dataset_name="dataset_kde",
                M=2000,
                L=51,
                T=0.75,
                seed=3303,
            ),
        )

    elif profile == "levi_velocity_grid":

        common = dict(
            system_order=2,
            N=10,
            D=2,
            t0=0.25,
            ic_generator_name="levi_velocity_grid",
            ic_params={
                "x_low":  0.0,
                "x_high":  2.0,
                "v_low":  -1.0,
                "v_high": 1.0,
            },
            solver_method="Radau",
            solver_rtol=1e-6,
            solver_atol=1e-6,
            overwrite=True,
            system_params={
                "u": 0.1,
                "el": (-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
                "I0": 1.0,
                "lamda": 1.0,
                "beta": 0.1,
                "eps_r": 1e-12,
            },
        )

        return DatasetCollectionProfile(
            system_name="levi",
            system_factory=LeviSystem,
            profile_name=profile,
            common_config=common,
            train=TrialSplitSpec(
                split="train",
                dataset_name_template="dataset_train_trial_{trial:03d}",
                num_trials=10,
                M=20,
                L=201,
                T=2.25,
                seed_start=1101,
            ),
            test=SplitSpec(
                split="test",
                dataset_name="dataset_test",
                M=10,
                L=201,
                T=2.25,
                seed=2202,
            ),
            kde=SplitSpec(
                split="kde",
                dataset_name="dataset_kde",
                M=2000,
                L=51,
                T=0.75,
                seed=3303,
            ),
            validation=SplitSpec(
                split="validation",
                dataset_name="dataset_validation",
                M=10,
                L=201,
                T=2.25,
                seed=4404,
            ),
        )
    
    raise ValueError(f"Unknown Phototaxis dataset profile: {profile}")
