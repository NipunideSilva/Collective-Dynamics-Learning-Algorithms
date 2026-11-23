from __future__ import annotations

from ips_learning.systems.cucker_smale import CuckerSmaleSystem
from ips_learning.experiments.generate_second_order_data import (
    SplitSpec,
    TrialSplitSpec,
    DatasetCollectionProfile,
    run_one_second_order_dataset,
)


def build_cs_dataset_profile(profile: str = "profile1") -> DatasetCollectionProfile:
    """
    Build CS model-selection dataset profile.

    Matches ModelSelectionSystems/CS parameters.py:

        profile1: beta=0.1, N=10
        X0 ~ Uniform(0.1, 10)
        V0 ~ Uniform(0.1, 5)
        T_fit = 0.5
        dt = 0.01
    """

    profiles = {
        "profile1": {"beta": 0.1, "N": 10},
        "profile2": {"beta": 0.5, "N": 10},
        "profile3": {"beta": 1.0, "N": 10},
    }

    if profile not in profiles:
        raise ValueError(f"Unknown CS dataset profile: {profile!r}")

    params = profiles[profile]

    common = dict(
        system_order=2,
        N=params["N"],
        D=2,
        t0=0.0,
        ic_generator_name="cs_uniform",
        ic_params={
            "x_low": 0.1,
            "x_high": 10.0,
            "v_low": 0.1,
            "v_high": 5.0,
        },
        solver_method="Radau",
        solver_rtol=1e-6,
        solver_atol=1e-6,
        overwrite=True,
        system_params={
            "beta": params["beta"],
        },
    )

    return DatasetCollectionProfile(
        system_name="cs",
        system_factory=CuckerSmaleSystem,
        profile_name=profile,
        common_config=common,
        train=TrialSplitSpec(
            split="train",
            dataset_name_template="dataset_train_trial_{trial:03d}",
            num_trials=10,
            M=20,
            L=201,
            T=2.0,
            seed_start=8101,
        ),
        test=SplitSpec(
            split="test",
            dataset_name="dataset_test",
            M=20,
            L=201,
            T=2.0,
            seed=9202,
        ),
        kde=SplitSpec(
            split="kde",
            dataset_name="dataset_kde",
            M=2000,
            L=51,
            T=0.5,
            seed=10303,
        ),

        validation=SplitSpec(
            split="validation",
            dataset_name="dataset_validation",
            M=10,
            L=201,
            T=2.0,
            seed=10404,
        ),
    )


def run_one_cs_dataset(**kwargs):
    return run_one_second_order_dataset(
        system_name="cs",
        system_factory=CuckerSmaleSystem,
        **kwargs,
    )