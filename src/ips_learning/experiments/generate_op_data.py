from __future__ import annotations

from ips_learning.systems.opinion_dynamics import OpinionDynamicsSystem
from ips_learning.experiments.generate_second_order_data import (
    SplitSpec,
    TrialSplitSpec,
    DatasetCollectionProfile,
)
from ips_learning.experiments.generate_first_order_data import (
    run_one_first_order_dataset,
)

import numpy as np
def build_op_dataset_profile(profile: str = "profile1") -> DatasetCollectionProfile:
    """
    Build Opinion Dynamics dataset-generation profile.

    First-order model:
        dX/dt = f_phi(X)

    Saved dataset:
        X   : (M, L, N, D)
        V   : None
        LHS : dX/dt
        U0  : (M, N*D)
    """
    if profile != "profile1":
        raise ValueError(
            f"Unknown opinion dynamics dataset profile: {profile}. "
            "Currently implemented: 'profile1'."
        )

    common = dict(
        system_order=1,
        N=10,
        D=1,
        t0=0.0,
        ic_generator_name="opinion_uniform",
        ic_params={
            "low": 0.0,
            "high": 10.0 ,
        },
        solver_method="BDF",
        solver_rtol=1e-6,
        solver_atol=1e-6,
        overwrite=True,
        system_params={},
    )

    return DatasetCollectionProfile(
        system_name="opinion",
        system_factory=OpinionDynamicsSystem,
        profile_name=profile,
        common_config=common,

        # Full train rollout/reference datasets
        train=TrialSplitSpec(
            split="train",
            dataset_name_template="dataset_train_trial_{trial:03d}",
            num_trials=10,
            M=20,
            L=401,
            T=20.0,
            seed_start=1101,
        ),

        # Held-out test dataset
        test=SplitSpec(
            split="test",
            dataset_name="dataset_test",
            M=10,
            L=401,
            T=20.0,
            seed=2202,
        ),

        # Large reference dataset for density/support
        kde=SplitSpec(
            split="kde",
            dataset_name="dataset_kde",
            M=2000,
            L=201,
            T=10.0,
            seed=3303,
        ),

        # Validation dataset (optional, can be removed if not needed)
        validation=SplitSpec(
            split="validation",
            dataset_name="dataset_validation",
            M=10,
            L=201,
            T=10.0,
            seed=4404,
        ),
    )

# can be removed if no code depends on them.
def run_one_kuramoto_dataset(**kwargs):
    return run_one_first_order_dataset(
        system_name="opinion",
        system_factory=OpinionDynamicsSystem,
        **kwargs,
    )