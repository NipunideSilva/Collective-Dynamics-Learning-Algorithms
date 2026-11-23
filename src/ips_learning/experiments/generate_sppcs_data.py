from __future__ import annotations

from ips_learning.systems.sppcs import SPPCSSystem
from ips_learning.experiments.generate_second_order_data import (
    SplitSpec,
    TrialSplitSpec,
    DatasetCollectionProfile,
    run_one_second_order_dataset,
)


def build_sppcs_dataset_profile(profile: str = "clumps") -> DatasetCollectionProfile:
    """
    Build SPPCS model-selection dataset profile.

    Uses Case 8 ICs:
        spp_velocity_grid_clumps
    """

    if profile == "clumps":
        common = dict(
            system_order=2,
            N=5,
            D=2,
            t0=0.0,
            ic_generator_name="spp_velocity_grid_clumps",
            ic_params={
                "v_eq": (1.0 / 0.5) ** 0.5,
                "L_values": (2.0, 4.0, 6.0, 8.0),
                "center_range": (-4.0, 4.0),
                "v_box_scale": 2.0,
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
                "beta": 1.0,
                "eps_r": 1e-1,
            },
        )

        return DatasetCollectionProfile(
            system_name="sppcs",
            system_factory=SPPCSSystem,
            profile_name=profile,
            common_config=common,
            train=TrialSplitSpec(
                split="train",
                dataset_name_template="dataset_train_trial_{trial:03d}",
                num_trials=10,
                M=20,
                L=201,
                T=2.0,
                seed_start=5101,
            ),
            test=SplitSpec(
                split="test",
                dataset_name="dataset_test",
                M=20,
                L=201,
                T=2.0,
                seed=6202,
            ),
            kde=SplitSpec(
                split="kde",
                dataset_name="dataset_kde",
                M=2000,
                L=51,
                T=0.5,
                seed=7303,
            ),

            validation=SplitSpec(
                split="validation",
                dataset_name="dataset_validation",
                M=10,
                L=201,                
                T=2.0,
                seed=8404,
            ),
        )

    raise ValueError(f"Unknown SPPCS dataset profile: {profile!r}")


def run_one_sppcs_dataset(**kwargs):
    return run_one_second_order_dataset(
        system_name="sppcs",
        system_factory=SPPCSSystem,
        **kwargs,
    )