from __future__ import annotations

from dataclasses import asdict
import numpy as np

from ips_learning.data.dataset_repository import DatasetRepository, RawTrajectoryDataset

from ips_learning.experiments.generate_second_order_data import (
    DatasetCollectionProfile,
    run_one_second_order_dataset,
)

from ips_learning.experiments.generate_first_order_data import (
    run_one_first_order_dataset,
)



def slice_raw_dataset_by_time(
    dataset: RawTrajectoryDataset,
    fit_end_time: float,
    dataset_role: str = "fit_window",
    atol: float = 1e-12,
) -> RawTrajectoryDataset:
    """
    Slice a raw trajectory dataset along the time axis using a fit end time.

    Parameters
    ----------
    dataset : RawTrajectoryDataset
        Full raw dataset.
    fit_end_time : float
        End time of the fitting window.
    dataset_role : str
        Metadata label for the sliced dataset.
    atol : float
        Small tolerance for floating point comparisons.

    Returns
    -------
    RawTrajectoryDataset
        Time-sliced raw dataset.
    """
    t = np.asarray(dataset.t, dtype=float)

    fit_mask = t <= fit_end_time + atol
    fit_indices = np.where(fit_mask)[0]

    if fit_indices.size == 0:
        raise ValueError(
            f"No time points found with t <= {fit_end_time}. "
            f"Dataset starts at t={t[0]}."
        )
    
    V_fit = None
    if dataset.V is not None:
        V_fit = dataset.V[:, fit_indices, :, :]

    return RawTrajectoryDataset(
        t=dataset.t[fit_indices],
        X=dataset.X[:, fit_indices, :, :],
        V=V_fit,
        LHS=dataset.LHS[:, fit_indices, :, :],
        U0=dataset.U0,
        metadata={
            **dataset.metadata,
            "dataset_role": dataset_role,
            "source_dataset_name": dataset.metadata.get("dataset_name", None),

            # Update top-level metadata to describe the sliced dataset.
            "t0": float(dataset.t[fit_indices[0]]),
            "T": float(dataset.t[fit_indices[-1]]),
            "L": int(fit_indices.size),

            "fit_time_policy": {
                "fit_end_time_requested": float(fit_end_time),
                "fit_t0": float(dataset.t[fit_indices[0]]),
                "fit_T": float(dataset.t[fit_indices[-1]]),
                "fit_num_time_points": int(fit_indices.size),
                "fit_indices_start": int(fit_indices[0]),
                "fit_indices_end": int(fit_indices[-1]),
            },
        },
    )

def run_one_dataset_from_profile(
    *,
    profile: DatasetCollectionProfile,
    split: str,
    dataset_name: str,
    M: int,
    L: int,
    T: float,
    seed: int,
    split_config: dict,
    N_override: int | None = None,
):
    """
    Dispatch to the correct single-dataset generator based on profile.common_config['system_order'].

    system_order=1:
        first-order generator

    system_order=2:
        second-order generator
    """
    system_order = profile.common_config.get("system_order")

    if system_order not in (1, 2):
        raise ValueError(
            f"profile.common_config['system_order'] must be 1 or 2, got {system_order}."
        )

    common_config = dict(profile.common_config)
    common_config.pop("system_order", None)
    if N_override is not None:
        common_config["N"] = int(N_override)

    if system_order == 1:
        return run_one_first_order_dataset(
            system_name=profile.system_name,
            system_factory=profile.system_factory,
            split=split,
            dataset_name=dataset_name,
            M=M,
            seed=seed,
            L=L,
            T=T,
            profile_name=profile.profile_name,
            split_config=split_config,
            **common_config,
        )

    return run_one_second_order_dataset(
        system_name=profile.system_name,
        system_factory=profile.system_factory,
        split=split,
        dataset_name=dataset_name,
        M=M,
        seed=seed,
        L=L,
        T=T,
        profile_name=profile.profile_name,
        split_config=split_config,
        **common_config,
    )


def generate_standard_dataset_collection(
    *,
    profile: DatasetCollectionProfile,
    save_train_fit_split: bool = False,
    fit_end_time: float | None = None,
    fit_split_name: str = "train_fit",
    fit_name_template: str = "dataset_train_fit_trial_{trial:03d}",
) -> dict:
    """
    Generic dataset collection runner.

    This file is intentionally system-agnostic.
    It orchestrates:
    - train trials
    - one test dataset
    - one KDE/reference dataset
    - one validation dataset
    - optional saved train-fit split derived from full train datasets

    Parameters
    ----------

    profile : DatasetCollectionProfile
        Profile carrying all system-specific defaults.
    save_train_fit_split : bool
        Whether to derive and save train_fit datasets from the full train datasets.
    fit_end_time : float or None
        End time of fitting window. Required if save_train_fit_split=True.
    fit_split_name : str
        Split name under
        artifacts/datasets/<system>/<profile>/<fit_split_name>/...
    fit_name_template : str
        Naming template for saved fit-window train datasets.
    """
    outputs = {
        "profile_name": profile.profile_name,
        "train": [],
        "train_fit": [],
        "test": None,
        "kde": None,
        "validation": None,
    }

    repo = DatasetRepository()
    system_name = profile.system_name

    if save_train_fit_split and fit_end_time is None:
        raise ValueError(
            "fit_end_time must be provided if save_train_fit_split=True."
        )

    # --------------------------------------------------
    # Train trials
    # --------------------------------------------------
    for q in range(profile.train.num_trials):
        trial_id = q + 1
        train_dataset_name = profile.train.dataset_name_template.format(trial=trial_id)

        save_dir = run_one_dataset_from_profile(
            profile=profile,
            split=profile.train.split,
            dataset_name=train_dataset_name,
            M=profile.train.M,
            seed=profile.train.seed_start + q,
            L=profile.train.L,
            T=profile.train.T,
            split_config=asdict(profile.train),
        )
        outputs["train"].append(str(save_dir))

        # ----------------------------------------------
        # Optional derived fit-window dataset
        # ----------------------------------------------
        if save_train_fit_split:
            full_train = repo.load(
                system_name=system_name,
                profile_name=profile.profile_name,
                split=profile.train.split,
                dataset_name=train_dataset_name,
            )

            train_fit = slice_raw_dataset_by_time(
                dataset=full_train,
                fit_end_time=fit_end_time,
                dataset_role="fit_window",
            )

            train_fit_name = fit_name_template.format(trial=trial_id)
            fit_save_dir = repo.save(
                dataset=train_fit,
                system_name=system_name,
                profile_name=profile.profile_name,
                split=fit_split_name,
                dataset_name=train_fit_name,
                overwrite=profile.common_config.get("overwrite", True),
            )
            outputs["train_fit"].append(str(fit_save_dir))

    # --------------------------------------------------
    # KDE/reference dataset
    # --------------------------------------------------

    kde_dir = run_one_dataset_from_profile(
        profile=profile,
        split=profile.kde.split,
        dataset_name=profile.kde.dataset_name,
        M=profile.kde.M,
        seed=profile.kde.seed,
        L=profile.kde.L,
        T=profile.kde.T,
        split_config=asdict(profile.kde),
    )
    outputs["kde"] = str(kde_dir)

    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------
    test_dir = run_one_dataset_from_profile(
        profile=profile,
        split=profile.test.split,
        dataset_name=profile.test.dataset_name,
        M=profile.test.M,
        seed=profile.test.seed,
        L=profile.test.L,
        T=profile.test.T,
        split_config=asdict(profile.test),
    )
    outputs["test"] = str(test_dir)

    # --------------------------------------------------
    # Validation dataset
    # --------------------------------------------------
    validation_dir = run_one_dataset_from_profile(
        profile=profile,
        split=profile.validation.split,
        dataset_name=profile.validation.dataset_name,
        M=profile.validation.M,
        seed=profile.validation.seed,
        L=profile.validation.L,
        T=profile.validation.T,
        split_config=asdict(profile.validation),
    )
    outputs["validation"] = str(validation_dir)

    return outputs
