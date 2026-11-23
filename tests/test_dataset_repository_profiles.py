import numpy as np
import pytest

from ips_learning.data.dataset_repository import (
    DatasetRepository,
    RawTrajectoryDataset,
)


def make_first_order_dataset():
    t = np.array([0.0, 0.1, 0.2])

    X = np.zeros((2, 3, 4, 1))
    LHS = np.ones((2, 3, 4, 1))
    U0 = np.zeros((2, 4))

    return RawTrajectoryDataset(
        t=t,
        X=X,
        V=None,
        LHS=LHS,
        U0=U0,
        metadata={
            "system_name": "test_system",
            "system_order": 1,
            "interaction_type": "energy",
            "N": 4,
            "D": 1,
            "state_dim": 4,
        },
    )


def test_dataset_repository_legacy_layout(tmp_path):
    repo = DatasetRepository(root_dir=tmp_path)
    dataset = make_first_order_dataset()

    path = repo.save(
        dataset,
        system_name="test_system",
        split="train",
        dataset_name="dataset_001",
    )

    expected = (
        tmp_path
        / "test_system"
        / "train"
        / "dataset_001"
    )

    assert path == expected

    loaded = repo.load(
        system_name="test_system",
        split="train",
        dataset_name="dataset_001",
    )

    assert loaded.metadata["system_name"] == "test_system"
    assert "profile_name" not in loaded.metadata

    np.testing.assert_allclose(loaded.X, dataset.X)
    np.testing.assert_allclose(loaded.LHS, dataset.LHS)


def test_dataset_repository_profile_layout(tmp_path):
    repo = DatasetRepository(root_dir=tmp_path)
    dataset = make_first_order_dataset()

    path = repo.save(
        dataset,
        system_name="spp",
        profile_name="spp_swarm_spread",
        split="train",
        dataset_name="dataset_train_trial_001",
    )

    expected = (
        tmp_path
        / "spp"
        / "spp_swarm_spread"
        / "train"
        / "dataset_train_trial_001"
    )

    assert path == expected

    loaded = repo.load(
        system_name="spp",
        profile_name="spp_swarm_spread",
        split="train",
        dataset_name="dataset_train_trial_001",
    )

    assert loaded.metadata["system_name"] == "spp"
    assert loaded.metadata["profile_name"] == "spp_swarm_spread"

    np.testing.assert_allclose(loaded.X, dataset.X)
    np.testing.assert_allclose(loaded.LHS, dataset.LHS)


def test_list_datasets_with_profile(tmp_path):
    repo = DatasetRepository(root_dir=tmp_path)
    dataset = make_first_order_dataset()

    repo.save(
        dataset,
        system_name="spp",
        profile_name="profile_a",
        split="train",
        dataset_name="dataset_001",
    )

    repo.save(
        dataset,
        system_name="spp",
        profile_name="profile_b",
        split="train",
        dataset_name="dataset_002",
    )

    results = repo.list_datasets(
        system_name="spp",
        profile_name="profile_a",
        split="train",
    )

    assert len(results) == 1
    assert results[0].name == "dataset_001"


def test_profile_requires_system_name_when_listing(tmp_path):
    repo = DatasetRepository(root_dir=tmp_path)

    with pytest.raises(ValueError, match="system_name"):
        repo.list_datasets(profile_name="profile_a")


def test_profile_load_does_not_fall_back_to_legacy_layout(tmp_path):
    repo = DatasetRepository(root_dir=tmp_path)
    dataset = make_first_order_dataset()

    repo.save(
        dataset,
        system_name="spp",
        split="train",
        dataset_name="dataset_001",
    )

    with pytest.raises(FileNotFoundError):
        repo.load(
            system_name="spp",
            profile_name="some_profile",
            split="train",
            dataset_name="dataset_001",
        )