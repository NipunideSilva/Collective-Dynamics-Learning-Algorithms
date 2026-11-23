import numpy as np
import pytest

from ips_learning.features.feature_builder import FeatureDataset
from ips_learning.features.feature_repository import FeatureRepository


def make_feature_dataset():
    M, L, N, D = 2, 3, 4, 1

    return FeatureDataset(
        t=np.array([0.0, 0.1, 0.2]),
        X=np.zeros((M, L, N, D)),
        V=None,
        LHS=np.ones((M, L, N, D)),
        R=np.zeros((M, L, N, N, 1)),
        R_D=np.zeros((M, L, N, N, D)),
        VR_D=None,
        metadata={
            "system_name": "test_system",
            "system_order": 1,
            "interaction_type": "energy",
            "N": N,
            "D": D,
        },
    )


def test_feature_repository_legacy_layout(tmp_path):
    repo = FeatureRepository(root_dir=tmp_path)
    features = make_feature_dataset()

    path = repo.save(
        features,
        system_name="test_system",
        split="train_fit",
        feature_name="features_001",
    )

    expected = (
        tmp_path
        / "test_system"
        / "train_fit"
        / "features_001"
    )

    assert path == expected

    loaded = repo.load(
        system_name="test_system",
        split="train_fit",
        feature_name="features_001",
    )

    assert loaded.metadata["system_name"] == "test_system"
    assert loaded.metadata["split"] == "train_fit"
    assert loaded.metadata["feature_name"] == "features_001"
    assert "profile_name" not in loaded.metadata

    np.testing.assert_allclose(loaded.t, features.t)
    np.testing.assert_allclose(loaded.X, features.X)
    np.testing.assert_allclose(loaded.LHS, features.LHS)
    np.testing.assert_allclose(loaded.R, features.R)
    np.testing.assert_allclose(loaded.R_D, features.R_D)

    assert loaded.V is None
    assert loaded.VR_D is None


def test_feature_repository_profile_layout(tmp_path):
    repo = FeatureRepository(root_dir=tmp_path)
    features = make_feature_dataset()

    path = repo.save(
        features,
        system_name="spp",
        profile_name="spp_swarm_spread",
        split="train_fit",
        feature_name="features_train_fit_trial_001",
    )

    expected = (
        tmp_path
        / "spp"
        / "spp_swarm_spread"
        / "train_fit"
        / "features_train_fit_trial_001"
    )

    assert path == expected

    loaded = repo.load(
        system_name="spp",
        profile_name="spp_swarm_spread",
        split="train_fit",
        feature_name="features_train_fit_trial_001",
    )

    assert loaded.metadata["system_name"] == "spp"
    assert loaded.metadata["profile_name"] == "spp_swarm_spread"
    assert loaded.metadata["split"] == "train_fit"
    assert (
        loaded.metadata["feature_name"]
        == "features_train_fit_trial_001"
    )

    np.testing.assert_allclose(loaded.t, features.t)
    np.testing.assert_allclose(loaded.X, features.X)
    np.testing.assert_allclose(loaded.LHS, features.LHS)
    np.testing.assert_allclose(loaded.R, features.R)
    np.testing.assert_allclose(loaded.R_D, features.R_D)


# tests/test_feature_repository_profiles.py

def test_feature_profile_load_does_not_fall_back_to_legacy_layout(tmp_path):
    repo = FeatureRepository(root_dir=tmp_path)
    features = make_feature_dataset()

    repo.save(
        features,
        system_name="spp",
        split="train_fit",
        feature_name="features_001",
    )

    with pytest.raises(FileNotFoundError):
        repo.load(
            system_name="spp",
            profile_name="some_profile",
            split="train_fit",
            feature_name="features_001",
        )
