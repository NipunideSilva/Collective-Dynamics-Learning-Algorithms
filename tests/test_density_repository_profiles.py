import json
import pytest
import numpy as np

from ips_learning.statistics.density_repository import DensityRepository


def test_density_repository_legacy_layout(tmp_path):
    repo = DensityRepository(root=tmp_path)

    dist_est = {"grid": np.array([0.0, 1.0])}
    vel_est = {"grid": np.array([-1.0, 1.0])}

    path = repo.save(
        dist_est,
        vel_est,
        system_name="test_system",
        split="kde",
        name="density_001",
        metadata={"M": 100},
    )

    expected = (
        tmp_path
        / "test_system"
        / "kde"
        / "density_001"
    )

    assert path == expected

    loaded_dist, loaded_vel = repo.load(
        system_name="test_system",
        split="kde",
        name="density_001",
    )

    np.testing.assert_allclose(
        loaded_dist["grid"],
        dist_est["grid"],
    )
    np.testing.assert_allclose(
        loaded_vel["grid"],
        vel_est["grid"],
    )

    with open(path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["system_name"] == "test_system"
    assert metadata["split"] == "kde"
    assert metadata["density_name"] == "density_001"
    assert "profile_name" not in metadata


def test_density_repository_profile_layout(tmp_path):
    repo = DensityRepository(root=tmp_path)

    dist_est = {"grid": np.array([0.0, 0.5, 1.0])}
    vel_est = {"grid": np.array([-2.0, 0.0, 2.0])}

    path = repo.save(
        dist_est,
        vel_est,
        system_name="spp",
        profile_name="spp_swarm_spread",
        split="kde",
        name="dataset_kde",
        metadata={"M": 2000},
    )

    expected = (
        tmp_path
        / "spp"
        / "spp_swarm_spread"
        / "kde"
        / "dataset_kde"
    )

    assert path == expected

    loaded_dist, loaded_vel = repo.load(
        system_name="spp",
        profile_name="spp_swarm_spread",
        split="kde",
        name="dataset_kde",
    )

    np.testing.assert_allclose(
        loaded_dist["grid"],
        dist_est["grid"],
    )
    np.testing.assert_allclose(
        loaded_vel["grid"],
        vel_est["grid"],
    )

    with open(path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["system_name"] == "spp"
    assert metadata["profile_name"] == "spp_swarm_spread"
    assert metadata["split"] == "kde"
    assert metadata["density_name"] == "dataset_kde"

# tests/test_density_repository_profiles.py

def test_density_profile_load_does_not_fall_back_to_legacy_layout(tmp_path):
    repo = DensityRepository(root=tmp_path)

    dist_est = {"grid": np.array([0.0, 1.0])}
    vel_est = {"grid": np.array([-1.0, 1.0])}

    repo.save(
        dist_est,
        vel_est,
        system_name="spp",
        split="kde",
        name="density_001",
        metadata={"M": 100},
    )

    with pytest.raises(FileNotFoundError):
        repo.load(
            system_name="spp",
            profile_name="some_profile",
            split="kde",
            name="density_001",
        )