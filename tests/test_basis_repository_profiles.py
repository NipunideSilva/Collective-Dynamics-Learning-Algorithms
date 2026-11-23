import numpy as np
import pytest

from ips_learning.bases.bundle import BasisBundle
from ips_learning.bases.repository import BasisRepository


def make_basis_bundle():
    # Valid degree-1 B-spline knot vector for 3 basis functions:
    # len(knots) = n_basis + degree + 1 = 5
    knots = [0.0, 0.0, 0.5, 1.0, 1.0]

    metadata = {
        "radial_family": "bspline",
        "radial_degree": 1,
        "n_radial_basis": 3,
        "r_min": 0.0,
        "r_max": 1.0,
        "radial_knots": knots,

        "environment_family": "bspline",
        "environment_degree": 1,
        "n_environment_basis_per_dim": 3,
        "env_min": [0.0, 0.0],
        "env_max": [1.0, 1.0],
        "environment_knots": [
            knots,
            knots,
        ],
    }

    return BasisBundle(
        localization_metadata=metadata,
    )


def test_basis_repository_legacy_layout(tmp_path):
    repo = BasisRepository(root_dir=tmp_path)
    bundle = make_basis_bundle()

    path = repo.save(
        bundle,
        system_name="test_system",
        split="train_fit",
        basis_name="basis_001",
    )

    expected = (
        tmp_path
        / "test_system"
        / "train_fit"
        / "basis_001"
    )

    assert path == expected

    loaded = repo.load(
        system_name="test_system",
        split="train_fit",
        basis_name="basis_001",
    )

    assert loaded.localization_metadata["system_name"] == "test_system"
    assert loaded.localization_metadata["split"] == "train_fit"
    assert loaded.localization_metadata["basis_name"] == "basis_001"
    assert "profile_name" not in loaded.localization_metadata

    assert loaded.radial_basis is not None
    assert loaded.environment_basis is not None

    assert loaded.radial_basis.n_basis == 3
    assert loaded.environment_basis.n_basis_per_dim == 3


def test_basis_repository_profile_layout(tmp_path):
    repo = BasisRepository(root_dir=tmp_path)
    bundle = make_basis_bundle()

    path = repo.save(
        bundle,
        system_name="spp",
        profile_name="spp_swarm_spread",
        split="train_fit",
        basis_name="basis_train_fit_trial_001",
    )

    expected = (
        tmp_path
        / "spp"
        / "spp_swarm_spread"
        / "train_fit"
        / "basis_train_fit_trial_001"
    )

    assert path == expected

    loaded = repo.load(
        system_name="spp",
        profile_name="spp_swarm_spread",
        split="train_fit",
        basis_name="basis_train_fit_trial_001",
    )

    metadata = loaded.localization_metadata

    assert metadata["system_name"] == "spp"
    assert metadata["profile_name"] == "spp_swarm_spread"
    assert metadata["split"] == "train_fit"
    assert metadata["basis_name"] == "basis_train_fit_trial_001"

    assert loaded.radial_basis is not None
    assert loaded.environment_basis is not None

    np.testing.assert_allclose(
        loaded.radial_basis.knots,
        np.array([0.0, 0.0, 0.5, 1.0, 1.0]),
    )

# tests/test_basis_repository_profiles.py

def test_basis_profile_load_does_not_fall_back_to_legacy_layout(tmp_path):
    repo = BasisRepository(root_dir=tmp_path)
    bundle = make_basis_bundle()

    repo.save(
        bundle,
        system_name="spp",
        split="train_fit",
        basis_name="basis_001",
    )

    with pytest.raises(FileNotFoundError):
        repo.load(
            system_name="spp",
            profile_name="some_profile",
            split="train_fit",
            basis_name="basis_001",
        )
