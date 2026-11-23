import pytest

from ips_learning.experiments.generate_spp_data import (
    build_spp_dataset_profile,
)


@pytest.mark.parametrize(
    "profile_name",
    [
        "baseline",
        "spp_velocity_grid_clumps",
        "spp_swarm",
        "spp_swarm_spread",
    ],
)
def test_all_spp_profiles_use_canonical_system_name(profile_name):
    profile = build_spp_dataset_profile(profile_name)

    assert profile.system_name == "spp"
    assert profile.profile_name == profile_name
