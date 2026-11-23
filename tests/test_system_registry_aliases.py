import pytest

from ips_learning.systems.registry import (
    canonical_system_name,
    get_system_factory,
    build_true_system_from_metadata,
)
from ips_learning.systems.spp import SPPSystem


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("spp", "spp"),
        ("spp_swarm", "spp"),
        ("spp_rings", "spp"),
        ("levi", "levi"),
        ("kuramoto", "kuramoto"),
    ],
)
def test_canonical_system_name(name, expected):
    assert canonical_system_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "spp",
        "spp_swarm",
        "spp_rings",
    ],
)
def test_spp_legacy_names_resolve_to_spp_factory(name):
    assert get_system_factory(name) is SPPSystem


@pytest.mark.parametrize(
    "system_name",
    [
        "spp",
        "spp_swarm",
        "spp_rings",
    ],
)
def test_spp_metadata_aliases_build_same_system(system_name):
    metadata = {
        "system_name": system_name,
        "N": 10,
        "D": 2,
        "cr": 1.0,
        "lr": 0.5,
        "ca": 0.5,
        "la": 2.0,
        "a": 1.6,
        "b": 0.5,
        "eps_r": 1e-1,
    }

    system = build_true_system_from_metadata(metadata)

    assert isinstance(system, SPPSystem)
    assert system.system_name == "spp"
