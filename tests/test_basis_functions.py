import numpy as np
import pytest

from ips_learning.bases.radial import BSplineRadialBasisBuilder
from ips_learning.bases.environment import BSplineEnvironmentBasisBuilder


def test_radial_bspline_shape_and_partition_of_unity():
    """
    B-spline basis functions should have the expected shape and,
    inside the fitted domain, sum to one.
    """
    basis = BSplineRadialBasisBuilder().build(
        degree=2,
        n_basis=4,
        r_min=0.0,
        r_max=1.0,
    )

    r = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    values = basis.evaluate(r)

    assert values.shape == (5, 4)

    np.testing.assert_allclose(
        values.sum(axis=-1),
        np.ones(5),
        atol=1e-12,
    )


def test_radial_bspline_accepts_trailing_singleton_dimension():
    """
    Radial distances in the pipeline often have shape (..., 1).
    The evaluator should remove that final singleton coordinate
    and return (..., n_basis).
    """
    basis = BSplineRadialBasisBuilder().build(
        degree=2,
        n_basis=4,
        r_min=0.0,
        r_max=1.0,
    )

    r = np.array([
        [0.0],
        [0.5],
        [1.0],
    ])

    values = basis.evaluate(r)

    assert values.shape == (3, 4)


def test_environment_bspline_shape_and_partition_of_unity():
    """
    Environment bases are built independently for each spatial
    coordinate.

    For D=2, evaluate_per_dim should therefore return two arrays,
    each with shape (..., n_basis_per_dim).
    """
    basis = BSplineEnvironmentBasisBuilder().build(
        degree=2,
        n_basis_per_dim=4,
        domain_min=np.array([-1.0, 0.0]),
        domain_max=np.array([1.0, 2.0]),
    )

    Z = np.array([
        [-1.0, 0.0],
        [0.0, 1.0],
        [1.0, 2.0],
    ])

    values = basis.evaluate_per_dim(Z)

    assert len(values) == 2
    assert values[0].shape == (3, 4)
    assert values[1].shape == (3, 4)

    np.testing.assert_allclose(
        values[0].sum(axis=-1),
        np.ones(3),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        values[1].sum(axis=-1),
        np.ones(3),
        atol=1e-12,
    )


def test_environment_basis_rejects_wrong_dimension():
    """
    A D=2 environment basis should reject data whose final
    coordinate dimension is not 2.
    """
    basis = BSplineEnvironmentBasisBuilder().build(
        degree=2,
        n_basis_per_dim=4,
        domain_min=np.array([-1.0, 0.0]),
        domain_max=np.array([1.0, 2.0]),
    )

    Z = np.zeros((5, 3))

    with pytest.raises(ValueError):
        basis.evaluate_per_dim(Z)
        