import numpy as np
import pytest

from ips_learning.fitting.solver import LinearSystemSolver


def test_lstsq_recovers_exact_coefficients():
    """
    Verify that the least-squares solver recovers a known coefficient
    vector from an exactly consistent full-rank system A theta = b.
    """
    A = np.array([
        [4.0, 1.0],
        [1.0, 3.0],
    ])

    theta_true = np.array([2.0, -1.0])
    b = A @ theta_true

    solver = LinearSystemSolver(method="lstsq")
    theta_hat, diagnostics = solver.solve(A, b)

    np.testing.assert_allclose(
        theta_hat,
        theta_true,
        atol=1e-12,
    )

    assert diagnostics.method == "lstsq"
    assert diagnostics.rank == 2
    assert np.isfinite(diagnostics.condition_number)


def test_ridge_matches_regularized_linear_system():
    """
    Ridge solves

        (A + lambda I) theta = b.

    Compare the class output directly with the corresponding
    NumPy solution.
    """
    A = np.array([
        [4.0, 1.0],
        [1.0, 3.0],
    ])

    b = np.array([7.0, -1.0])
    ridge_lambda = 1e-3

    expected = np.linalg.solve(
        A + ridge_lambda * np.eye(2),
        b,
    )

    solver = LinearSystemSolver(
        method="ridge",
        ridge_lambda=ridge_lambda,
    )

    theta_hat, diagnostics = solver.solve(A, b)

    np.testing.assert_allclose(
        theta_hat,
        expected,
        atol=1e-12,
    )

    assert diagnostics.method == "ridge"
    assert diagnostics.ridge_lambda == ridge_lambda


def test_truncated_svd_removes_small_singular_direction():
    """
    The SVD solver should discard singular values below svd_tol.

    Here the second singular value is 1e-14 and the tolerance is
    1e-12, so the second coefficient should be set to zero.
    """
    A = np.array([
        [2.0, 0.0],
        [0.0, 1e-14],
    ])

    b = np.array([
        4.0,
        3.0,
    ])

    solver = LinearSystemSolver(
        method="svd",
        svd_tol=1e-12,
    )

    theta_hat, diagnostics = solver.solve(A, b)

    expected = np.array([
        2.0,
        0.0,
    ])

    np.testing.assert_allclose(
        theta_hat,
        expected,
        atol=1e-12,
    )

    assert diagnostics.method == "svd"
    assert diagnostics.svd_tol == 1e-12
    # This is the raw diagnostic rank of A,
    # independent of the SVD truncation threshold.
    assert diagnostics.rank == 2


def test_unknown_solver_method_raises_error():
    solver = LinearSystemSolver(method="not_a_solver")

    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(ValueError, match="Unknown solver method"):
        solver.solve(A, b)