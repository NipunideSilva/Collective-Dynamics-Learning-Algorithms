import numpy as np

from ips_learning.systems.base import FirstOrderSystemBase


class DummyFirstOrderSystem(FirstOrderSystemBase):
    """Minimal concrete system used only to test shared geometry utilities."""

    def f_env(self, X: np.ndarray) -> np.ndarray:
        return np.zeros_like(X)

    def kernel(self, R: np.ndarray) -> np.ndarray:
        return np.ones_like(R)

    def interaction_force(self, X: np.ndarray) -> np.ndarray:
        return np.zeros_like(X)


def test_pairwise_position_differences():
    """
    Verify

        R_D[i, j] = X_j - X_i
        R[i, j]   = ||X_j - X_i||_2

    for a hand-computable 2D configuration.
    """
    system = DummyFirstOrderSystem(N=3, D=2)

    X = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [-1.0, 0.0],
    ])

    R_D, R = system.pairwise_position_differences(X)

    expected_R_D = np.array([
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [-1.0, 0.0],
        ],
        [
            [-3.0, -4.0],
            [0.0, 0.0],
            [-4.0, -4.0],
        ],
        [
            [1.0, 0.0],
            [4.0, 4.0],
            [0.0, 0.0],
        ],
    ])

    expected_R = np.array([
        [[0.0], [5.0], [1.0]],
        [[5.0], [0.0], [np.sqrt(32.0)]],
        [[1.0], [np.sqrt(32.0)], [0.0]],
    ])

    np.testing.assert_allclose(R_D, expected_R_D)
    np.testing.assert_allclose(R, expected_R)


def test_pairwise_geometry_symmetry():
    system = DummyFirstOrderSystem(N=3, D=2)

    X = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
        [-1.0, 0.0],
    ])

    R_D, R = system.pairwise_position_differences(X)

    # Relative vectors are antisymmetric:
    # X_j - X_i = -(X_i - X_j)
    np.testing.assert_allclose(
        R_D,
        -np.swapaxes(R_D, 0, 1),
    )

    # Distances are symmetric.
    np.testing.assert_allclose(
        R,
        np.swapaxes(R, 0, 1),
    )

    # Self-distance and self-difference are zero.
    diagonal = np.arange(system.N)

    np.testing.assert_allclose(
        R_D[diagonal, diagonal],
        0.0,
    )

    np.testing.assert_allclose(
        R[diagonal, diagonal],
        0.0,
    )