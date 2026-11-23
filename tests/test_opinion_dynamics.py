import numpy as np

from ips_learning.systems.opinion_dynamics import OpinionDynamicsSystem


def test_two_agent_opinion_interaction():
    """
    For N=2 with opinions X1=0 and X2=0.5, the pairwise distance
    lies inside the inner confidence region, so phi(0.5)=1.

    Therefore,

        F1 = (1/2) * 1 * (0.5 - 0) = 0.25
        F2 = (1/2) * 1 * (0 - 0.5) = -0.25
    """
    system = OpinionDynamicsSystem(N=2, D=1)

    X = np.array([
        [0.0],
        [0.5],
    ])

    force = system.interaction_force(X)

    expected = np.array([
        [0.25],
        [-0.25],
    ])

    np.testing.assert_allclose(force, expected)

def test_equal_opinions_have_zero_interaction():
    """
    If every agent has the same opinion, all pairwise differences
    are zero, so the collective interaction force must be zero.
    """
    system = OpinionDynamicsSystem(N=5, D=1)

    X = np.zeros((5, 1))

    force = system.interaction_force(X)

    expected = np.zeros_like(X)

    np.testing.assert_allclose(force, expected)

def test_opinion_kernel_regions():
    system = OpinionDynamicsSystem(N=2, D=1)

    R = np.array([
        0.5,
        0.8,
        1.2,
    ])

    values = system.kernel(R)

    expected = np.array([
        1.0,
        0.1,
        0.0,
    ])

    np.testing.assert_allclose(values, expected)