import numpy as np

from ips_learning.data.simulator import Simulator
from ips_learning.models.learned_system import (
    LearnedFirstOrderSystem,
    LearnedSecondOrderSystem,
)
from ips_learning.rollout.simulator import LearnedSystemRollout
from ips_learning.rollout.scenarios import RolloutScenarioBuilder
from ips_learning.systems.base import (
    FirstOrderSystemBase,
    SecondOrderSystemBase,
)

class ConstantDriftSystem(FirstOrderSystemBase):
    """
    Analytically solvable first-order system:

        dX/dt = 1.
    """

    def f_env(self, X):
        return np.ones_like(X)

    def kernel(self, R):
        return np.zeros_like(R)

    def interaction_force(self, X):
        return np.zeros_like(X)

# Test that the first-order simulator reproduces constant drift motion.
def test_first_order_simulator_matches_constant_drift_solution():
    system = ConstantDriftSystem(N=2, D=1)
    simulator = Simulator(system)

    u0 = np.array([
        1.0,
        -2.0,
    ])

    t_eval = np.linspace(0.0, 1.0, 6)

    t, X, V, LHS = simulator.simulate_one(
        u0=u0,
        t_eval=t_eval,
    )

    expected_X = np.stack(
        [
            np.array([
                [1.0 + t],
                [-2.0 + t],
            ])
            for t in t_eval
        ],
        axis=0,
    )

    expected_LHS = np.ones_like(expected_X)

    np.testing.assert_allclose(t, t_eval)

    np.testing.assert_allclose(
        X,
        expected_X,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        LHS,
        expected_LHS,
        atol=1e-12,
    )

    assert V is None

    assert X.shape == (6, 2, 1)
    assert LHS.shape == (6, 2, 1)


class ConstantVelocitySystem(SecondOrderSystemBase):
    """
    Analytically solvable second-order system:

        dX/dt = V
        dV/dt = 0.
    """

    def f_env(self, V):
        return np.zeros_like(V)

    def kernel(self, R):
        return np.zeros_like(R)

    def interaction_force(self, X, V):
        return np.zeros_like(X)
# Test that the second-order simulator reproduces constant velocity motion.
def test_second_order_simulator_matches_constant_velocity_solution():
    system = ConstantVelocitySystem(N=2, D=1)
    simulator = Simulator(system)

    X0 = np.array([
        [1.0],
        [-1.0],
    ])

    V0 = np.array([
        [0.5],
        [-0.25],
    ])

    u0 = system.join_state(X0, V0)

    t_eval = np.linspace(0.0, 2.0, 9)

    t, X, V, LHS = simulator.simulate_one(
        u0=u0,
        t_eval=t_eval,
    )

    expected_X = np.stack(
        [
            X0 + t * V0
            for t in t_eval
        ],
        axis=0,
    )

    expected_V = np.stack(
        [
            V0
            for _ in t_eval
        ],
        axis=0,
    )

    expected_LHS = np.zeros_like(expected_V)

    np.testing.assert_allclose(
        X,
        expected_X,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        V,
        expected_V,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        LHS,
        expected_LHS,
        atol=1e-12,
    )

    assert X.shape == (9, 2, 1)
    assert V.shape == (9, 2, 1)
    assert LHS.shape == (9, 2, 1)
# Test that the simulator can handle a batch of initial conditions and returns the expected shapes.

def test_simulator_batch_shapes_and_initial_conditions():
    system = ConstantDriftSystem(N=2, D=1)
    simulator = Simulator(system)

    U0 = np.array([
        [0.0, 1.0],
        [2.0, -1.0],
        [5.0, 3.0],
    ])

    t_eval = np.linspace(0.0, 1.0, 5)

    batch = simulator.simulate_many(
        U0=U0,
        t_eval=t_eval,
        show_progress=False,
    )

    assert batch.X.shape == (3, 5, 2, 1)
    assert batch.LHS.shape == (3, 5, 2, 1)
    assert batch.V is None

    np.testing.assert_allclose(
        batch.t,
        t_eval,
    )

    np.testing.assert_allclose(
        batch.U0,
        U0,
    )

    # First simulated point must equal each supplied initial condition.
    np.testing.assert_allclose(
        batch.X[:, 0, :, 0],
        U0,
        atol=1e-12,
    )

# Test that a learned first-order system with zero interaction and constant environment reproduces constant drift motion.

def test_learned_first_order_rollout_matches_known_solution():
    learned_system = LearnedFirstOrderSystem(
        N=2,
        D=1,
        learned_kernel=lambda R: np.zeros_like(R),
        learned_environment=lambda X: np.ones_like(X),
        interaction_type="energy",
    )

    rollout = LearnedSystemRollout(learned_system)

    u0 = np.array([
        1.0,
        -2.0,
    ])

    t_eval = np.linspace(0.0, 1.0, 6)

    t, X_hat, V_hat = rollout.simulate_one(
        u0=u0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )

    expected_X = np.stack(
        [
            np.array([
                [1.0 + t],
                [-2.0 + t],
            ])
            for t in t_eval
        ],
        axis=0,
    )

    np.testing.assert_allclose(
        t,
        t_eval,
    )

    np.testing.assert_allclose(
        X_hat,
        expected_X,
        atol=1e-7,
    )

    assert V_hat is None

# Test that a learned second-order system with zero interaction and zero environment reproduces constant velocity motion.

def test_learned_second_order_rollout_matches_constant_velocity():
    learned_system = LearnedSecondOrderSystem(
        N=2,
        D=1,
        learned_kernel=lambda R: np.zeros_like(R),
        learned_environment=lambda V: np.zeros_like(V),
        interaction_type="energy",
    )

    rollout = LearnedSystemRollout(learned_system)

    X0 = np.array([
        [1.0],
        [-1.0],
    ])

    V0 = np.array([
        [0.5],
        [-0.25],
    ])

    u0 = learned_system.join_state(
        X0,
        V0,
    )

    t_eval = np.linspace(0.0, 2.0, 9)

    t, X_hat, V_hat = rollout.simulate_one(
        u0=u0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )

    expected_X = np.stack(
        [
            X0 + t * V0
            for t in t_eval
        ],
        axis=0,
    )

    expected_V = np.stack(
        [
            V0
            for _ in t_eval
        ],
        axis=0,
    )

    np.testing.assert_allclose(
        X_hat,
        expected_X,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        V_hat,
        expected_V,
        atol=1e-7,
    )

# Test that the rollout artifact has the expected shape and metadata.

def test_learned_rollout_batch_artifact():
    learned_system = LearnedFirstOrderSystem(
        N=2,
        D=1,
        learned_kernel=lambda R: np.zeros_like(R),
        learned_environment=lambda X: np.ones_like(X),
        interaction_type="energy",
    )

    rollout = LearnedSystemRollout(learned_system)

    U0 = np.array([
        [0.0, 1.0],
        [2.0, -1.0],
    ])

    t_eval = np.linspace(0.0, 1.0, 5)

    artifact = rollout.simulate_many(
        U0=U0,
        t_eval=t_eval,
        show_progress=False,
        metadata={
            "scenario": "unit_test",
        },
    )

    assert artifact.X_hat.shape == (2, 5, 2, 1)
    assert artifact.V_hat is None
    assert artifact.U0.shape == (2, 2)

    np.testing.assert_allclose(
        artifact.t,
        t_eval,
    )

    np.testing.assert_allclose(
        artifact.X_hat[:, 0, :, 0],
        U0,
        atol=1e-12,
    )

    assert artifact.metadata["stage"] == "2.5"
    assert artifact.metadata["artifact_type"] == "rollout"
    assert artifact.metadata["system_order"] == 1
    assert artifact.metadata["n_rollouts"] == 2
    assert artifact.metadata["scenario"] == "unit_test"

#Rollout-scenario builder

def test_longer_horizon_rollout_scenario():
    builder = RolloutScenarioBuilder()

    U0 = np.array([
        [0.0, 1.0],
        [2.0, 3.0],
    ])

    returned_U0, t_eval, metadata = builder.longer_horizon(
        U0=U0,
        t0=0.0,
        T_new=2.0,
        L_new=21,
        metadata={
            "system_name": "test_system",
            "system_order": 1,
        },
    )

    np.testing.assert_array_equal(
        returned_U0,
        U0,
    )

    assert t_eval.shape == (21,)
    assert t_eval[0] == 0.0
    assert t_eval[-1] == 2.0

    assert metadata["scenario"] == "trajectory_prediction"
    assert metadata["horizon_type"] == "extended"
    assert metadata["system_name"] == "test_system"
    assert metadata["system_order"] == 1