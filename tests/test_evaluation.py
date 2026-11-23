import numpy as np
import pytest

from types import SimpleNamespace

from ips_learning.evaluation.function_errors import (
    KernelErrorEvaluator,
    EnvironmentErrorEvaluator,
)
from ips_learning.evaluation.residuals import ResidualEvaluator
from ips_learning.evaluation.trajectory_errors import TrajectoryErrorEvaluator
from ips_learning.evaluation.weighting import WeightNormalizer


# Weight normalization

def test_weight_normalizer():
    weights = np.array([1.0, 2.0, 1.0])

    normalized = WeightNormalizer.normalize(weights)

    expected = np.array([
        0.25,
        0.50,
        0.25,
    ])

    np.testing.assert_allclose(
        normalized,
        expected,
    )

    assert np.sum(normalized) == pytest.approx(1.0)


def test_weight_normalizer_rejects_nonpositive_total():
    with pytest.raises(
        ValueError,
        match="Weights must sum to a positive value",
    ):
        WeightNormalizer.normalize(
            np.array([0.0, 0.0])
        )
# Kernel-error metric

def test_kernel_error_without_radial_measure():
    evaluator = KernelErrorEvaluator()

    true_kernel = (
        lambda r: 2.0 * np.ones_like(r)
    )

    learned_kernel = (
        lambda r: np.ones_like(r)
    )

    r_grid = np.array([
        0.25,
        0.50,
        0.75,
    ])

    result = evaluator.evaluate(
        true_kernel=true_kernel,
        learned_kernel=learned_kernel,
        r_grid=r_grid,
        include_radial_measure=False,
    )

    assert result["E_phi"] == pytest.approx(1.0)

    assert result["E_phi_relative"] == pytest.approx(
        0.5
    )

    assert result["mode"] == "grid"

    np.testing.assert_allclose(
        result["phi_true"],
        np.array([2.0, 2.0, 2.0]),
    )

    np.testing.assert_allclose(
        result["phi_hat"],
        np.array([1.0, 1.0, 1.0]),
    )

# Test the radial measure

def test_kernel_error_with_radial_measure():
    evaluator = KernelErrorEvaluator()

    true_kernel = (
        lambda r: 2.0 * np.ones_like(r)
    )

    learned_kernel = (
        lambda r: np.ones_like(r)
    )

    result = evaluator.evaluate(
        true_kernel=true_kernel,
        learned_kernel=learned_kernel,
        r_grid=np.array([1.0, 2.0]),
        include_radial_measure=True,
    )

    assert result["E_phi"] == pytest.approx(
        np.sqrt(2.5)
    )

    # True norm:
    #
    # sqrt(mean([4*1^2, 4*2^2]))
    # = sqrt(10)
    #
    # therefore relative error = 0.5.
    assert result["E_phi_relative"] == pytest.approx(
        0.5
    )

# Environmental-force error

def test_environment_error_on_known_samples():
    evaluator = EnvironmentErrorEvaluator()

    true_env = lambda Z: 2.0 * Z
    learned_env = lambda Z: Z

    Z_samples = np.array([
        [1.0],
        [2.0],
    ])

    result = evaluator.evaluate(
        true_env=true_env,
        learned_env=learned_env,
        Z_samples=Z_samples,
    )

    assert result["E_env"] == pytest.approx(
        np.sqrt(2.5)
    )

    assert result["E_env_relative"] == pytest.approx(
        0.5
    )

    assert result["mode"] == "samples"

    np.testing.assert_allclose(
        result["f_true"],
        np.array([
            [2.0],
            [4.0],
        ]),
    )

    np.testing.assert_allclose(
        result["f_hat"],
        np.array([
            [1.0],
            [2.0],
        ]),
    )

# First-order residual error

class ZeroFirstOrderModel:
    def lhs(self, X):
        return np.zeros_like(X)


def test_first_order_residual_error():
    X = np.array([
        [
            [[0.0], [1.0]],
            [[0.5], [1.5]],
        ]
    ])

    LHS = np.ones_like(X)

    features = SimpleNamespace(
        X=X,
        V=None,
        LHS=LHS,
        metadata={
            "system_order": 1,
        },
    )

    result = ResidualEvaluator().evaluate(
        features=features,
        learned_model=ZeroFirstOrderModel(),
    )

    assert result["E_res"] == pytest.approx(1.0)

    assert result["E_res_relative"] == pytest.approx(
        1.0
    )

    np.testing.assert_allclose(
        result["per_trajectory_E_res"],
        np.array([1.0]),
    )

    assert result["system_order"] == 1
    assert result["n_trajectories"] == 1
    assert result["n_time_points"] == 2
    assert result["N"] == 2
    assert result["D"] == 1

# Second-order residual
class ExactSecondOrderModel:
    def acceleration(self, X, V):
        return 2.0 * V


def test_second_order_exact_residual_is_zero():
    X = np.array([
        [
            [[0.0], [1.0]],
            [[0.2], [1.3]],
        ]
    ])

    V = np.array([
        [
            [[0.5], [-0.25]],
            [[0.7], [0.10]],
        ]
    ])

    LHS = 2.0 * V

    features = SimpleNamespace(
        X=X,
        V=V,
        LHS=LHS,
        metadata={
            "system_order": 2,
        },
    )

    result = ResidualEvaluator().evaluate(
        features=features,
        learned_model=ExactSecondOrderModel(),
    )

    assert result["E_res"] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert result["E_res_relative"] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert result["system_order"] == 2

# Pointwise trajectory error
def test_pointwise_trajectory_error_curve():
    X_true = np.zeros(
        (3, 2, 1)
    )

    X_hat = np.array([
        [[1.0], [-1.0]],
        [[2.0], [ 0.0]],
        [[2.0], [ 2.0]],
    ])

    e_t = TrajectoryErrorEvaluator.pointwise_error_curve(
        X_true,
        X_hat,
    )

    expected = np.array([
        1.0,
        np.sqrt(2.0),
        2.0,
    ])

    np.testing.assert_allclose(
        e_t,
        expected,
    )

# Reconstruction vs prediction horizon

def test_trajectory_reconstruction_and_prediction_errors():
    evaluator = TrajectoryErrorEvaluator()

    t = np.array([
        0.0,
        0.5,
        1.0,
    ])

    X_true = np.zeros(
        (1, 3, 2, 1)
    )

    X_hat = np.array([
        [
            [[1.0], [-1.0]],
            [[2.0], [ 0.0]],
            [[2.0], [ 2.0]],
        ]
    ])

    result = evaluator.evaluate_batch(
        t_true=t,
        X_true_batch=X_true,
        t_hat=t,
        X_hat_batch=X_hat,
        split_index=1,
    )

    expected_curve = np.array([
        1.0,
        np.sqrt(2.0),
        2.0,
    ])

    np.testing.assert_allclose(
        result["pointwise_error_curves"][0],
        expected_curve,
    )

    np.testing.assert_allclose(
        result["mean_pointwise_curve"],
        expected_curve,
    )

    assert (
        result["reconstruction_Linf_mean"]
        == pytest.approx(np.sqrt(2.0))
    )

    assert (
        result["prediction_Linf_mean"]
        == pytest.approx(2.0)
    )

    assert (
        result["reconstruction_Linf_std"]
        == pytest.approx(0.0)
    )

    assert (
        result["prediction_Linf_std"]
        == pytest.approx(0.0)
    )
