import numpy as np
import pytest

from ips_learning.data.dataset_repository import RawTrajectoryDataset
from ips_learning.data.noise import AdditiveGaussianNoise

from scipy.ndimage import uniform_filter1d
from ips_learning.data.sliding_window_noise import SlidingWindowNoise
from ips_learning.data.kalman_filter_noise import KalmanFilterNoise


def make_first_order_dataset():
    """Small deterministic first-order dataset for noise tests."""
    t = np.array([0.0, 0.5, 1.0])

    X = np.array([
        [
            [[1.0], [2.0]],
            [[2.0], [3.0]],
            [[3.0], [4.0]],
        ]
    ])

    LHS = np.array([
        [
            [[0.5], [1.0]],
            [[1.0], [1.5]],
            [[1.5], [2.0]],
        ]
    ])

    U0 = np.array([[1.0, 2.0]])

    return RawTrajectoryDataset(
        t=t,
        X=X,
        V=None,
        LHS=LHS,
        U0=U0,
        metadata={
            "system_name": "test_system",
            "system_order": 1,
        },
    )


@pytest.mark.parametrize(
    "noise_type",
    [
        "additive_gaussian",
        "multiplicative_uniform",
    ],
)
def test_zero_noise_leaves_dataset_unchanged(noise_type):
    """
    A noise level of zero must leave all numerical observations
    unchanged.
    """
    dataset = make_first_order_dataset()

    noise = AdditiveGaussianNoise(
        noise_level=0.0,
        seed=123,
        noise_type=noise_type,
    )

    perturbed = noise.perturb(dataset)

    np.testing.assert_allclose(perturbed.X, dataset.X)
    np.testing.assert_allclose(perturbed.LHS, dataset.LHS)

    assert perturbed.V is None

    # Time grid and initial conditions are not perturbed.
    np.testing.assert_array_equal(perturbed.t, dataset.t)
    np.testing.assert_array_equal(perturbed.U0, dataset.U0)


def test_additive_gaussian_noise_is_reproducible():
    """
    Two independent noise objects with the same seed must produce
    identical noisy datasets.
    """
    dataset = make_first_order_dataset()

    noise_1 = AdditiveGaussianNoise(
        noise_level=0.1,
        seed=42,
        noise_type="additive_gaussian",
    )

    noise_2 = AdditiveGaussianNoise(
        noise_level=0.1,
        seed=42,
        noise_type="additive_gaussian",
    )

    perturbed_1 = noise_1.perturb(dataset)
    perturbed_2 = noise_2.perturb(dataset)

    np.testing.assert_allclose(
        perturbed_1.X,
        perturbed_2.X,
    )

    np.testing.assert_allclose(
        perturbed_1.LHS,
        perturbed_2.LHS,
    )


def test_additive_gaussian_matches_defined_formula():
    """
    Verify the implemented model

        A_noisy = A + zeta * std(A) * eta,

    where eta ~ N(0,1).
    """
    dataset = make_first_order_dataset()

    zeta = 0.2
    seed = 7

    # Reproduce the expected RNG sequence independently.
    rng = np.random.default_rng(seed)

    expected_X = (
        dataset.X
        + zeta
        * np.std(dataset.X)
        * rng.normal(size=dataset.X.shape)
    )

    # For first-order data V is None, so LHS is the next RNG draw.
    expected_LHS = (
        dataset.LHS
        + zeta
        * np.std(dataset.LHS)
        * rng.normal(size=dataset.LHS.shape)
    )

    noise = AdditiveGaussianNoise(
        noise_level=zeta,
        seed=seed,
        noise_type="additive_gaussian",
    )

    perturbed = noise.perturb(dataset)

    np.testing.assert_allclose(
        perturbed.X,
        expected_X,
    )

    np.testing.assert_allclose(
        perturbed.LHS,
        expected_LHS,
    )


def test_multiplicative_uniform_noise_stays_within_bounds():
    """
    The multiplicative model is

        A_noisy = A * (1 + epsilon),

    where

        epsilon in [-zeta, zeta].

    Therefore, for positive A,

        1-zeta <= A_noisy/A <= 1+zeta.
    """
    dataset = make_first_order_dataset()

    zeta = 0.15

    noise = AdditiveGaussianNoise(
        noise_level=zeta,
        seed=11,
        noise_type="multiplicative_uniform",
    )

    perturbed = noise.perturb(dataset)

    X_ratio = perturbed.X / dataset.X
    LHS_ratio = perturbed.LHS / dataset.LHS

    assert np.all(X_ratio >= 1.0 - zeta)
    assert np.all(X_ratio <= 1.0 + zeta)

    assert np.all(LHS_ratio >= 1.0 - zeta)
    assert np.all(LHS_ratio <= 1.0 + zeta)


def test_noise_metadata_is_recorded():
    dataset = make_first_order_dataset()

    noise = AdditiveGaussianNoise(
        noise_level=0.1,
        seed=42,
        noise_type="additive_gaussian",
    )

    perturbed = noise.perturb(dataset)

    assert perturbed.metadata["noise_type"] == "additive_gaussian"
    assert perturbed.metadata["noise_level"] == pytest.approx(0.1)

    assert (
        perturbed.metadata["noise_model"]
        == "additive_gaussian_scaled_by_global_std"
    )

    # Original metadata should still be present.
    assert perturbed.metadata["system_name"] == "test_system"
    assert perturbed.metadata["system_order"] == 1


def test_invalid_noise_type_is_rejected():
    with pytest.raises(ValueError):
        AdditiveGaussianNoise(
            noise_level=0.1,
            noise_type="unknown_noise_model",
        )

#--------Sliding Window Noise Tests --------------------------------------------------

def test_sliding_window_first_order_reconstructs_lhs():
    """
    For a first-order system:

        X -> smooth X -> dX/dt -> smooth LHS.

    Fix the window at 3 so the expected result can be computed
    independently.
    """
    t = np.linspace(0.0, 1.0, 5)

    X = np.array([
        [
            [[0.0], [1.0]],
            [[0.2], [1.1]],
            [[0.5], [1.4]],
            [[0.9], [1.8]],
            [[1.4], [2.3]],
        ]
    ])

    dataset = RawTrajectoryDataset(
        t=t,
        X=X,
        V=None,
        LHS=np.zeros_like(X),
        U0=np.array([[0.0, 1.0]]),
        metadata={
            "system_name": "test_first_order",
            "system_order": 1,
        },
    )

    noise = SlidingWindowNoise(
        noise_level=0.0,
        seed=123,
        noise_type="additive_gaussian",
    )

    # Isolate the filtering/reconstruction algorithm from the
    # automatic window-selection heuristic.
    noise.estimate_dataset_window = (
        lambda *_args, **_kwargs: 3
    )

    result = noise.perturb(dataset)

    expected_X = uniform_filter1d(
        X,
        size=3,
        axis=1,
        mode="nearest",
    )

    expected_LHS_raw = np.gradient(
        expected_X,
        t,
        axis=1,
        edge_order=2,
    )

    expected_LHS = uniform_filter1d(
        expected_LHS_raw,
        size=3,
        axis=1,
        mode="nearest",
    )

    np.testing.assert_allclose(
        result.X,
        expected_X,
    )

    np.testing.assert_allclose(
        result.LHS,
        expected_LHS,
    )

    assert result.V is None

    assert result.metadata["smoothing_windows"] == {
        "X": 3,
        "LHS": 3,
    }


def test_sliding_window_second_order_reconstructs_velocity_and_lhs():
    """
    For a second-order system:

        X
        -> smooth X
        -> differentiate to obtain V
        -> smooth V
        -> differentiate to obtain LHS
        -> smooth LHS.

    The current implementation reconstructs V from X rather than
    smoothing the supplied dataset.V directly.
    """
    t = np.linspace(0.0, 1.0, 5)

    X = np.array([
        [
            [[0.0], [1.0]],
            [[0.1], [1.2]],
            [[0.4], [1.5]],
            [[0.9], [1.9]],
            [[1.6], [2.4]],
        ]
    ])

    # V only needs to exist to identify this as second-order.
    V = np.zeros_like(X)

    dataset = RawTrajectoryDataset(
        t=t,
        X=X,
        V=V,
        LHS=np.zeros_like(X),
        U0=np.array([[0.0, 1.0, 0.0, 0.0]]),
        metadata={
            "system_name": "test_second_order",
            "system_order": 2,
        },
    )

    noise = SlidingWindowNoise(
        noise_level=0.0,
        seed=123,
        noise_type="additive_gaussian",
    )

    noise.estimate_dataset_window = (
        lambda *_args, **_kwargs: 3
    )

    result = noise.perturb(dataset)

    expected_X = uniform_filter1d(
        X,
        size=3,
        axis=1,
        mode="nearest",
    )

    V_raw = np.gradient(
        expected_X,
        t,
        axis=1,
        edge_order=2,
    )

    expected_V = uniform_filter1d(
        V_raw,
        size=3,
        axis=1,
        mode="nearest",
    )

    LHS_raw = np.gradient(
        expected_V,
        t,
        axis=1,
        edge_order=2,
    )

    expected_LHS = uniform_filter1d(
        LHS_raw,
        size=3,
        axis=1,
        mode="nearest",
    )

    np.testing.assert_allclose(
        result.X,
        expected_X,
    )

    np.testing.assert_allclose(
        result.V,
        expected_V,
    )

    np.testing.assert_allclose(
        result.LHS,
        expected_LHS,
    )

    assert result.metadata["smoothing_windows"] == {
        "X": 3,
        "V": 3,
        "LHS": 3,
    }


def test_sliding_window_estimator_returns_valid_odd_window():
    rng = np.random.default_rng(123)

    X = rng.normal(
        size=(2, 21, 3, 2),
    )

    noise = SlidingWindowNoise(
        noise_level=0.1,
        seed=123,
    )

    window = noise.estimate_dataset_window(
        X,
        max_window=11,
    )

    assert window in {3, 5, 7, 9, 11}
    assert window % 2 == 1


def test_sliding_window_preserves_time_initial_conditions_and_metadata():
    t = np.linspace(0.0, 1.0, 5)

    X = np.ones((1, 5, 2, 1))

    U0 = np.array([
        [1.0, 1.0],
    ])

    dataset = RawTrajectoryDataset(
        t=t,
        X=X,
        V=None,
        LHS=np.zeros_like(X),
        U0=U0,
        metadata={
            "system_name": "test_system",
            "system_order": 1,
            "split": "train",
        },
    )

    noise = SlidingWindowNoise(
        noise_level=0.0,
        seed=123,
    )

    result = noise.perturb(dataset)

    np.testing.assert_array_equal(
        result.t,
        dataset.t,
    )

    np.testing.assert_array_equal(
        result.U0,
        dataset.U0,
    )

    assert result.metadata["system_name"] == "test_system"
    assert result.metadata["system_order"] == 1
    assert result.metadata["split"] == "train"

    assert result.metadata["noise_type"] == "additive_gaussian"
    assert result.metadata["noise_level"] == 0.0
    assert "smoothing_windows" in result.metadata


    #--- Kalman Filter Noise Tests --------------------------------------------------
def test_kalman_first_order_constant_trajectory():
    """
    A constant first-order trajectory should remain constant and
    have zero reconstructed derivative when noise_level=0.
    """
    t = np.linspace(0.0, 1.0, 7)

    X = np.full(
        (1, 7, 2, 1),
        2.0,
    )

    dataset = RawTrajectoryDataset(
        t=t,
        X=X,
        V=None,
        LHS=np.zeros_like(X),
        U0=np.array([[2.0, 2.0]]),
        metadata={
            "system_name": "constant_first_order",
            "system_order": 1,
        },
    )

    noise = KalmanFilterNoise(
        noise_level=0.0,
        seed=123,
        noise_type="additive_gaussian",
    )

    noisy_dataset, smoothed_dataset = noise.perturb(dataset)

    # No injected noise.
    np.testing.assert_allclose(
        noisy_dataset.X,
        X,
        atol=1e-12,
    )

    # First-order noisy target is reconstructed as dX/dt.
    np.testing.assert_allclose(
        noisy_dataset.LHS,
        0.0,
        atol=1e-12,
    )

    assert noisy_dataset.V is None

    # RTS reconstruction.
    np.testing.assert_allclose(
        smoothed_dataset.X,
        X,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        smoothed_dataset.LHS,
        0.0,
        atol=1e-12,
    )

    assert smoothed_dataset.V is None

def test_kalman_second_order_constant_trajectory():
    """
    A constant second-order trajectory should reconstruct

        V = 0
        LHS = acceleration = 0.
    """
    t = np.linspace(0.0, 1.0, 7)

    X = np.full(
        (1, 7, 2, 1),
        2.0,
    )

    V = np.zeros_like(X)

    dataset = RawTrajectoryDataset(
        t=t,
        X=X,
        V=V,
        LHS=np.zeros_like(X),
        U0=np.array([
            [2.0, 2.0, 0.0, 0.0],
        ]),
        metadata={
            "system_name": "constant_second_order",
            "system_order": 2,
        },
    )

    noise = KalmanFilterNoise(
        noise_level=0.0,
        seed=123,
        noise_type="additive_gaussian",
    )

    noisy_dataset, smoothed_dataset = noise.perturb(dataset)

    np.testing.assert_allclose(
        noisy_dataset.X,
        X,
        atol=1e-12,
    )

    # The raw noisy V is reconstructed from X.
    np.testing.assert_allclose(
        noisy_dataset.V,
        0.0,
        atol=1e-12,
    )

    # The raw noisy LHS is reconstructed from the noisy velocity.
    np.testing.assert_allclose(
        noisy_dataset.LHS,
        0.0,
        atol=1e-11,
    )

    # RTS smoothed state.
    np.testing.assert_allclose(
        smoothed_dataset.X,
        X,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        smoothed_dataset.V,
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        smoothed_dataset.LHS,
        0.0,
        atol=1e-12,
    )


def test_kalman_noise_is_reproducible():
    """
    Independent KalmanFilterNoise objects with the same seed should
    generate identical noisy and smoothed datasets.
    """
    dataset = make_first_order_dataset()

    filter_1 = KalmanFilterNoise(
        noise_level=0.1,
        seed=42,
        noise_type="additive_gaussian",
    )

    filter_2 = KalmanFilterNoise(
        noise_level=0.1,
        seed=42,
        noise_type="additive_gaussian",
    )

    noisy_1, smooth_1 = filter_1.perturb(dataset)
    noisy_2, smooth_2 = filter_2.perturb(dataset)

    np.testing.assert_allclose(
        noisy_1.X,
        noisy_2.X,
    )

    np.testing.assert_allclose(
        noisy_1.LHS,
        noisy_2.LHS,
    )

    np.testing.assert_allclose(
        smooth_1.X,
        smooth_2.X,
    )

    np.testing.assert_allclose(
        smooth_1.LHS,
        smooth_2.LHS,
    )

def test_kalman_preserves_dataset_information_and_records_filter():
    dataset = make_first_order_dataset()

    noise = KalmanFilterNoise(
        noise_level=0.1,
        seed=42,
        noise_type="additive_gaussian",
    )

    noisy_dataset, smoothed_dataset = noise.perturb(dataset)

    # Time grid and initial conditions are unchanged.
    np.testing.assert_array_equal(
        noisy_dataset.t,
        dataset.t,
    )

    np.testing.assert_array_equal(
        smoothed_dataset.t,
        dataset.t,
    )

    np.testing.assert_array_equal(
        noisy_dataset.U0,
        dataset.U0,
    )

    np.testing.assert_array_equal(
        smoothed_dataset.U0,
        dataset.U0,
    )

    # Original metadata survives.
    assert noisy_dataset.metadata["system_name"] == "test_system"
    assert smoothed_dataset.metadata["system_name"] == "test_system"

    assert noisy_dataset.metadata["system_order"] == 1
    assert smoothed_dataset.metadata["system_order"] == 1

    # Noise metadata.
    assert noisy_dataset.metadata["noise_type"] == "additive_gaussian"
    assert smoothed_dataset.metadata["noise_type"] == "additive_gaussian"

    assert noisy_dataset.metadata["noise_level"] == pytest.approx(0.1)
    assert smoothed_dataset.metadata["noise_level"] == pytest.approx(0.1)

    # Distinguish raw noisy observations from RTS-smoothed output.
    assert (
        noisy_dataset.metadata["filter_applied"]
        == "None (Raw Noisy Data)"
    )

    assert (
        smoothed_dataset.metadata["filter_applied"]
        == "Kinematic RTS Smoother"
    )