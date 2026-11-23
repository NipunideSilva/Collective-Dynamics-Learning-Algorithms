from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import numpy as np


@dataclass(frozen=True)
class InformationCriterionConfig:
    """
    Configuration for AIC/BIC evaluation.

    AIC and BIC are computed from the training reconstruction
    of the observed position trajectories.
    """

    rss_metric: str = "trajectory_reconstruction_rss"
    n_observations_metric: str = "trajectory_reconstruction_n_obs"

    # The Gaussian observation variance is estimated from RSS/n.
    # It is therefore formally one additional likelihood parameter.
    include_noise_variance_parameter: bool = True


def count_model_coefficients(fit_result: Any) -> int:
    """
    Count the fitted coefficients in a candidate model.

    This counts all fitted energy-kernel, alignment-kernel, and
    environmental-force coefficients contained in fit_result.theta.

    It does not count:
        - initial conditions,
        - basis degree,
        - basis knot locations,
        - ridge lambda,
        - solver tolerances,
        - the ordinal framework-complexity rank.
    """
    theta = getattr(fit_result, "theta", None)

    if theta is None:
        raise ValueError(
            "fit_result.theta is required to count model coefficients."
        )

    theta = np.asarray(theta)

    if theta.size == 0:
        raise ValueError(
            "fit_result.theta contains no fitted coefficients."
        )

    return int(theta.size)


def compute_gaussian_information_criteria(
    *,
    rss: float,
    n_observations: int,
    n_model_coefficients: int,
    include_noise_variance_parameter: bool = True,
) -> dict[str, float | int]:
    """
    Compute Gaussian maximum-likelihood AIC and BIC.

    The working likelihood assumes that the scalar position-coordinate
    reconstruction errors are independent Gaussian errors with common,
    unknown variance.

    Parameters
    ----------
    rss
        Position-trajectory reconstruction residual sum of squares,

            RSS = sum ||X_true - X_hat||^2,

        calculated over the training reconstruction interval.

    n_observations
        Number of scalar position observations used in RSS,

            n = M_train * L_fit * N * D.

    n_model_coefficients
        Number of fitted model coefficients, normally fit_result.theta.size.

    include_noise_variance_parameter
        Whether the estimated Gaussian variance is counted as one
        additional likelihood parameter.

    Returns
    -------
    dict
        Parameter counts, estimated variance, maximized log-likelihood,
        AIC, and BIC.
    """
    rss = float(rss)
    n = int(n_observations)
    n_coefficients = int(n_model_coefficients)

    if not math.isfinite(rss) or rss < 0.0:
        raise ValueError(
            f"rss must be finite and nonnegative, got {rss}."
        )

    if n <= 0:
        raise ValueError(
            f"n_observations must be positive, got {n}."
        )

    if n_coefficients < 0:
        raise ValueError(
            "n_model_coefficients must be nonnegative."
        )

    variance_parameter_count = (
        1 if include_noise_variance_parameter else 0
    )

    k = n_coefficients + variance_parameter_count

    # Maximum-likelihood estimate of the Gaussian variance.
    # The lower bound prevents log(0) for a numerically exact fit.
    sigma2_mle = max(
        rss / float(n),
        np.finfo(float).tiny,
    )

    # For independent Gaussian scalar errors:
    #
    # -2 log(L_hat)
    # =
    # n [log(2*pi) + 1 + log(RSS/n)].
    negative_two_log_likelihood = float(
        n
        * (
            math.log(2.0 * math.pi)
            + 1.0
            + math.log(sigma2_mle)
        )
    )

    log_likelihood = -0.5 * negative_two_log_likelihood

    aic = negative_two_log_likelihood + 2.0 * k
    bic = negative_two_log_likelihood + k * math.log(n)

    return {
        "trajectory_reconstruction_rss": rss,
        "trajectory_reconstruction_n_obs": n,
        "n_model_coefficients": n_coefficients,
        "n_variance_parameters": variance_parameter_count,
        "n_likelihood_parameters": k,
        "sigma2_mle": float(sigma2_mle),
        "log_likelihood": float(log_likelihood),
        "negative_two_log_likelihood": negative_two_log_likelihood,
        "AIC": float(aic),
        "BIC": float(bic),
    }