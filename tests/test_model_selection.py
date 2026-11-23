import math
from types import SimpleNamespace

import numpy as np
import pytest

from ips_learning.model_selection.candidates import (
    ModelCandidate,
    FIRST_ORDER_ENERGY_ONLY,
    FIRST_ORDER_ENERGY_F,
    SECOND_ORDER_ENERGY_ONLY,
    get_candidates,
    validate_candidates,
)
from ips_learning.model_selection.information_criteria import (
    count_model_coefficients,
    compute_gaussian_information_criteria,
)
from ips_learning.model_selection.repository import (
    ModelSelectionRepository,
)
from ips_learning.model_selection.scoring import (
    SelectionConfig,
    attach_information_criterion_rankings,
    score_candidate,
    select_and_save,
    select_model_from_metrics,
)


def valid_metrics(**overrides):
    """
    Minimal finite metric dictionary accepted by model selection.
    """
    metrics = {
        "E_traj_full": 0.10,
        "E_res": 0.01,
        "condition_number": 100.0,
        "E_traj_reconstruction": 0.02,
        "E_traj_reconstruction_std": 0.0,
        "E_traj_reconstruction_rel": 0.02,
        "E_traj_prediction": 0.10,
        "E_traj_prediction_std": 0.0,
        "E_traj_full_std": 0.0,
        "E_res_rel": 0.01,
        "trajectory_reconstruction_rss": 1.0,
        "trajectory_reconstruction_n_obs": 100,
        "n_model_coefficients": 3,
        "n_likelihood_parameters": 4,
        "AIC": 10.0,
        "BIC": 12.0,
        "status": "ok",
        "failed": False,
    }

    metrics.update(overrides)
    return metrics



#--------------Candidate groups and validation------------------

def test_model_selection_candidate_groups():
    first_order = get_candidates(
        "first_order_structure_selection"
    )

    second_order = get_candidates(
        "second_order_structure_selection"
    )

    full = get_candidates(
        "full_structure_selection"
    )

    assert len(first_order) == 2
    assert len(second_order) == 6
    assert len(full) == 8

    assert all(
        candidate.system_order == 1
        for candidate in first_order
    )

    assert all(
        candidate.system_order == 2
        for candidate in second_order
    )

    assert len({
        candidate.name
        for candidate in full
    }) == 8

def test_invalid_first_order_alignment_candidate_is_rejected():
    candidate = ModelCandidate(
        name="invalid_first_order_alignment",
        system_order=1,
        interaction_type="alignment",
    )

    with pytest.raises(
        ValueError,
        match="First-order candidates",
    ):
        candidate.validate()


def test_duplicate_candidate_names_are_rejected():
    candidate_1 = ModelCandidate(
        name="duplicate",
        system_order=1,
        interaction_type="energy",
    )

    candidate_2 = ModelCandidate(
        name="duplicate",
        system_order=2,
        interaction_type="energy",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate candidate name",
    ):
        validate_candidates([
            candidate_1,
            candidate_2,
        ])


#----------AIC/BIC Number of fitted parameters--------------------
def test_count_model_coefficients_uses_theta_size():
    fit_result = SimpleNamespace(
        theta=np.array([
            1.0,
            2.0,
            3.0,
            4.0,
        ])
    )

    assert count_model_coefficients(
        fit_result
    ) == 4

#----Verify AIC/BIC numerically------------------------------
def test_gaussian_information_criteria():
    result = compute_gaussian_information_criteria(
        rss=4.0,
        n_observations=4,
        n_model_coefficients=2,
        include_noise_variance_parameter=True,
    )

    expected_neg2logL = (
        4.0
        * (
            math.log(2.0 * math.pi)
            + 1.0
        )
    )

    expected_AIC = (
        expected_neg2logL
        + 2.0 * 3
    )

    expected_BIC = (
        expected_neg2logL
        + 3.0 * math.log(4.0)
    )

    assert result["sigma2_mle"] == pytest.approx(
        1.0
    )

    assert result["n_model_coefficients"] == 2
    assert result["n_variance_parameters"] == 1
    assert result["n_likelihood_parameters"] == 3

    assert (
        result["negative_two_log_likelihood"]
        == pytest.approx(expected_neg2logL)
    )

    assert result["AIC"] == pytest.approx(
        expected_AIC
    )

    assert result["BIC"] == pytest.approx(
        expected_BIC
    )

#--------Test every rejection rule-----------------------
@pytest.mark.parametrize(
    (
        "metric_changes",
        "config",
        "expected_reason",
    ),
    [
        (
            {"failed": True},
            SelectionConfig(),
            "failed_rollout",
        ),
        (
            {"condition_number": 1e13},
            SelectionConfig(
                max_condition_number=1e12
            ),
            "ill_conditioned",
        ),
        (
            {
                "E_traj_reconstruction_rel": 0.20
            },
            SelectionConfig(
                max_reconstruction_rel=0.10
            ),
            "poor_reconstruction",
        ),
        (
            {"E_res_rel": 0.20},
            SelectionConfig(
                max_residual_rel=0.10
            ),
            "large_residual",
        ),
        (
            {"E_traj_full": np.nan},
            SelectionConfig(),
            "invalid_primary_metric",
        ),
    ],
)
def test_candidate_rejection_rules(
    metric_changes,
    config,
    expected_reason,
):
    metrics = valid_metrics(
        **metric_changes
    )

    row = score_candidate(
        candidate=FIRST_ORDER_ENERGY_ONLY,
        metrics=metrics,
        config=config,
    )

    assert row["accepted"] is False

    assert (
        row["rejection_reason"]
        == expected_reason
    )

#--------Test simpler-model preference within a tie-----------------------
def test_simpler_candidate_wins_within_tie_tolerance():
    candidates = [
        FIRST_ORDER_ENERGY_ONLY,
        FIRST_ORDER_ENERGY_F,
    ]

    metrics = {
        FIRST_ORDER_ENERGY_ONLY.name:
            valid_metrics(
                E_traj_full=0.103,
                E_res=0.02,
            ),

        FIRST_ORDER_ENERGY_F.name:
            valid_metrics(
                E_traj_full=0.100,
                E_res=0.01,
            ),
    }

    config = SelectionConfig(
        relative_tie_tol=0.05,
        absolute_tie_tol=0.0,
        prefer_simpler_on_tie=True,
    )

    selected, rows = select_model_from_metrics(
        candidates=candidates,
        metrics_by_candidate=metrics,
        config=config,
    )

    assert selected["selection_status"] == "selected"

    assert (
        selected["candidate"]
        == FIRST_ORDER_ENERGY_ONLY.name
    )

    assert selected["complexity"] == 1

    selected_rows = [
        row
        for row in rows
        if row["selected"]
    ]

    assert len(selected_rows) == 1


#------------Missing metrics must not silently select a candidate----------
#\[
# \text{candidate metrics}
# \rightarrow
# \text{scoring}
# \rightarrow
# \text{ranking}
# \rightarrow
# \text{selection}
# \rightarrow
# \text{summary files}
# \rightarrow
# \texttt{selected\_model.json}.
# \]
def test_missing_candidate_metrics_are_rejected():
    candidates = [
        FIRST_ORDER_ENERGY_ONLY,
        FIRST_ORDER_ENERGY_F,
    ]

    metrics = {
        FIRST_ORDER_ENERGY_ONLY.name:
            valid_metrics(
                E_traj_full=0.1
            ),
    }

    selected, rows = select_model_from_metrics(
        candidates=candidates,
        metrics_by_candidate=metrics,
    )

    assert (
        selected["candidate"]
        == FIRST_ORDER_ENERGY_ONLY.name
    )

    missing_row = next(
        row
        for row in rows
        if row["candidate"]
        == FIRST_ORDER_ENERGY_F.name
    )

    assert missing_row["accepted"] is False

    assert (
        missing_row["rejection_reason"]
        == "missing_metrics"
    )



#--------AIC/BIC rankings are independent of your proposed screening---------

def test_information_criterion_ranking_is_independent_of_acceptance():
    rows = [
        {
            "candidate": "candidate_A",
            "accepted": False,
            "AIC": 10.0,
            "BIC": 20.0,
            "n_likelihood_parameters": 5,
        },
        {
            "candidate": "candidate_B",
            "accepted": True,
            "AIC": 12.0,
            "BIC": 18.0,
            "n_likelihood_parameters": 3,
        },
    ]

    ranked = attach_information_criterion_rankings(
        rows
    )

    row_A = next(
        row
        for row in ranked
        if row["candidate"] == "candidate_A"
    )

    row_B = next(
        row
        for row in ranked
        if row["candidate"] == "candidate_B"
    )

    # Lowest AIC is candidate A,
    # even though the proposed framework rejected it.
    assert row_A["aic_rank"] == 1
    assert row_A["aic_selected"] is True

    # Lowest BIC is candidate B.
    assert row_B["bic_rank"] == 1
    assert row_B["bic_selected"] is True


#-------------Repository + selected-model bookkeeping-----------------

def test_select_and_save_round_trip(tmp_path):
    repo = ModelSelectionRepository(
        root_dir=tmp_path
    )

    candidates = [
        FIRST_ORDER_ENERGY_ONLY,
        FIRST_ORDER_ENERGY_F,
    ]

    # ---------------------------------------------------------
    # Proposed method:
    # candidate F should win because its trajectory error
    # is substantially lower.
    #
    # AIC/BIC:
    # candidate E should win.
    #
    # This deliberately verifies that the selections remain
    # independent.
    # ---------------------------------------------------------

    metrics_energy_only = valid_metrics(
        E_traj_full=0.30,
        AIC=10.0,
        BIC=11.0,
    )

    metrics_energy_f = valid_metrics(
        E_traj_full=0.10,
        AIC=14.0,
        BIC=15.0,
    )

    for candidate, metrics in [
        (
            FIRST_ORDER_ENERGY_ONLY,
            metrics_energy_only,
        ),
        (
            FIRST_ORDER_ENERGY_F,
            metrics_energy_f,
        ),
    ]:
        repo.save_candidate_metrics(
            system_name="test_system",
            profile_name="test_profile",
            dataset_name="test_dataset",
            candidate_name=candidate.name,
            metrics=metrics,
            split="validation",
        )

    selected, summary = select_and_save(
        repo=repo,
        system_name="test_system",
        profile_name="test_profile",
        dataset_name="test_dataset",
        candidates=candidates,
        split="validation",
    )

    # Proposed procedure selects the lower trajectory error.
    assert selected["selection_status"] == "selected"

    assert (
        selected["candidate"]
        == FIRST_ORDER_ENERGY_F.name
    )

    # Check persisted selected-model record.
    loaded_selected = repo.load_selected_model(
        system_name="test_system",
        profile_name="test_profile",
        dataset_name="test_dataset",
    )

    assert (
        loaded_selected["candidate"]
        == FIRST_ORDER_ENERGY_F.name
    )

    # Summary contains both candidates.
    assert len(summary) == 2

    assert {
        "rank",
        "selected",
        "aic_rank",
        "aic_selected",
        "bic_rank",
        "bic_selected",
    }.issubset(summary.columns)

    energy_only_row = summary.loc[
        summary["candidate"]
        == FIRST_ORDER_ENERGY_ONLY.name
    ].iloc[0]

    energy_f_row = summary.loc[
        summary["candidate"]
        == FIRST_ORDER_ENERGY_F.name
    ].iloc[0]

    # Proposed framework.
    assert bool(
        energy_f_row["selected"]
    )

    # AIC/BIC baselines.
    assert bool(
        energy_only_row["aic_selected"]
    )

    assert bool(
        energy_only_row["bic_selected"]
    )

    # Verify saved summary can be loaded.
    loaded_summary = repo.load_summary_table(
        system_name="test_system",
        profile_name="test_profile",
        dataset_name="test_dataset",
    )

    assert len(loaded_summary) == 2