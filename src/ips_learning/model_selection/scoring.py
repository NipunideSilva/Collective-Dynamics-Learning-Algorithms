from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import pandas as pd

from ips_learning.model_selection.candidates import ModelCandidate
from ips_learning.model_selection.repository import ModelSelectionRepository


#---------------------------------
# all the tables
#--------------------

@dataclass(frozen=True)
class SelectionConfig:
    """
    Configuration for model-selection scoring.

    Primary metric should usually be a held-out prediction error.
    """

    primary_metric: str = "E_traj_full"
    residual_metric: str = "E_res"
    condition_metric: str = "condition_number"

    relative_tie_tol: float = 0.05
    absolute_tie_tol: float = 1e-3

    max_condition_number: float | None = 1e12
    reject_failed_rollouts: bool = True

    prefer_simpler_on_tie: bool = True

    max_reconstruction_rel: float | None = None
    max_residual_rel: float | None = None


def candidate_complexity(candidate_name: str) -> int:
    """
    Complexity profile.

    Smaller values are preferred when validation trajectory errors
    are tied within tolerance.
    """
    complexity_map = {
        # New full-structure candidates
        "first_order_energy_only": 1,
        "first_order_energy_f": 2,

        "second_order_energy_only": 3,
        "second_order_alignment_only": 3,

        "second_order_energy_f": 4,
        "second_order_alignment_f": 4,

        "second_order_energy_alignment_only": 5,
        "second_order_energy_alignment_f": 6,
    }

    return complexity_map.get(candidate_name, 99)


def _as_float(value: Any, default: float = math.inf) -> float:
    if value is None:
        return default

    try:
        out = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(out):
        return default

    return out


def _candidate_failed(metrics: dict[str, Any]) -> bool:
    if bool(metrics.get("failed", False)):
        return True

    status = str(metrics.get("status", "ok")).lower()
    if status in {"failed", "error", "nan", "diverged"}:
        return True

    return False


def score_candidate(
    *,
    candidate: ModelCandidate,
    metrics: dict[str, Any],
    config: SelectionConfig,
) -> dict[str, Any]:
    """
    Convert raw metrics into one sortable row.
    """

    primary_value = _as_float(metrics.get(config.primary_metric))
    residual_value = _as_float(metrics.get(config.residual_metric))
    condition_value = _as_float(metrics.get(config.condition_metric))

    #-----------
    E_recon = _as_float(
    metrics.get("E_traj_reconstruction"),
    default=math.nan,
    )

    E_recon_std = _as_float(
        metrics.get("E_traj_reconstruction_std"),
        default=math.nan,
    )

    E_recon_rel = _as_float(
        metrics.get("E_traj_reconstruction_rel"),
        default=math.nan,
    )

    E_pred = _as_float(
        metrics.get("E_traj_prediction"),
        default=math.nan,
    )

    E_pred_std = _as_float(
        metrics.get("E_traj_prediction_std"),
        default=math.nan,
    )

    E_full = _as_float(
        metrics.get("E_traj_full"),
        default=math.nan,
    )

    E_full_std = _as_float(
        metrics.get("E_traj_full_std"),
        default=math.nan,
    )

    E_res_rel = _as_float(
        metrics.get("E_res_rel"),
        default=math.nan,
    )

    # AIC/BIC baseline diagnostics
    aic_value = _as_float(
        metrics.get("AIC"),
        default=math.nan,
    )

    bic_value = _as_float(
        metrics.get("BIC"),
        default=math.nan,
    )

    trajectory_reconstruction_rss = _as_float(
        metrics.get("trajectory_reconstruction_rss"),
        default=math.nan,
    )

    trajectory_reconstruction_n_obs = metrics.get(
        "trajectory_reconstruction_n_obs"
    )

    n_model_coefficients = metrics.get(
        "n_model_coefficients"
    )

    n_likelihood_parameters = metrics.get(
        "n_likelihood_parameters"
    )

    #-----------
    # rejection criterias
    failed = _candidate_failed(metrics)

    if config.reject_failed_rollouts and failed:
        accepted = False
        rejection_reason = "failed_rollout"

    elif (
        config.max_condition_number is not None
        and condition_value > config.max_condition_number
    ):
        accepted = False
        rejection_reason = "ill_conditioned"

    elif (
        config.max_reconstruction_rel is not None
        and math.isfinite(E_recon_rel)
        and E_recon_rel > config.max_reconstruction_rel
    ):
        accepted = False
        rejection_reason = "poor_reconstruction"

    elif (
        config.max_residual_rel is not None
        and math.isfinite(E_res_rel)
        and E_res_rel > config.max_residual_rel
    ):
        accepted = False
        rejection_reason = "large_residual"

    elif not math.isfinite(primary_value):
        accepted = False
        rejection_reason = "invalid_primary_metric"

    else:
        accepted = True
        rejection_reason = ""

    return {
        "candidate": candidate.name,
        "system_order": candidate.system_order,
        "interaction_type": candidate.interaction_type,
        "algorithm": candidate.algorithm,
        "include_environment": getattr(candidate, "include_environment", True),

        "primary_metric": config.primary_metric,
        "primary_value": primary_value,

        # ==========================================================
        # Train-fit / train trajectory diagnostics
        # ==========================================================
        "E_traj_reconstruction": E_recon, #Fit-window reconstruction
        "E_traj_reconstruction_std": E_recon_std, 
        "E_traj_reconstruction_rel": E_recon_rel, #Framework compatibility

        "E_traj_prediction": E_pred,#Long-time validation forecasting
        "E_traj_prediction_std": E_pred_std,

        # Equation-consistency errors-train-fit
        "E_res": residual_value, #Equation consistency
        "E_res_rel": E_res_rel, #Framework compatibility


        # ==========================================================
        # Validation diagnostic used for model selection
        # ==========================================================

        "E_traj_full": E_full, #Primary validation metric
        "E_traj_full_std": E_full_std,

        

        # Numerical diagnostics
        "condition_number": condition_value,

        # Complexity profile
        "complexity": getattr(
            candidate,
            "complexity",
            candidate_complexity(candidate.name),
        ),

        # ==========================================================
        # Information-criterion baseline diagnostics
        # ==========================================================
        "trajectory_reconstruction_rss": (
            trajectory_reconstruction_rss
        ),
        "trajectory_reconstruction_n_obs": (
            trajectory_reconstruction_n_obs
        ),
        "n_model_coefficients": n_model_coefficients,
        "n_likelihood_parameters": n_likelihood_parameters,
        "AIC": aic_value,
        "BIC": bic_value,

        "accepted": accepted,
        "rejection_reason": rejection_reason,

        # Activity diagnostics
        "phi_E_norm": _as_float(metrics.get("phi_E_norm"), default=math.nan),
        "phi_A_norm": _as_float(metrics.get("phi_A_norm"), default=math.nan),
        "f_norm": _as_float(metrics.get("f_norm"), default=math.nan),
    }

def rank_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    config: SelectionConfig,
) -> list[dict[str, Any]]:
    """
    Rank candidate rows.

    Accepted candidates are ranked before rejected candidates.
    Among accepted candidates:
        1. lower primary metric wins,
        2. if tied, simpler model wins,
        3. if still tied, lower residual wins.
    """

    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]

    if not accepted:
        ranked = sorted(
            rejected,
            key=lambda row: (
                row["primary_value"],
                row["condition_number"],
                row["complexity"],
            ),
        )
        for i, row in enumerate(ranked, start=1):
            row["rank"] = i
            row["selected"] = False
        return ranked

    best_primary = min(row["primary_value"] for row in accepted)

    def is_tied_with_best(row: dict[str, Any]) -> bool:
        abs_close = (
            row["primary_value"]
            <= best_primary + config.absolute_tie_tol
        )

        rel_close = (
            row["primary_value"]
            <= best_primary * (1.0 + config.relative_tie_tol)
        )

        return abs_close or rel_close

    tied = [row for row in accepted if is_tied_with_best(row)]
    not_tied = [row for row in accepted if not is_tied_with_best(row)]

    if config.prefer_simpler_on_tie:
        tied_sorted = sorted(
            tied,
            key=lambda row: (
                row["complexity"],
                row["primary_value"],
                row["E_res"],
            ),
        )
    else:
        tied_sorted = sorted(
            tied,
            key=lambda row: (
                row["primary_value"],
                row["complexity"],
                row["E_res"],
            ),
        )

    remaining_sorted = sorted(
        not_tied,
        key=lambda row: (
            row["primary_value"],
            row["complexity"],
            row["E_res"],
        ),
    )

    rejected_sorted = sorted(
        rejected,
        key=lambda row: (
            row["rejection_reason"],
            row["primary_value"],
            row["condition_number"],
        ),
    )

    ranked = tied_sorted + remaining_sorted + rejected_sorted

    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
        row["selected"] = i == 1 and row["accepted"]

    return ranked


def select_model_from_metrics(
    *,
    candidates: list[ModelCandidate],
    metrics_by_candidate: dict[str, dict[str, Any]],
    config: SelectionConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Select the best candidate from already-computed metrics.
    """

    config = config or SelectionConfig()

    rows = []
    for candidate in candidates:
        if candidate.name not in metrics_by_candidate:
            rows.append(
                {
                    "candidate": candidate.name,
                    "system_order": candidate.system_order,
                    "interaction_type": candidate.interaction_type,
                    "algorithm": candidate.algorithm,
                    "include_environment": getattr(candidate, "include_environment", True),

                    "primary_metric": config.primary_metric,
                    "primary_value": math.nan,

                    "E_traj_reconstruction": math.nan,
                    "E_traj_reconstruction_std": math.nan,
                    "E_traj_reconstruction_rel": math.nan,

                    "E_traj_prediction": math.nan,
                    "E_traj_prediction_std": math.nan,

                    "E_traj_full": math.nan,
                    "E_traj_full_std": math.nan,

                    "E_res": math.inf,
                    "E_res_rel": math.nan,

                    "condition_number": math.nan,

                    "complexity": getattr(
                        candidate,
                        "complexity",
                        candidate_complexity(candidate.name),
                    ),
                    "trajectory_reconstruction_rss": math.nan,
                    "trajectory_reconstruction_n_obs": None,
                    "n_model_coefficients": None,
                    "n_likelihood_parameters": None,
                    "AIC": math.nan,
                    "BIC": math.nan,

                    "accepted": False,
                    "rejection_reason": "missing_metrics",

                    "phi_E_norm": math.nan,
                    "phi_A_norm": math.nan,
                    "f_norm": math.nan,
                }
            )
            continue

        rows.append(
            score_candidate(
                candidate=candidate,
                metrics=metrics_by_candidate[candidate.name],
                config=config,
            )
        )

    ranked = rank_candidate_rows(rows, config=config)

    selected_rows = [row for row in ranked if row.get("selected", False)]

    if selected_rows:
        selected = dict(selected_rows[0])
        selected["selection_status"] = "selected"
    else:
        selected = {
            "selection_status": "no_valid_candidate",
            "reason": "All candidates were rejected or missing metrics.",
            "primary_metric": config.primary_metric,
        }

    return selected, ranked

PAPER_CANDIDATE_LABELS = {
    "first_order_energy_only": "F1",
    "first_order_energy_f": "F2",
    "second_order_energy_only": "S1",
    "second_order_alignment_only": "S2",
    "second_order_energy_f": "S3",
    "second_order_alignment_f": "S4",
    "second_order_energy_alignment_only": "S5",
    "second_order_energy_alignment_f": "S6",
}

def attach_information_criterion_rankings(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Attach independent AIC and BIC ranks to candidate rows.

    These rankings are separate from the proposed algorithm.
    Every candidate with a finite information-criterion value is
    eligible, regardless of whether it was rejected by the proposed
    screening procedure.
    """
    output_rows = [dict(row) for row in rows]

    criterion_specs = [
        ("AIC", "aic"),
        ("BIC", "bic"),
    ]

    for criterion, prefix in criterion_specs:
        eligible_rows = [
            row
            for row in output_rows
            if math.isfinite(
                _as_float(
                    row.get(criterion),
                    default=math.nan,
                )
            )
        ]

        ordered_rows = sorted(
            eligible_rows,
            key=lambda row: (
                _as_float(
                    row.get(criterion),
                    default=math.inf,
                ),
                _as_float(
                    row.get("n_likelihood_parameters"),
                    default=math.inf,
                ),
                str(row.get("candidate", "")),
            ),
        )

        rank_by_candidate = {
            row["candidate"]: rank
            for rank, row in enumerate(ordered_rows, start=1)
        }

        for row in output_rows:
            rank = rank_by_candidate.get(row["candidate"])

            row[f"{prefix}_rank"] = (
                rank if rank is not None else math.nan
            )

            row[f"{prefix}_selected"] = (
                rank == 1 if rank is not None else False
            )

    return output_rows


def build_paper_summary_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()

    def yes_no(series: pd.Series) -> pd.Series:
        return series.fillna(False).map(
            lambda value: "Yes" if bool(value) else "No"
        )

    out = pd.DataFrame(
        {
            "Candidate": df["candidate"].map(PAPER_CANDIDATE_LABELS).fillna(df["candidate"]),
            "Order": df["system_order"],
            r"$E_{\mathrm{res}}^{\mathrm{rel}}$": df["E_res_rel"],
            r"$E_{\mathrm{recon}}^{\mathrm{rel}}$": df["E_traj_reconstruction_rel"],
            r"$E_{\mathrm{full}}(M_{\mathrm{val}})$": df["E_traj_full"],
            "Complexity": df["complexity"],
            r"$E_{\mathrm{res}}$": df["E_res"],
            "Rank": df["rank"],
            "Selected": yes_no(df["selected"]),
            # Training position-trajectory goodness of fit
            r"$\mathrm{RSS}_{X}$": (
                df["trajectory_reconstruction_rss"]
            ),

            # Number of likelihood parameters used by AIC/BIC
            r"$k_c$": df["n_likelihood_parameters"],

            r"$\mathrm{AIC}$": df["AIC"],
            r"$\mathrm{BIC}$": df["BIC"],
            "AIC selected": yes_no(df["aic_selected"]),
            "BIC selected": yes_no(df["bic_selected"]),


        }
    )

    return out


def select_and_save(
    *,
    repo: ModelSelectionRepository,
    system_name: str,
    profile_name: str,
    dataset_name: str,
    candidates: list[ModelCandidate],
    split: str = "validation",
    config: SelectionConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Load candidate metrics from the repository, select models using
    the proposed method, AIC, and BIC, and save summary files.
    """

    config = config or SelectionConfig()

    metrics_by_candidate: dict[str, dict[str, Any]] = {}

    # ==========================================================
    # First load metrics for ALL candidates
    # ==========================================================
    for candidate in candidates:
        try:
            metrics_by_candidate[candidate.name] = (
                repo.load_candidate_metrics(
                    system_name=system_name,
                    profile_name=profile_name,
                    dataset_name=dataset_name,
                    candidate_name=candidate.name,
                    split=split,
                )
            )

            # Add feature-norm diagnostics when available.
            try:
                kernel_diag = repo.load_kernel_diagnostics(
                    system_name=system_name,
                    profile_name=profile_name,
                    dataset_name=dataset_name,
                    candidate_name=candidate.name,
                )

                for key in [
                    "phi_E_norm",
                    "phi_A_norm",
                    "f_norm",
                ]:
                    if key in kernel_diag:
                        metrics_by_candidate[
                            candidate.name
                        ][key] = float(kernel_diag[key])

            except FileNotFoundError:
                pass

        except FileNotFoundError:
            metrics_by_candidate[candidate.name] = {
                "failed": True,
                "status": "missing_metrics",
            }

    # ==========================================================
    # These operations must occur AFTER the for-loop
    # ==========================================================

    selected, ranked_rows = select_model_from_metrics(
        candidates=candidates,
        metrics_by_candidate=metrics_by_candidate,
        config=config,
    )

    # Attach independent AIC and BIC rankings.
    ranked_rows = attach_information_criterion_rankings(
        ranked_rows
    )

    # Refresh the proposed selected row so that selected_model.json
    # also contains the AIC/BIC values and rankings.
    if selected.get("selection_status") == "selected":
        selected_candidate_name = selected.get("candidate")

        selected_row = next(
            (
                row
                for row in ranked_rows
                if row.get("candidate")
                == selected_candidate_name
            ),
            None,
        )

        if selected_row is not None:
            selected = {
                **selected_row,
                "selection_status": "selected",
            }

    # ==========================================================
    # Save complete summary
    # ==========================================================

    repo.save_summary_table(
        system_name=system_name,
        profile_name=profile_name,
        dataset_name=dataset_name,
        rows=ranked_rows,
    )

    # ==========================================================
    # Save paper-ready summary
    # ==========================================================

    paper_df = build_paper_summary_table(
        ranked_rows
    )

    repo.save_summary_table(
        system_name=system_name,
        profile_name=profile_name,
        dataset_name=dataset_name,
        rows=paper_df.to_dict(orient="records"),
        file_stem="model_selection_summary_paper",
    )

    # Save candidate selected by the proposed method.
    repo.save_selected_model(
        system_name=system_name,
        profile_name=profile_name,
        dataset_name=dataset_name,
        selected=selected,
    )

    return selected, pd.DataFrame(ranked_rows)