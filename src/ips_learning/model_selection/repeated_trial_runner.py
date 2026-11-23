from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Any

import pandas as pd

from ips_learning.data.dataset_repository import DatasetRepository
from ips_learning.features.feature_builder import FeatureBuilder
from ips_learning.bases.specs import (
    BasisConfig,
    RadialBasisSpec,
    EnvironmentBasisSpec,
)
from ips_learning.bases.builders import BasisBuilder

from ips_learning.model_selection.candidates import get_candidates
from ips_learning.model_selection.runner import (
    CandidateRunConfig,
    ModelSelectionRunner,
)
from ips_learning.model_selection.scoring import SelectionConfig
from ips_learning.model_selection.callbacks import (
    model_selection_rollout_fn,
    model_selection_evaluate_fn,
    model_selection_kernel_diagnostics_fn,
)


@dataclass(frozen=True)
class ModelSelectionRepeatedTrialRunConfig:
    system_name: str
    profile_name: str

    model_selection_mode: str = "second_order"

    train_fit_split: str = "train_fit"
    validation_split: str = "validation"
    test_split: str = "test"

    train_fit_name_template: str = "dataset_train_fit_trial_{trial:03d}"
    validation_dataset_name: str = "dataset_validation"
    test_dataset_name: str = "dataset_test"

    num_trials: int = 10
    trial_start: int = 1

    fit_end_time: float = 0.5
    prediction_end_time: float | None = None

    radial_basis_family: str = "bspline"
    radial_degree: int = 1
    radial_n_basis: int = 10

    energy_radial_n_basis: int | None = None
    alignment_radial_n_basis: int | None = None

    environment_basis_family: str = "bspline"
    environment_degree: int = 1
    environment_n_basis_per_dim: int = 10

    solver_method: str = "lstsq"
    ridge_lambda: float = 1e-6
    svd_tol: float = 1e-12

    parallel_jobs: int = 1
    overwrite_fit: bool = False
    overwrite_metrics: bool = True

    # Gaussian working-likelihood configuration
    include_noise_variance_parameter: bool = True

    primary_metric: str = "E_traj_full"
    relative_tie_tol: float = 0.05
    absolute_tie_tol: float = 1e-3
    max_condition_number: float | None = 1e12
    prefer_simpler_on_tie: bool = True

    artifact_root: str | Path = "artifacts"


class ModelSelectionRepeatedTrialRunner:
    """
    Repeated-trial runner for model selection.

    For each trial:
        1. load train_fit dataset,
        2. build features,
        3. build basis,
        4. run candidate model selection,
        5. save candidate artifacts,
        6. return selected candidate and summary metrics.

    This runner does not do N-scaling.
    """

    def __init__(self, config: ModelSelectionRepeatedTrialRunConfig):
        self.config = config

        root = Path(config.artifact_root)

        self.dataset_repo = DatasetRepository(root_dir=root / "datasets")
        self.feature_builder = FeatureBuilder()
        self.basis_builder = BasisBuilder(self._build_basis_config())

        self.ms_runner = ModelSelectionRunner()

    def run_all_trials(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for q in range(self.config.num_trials):
            trial = self.config.trial_start + q
            print(f"\nRunning model selection trial {trial:03d}")
            rows.append(self.run_one_trial(trial))

        return pd.DataFrame(rows)

    def run_one_trial(self, trial: int) -> dict[str, Any]:
        cfg = self.config

        row: dict[str, Any] = {
            "trial": trial,
            "system_name": cfg.system_name,
            "profile_name": cfg.profile_name,
            "model_selection_mode": cfg.model_selection_mode,
            "failed": False,
            "failure_message": "",
        }

        try:
            train_fit_name = cfg.train_fit_name_template.format(trial=trial)

            train_fit_dataset = self.dataset_repo.load(
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split=cfg.train_fit_split,
                dataset_name=train_fit_name,
            )

            validation_dataset = self.dataset_repo.load(
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split=cfg.validation_split,
                dataset_name=cfg.validation_dataset_name,
            )

            test_dataset = self.dataset_repo.load(
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split=cfg.test_split,
                dataset_name=cfg.test_dataset_name,
            )

            row.update(
                {
                    "validation_dataset": cfg.validation_dataset_name,
                    "test_dataset": cfg.test_dataset_name,
                    "validation_Tf": float(validation_dataset.t[-1]),
                    "test_Tf": float(test_dataset.t[-1]),
                    "validation_M": int(validation_dataset.X.shape[0]),
                    "test_M": int(test_dataset.X.shape[0]),
                }
            )

            train_fit_features = self.feature_builder.build(train_fit_dataset)

            # Allow candidate interaction types to differ from the generating system.
            train_fit_features.metadata["model_selection_mode"] = True
            train_fit_features.metadata["generating_interaction_type"] = (
                train_fit_features.metadata.get("interaction_type")
            )

            basis_bundle = self.basis_builder.build(train_fit_features)

            candidates = get_candidates(cfg.model_selection_mode)

            run_config = CandidateRunConfig(
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                dataset_name=train_fit_name,
                overwrite_fit=cfg.overwrite_fit,
                overwrite_metrics=cfg.overwrite_metrics,
                ridge_lambda=cfg.ridge_lambda,
                solver_method=cfg.solver_method,
                svd_tol=cfg.svd_tol,
                parallel_jobs=cfg.parallel_jobs,
                primary_metric=cfg.primary_metric,
                include_noise_variance_parameter=(
                        cfg.include_noise_variance_parameter
                    ),
            )

            run_config.fit_end_time = cfg.fit_end_time
            run_config.prediction_end_time = cfg.prediction_end_time

            selection_config = SelectionConfig(
                primary_metric=cfg.primary_metric,
                relative_tie_tol=cfg.relative_tie_tol,
                absolute_tie_tol=cfg.absolute_tie_tol,
                max_condition_number=cfg.max_condition_number,
                prefer_simpler_on_tie=cfg.prefer_simpler_on_tie,
            )

            result = self.ms_runner.run_select_and_save(
                candidates=candidates,
                train_dataset=train_fit_dataset,
                # features_train_fit=train_fit_features,
                basis_bundle=basis_bundle,
                validation_dataset=validation_dataset,
                test_dataset=test_dataset,
                config=run_config,
                rollout_fn=model_selection_rollout_fn,
                evaluate_fn=model_selection_evaluate_fn,
                kernel_diagnostics_fn=model_selection_kernel_diagnostics_fn,
                selection_config=selection_config,
            )

            selected = result["selected"]
            summary = result["summary"]

            # ============================================================
            # Extract candidates independently selected by AIC and BIC
            # ============================================================

            def selected_row_by_flag(
                summary_df: pd.DataFrame,
                flag_column: str,
            ) -> dict[str, Any] | None:
                if flag_column not in summary_df.columns:
                    return None

                mask = summary_df[flag_column].map(
                    lambda value: (
                        bool(value)
                        if isinstance(value, bool)
                        else str(value).strip().lower() == "true"
                    )
                )

                selected_rows = summary_df.loc[mask]

                if selected_rows.empty:
                    return None

                if len(selected_rows) > 1:
                    raise RuntimeError(
                        f"Expected one candidate with {flag_column}=True, "
                        f"but found {len(selected_rows)}."
                    )

                return selected_rows.iloc[0].to_dict()


            aic_selected_row = selected_row_by_flag(
                summary,
                "aic_selected",
            )

            bic_selected_row = selected_row_by_flag(
                summary,
                "bic_selected",
            )

            row.update(
                {
                    # "train_fit_dataset": train_fit_name,
                    # "validation_dataset": cfg.validation_dataset_name,
                    # "test_dataset": cfg.test_dataset_name,

                    # "selected_candidate": selected.get("candidate"),
                    # "selection_status": selected.get("selection_status"),
                    # "selected_primary_metric": selected.get("primary_metric"),
                    # "selected_primary_value": selected.get("primary_value"),

                    # "selected_validation_E_traj_reconstruction": selected.get("E_traj_reconstruction"),
                    # "selected_validation_E_traj_reconstruction_std": selected.get("E_traj_reconstruction_std"),
                    # "selected_validation_E_traj_prediction": selected.get("E_traj_prediction"),
                    # "selected_validation_E_traj_prediction_std": selected.get("E_traj_prediction_std"),
                    # "selected_validation_E_traj_full": selected.get("E_traj_full"),
                    # "selected_validation_E_traj_full_std": selected.get("E_traj_full_std"),

                    # "selected_E_res": selected.get("E_res"),
                    # "selected_condition_number": selected.get("condition_number"),
                    # "selected_phi_E_norm": selected.get("phi_E_norm"),
                    # "selected_phi_A_norm": selected.get("phi_A_norm"),
                    # "selected_f_norm": selected.get("f_norm"),
                    # "num_candidates": int(len(candidates)),

                    "selected_candidate":
                        selected.get("candidate"),

                    "selected_system_order":
                        selected.get("system_order"),

                    "selected_interaction_type":
                        selected.get("interaction_type"),

                    "selected_include_environment":
                        selected.get("include_environment"),

                    "selected_complexity":
                        selected.get("complexity"),

                    "selected_E_traj_full":
                        selected.get("E_traj_full"),

                    "selected_E_traj_full_std":
                        selected.get("E_traj_full_std"),

                    "selected_E_traj_reconstruction":
                        selected.get("E_traj_reconstruction"),

                    "selected_E_traj_reconstruction_rel":
                        selected.get("E_traj_reconstruction_rel"),

                    "selected_E_traj_prediction":
                        selected.get("E_traj_prediction"),

                    "selected_E_res":
                        selected.get("E_res"),

                    "selected_E_res_rel":
                        selected.get("E_res_rel"),

                    "selected_accepted":
                        selected.get("accepted"),

                    "selected_rejection_reason":
                        selected.get("rejection_reason"),

                    "selected_phi_E_norm":
                        selected.get("phi_E_norm"),

                    "selected_phi_A_norm":
                        selected.get("phi_A_norm"),

                    "selected_f_norm":
                        selected.get("f_norm"),

                    # ============================================================
                    # Proposed-method information-criterion diagnostics
                    # ============================================================

                    "selected_trajectory_reconstruction_rss":
                        selected.get("trajectory_reconstruction_rss"),

                    "selected_trajectory_reconstruction_n_obs":
                        selected.get("trajectory_reconstruction_n_obs"),

                    "selected_n_model_coefficients":
                        selected.get("n_model_coefficients"),

                    "selected_n_likelihood_parameters":
                        selected.get("n_likelihood_parameters"),

                    "selected_AIC":
                        selected.get("AIC"),

                    "selected_BIC":
                        selected.get("BIC"),

                    # ============================================================
                    # AIC-selected candidate
                    # ============================================================

                    "aic_selected_candidate": (
                        None
                        if aic_selected_row is None
                        else aic_selected_row.get("candidate")
                    ),

                    "aic_selected_AIC": (
                        None
                        if aic_selected_row is None
                        else aic_selected_row.get("AIC")
                    ),

                    "aic_selected_BIC": (
                        None
                        if aic_selected_row is None
                        else aic_selected_row.get("BIC")
                    ),

                    "aic_selected_RSS": (
                        None
                        if aic_selected_row is None
                        else aic_selected_row.get(
                            "trajectory_reconstruction_rss"
                        )
                    ),

                    "aic_selected_k": (
                        None
                        if aic_selected_row is None
                        else aic_selected_row.get(
                            "n_likelihood_parameters"
                        )
                    ),

                    # ============================================================
                    # BIC-selected candidate
                    # ============================================================

                    "bic_selected_candidate": (
                        None
                        if bic_selected_row is None
                        else bic_selected_row.get("candidate")
                    ),

                    "bic_selected_AIC": (
                        None
                        if bic_selected_row is None
                        else bic_selected_row.get("AIC")
                    ),

                    "bic_selected_BIC": (
                        None
                        if bic_selected_row is None
                        else bic_selected_row.get("BIC")
                    ),

                    "bic_selected_RSS": (
                        None
                        if bic_selected_row is None
                        else bic_selected_row.get(
                            "trajectory_reconstruction_rss"
                        )
                    ),

                    "bic_selected_k": (
                        None
                        if bic_selected_row is None
                        else bic_selected_row.get(
                            "n_likelihood_parameters"
                        )
                    ),

                    # ============================================================
                    # Agreement among selection methods
                    # ============================================================

                    "proposed_aic_agree": (
                        aic_selected_row is not None
                        and selected.get("candidate")
                        == aic_selected_row.get("candidate")
                    ),

                    "proposed_bic_agree": (
                        bic_selected_row is not None
                        and selected.get("candidate")
                        == bic_selected_row.get("candidate")
                    ),

                    "aic_bic_agree": (
                        aic_selected_row is not None
                        and bic_selected_row is not None
                        and aic_selected_row.get("candidate")
                        == bic_selected_row.get("candidate")
                    ),
                }
            )

            for _, srow in summary.iterrows():
                cname = srow["candidate"]
                prefix = f"{cname}__"

                for key in [
                    "rank",
                    "selected",
                    "include_environment",
                    "complexity",
                    "primary_value",

                    "E_traj_reconstruction",
                    "E_traj_reconstruction_std",
                    "E_traj_reconstruction_rel",

                    "E_traj_prediction",
                    "E_traj_prediction_std",

                    "E_traj_full",
                    "E_traj_full_std",

                    "E_res",
                    "E_res_rel",

                    "condition_number",
                    "phi_E_norm",
                    "phi_A_norm",
                    "f_norm",
                    "accepted",
                    "rejection_reason",
                    # Information-criterion inputs
                    "trajectory_reconstruction_rss",
                    "trajectory_reconstruction_n_obs",
                    "n_model_coefficients",
                    "n_likelihood_parameters",

                    # Information criteria
                    "AIC",
                    "BIC",

                    # Independent criterion rankings
                    "aic_rank",
                    "bic_rank",
                    "aic_selected",
                    "bic_selected",
                ]:
                    if key in srow:
                        row[prefix + key] = srow[key]

        except Exception as exc:
            row["failed"] = True
            row["failure_message"] = str(exc)
            row["traceback"] = traceback.format_exc()
            print(f"Trial {trial:03d} failed: {exc}")

        return row

    def _build_basis_config(self) -> BasisConfig:
        cfg = self.config

        radial = RadialBasisSpec(
            family=cfg.radial_basis_family,
            degree=cfg.radial_degree,
            n_basis=cfg.radial_n_basis,
        )

        energy = RadialBasisSpec(
            family=cfg.radial_basis_family,
            degree=cfg.radial_degree,
            n_basis=cfg.energy_radial_n_basis or cfg.radial_n_basis,
        )

        alignment = RadialBasisSpec(
            family=cfg.radial_basis_family,
            degree=cfg.radial_degree,
            n_basis=cfg.alignment_radial_n_basis or cfg.radial_n_basis,
        )

        environment = EnvironmentBasisSpec(
            family=cfg.environment_basis_family,
            degree=cfg.environment_degree,
            n_basis_per_dim=cfg.environment_n_basis_per_dim,
        )

        return BasisConfig(
            radial=radial,
            energy=energy,
            alignment=alignment,
            environment=environment,
        )