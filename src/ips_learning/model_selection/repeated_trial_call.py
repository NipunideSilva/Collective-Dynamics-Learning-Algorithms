from __future__ import annotations

import json
from pathlib import Path

from ips_learning.model_selection.repeated_trial_runner import (
    ModelSelectionRepeatedTrialRunConfig,
    ModelSelectionRepeatedTrialRunner,
)
from ips_learning.model_selection.repeated_trial_summary import (
    ModelSelectionRepeatedTrialSummaryBuilder,
)


class ModelSelectionRepeatedTrialEvaluationRunner:
    """
    Wrapper for running repeated-trial model-selection experiments.

    This class:
        1. runs model selection for all trials,
        2. saves raw trial metrics,
        3. builds repeated-trial summaries,
        4. saves config metadata.

    It does not perform N-scaling.
    """

    def __init__(
        self,
        config: ModelSelectionRepeatedTrialRunConfig,
    ) -> None:
        self.config = config

    def run(self) -> dict:
        cfg = self.config

        artifact_root = Path(cfg.artifact_root)

        output_dir = (
            artifact_root
            / "model_selection"
            / cfg.system_name
            / cfg.profile_name
            / "repeated_trials"
            / "summaries"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        runner = ModelSelectionRepeatedTrialRunner(cfg)
        trial_df = runner.run_all_trials()

        metrics_path = output_dir / "repeated_trial_model_selection_metrics.csv"
        trial_df.to_csv(metrics_path, index=False)

        summary_builder = ModelSelectionRepeatedTrialSummaryBuilder()
        summaries = summary_builder.build_all_summaries(trial_df)
        summary_paths = summary_builder.save_all_summaries(
            summaries=summaries,
            output_dir=output_dir,
        )

        config_metadata = self._config_metadata(
            metrics_path=metrics_path,
            summary_paths=summary_paths,
        )

        config_path = output_dir / "repeated_trial_model_selection_config.json"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_metadata, f, indent=2, sort_keys=True)

        expected_files = [
            metrics_path,
            config_path,
            *summary_paths.values(),
        ]

        for path in expected_files:
            print("FOUND" if Path(path).exists() else "MISSING", path)

        assert all(Path(path).exists() for path in expected_files)

        return {
            "trial_df": trial_df,
            "summaries": summaries,
            "metrics_path": metrics_path,
            "summary_paths": summary_paths,
            "config_path": config_path,
            "output_dir": output_dir,
        }

    def _config_metadata(
        self,
        *,
        metrics_path: Path,
        summary_paths: dict[str, Path],
    ) -> dict:
        cfg = self.config

        return {
            "artifact_type": "repeated_trials_model_selection",
            "stage": "model_selection_repeated_trials",
            "system_name": cfg.system_name,
            "profile_name": cfg.profile_name,
            "model_selection_mode": cfg.model_selection_mode,
            "num_trials": cfg.num_trials,
            "trial_start": cfg.trial_start,
            "fit_end_time": cfg.fit_end_time,
            "prediction_end_time": cfg.prediction_end_time,
            "splits": {
                "train_fit_split": cfg.train_fit_split,
                "validation_split": cfg.validation_split,
                "test_split": cfg.test_split,
            },
            "dataset_names": {
                "train_fit_name_template": cfg.train_fit_name_template,
                "validation_dataset_name": cfg.validation_dataset_name,
                "test_dataset_name": cfg.test_dataset_name,
            },
            "basis": {
                "radial_basis_family": cfg.radial_basis_family,
                "radial_degree": cfg.radial_degree,
                "radial_n_basis": cfg.radial_n_basis,
                "energy_radial_n_basis": cfg.energy_radial_n_basis,
                "alignment_radial_n_basis": cfg.alignment_radial_n_basis,
                "environment_basis_family": cfg.environment_basis_family,
                "environment_degree": cfg.environment_degree,
                "environment_n_basis_per_dim": cfg.environment_n_basis_per_dim,
            },
            "solver": {
                "solver_method": cfg.solver_method,
                "ridge_lambda": cfg.ridge_lambda,
                "svd_tol": cfg.svd_tol,
            },
            "selection": {
                "primary_metric": cfg.primary_metric,
                "relative_tie_tol": cfg.relative_tie_tol,
                "absolute_tie_tol": cfg.absolute_tie_tol,
                "max_condition_number": cfg.max_condition_number,
                
                "max_reconstruction_rel": getattr(cfg, "max_reconstruction_rel", None),
                "max_residual_rel": getattr(cfg, "max_residual_rel", None),

                "prefer_simpler_on_tie": cfg.prefer_simpler_on_tie,
            },
            "information_criteria": {
                "likelihood": (
                    "independent scalar Gaussian working likelihood"
                ),
                "rss_metric": (
                    "training position-trajectory reconstruction RSS"
                ),
                "n_observations_definition": (
                    "M_train * L_fit * N * D"
                ),
                "parameter_count_definition": (
                    "fit_result.theta.size plus estimated variance"
                ),
                "include_noise_variance_parameter": (
                    cfg.include_noise_variance_parameter
                ),
            },
            "outputs": {
                "repeated_trial_metrics": str(metrics_path),
                **{key: str(value) for key, value in summary_paths.items()},
            },
        }