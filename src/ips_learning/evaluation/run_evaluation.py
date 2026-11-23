import numpy as np

from ips_learning.evaluation.report import EvaluationReport
from ips_learning.evaluation.residuals import ResidualEvaluator
from ips_learning.evaluation.function_errors import (
    KernelErrorEvaluator,
    EnvironmentErrorEvaluator,
)
from ips_learning.evaluation.trajectory_errors import TrajectoryErrorEvaluator


class EvaluationRunner:
    """
    Orchestrate evaluation for one learned model / one run.

    Supports:
        first-order:  dot X  = f(X) + f_phi(X)
        second-order: ddot X = f(V) + f_phi(X)
    """

    def __init__(self):
        self.residual_eval = ResidualEvaluator()
        self.kernel_eval = KernelErrorEvaluator()
        self.env_eval = EnvironmentErrorEvaluator()
        self.traj_eval = TrajectoryErrorEvaluator()

    def run(
        self,
        features,
        learned_model,
        reconstruction_artifact=None,
        prediction_artifact=None,
        prediction_true_dataset=None,
        true_kernel=None,
        true_env=None,
        kernel_grid=None,
        kernel_samples=None,
        kernel_weights=None,
        z1_grid=None,
        z2_grid=None,
        Z_samples=None,
        env_weights=None,
        # backward-compatible aliases
        v1_grid=None,
        v2_grid=None,
        V_samples=None,
        split_index=None,
    ) -> EvaluationReport:
        source_metadata = getattr(features, "metadata", {})
        system_order = source_metadata.get("system_order", 2)

        if system_order not in (1, 2):
            raise ValueError(
                f"features.metadata['system_order'] must be 1 or 2, got {system_order}."
            )

        residual_metrics = self.residual_eval.evaluate(features, learned_model)

        function_metrics = {}

        if true_kernel is not None and (
            kernel_grid is not None or kernel_samples is not None
        ):
            function_metrics.update(
                self.kernel_eval.evaluate(
                    true_kernel=true_kernel,
                    learned_kernel=learned_model.learned_kernel,
                    r_grid=kernel_grid,
                    r_samples=kernel_samples,
                    weights=kernel_weights,
                )
            )

        # Backward-compatible aliases
        if Z_samples is None and V_samples is not None:
            Z_samples = V_samples

        if z1_grid is None and v1_grid is not None:
            z1_grid = v1_grid

        if z2_grid is None and v2_grid is not None:
            z2_grid = v2_grid

        if true_env is not None and (Z_samples is not None or z1_grid is not None):
            function_metrics.update(
                self.env_eval.evaluate(
                    true_env=true_env,
                    learned_env=learned_model.learned_environment,
                    z1_grid=z1_grid,
                    z2_grid=z2_grid,
                    Z_samples=Z_samples,
                    weights=env_weights,
                )
            )

        trajectory_metrics = {} # For ML sweep this is unnecessary. hence empty when passing

        if reconstruction_artifact is not None:
            trajectory_metrics["reconstruction"] = self.traj_eval.evaluate_batch(
                t_true=features.t,
                X_true_batch=features.X,
                t_hat=reconstruction_artifact.t,
                X_hat_batch=reconstruction_artifact.X_hat,
                split_index=split_index,
            )

        if prediction_artifact is not None and prediction_true_dataset is not None:
            trajectory_metrics["prediction"] = self.traj_eval.evaluate_batch(
                t_true=prediction_true_dataset.t,
                X_true_batch=prediction_true_dataset.X,
                t_hat=prediction_artifact.t,
                X_hat_batch=prediction_artifact.X_hat,
                split_index=None,
            )

        metadata = {
            "stage": "2.6",
            "artifact_type": "evaluation_report",
            "model_family": (
                "first_order_nonparametric"
                if system_order == 1
                else "second_order_nonparametric"
            ),
            "system_name": source_metadata.get("system_name"),
            "system_order": system_order,
            "environment_variable": (
                "position" if system_order == 1 else "velocity"
            ),
            "interaction_type": getattr(learned_model, "interaction_type", None),
            "source_feature_metadata": source_metadata,
        }

        return EvaluationReport(
            residual_metrics=residual_metrics,
            function_metrics=function_metrics,
            trajectory_metrics=trajectory_metrics,
            metadata=metadata,
        )
