from ips_learning.data.dataset_repository import DatasetRepository
from ips_learning.features.feature_repository import FeatureRepository
from ips_learning.bases.repository import BasisRepository
from ips_learning.fitting.system_repository import NormalSystemRepository
from ips_learning.models.factories import LearnedModelFactory
from ips_learning.rollout.repository import RolloutRepository
from ips_learning.statistics.density import build_empirical_error_density

from ips_learning.evaluation.residuals import ResidualEvaluator
from ips_learning.evaluation.function_errors import KernelErrorEvaluator, EnvironmentErrorEvaluator
from ips_learning.evaluation.trajectory_errors import TrajectoryErrorEvaluator

import numpy as np
import pandas as pd
import json

from ips_learning.systems.registry import (
    true_kernel_from_metadata,
    true_env_from_metadata,
)


class GeneralRunOneBaseline:
    def __init__(
        self,
        SYSTEM_NAME,
        PROFILE_NAME,
        FIT_CASE_NAME,
        TRAIN_FIT_SPLIT,
        TRAIN_FIT_DATASET_NAME,
        TRAIN_SPLIT,
        TRAIN_DATASET_NAME,
        TEST_SPLIT,
        TEST_DATASET_NAME,
        FIT_END_TIME,
        ARTIFACTS,
        SYSTEM_ORDER: int | None = None,
    ):

        self.one_run_df = self._run(
            SYSTEM_NAME,
            PROFILE_NAME,
            FIT_CASE_NAME,
            TRAIN_FIT_SPLIT,
            TRAIN_FIT_DATASET_NAME,
            TRAIN_SPLIT,
            TRAIN_DATASET_NAME,
            TEST_SPLIT,
            TEST_DATASET_NAME,
            FIT_END_TIME,
            ARTIFACTS,
            SYSTEM_ORDER
            )

    def _run(self,
        SYSTEM_NAME,
        PROFILE_NAME,
        FIT_CASE_NAME,
        TRAIN_FIT_SPLIT,
        TRAIN_FIT_DATASET_NAME,
        TRAIN_SPLIT,
        TRAIN_DATASET_NAME,
        TEST_SPLIT,
        TEST_DATASET_NAME,
        FIT_END_TIME,
        ARTIFACTS,
        SYSTEM_ORDER: int | None = None,
                ):
        # Load saved artifacts
        dataset_repo = DatasetRepository(root_dir=ARTIFACTS / "datasets")
        feature_repo = FeatureRepository(root_dir=ARTIFACTS / "features")

        fit_repo = NormalSystemRepository(root_dir=ARTIFACTS / "fits")
        rollout_repo = RolloutRepository(root_dir=ARTIFACTS / "rollouts")

        train_fit_dataset = dataset_repo.load(
            system_name=SYSTEM_NAME,
            profile_name=PROFILE_NAME,
            split=TRAIN_FIT_SPLIT,
            dataset_name=TRAIN_FIT_DATASET_NAME,
        )

        if SYSTEM_ORDER is None:
            SYSTEM_ORDER = train_fit_dataset.metadata.get("system_order")

        if SYSTEM_ORDER not in (1, 2):
            raise ValueError(f"SYSTEM_ORDER must be 1 or 2, got {SYSTEM_ORDER}.")

        environment_variable = "position" if SYSTEM_ORDER == 1 else "velocity"


        train_dataset = dataset_repo.load(
            system_name=SYSTEM_NAME,
            profile_name=PROFILE_NAME,
            split=TRAIN_SPLIT,
            dataset_name=TRAIN_DATASET_NAME,
        )

        test_dataset = dataset_repo.load(
            system_name=SYSTEM_NAME,
            profile_name=PROFILE_NAME,
            split=TEST_SPLIT,
            dataset_name=TEST_DATASET_NAME,
        )

        features = feature_repo.load(
            system_name=SYSTEM_NAME,
            profile_name=PROFILE_NAME,
            split=TRAIN_FIT_SPLIT,
            feature_name=TRAIN_FIT_DATASET_NAME,
        )

        fit_result = fit_repo.load_fit_result(
            system_name=SYSTEM_NAME,
            profile_name=PROFILE_NAME,
            fit_case_name=FIT_CASE_NAME,
        )

        learned_model = LearnedModelFactory().build_from_fit_result(
            fit_result=fit_result,
            N=int(fit_result.metadata["N"]),
            D=int(fit_result.metadata["D"]),
        )

        print("Loaded datasets, features, basis, fit_result, learned model.")

        # Load rollout artifacts from Stage 5

        FIT_RECON_ROLLOUT_NAME = f"{FIT_CASE_NAME}_train_fit_reconstruction"
        TRAIN_FULL_ROLLOUT_NAME = f"{FIT_CASE_NAME}_train_full_prediction"
        TEST_ROLLOUT_NAME = f"{FIT_CASE_NAME}_test_prediction"

        fit_recon_artifact = rollout_repo.load_case(
            system_name=SYSTEM_NAME,
            profile_name=PROFILE_NAME,
            case_name=FIT_CASE_NAME,
            scenario="train_fit_reconstruction",
            rollout_name=FIT_RECON_ROLLOUT_NAME,
        )

        train_full_artifact = rollout_repo.load_case(
            system_name=SYSTEM_NAME,
            profile_name=PROFILE_NAME,
            case_name=FIT_CASE_NAME,
            scenario="train_full_prediction",
            rollout_name=TRAIN_FULL_ROLLOUT_NAME,
        )

        test_artifact = rollout_repo.load_case(
            system_name=SYSTEM_NAME,
            profile_name=PROFILE_NAME,
            case_name=FIT_CASE_NAME,
            scenario="test_prediction",
            rollout_name=TEST_ROLLOUT_NAME,
        )

        print("Loaded rollout artifacts.")


        true_phi = true_kernel_from_metadata(train_fit_dataset.metadata)
        true_f = true_env_from_metadata(train_fit_dataset.metadata)

        print("Prepared true functions.")

        # Build empirical error weights directly from the current
        # training features used for fitting.
        empirical_density = build_empirical_error_density(
            features=features,
            localization_metadata=fit_result.basis_bundle.localization_metadata,
        )

        r_grid = empirical_density.r_grid
        kernel_weights = empirical_density.kernel_weights
        z1_grid = empirical_density.z1_grid
        z2_grid = empirical_density.z2_grid
        env_weights = empirical_density.environment_weights

        print("Built empirical evaluation grids and weights from training features.")

        # Compute residual and function errors

        residual_eval = ResidualEvaluator()
        kernel_eval = KernelErrorEvaluator()
        env_eval = EnvironmentErrorEvaluator()

        residual_metrics = residual_eval.evaluate(
            features=features,
            learned_model=learned_model,
        )

        kernel_metrics = kernel_eval.evaluate(
            true_kernel=true_phi,
            learned_kernel=learned_model.learned_kernel,
            r_grid=r_grid,
            weights=kernel_weights,
            include_radial_measure=True,
        )

        env_metrics = env_eval.evaluate(
            true_env=true_f,
            learned_env=learned_model.learned_environment,
            z1_grid=z1_grid,
            z2_grid=z2_grid,
            weights=env_weights,
        )

        print("Residual metrics:", residual_metrics.keys())
        print("Kernel metrics:", kernel_metrics.keys())
        print("Environment metrics:", env_metrics.keys())

        # Compute trajectory errors
        traj_eval = TrajectoryErrorEvaluator()

        train_split_idx = int(np.searchsorted(train_dataset.t, FIT_END_TIME, side="right") - 1)
        test_split_idx = int(np.searchsorted(test_dataset.t, FIT_END_TIME, side="right") - 1)

        fit_recon_metrics = traj_eval.evaluate_batch(
            t_true=train_fit_dataset.t,
            X_true_batch=train_fit_dataset.X,
            t_hat=fit_recon_artifact.t,
            X_hat_batch=fit_recon_artifact.X_hat,
            split_index=None,
        )

        train_full_metrics = traj_eval.evaluate_batch(
            t_true=train_dataset.t,
            X_true_batch=train_dataset.X,
            t_hat=train_full_artifact.t,
            X_hat_batch=train_full_artifact.X_hat,
            split_index=train_split_idx,
        )

        test_metrics = traj_eval.evaluate_batch(
            t_true=test_dataset.t,
            X_true_batch=test_dataset.X,
            t_hat=test_artifact.t,
            X_hat_batch=test_artifact.X_hat,
            split_index=test_split_idx,
        )

        print("Trajectory errors computed.")

        eval_dir = ARTIFACTS / SYSTEM_NAME / "evaluations" / "one_run" / FIT_CASE_NAME
        eval_dir.mkdir(parents=True, exist_ok=True)

        diag = fit_result.diagnostics

        diagnostics_metrics = {
            "method": diag.method,
            "condition_number": float(diag.condition_number),
            "rank": int(diag.rank),
            "min_singular": float(diag.min_singular),
            "max_singular": float(diag.max_singular),
            "ridge_lambda": diag.ridge_lambda,
            "svd_tol": diag.svd_tol,
        }

        metric_groups = {
            "residual": residual_metrics,
            "kernel": kernel_metrics,
            "environment": env_metrics,
            "trajectory_fit_reconstruction": fit_recon_metrics,
            "trajectory_train_full": train_full_metrics,
            "trajectory_test": test_metrics,
            "diagnostics": diagnostics_metrics,
        }

        summary = {
            "system_name": SYSTEM_NAME,
            "profile_name": PROFILE_NAME,
            "fit_case_name": FIT_CASE_NAME,
            "system_order": SYSTEM_ORDER,
            "environment_variable": environment_variable,
            "interaction_type": learned_model.interaction_type,
        }

        array_payload = {}

        for group_name, metrics in metric_groups.items():
            scalars, arrays = self._scalarize_metric_dict(metrics)
            summary[group_name] = scalars

            for key, value in arrays.items():
                array_payload[f"{group_name}__{key}"] = value

        with open(eval_dir / "one_run_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        if array_payload:
            np.savez_compressed(eval_dir / "arrays.npz", **array_payload)

        output_files = ["one_run_metrics.json", "one_run_metrics.csv", "metadata.json"]
        if array_payload:
            output_files.append("arrays.npz")

        metadata = {
            "artifact_type": "one_run_evaluation",
            "system_name": SYSTEM_NAME,
            "profile_name": PROFILE_NAME,
            "fit_case_name": FIT_CASE_NAME,
            "source_fit_case": FIT_CASE_NAME,
            "source_rollouts": {
                "train_fit_reconstruction": FIT_RECON_ROLLOUT_NAME,
                "train_full_prediction": TRAIN_FULL_ROLLOUT_NAME,
                "test_prediction": TEST_ROLLOUT_NAME,
            },
            "output_files": output_files,
        }

        with open(eval_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        print("Saved one-run evaluation artifact:")
        print(eval_dir)

        # Save flat CSV summary
        flat_row = {
            "system_name": SYSTEM_NAME,
            "profile_name": PROFILE_NAME,
            "fit_case_name": FIT_CASE_NAME,
            "interaction_type": learned_model.interaction_type,

            "system_order": SYSTEM_ORDER,
            "environment_variable": environment_variable,

            "E_res": residual_metrics.get("E_res", np.nan),
            "E_res_relative": residual_metrics.get("E_res_relative", np.nan),

            "E_phi": kernel_metrics.get("E_phi", np.nan),
            "E_env": env_metrics.get("E_env", np.nan),

            "E_recon_train_fit": fit_recon_metrics.get("reconstruction_Linf_mean", np.nan),

            "E_recon_train_full_0T": train_full_metrics.get("reconstruction_Linf_mean", np.nan),
            "E_pred_train_full_TTf": train_full_metrics.get("prediction_Linf_mean", np.nan),

            "E_recon_test_0T": test_metrics.get("reconstruction_Linf_mean", np.nan),
            "E_pred_test_TTf": test_metrics.get("prediction_Linf_mean", np.nan),

            "condition_number": diagnostics_metrics["condition_number"],
            "rank": diagnostics_metrics["rank"],
            "min_singular": diagnostics_metrics["min_singular"],
            "max_singular": diagnostics_metrics["max_singular"],
            "ridge_lambda": diagnostics_metrics["ridge_lambda"],
        }

        one_run_df = pd.DataFrame([flat_row])

        csv_path = eval_dir / "one_run_metrics.csv"
        one_run_df.to_csv(csv_path, index=False)

        return one_run_df



    def _build_true_system(self, system_name: str, profile_name: str):
            if system_name == "spp":
                from ips_learning.systems.spp import SPPSystem
                from ips_learning.experiments.generate_spp_data import build_spp_dataset_profile

                profile_obj = build_spp_dataset_profile(profile_name)
                cfg = profile_obj.common_config
                sp = cfg["system_params"]

                return SPPSystem(
                    N=cfg["N"],
                    D=cfg["D"],
                    cr=sp["cr"],
                    lr=sp["lr"],
                    ca=sp["ca"],
                    la=sp["la"],
                    a=sp["a"],
                    b=sp["b"],
                    eps_r=sp["eps_r"],
                )

            if system_name == "levi":
                from ips_learning.systems.levi import LeviSystem
                from ips_learning.experiments.generate_levi_data import build_levi_dataset_profile

                profile_obj = build_levi_dataset_profile(profile_name)
                cfg = profile_obj.common_config
                sp = cfg["system_params"]

                return LeviSystem(
                    N=cfg["N"],
                    D=cfg["D"],
                    u=sp["u"],
                    el=sp["el"],
                    I0=sp["I0"],
                    beta=sp["beta"],
                    lamda=sp["lamda"],
                    eps_r=sp["eps_r"],
                )

            raise ValueError(f"Unknown system_name: {system_name}")

    # Save one-run evaluation artifact
    def _to_jsonable_scalar(self, value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value


    def _scalarize_metric_dict(self, metrics: dict):
        out = {}
        arrays = {}

        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            elif np.isscalar(value) or value is None:
                out[key] = self._to_jsonable_scalar(value)
            else:
                try:
                    arr = np.asarray(value)
                    if arr.ndim == 0:
                        out[key] = arr.item()
                    else:
                        arrays[key] = arr
                except Exception:
                    out[key] = str(value)

        return out, arrays
