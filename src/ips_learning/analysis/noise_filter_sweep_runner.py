from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

from ips_learning.data.dataset_repository import DatasetRepository
from ips_learning.data.sliding_window_noise import SlidingWindowNoise
from ips_learning.data.kalman_filter_noise import KalmanFilterNoise


from ips_learning.features.feature_builder import FeatureBuilder

from ips_learning.bases.specs import (
    BasisConfig,
    RadialBasisSpec,
    EnvironmentBasisSpec,
)
from ips_learning.bases.builders import BasisBuilder
from ips_learning.fitting.solver import LinearSystemSolver
from ips_learning.fitting.assembler import NormalEquationAssembler
from ips_learning.models.factories import LearnedModelFactory
from ips_learning.rollout.simulator import LearnedSystemRollout
from ips_learning.evaluation.run_evaluation import EvaluationRunner
from ips_learning.evaluation.trajectory_errors import TrajectoryErrorEvaluator


from ips_learning.features.feature_repository import FeatureRepository
from ips_learning.bases.repository import BasisRepository
from ips_learning.fitting.system_repository import NormalSystemRepository
from ips_learning.rollout.repository import RolloutRepository
from ips_learning.statistics.density import build_empirical_error_density
from ips_learning.systems.registry import (
    true_kernel_from_metadata,
    true_env_from_metadata,
)


@dataclass(frozen=True)
class NoiseFilterSweepConfig:
    """
    Stage 6D noise robustness sweep configuration.

    This runner perturbs only the train-fit dataset, fits on noisy data,
    and evaluates against clean truth.
    """

    system_name: str = "spp"
    profile_name: str = "baseline"
    interaction_type: str = "energy"
    artifact_root: str | Path = "artifacts"
    system_order: int = 1

    train_fit_split: str = "train_fit"
    train_fit_dataset_name: str = "dataset_train_fit_trial_000"

    train_split: str = "train"
    train_dataset_name: str = "dataset_train_trial_000"

    test_split: str = "test"
    test_dataset_name: str = "dataset_test"

    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10)
    noise_type: str = "multiplicative_uniform"

    train_horizon_end: float = 0.5

    radial_basis_family: str = "bspline"
    radial_degree: int = 1
    radial_n_basis: int = 10

    environment_basis_family: str = "bspline"
    environment_degree: int = 1
    environment_n_basis_per_dim: int = 10

    linear_solver_method: str = "ridge"
    ridge_lambda: float = 1e-6
    svd_tol: float = 1e-12

    rollout_method: str = "Radau"
    rollout_rtol: float = 1e-6
    rollout_atol: float = 1e-6

    seed_base: int = 9000

    kernel_grid_size: int = 300
    environment_grid_size: int = 120


    overwrite: bool = True


class NoiseFilterSweepRunner:
    """
    Stage: one-trial noise robustness sweep.

    This runner follows the notebook logic exactly:
    1. Load clean train_fit, full train, and test datasets.
    2. Perturb only train_fit.
    3. Fit on noisy train_fit.
    4. Evaluate against clean truth.
    5. Save one row per noise level.
    """

    def __init__(self, config: NoiseFilterSweepConfig) -> None:
        self.config = config
        root = Path(config.artifact_root)

        self.feature_builder = FeatureBuilder()

        self.basis_config = BasisConfig(
            radial=RadialBasisSpec(
                family=config.radial_basis_family,
                degree=config.radial_degree,
                n_basis=config.radial_n_basis,
            ),
            environment=EnvironmentBasisSpec(
                family=config.environment_basis_family,
                degree=config.environment_degree,
                n_basis_per_dim=config.environment_n_basis_per_dim,
            ),
        )

        self.basis_builder = BasisBuilder(config=self.basis_config)

        self.solver = LinearSystemSolver(
            method=config.linear_solver_method,
            ridge_lambda=config.ridge_lambda,
            svd_tol=config.svd_tol,
        )

        self.model_factory = LearnedModelFactory()
        self.eval_runner = EvaluationRunner()
        self.traj_eval = TrajectoryErrorEvaluator()

        self.dataset_repo = DatasetRepository(root_dir=root / "datasets")
        self.feature_repo = FeatureRepository(root_dir=root / "features")
        self.basis_repo = BasisRepository(root_dir=root / "bases")
        self.normal_repo = NormalSystemRepository(root_dir=root / "fits")
        self.rollout_repo = RolloutRepository(root_dir=root / "rollouts")

    def run(self) -> pd.DataFrame:
        cfg = self.config

        train_fit_clean = self.dataset_repo.load(
            system_name=cfg.system_name,
            profile_name=cfg.profile_name,
            split=cfg.train_fit_split,
            dataset_name=cfg.train_fit_dataset_name,
        )

        train_full_clean = self.dataset_repo.load(
            system_name=cfg.system_name,
            profile_name=cfg.profile_name,
            split=cfg.train_split,
            dataset_name=cfg.train_dataset_name,
        )

        test_clean = self.dataset_repo.load(
            system_name=cfg.system_name,
            profile_name=cfg.profile_name,
            split=cfg.test_split,
            dataset_name=cfg.test_dataset_name,
        )

        # Build one clean empirical measure for function-error weighting.
        # Reuse it for every noise/filter case so all variants are compared
        # under the same clean training-data distribution.
        features_clean = self.feature_builder.build(train_fit_clean)
        basis_clean = self.basis_builder.build(features_clean)

        empirical_density_clean = build_empirical_error_density(
            features=features_clean,
            localization_metadata=basis_clean.localization_metadata,
            kernel_grid_size=cfg.kernel_grid_size,
            environment_grid_size=cfg.environment_grid_size,
        )

        rows = []

        for noise_level in cfg.noise_levels:
            print(f"\nRunning noise level ζ={noise_level}")

            """
            ζ=0.05, raw
            ζ=0.05, sliding_window
            ζ=0.05, kalman_filter
            """

            case_rows = self._run_one_noise_level(
                noise_level=noise_level,
                train_fit_clean=train_fit_clean,
                train_full_clean=train_full_clean,
                test_clean=test_clean,
                empirical_density_clean=empirical_density_clean,
            )

            rows.extend(case_rows)

        df = pd.DataFrame(rows)
        self.save_results(df)
        return df

    def _run_one_noise_level(
        self,
        noise_level: float,
        train_fit_clean,
        train_full_clean,
        test_clean,
        empirical_density_clean,
    ) -> list[dict]:
        cfg = self.config
        rows = []
        # --------------------------------------------------
        # 1. Create noisy training data only
        # --------------------------------------------------
        noise_model1 = SlidingWindowNoise(
            noise_level=noise_level,
            seed=cfg.seed_base + int(noise_level * 1000),
            noise_type=cfg.noise_type,
        )
        train_fit_noisy_sliding = noise_model1.perturb(train_fit_clean)

        noise_model2 = KalmanFilterNoise(
            noise_level=noise_level,
            seed=cfg.seed_base + int(noise_level * 1000),
            noise_type=cfg.noise_type,
        )
        train_fit_noisy_raw, train_fit_noisy_kalman = noise_model2.perturb(train_fit_clean)

        noise_variants = {
            "raw": train_fit_noisy_raw,
            "sliding_window": train_fit_noisy_sliding,
            "kalman_filter": train_fit_noisy_kalman,
        }
        #SAVE NOISE AND FILTER DATASETS
        for preprocessing, dataset in noise_variants.items():
            case_name = self._case_name(noise_level, preprocessing)

            self.dataset_repo.save(
                dataset=dataset,
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split="noise_filter_sweep",
                dataset_name=case_name,
                overwrite=cfg.overwrite,
            )


        
        for preprocessing, train_fit_variant in noise_variants.items():
            case_name = self._case_name(noise_level, preprocessing)


            # --------------------------------------------------
            # 2. Build features and basis from noisy train_fit
            # -------------------------------------------------
            features_noisy = self.feature_builder.build(train_fit_variant)

            self.feature_repo.save(
                features_noisy,
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split="noise_filter_sweep",
                feature_name=case_name,
                overwrite=True,
            )

            basis_bundle = self.basis_builder.build(features_noisy)
            self.basis_repo.save(
                basis_bundle=basis_bundle,
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split="noise_filter_sweep",
                basis_name=case_name,
                overwrite=True,
            )

            # --------------------------------------------------
            # 3. Fit on noisy data
            # --------------------------------------------------
            interaction_type = features_noisy.metadata.get(
                    "interaction_type",
                    train_fit_clean.metadata.get("interaction_type", cfg.interaction_type),
                )

            assembler = NormalEquationAssembler(
                    parallel_jobs=1,
                    interaction_type=interaction_type,
                )
            fit_result = assembler.fit(
                    features=features_noisy,
                    basis_bundle=basis_bundle,
                    solver=self.solver,
                    system_name=cfg.system_name,
                )

            if fit_result.A is None or fit_result.b is None:
                    raise RuntimeError(
                        "fit_result.A and fit_result.b are required for Stage 2.7D artifact saving and conditioning."
                    )

            diagnostics_dict = NormalSystemRepository._diagnostics_to_dict(
                    fit_result.diagnostics
                )
            system_order = train_fit_clean.metadata.get("system_order", cfg.system_order)
            environment_variable = "position" if system_order == 1 else "velocity"

            self.normal_repo.save(
                    system_name=cfg.system_name,
                    profile_name=cfg.profile_name,
                    fit_case_name=case_name,
                    A=fit_result.A,
                    b=fit_result.b,
                    theta=fit_result.theta,
                    n_radial=fit_result.n_radial,
                    n_environment=fit_result.n_environment,
                    diagnostics=diagnostics_dict,
                    metadata={
                        "system_name": cfg.system_name,
                        "profile_name": cfg.profile_name,
                        "interaction_type": cfg.interaction_type,
                        "system_order": system_order,
                        "environment_variable": environment_variable,
                        "case_name": case_name,
                        "noise_level": float(noise_level),
                        "basis_metadata": basis_bundle.localization_metadata,
                        "fit_metadata": fit_result.metadata,
                    },
                    overwrite=True,
                )

            self.normal_repo.save_fit_result(
                    fit_result=fit_result,
                    system_name=cfg.system_name,
                    profile_name=cfg.profile_name,
                    fit_case_name=case_name,
                    overwrite=cfg.overwrite,
                )

            learned_model = self.model_factory.build_from_fit_result(
                                fit_result=fit_result,
                                N=features_noisy.X.shape[2],
                                D=features_noisy.X.shape[3],
                            )


            # --------------------------------------------------
            # 4. Build true system from clean metadata
            # -------------------------------------------------
            true_kernel = true_kernel_from_metadata(train_fit_clean.metadata)
            true_env = true_env_from_metadata(train_fit_clean.metadata)
        


            # Use the fixed empirical measure built from clean training data.
            r_grid = empirical_density_clean.r_grid
            kernel_weights = empirical_density_clean.kernel_weights
            z1_grid = empirical_density_clean.z1_grid
            z2_grid = empirical_density_clean.z2_grid
            env_weights = empirical_density_clean.environment_weights

            # --------------------------------------------------
            # 6. Rollouts
            # --------------------------------------------------
     
            rollout_failed = False
            failure_message = ""

            try:
                rollout_engine = LearnedSystemRollout(learned_model)

                recon_artifact = rollout_engine.simulate_many(
                    U0=train_fit_clean.U0,
                    t_eval=train_fit_clean.t,
                    method=cfg.rollout_method,
                    rtol=cfg.rollout_rtol,
                    atol=cfg.rollout_atol,
                    metadata={
                        "scenario": "noise_reconstruction",
                        "noise_level": float(noise_level),
                        "artifact_type": "rollout",
                        "stage": "step_6D",
                        "system_name": cfg.system_name,
                        "profile_name": cfg.profile_name,
                        "system_order": system_order,
                        "environment_variable": environment_variable,
                        "interaction_type": interaction_type,
                        "case_name": case_name,

                    },
                )

                train_pred_artifact = rollout_engine.simulate_many(
                    U0=train_full_clean.U0,
                    t_eval=train_full_clean.t,
                    method=cfg.rollout_method,
                    rtol=cfg.rollout_rtol,
                    atol=cfg.rollout_atol,
                    metadata={
                        "scenario": "noise_same_ic_prediction",
                        "noise_level": float(noise_level),
                        "artifact_type": "rollout",
                        "stage": "step_6D",
                        "system_name": cfg.system_name,
                        "profile_name": cfg.profile_name,
                        "system_order": system_order,
                        "environment_variable": environment_variable,
                        "interaction_type": interaction_type,
                        "case_name": case_name,

                    },
                )

                test_artifact = rollout_engine.simulate_many(
                    U0=test_clean.U0,
                    t_eval=test_clean.t,
                    method=cfg.rollout_method,
                    rtol=cfg.rollout_rtol,
                    atol=cfg.rollout_atol,
                    metadata={
                        "scenario": "noise_unseen_ic_prediction",
                        "noise_level": float(noise_level),
                        "artifact_type": "rollout",
                        "stage": "step_6D",
                        "system_name": cfg.system_name,
                        "profile_name": cfg.profile_name,
                        "system_order": system_order,
                        "environment_variable": environment_variable,
                        "interaction_type": interaction_type,
                        "case_name": case_name,

                    },
                )

                # --------------------------------------------------
                # Save rollout artifacts for this noise level
                # --------------------------------------------------
                self.rollout_repo.save_case(
                    artifact=recon_artifact,
                    system_name=cfg.system_name,
                    profile_name=cfg.profile_name,
                    case_name=case_name,
                    rollout_name=f"{case_name}_noise_reconstruction",
                    overwrite=cfg.overwrite,
                )

                self.rollout_repo.save_case(
                    artifact=train_pred_artifact,
                    system_name=cfg.system_name,
                    profile_name=cfg.profile_name,
                    case_name=case_name,
                    rollout_name=f"{case_name}_noise_same_ic_prediction",
                    overwrite=cfg.overwrite,
                )

                self.rollout_repo.save_case(
                    artifact=test_artifact,
                    system_name=cfg.system_name,
                    profile_name=cfg.profile_name,
                    case_name=case_name,
                    rollout_name=f"{case_name}_noise_unseen_ic_prediction",
                    overwrite=cfg.overwrite,
                )

                print(f"Saved rollouts for noise level ζ={noise_level}")

                train_split_idx = int(
                    np.argmin(np.abs(train_full_clean.t - cfg.train_horizon_end))
                )
                test_split_idx = int(
                    np.argmin(np.abs(test_clean.t - cfg.train_horizon_end))
                )

                metrics_train = self.traj_eval.evaluate_batch(
                    t_true=train_full_clean.t,
                    X_true_batch=train_full_clean.X,
                    t_hat=train_pred_artifact.t,
                    X_hat_batch=train_pred_artifact.X_hat,
                    split_index=train_split_idx,
                )

                metrics_test = self.traj_eval.evaluate_batch(
                    t_true=test_clean.t,
                    X_true_batch=test_clean.X,
                    t_hat=test_artifact.t,
                    X_hat_batch=test_artifact.X_hat,
                    split_index=test_split_idx,
                )

            except Exception as e:
                rollout_failed = True
                failure_message = str(e)
                recon_artifact = None
                metrics_train = {}
                metrics_test = {}

            # --------------------------------------------------
            # 7. Residual/function metrics
            # --------------------------------------------------
            try:
                report = self.eval_runner.run(
                    features=features_noisy,
                    learned_model=learned_model,
                    reconstruction_artifact=recon_artifact,
                    prediction_artifact=None,
                    prediction_true_dataset=None,
                    true_kernel=true_kernel,
                    true_env=true_env,
                    kernel_grid=r_grid,
                    z1_grid=z1_grid,
                    z2_grid=z2_grid,
                    kernel_weights=kernel_weights,
                    env_weights=env_weights,
                    split_index=None,
                )

                E_res = report.residual_metrics.get("E_res", np.nan)
                E_phi = report.function_metrics.get("E_phi", np.nan)
                E_phi_relative = report.function_metrics.get("E_phi_relative", np.nan)
                E_env = report.function_metrics.get("E_env", np.nan)
                E_env_relative = report.function_metrics.get("E_env_relative", np.nan)

                if recon_artifact is not None:
                    E_recon = report.trajectory_metrics["reconstruction"].get(
                        "reconstruction_Linf_mean",
                        np.nan,
                    )
                else:
                    E_recon = np.nan

            except Exception as e:
                E_res = np.nan
                E_phi_relative = np.nan
                E_env_relative = np.nan
                E_recon = np.nan
                rollout_failed = rollout_failed
                failure_message = (failure_message + " | " if failure_message else "") + "eval failed: " + str(e)

            # --------------------------------------------------
            # 8. Condition number
            # --------------------------------------------------
            A = np.asarray(fit_result.A, dtype=float)

            cond_A = self._safe_condition_number(A)

            row ={
                "noise_level": noise_level,
                "preprocessing": preprocessing,
                "case_name": case_name,
                "system_order": system_order,
                "environment_variable": environment_variable,
                "E_res": E_res,
                "E_phi": E_phi,
                "E_phi_relative": E_phi_relative,
                "E_env": E_env,
                "E_env_relative": E_env_relative,
                "E_recon_train_fit": E_recon,
                "E_pred_train_same_ic": metrics_train.get(
                    "prediction_Linf_mean",
                    np.nan,
                ),
                "E_test_preT_unseen_ic": metrics_test.get(
                    "reconstruction_Linf_mean",
                    np.nan,
                ),
                "E_test_pred_unseen_ic": metrics_test.get(
                    "prediction_Linf_mean",
                    np.nan,
                ),
                "cond_A": cond_A,
                "rollout_failed": rollout_failed,
                "failure_message": failure_message,
            }
            # --------------------------------------------------
            # 9. Save per-noise metrics artifact
            # -------------------------------------------------
            metrics_dir = (
                Path(cfg.artifact_root)
                / cfg.system_name
                / "evaluations"
                / "noise_filter_sweep"
                / "cases"
                / cfg.profile_name
                / case_name
            )

            metrics_dir.mkdir(parents=True, exist_ok=True)

            with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)

            rows.append(row)
        return rows

    def save_results(
        self,
        df: pd.DataFrame,
        output_dir: str | Path | None = None,
        file_stem: str = "noise_filter_sweep_metrics",
    ) -> tuple[Path, Path]:
        """
        Save aggregate noise-sweep evaluation artifacts.

        Path:
            artifacts/{system}/evaluations/noise_filter_sweep/noise_filter_sweep_metrics.csv
        """
        if output_dir is None:
            output_dir = (
                Path(self.config.artifact_root)
                / self.config.system_name
                / "evaluations"
                / "noise_filter_sweep"
            )
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"{file_stem}.csv"
        config_path = output_dir / "noise_filter_sweep_settings.json"

        df.to_csv(csv_path, index=False)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

        return csv_path, config_path
    
    @staticmethod
    def _safe_condition_number(A: np.ndarray, tol: float = 1e-20) -> float:
        """
        Compute condition number using singular values.

        Works for both square and rectangular matrices.
        """
        try:
            A = np.asarray(A, dtype=float)

            if A.size == 0:
                return float("nan")

            s = np.linalg.svd(A, compute_uv=False)

            if s.size == 0:
                return float("nan")

            s_max = float(np.max(s))
            s_min = float(np.min(s))

            if s_min <= tol:
                return float("inf")

            return s_max / s_min

        except np.linalg.LinAlgError:
            return float("inf")
        
    @staticmethod
    def _case_name(noise_level: float, preprocessing: str) -> str:
        noise_tag = f"noise_{str(noise_level).replace('.', 'p')}"
        return f"trial_000_{noise_tag}_{preprocessing}"
        


        
    