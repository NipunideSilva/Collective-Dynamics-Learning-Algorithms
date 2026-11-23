from __future__ import annotations

from dataclasses import dataclass
import traceback

import numpy as np
import pandas as pd
from pathlib import Path

from ips_learning.data.dataset_repository import DatasetRepository
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
from ips_learning.rollout.repository import RolloutRepository
from ips_learning.evaluation.residuals import ResidualEvaluator

from ips_learning.evaluation.trajectory_errors import TrajectoryErrorEvaluator
from ips_learning.evaluation.run_evaluation import EvaluationRunner

from ips_learning.systems.registry import (
    true_kernel_from_metadata,
    true_env_from_metadata,
)

@dataclass(frozen=True)
class ScalabilityTrialConfig:
    system_name: str
    profile_name: str # "baseline"
    interaction_type: str

    train_fit_split: str = "train_fit"
    train_full_split: str = "train"
    test_split: str = "test"
    test_4N_split: str = "test_4N"

    train_fit_name_template: str = "dataset_train_fit_trial_{trial:03d}"
    train_full_name_template: str = "dataset_train_trial_{trial:03d}"
    test_dataset_name: str = "dataset_test"
    test_4N_dataset_name: str = "dataset_test_4N"

    num_trials: int = 10
    train_horizon_end: float = 0.75

    radial_basis_family: str = "bspline"
    radial_degree: int = 1
    radial_n_basis: int = 10

    environment_basis_family: str = "bspline"
    environment_degree: int = 1
    environment_n_basis_per_dim: int = 10

    kernel_grid_size: int = 300
    environment_grid_size: int = 120
    system_order: int | None = None

    solver_method: str = "ridge"
    ridge_lambda: float = 1e-8
    svd_tol: float = 1e-12

    rollout_method: str = "Radau"
    rollout_rtol: float = 1e-3
    rollout_atol: float = 1e-6

    test_4N_N: int | None = None

    save_rollouts: bool = False
    rollout_model_name: str | None = None
    artifact_root: str | Path = "artifacts"

    save_function_values: bool = True
    function_value_dir: str | Path | None = None

    N4_kernel_scale: float = 1

    use_global_function_grids: bool = True


class ScalabilityTrialRunner:
    """
    General repeated-trial runner for first- and second-order systems.

    Supports:
        Kuramoto : system_order=1, interaction_type="energy"
        SPP      : system_order=2, interaction_type="energy"
        Levi     : system_order=2, interaction_type="alignment"
    """

    def __init__(self, config: ScalabilityTrialConfig):
        self.config = config

        if config.interaction_type not in {"energy", "alignment"}:
            raise ValueError(
                "interaction_type must be 'energy' or 'alignment', "
                f"got {config.interaction_type!r}."
            )
        if config.system_order == 1 and config.interaction_type == "alignment":
            raise ValueError("First-order systems currently support only energy interaction.")
        
        root = Path(config.artifact_root)

        #----------------------------------------------
        if config.function_value_dir is None:
            self.function_value_dir = (
                root
                / config.system_name
                / config.profile_name
               / "evaluations"
               / "repeated_trials"
                / "function_values"
            )
        else:
            self.function_value_dir = Path(config.function_value_dir)

        if config.save_function_values:
            self.function_value_dir.mkdir(parents=True, exist_ok=True)

        self.global_function_grids = None
        # we want a coomon grid for mean std of function values across trials,
        
        #----------------------------------------------

        self.dataset_repo = DatasetRepository(root_dir=root / "datasets")
        self.rollout_repo = RolloutRepository(root_dir=root / "rollouts")
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

        self.basis_builder = BasisBuilder(self.basis_config)

        self.solver = LinearSystemSolver(
            method=config.solver_method,
            ridge_lambda=config.ridge_lambda,
            svd_tol=config.svd_tol,
        )


        self.model_factory = LearnedModelFactory()
        self.eval_runner = EvaluationRunner()
        self.traj_eval = TrajectoryErrorEvaluator()

    def run_all_trials(self) -> pd.DataFrame:
        if self.config.save_function_values:
            self.global_function_grids = self._build_global_function_grids()

        rows = []

        for q in range(self.config.num_trials):
            trial = q + 1
            print(f"\nRunning {self.config.system_name} trial {trial:03d}")
            row = self.run_one_trial(trial)
            rows.append(row)

        return pd.DataFrame(rows)


    def run_one_trial(self, trial: int) -> dict:
        cfg = self.config

        row = {
            "trial": trial,
            "system_name": cfg.system_name,
            "profile_name": cfg.profile_name,
            "interaction_type": cfg.interaction_type,
            "configured_system_order": cfg.system_order,
            "failed": False,
            "failure_message": "",
        }

        try:
            train_fit_name = cfg.train_fit_name_template.format(trial=trial)
            train_full_name = cfg.train_full_name_template.format(trial=trial)

            train_fit = self.dataset_repo.load(
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split=cfg.train_fit_split,
                dataset_name=train_fit_name,
            )

            train_full = self.dataset_repo.load(
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split=cfg.train_full_split,
                dataset_name=train_full_name,
            )

            test = self.dataset_repo.load(
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split=cfg.test_split,
                dataset_name=cfg.test_dataset_name,
            )

            test_4N = self.dataset_repo.load(
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split=cfg.test_4N_split,
                dataset_name=cfg.test_4N_dataset_name,
            )

            system_order = cfg.system_order
            if system_order is None:
                system_order = train_fit.metadata.get("system_order")

            if system_order not in (1, 2):
                raise ValueError(f"system_order must be 1 or 2, got {system_order}.")

            environment_variable = "position" if system_order == 1 else "velocity"

            row["system_order"] = int(system_order)
            row["environment_variable"] = environment_variable

            row.update(
                {
                    "train_fit_dataset": train_fit_name,
                    "train_full_dataset": train_full_name,
                    "test_dataset": cfg.test_dataset_name,
                    "test_4N_dataset": cfg.test_4N_dataset_name,
                    "train_N": int(train_fit.metadata["N"]),
                    "test_N": int(test.metadata["N"]),
                    "test_4N_N": int(test_4N.metadata["N"]),
                    "train_fit_t0": float(train_fit.t[0]),
                    "train_fit_T": float(train_fit.t[-1]),
                    "train_full_Tf": float(train_full.t[-1]),
                    "test_Tf": float(test.t[-1]),
                    "test_4N_Tf": float(test_4N.t[-1]),
                }
            )

            # --------------------------------------------------
            # Features, basis, fit
            # --------------------------------------------------
            features = self.feature_builder.build(train_fit)

            interaction_type = features.metadata.get(
                                "interaction_type",
                                train_fit.metadata.get("interaction_type", cfg.interaction_type),
                            )
            row["interaction_type"] = interaction_type

            basis_bundle = self.basis_builder.build(features, None)

            assembler = NormalEquationAssembler(
                parallel_jobs=1,
                interaction_type=interaction_type,
            )

            fit_result = assembler.fit(
                features=features,
                basis_bundle=basis_bundle,
                solver=self.solver,
                system_name=cfg.system_name,
            )

            learned_model = self._build_learned_model(
                fit_result=fit_result,
                N=features.X.shape[2],
                D=features.X.shape[3],
            )

            learned_model_4N = self._build_learned_model(
                fit_result=fit_result,
                N=int(test_4N.metadata["N"]),
                D=features.X.shape[3],
                N4_kernel_scale= self.config.N4_kernel_scale,
            )

            diagnostics = fit_result.diagnostics

            row.update(
                {
                    "condition_number": diagnostics.condition_number,
                    "rank": diagnostics.rank,
                    "min_singular": diagnostics.min_singular,
                    "max_singular": diagnostics.max_singular,
                    "solver_method": diagnostics.method,
                    "ridge_lambda": diagnostics.ridge_lambda,
                    "svd_tol": diagnostics.svd_tol,
                    "radial_n_basis": cfg.radial_n_basis,
                    "environment_n_basis_per_dim": cfg.environment_n_basis_per_dim,
                    "n_environment": fit_result.n_environment,
                    "n_radial": fit_result.n_radial,
                }
            )

            # --------------------------------------------------
            # Residual and function errors on fit window
            # -------------------------------------------------

            true_kernel = true_kernel_from_metadata(train_fit.metadata)
            true_env = true_env_from_metadata(train_fit.metadata)

            (
                kernel_grid,
                kernel_weights,
                z1_grid,
                z2_grid,
                env_weights,
            ) = self._build_case_empirical_weighted_grids(
                features=features,
                fit_result=fit_result,
            )

            # Evaluate learned functions on common grids for all trials
            if cfg.save_function_values:
                if self.global_function_grids is not None:
                    function_kernel_grid, function_z1_grid, function_z2_grid = (
                        self.global_function_grids
                    )
                else:
                    # unnecessary
                    function_kernel_grid = kernel_grid
                    function_z1_grid = z1_grid
                    function_z2_grid = z2_grid

                function_values = self._evaluate_learned_functions_on_grids(
                    learned_model=learned_model,
                    kernel_grid=function_kernel_grid,
                    z1_grid=function_z1_grid,
                    z2_grid=function_z2_grid,
                )

                function_values_path = self._save_trial_function_values(
                    trial=trial,
                    function_values=function_values,
                )

                row["function_values_path"] = str(function_values_path)


            report = self.eval_runner.run(
                features=features,
                learned_model=learned_model,
                reconstruction_artifact=None,
                prediction_artifact=None,
                prediction_true_dataset=None,
                true_kernel=true_kernel,
                true_env=true_env,
                kernel_grid=kernel_grid,
                z1_grid=z1_grid,
                z2_grid=z2_grid,
                kernel_weights=kernel_weights,
                env_weights=env_weights,
                split_index=None,
            )

            row.update(
                {   "E_res": report.residual_metrics.get("E_res", np.nan),
                    "E_res_relative": report.residual_metrics.get("E_res_relative", np.nan),
                    "E_phi": report.function_metrics.get("E_phi", np.nan),
                    "E_phi_relative": report.function_metrics.get("E_phi_relative", np.nan),
                    "E_env": report.function_metrics.get("E_env", np.nan),
                    "E_env_relative": report.function_metrics.get("E_env_relative", np.nan),
                }
            )

            # --------------------------------------------------
            # Rollouts
            # --------------------------------------------------
            fit_recon = self._rollout(
                learned_model=learned_model,
                dataset=train_fit,
                scenario="train_fit_reconstruction",
                trial=trial,
                rollout_name=f"trial_{trial:03d}_train_fit_reconstruction",
            )

            train_pred = self._rollout(
                learned_model=learned_model,
                dataset=train_full,
                scenario="train_full_prediction",
                trial=trial,
                rollout_name=f"trial_{trial:03d}_train_full_prediction",
            )

            test_pred = self._rollout(
                learned_model=learned_model,
                dataset=test,
                scenario="test_prediction",
                trial=trial,
                rollout_name=f"trial_{trial:03d}_test_prediction",
            )
            test_4N_pred = self._rollout(
                learned_model=learned_model_4N,
                dataset=test_4N,
                scenario="test_4N_prediction",
                trial=trial,
                rollout_name=f"trial_{trial:03d}_test_4N",
            )

            # --------------------------------------------------
            # Trajectory errors
            # --------------------------------------------------
            fit_metrics = self.traj_eval.evaluate_batch(
                t_true=train_fit.t,
                X_true_batch=train_fit.X,
                t_hat=fit_recon.t,
                X_hat_batch=fit_recon.X_hat,
                split_index=None,
            )

            train_split_idx = self._split_index(train_full.t, cfg.train_horizon_end)
            test_split_idx = self._split_index(test.t, cfg.train_horizon_end)
            test_4N_split_idx = self._split_index(test_4N.t, cfg.train_horizon_end)

            train_metrics = self.traj_eval.evaluate_batch(
                t_true=train_full.t,
                X_true_batch=train_full.X,
                t_hat=train_pred.t,
                X_hat_batch=train_pred.X_hat,
                split_index=train_split_idx,
            )

            test_metrics = self.traj_eval.evaluate_batch(
                t_true=test.t,
                X_true_batch=test.X,
                t_hat=test_pred.t,
                X_hat_batch=test_pred.X_hat,
                split_index=test_split_idx,
            )

            test_4N_metrics = self.traj_eval.evaluate_batch(
                t_true=test_4N.t,
                X_true_batch=test_4N.X,
                t_hat=test_4N_pred.t,
                X_hat_batch=test_4N_pred.X_hat,
                split_index=test_4N_split_idx,
            )

            row.update(
                {
                    # Fit-window reconstruction
                    "train_fit_mean_traj_0T": fit_metrics[
                        "reconstruction_Linf_mean"
                    ],
                    "train_fit_std_traj_0T": fit_metrics[
                        "reconstruction_Linf_std"
                    ],
                    # Seen train ICs, full horizon
                    "train_seen_mean_traj_0T": train_metrics[
                        "reconstruction_Linf_mean"
                    ],
                    "train_seen_std_traj_0T": train_metrics[
                        "reconstruction_Linf_std"
                    ],
                    "train_seen_mean_traj_TTf": train_metrics[
                        "prediction_Linf_mean"
                    ],
                    "train_seen_std_traj_TTf": train_metrics[
                        "prediction_Linf_std"
                    ],
                    # Unseen N-agent test ICs
                    "test_unseen_mean_traj_0T": test_metrics[
                        "reconstruction_Linf_mean"
                    ],
                    "test_unseen_std_traj_0T": test_metrics[
                        "reconstruction_Linf_std"
                    ],
                    "test_unseen_mean_traj_TTf": test_metrics[
                        "prediction_Linf_mean"
                    ],
                    "test_unseen_std_traj_TTf": test_metrics[
                        "prediction_Linf_std"
                    ],
                    # 4N test ICs
                    "test_4N_mean_traj_0T": test_4N_metrics[
                        "reconstruction_Linf_mean"
                    ],
                    "test_4N_std_traj_0T": test_4N_metrics[
                        "reconstruction_Linf_std"
                    ],
                    "test_4N_mean_traj_TTf": test_4N_metrics[
                        "prediction_Linf_mean"
                    ],
                    "test_4N_std_traj_TTf": test_4N_metrics[
                        "prediction_Linf_std"
                    ],
                }
            )

        except Exception as exc:
            row["failed"] = True
            row["failure_message"] = str(exc)
            row["traceback"] = traceback.format_exc()
            print(f"Trial {trial:03d} failed: {exc}")

        return row
    
    def _build_learned_model(self, fit_result, N: int, D: int,  N4_kernel_scale: float =1):
        return self.model_factory.build_from_fit_result(
                fit_result=fit_result,
                N=N,
                D=D,
                N4_kernel_scale =N4_kernel_scale,
            )

    def _rollout(
        self,
        learned_model,
        dataset,
        scenario: str,
        trial: int | None = None,
        rollout_name: str | None = None,
    ):
        rollout = LearnedSystemRollout(learned_model)

        artifact = rollout.simulate_many(
            U0=dataset.U0,
            t_eval=dataset.t,
            method=self.config.rollout_method,
            rtol=self.config.rollout_rtol,
            atol=self.config.rollout_atol,
            show_progress=False,
            metadata={
                "artifact_type": "rollout",
                "stage": "step_6B",
                "scenario": scenario,
                "system_name": self.config.system_name,
                "profile_name": self.config.profile_name,
                "interaction_type": learned_model.interaction_type,
                "trial": trial,
                "case_name": f"trial_{trial:03d}" if trial is not None else None,
                "system_order": getattr(learned_model, "system_order", None),
                "environment_variable": (
                    "position" if getattr(learned_model, "system_order", None) == 1 else "velocity"
                ),
            },
            )

        if self.config.save_rollouts:
            model_name = (
                self.config.rollout_model_name
                if self.config.rollout_model_name is not None
                else f"{self.config.system_name}_{self.config.interaction_type}"
            )

            if rollout_name is None:
                if trial is None:
                    rollout_name = scenario
                else:
                    rollout_name = f"trial_{trial:03d}_{scenario}"

            case_name = f"trial_{trial:03d}" if trial is not None else "unknown_trial"

            self.rollout_repo.save_case(
                artifact=artifact,
                system_name=self.config.system_name,
                profile_name=self.config.profile_name,
                case_name=case_name,
                rollout_name=rollout_name,
                overwrite=True,
            )

        return artifact

    @staticmethod
    def _split_index(t: np.ndarray, T_fit: float) -> int:
        return int(np.searchsorted(t, T_fit, side="right") - 1)

    def _build_case_empirical_weighted_grids(self, features, fit_result):
        """Build empirical error weights from the current training features."""
        from ips_learning.statistics.density import build_empirical_error_density

        density = build_empirical_error_density(
            features=features,
            localization_metadata=fit_result.basis_bundle.localization_metadata,
            kernel_grid_size=self.config.kernel_grid_size,
            environment_grid_size=self.config.environment_grid_size,
        )

        return (
            density.r_grid,
            density.kernel_weights,
            density.z1_grid,
            density.z2_grid,
            density.environment_weights,
        )
    def _evaluate_learned_functions_on_grids(
        self,
        *,
        learned_model,
        kernel_grid: np.ndarray,
        z1_grid: np.ndarray,
        z2_grid: np.ndarray | None,
    ) -> dict:
        """
        Evaluate learned interaction kernel and environment force on fixed grids.

        Returns
        -------
        dict with:
            kernel_grid : (K,)
            phi_values  : (K,)
            z1_grid     : (G1,)
            z2_grid     : None or (G2,)
            f_values    : first-order 1D: (G1, D)
                        second-order 2D: (G1, G2, D)
        """
        kernel_grid = np.asarray(kernel_grid, dtype=float).reshape(-1)
        r_eval = kernel_grid.reshape(-1, 1)

        phi_values = np.asarray(
            learned_model.learned_kernel(r_eval),
            dtype=float,
        ).reshape(-1)

        z1_grid = np.asarray(z1_grid, dtype=float).reshape(-1)

        if z2_grid is None:
            Z_grid = z1_grid.reshape(-1, 1)

            f_values = np.asarray(
                learned_model.learned_environment(Z_grid),
                dtype=float,
            )

            return {
                "kernel_grid": kernel_grid,
                "phi_values": phi_values,
                "z1_grid": z1_grid,
                "z2_grid": None,
                "f_values": f_values,
            }

        z2_grid = np.asarray(z2_grid, dtype=float).reshape(-1)
        Z1, Z2 = np.meshgrid(z1_grid, z2_grid, indexing="ij")
        Z_grid = np.column_stack([Z1.ravel(), Z2.ravel()])

        f_flat = np.asarray(
            learned_model.learned_environment(Z_grid),
            dtype=float,
        )

        if f_flat.ndim == 1:
            f_flat = f_flat.reshape(-1, 1)

        f_values = f_flat.reshape(
            len(z1_grid),
            len(z2_grid),
            f_flat.shape[-1],
        )

        return {
            "kernel_grid": kernel_grid,
            "phi_values": phi_values,
            "z1_grid": z1_grid,
            "z2_grid": z2_grid,
            "f_values": f_values,
        }
    
    def _save_trial_function_values(
        self,
        *,
        trial: int,
        function_values: dict,
    ) -> Path:
        """
        Save learned phi and f values for one trial.
        """
        trial_path = self.function_value_dir / f"trial_{trial:03d}_function_values.npz"

        save_kwargs = {
            "trial": np.array(trial, dtype=int),
            "kernel_grid": function_values["kernel_grid"],
            "phi_values": function_values["phi_values"],
            "z1_grid": function_values["z1_grid"],
            "f_values": function_values["f_values"],
        }

        if function_values["z2_grid"] is not None:
            save_kwargs["z2_grid"] = function_values["z2_grid"]

        np.savez(trial_path, **save_kwargs)

        return trial_path
    

    def _build_global_function_grids(self):
        cfg = self.config

        r_min_all, r_max_all = np.inf, -np.inf
        z_min_all, z_max_all = None, None

        for q in range(cfg.num_trials):
            trial = q + 1
            train_fit_name = cfg.train_fit_name_template.format(trial=trial)

            train_fit = self.dataset_repo.load(
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split=cfg.train_fit_split,
                dataset_name=train_fit_name,
            )

            features = self.feature_builder.build(train_fit)

            R = np.asarray(features.R, dtype=float).reshape(-1)
            R = R[np.isfinite(R)]
            R = R[R > 0]

            r_min_all = min(r_min_all, float(np.min(R)))
            r_max_all = max(r_max_all, float(np.max(R)))

            system_order = features.metadata.get("system_order")
            Z = features.X if system_order == 1 else features.V

            Z_flat = np.asarray(Z, dtype=float).reshape(-1, Z.shape[-1])
            z_min = np.min(Z_flat, axis=0)
            z_max = np.max(Z_flat, axis=0)

            if z_min_all is None:
                z_min_all = z_min
                z_max_all = z_max
            else:
                z_min_all = np.minimum(z_min_all, z_min)
                z_max_all = np.maximum(z_max_all, z_max)

        kernel_grid = np.linspace(
            r_min_all,
            r_max_all,
            cfg.kernel_grid_size,
        )

        z1_grid = np.linspace(
            z_min_all[0],
            z_max_all[0],
            cfg.environment_grid_size,
        )

        if z_min_all.shape[0] == 1:
            z2_grid = None
        else:
            z2_grid = np.linspace(
                z_min_all[1],
                z_max_all[1],
                cfg.environment_grid_size,
            )

        return kernel_grid, z1_grid, z2_grid
    
    def summarize_function_values(self) -> dict[str, np.ndarray | Path]:
        """
        Load saved trial function values and compute pointwise mean/std.

        Saves:
            function_value_summary.npz

        Returns
        -------
        dict containing grids, mean, std, and output path.
        """
        paths = sorted(
            self.function_value_dir.glob("trial_*_function_values.npz")
        )

        if len(paths) == 0:
            raise FileNotFoundError(
                f"No function value files found in {self.function_value_dir}"
            )

        phi_list = []
        f_list = []

        kernel_grid_ref = None
        z1_grid_ref = None
        z2_grid_ref = None

        for path in paths:
            data = np.load(path)

            kernel_grid = data["kernel_grid"]
            z1_grid = data["z1_grid"]
            z2_grid = data["z2_grid"] if "z2_grid" in data.files else None

            if kernel_grid_ref is None:
                kernel_grid_ref = kernel_grid
                z1_grid_ref = z1_grid
                z2_grid_ref = z2_grid
            else:
                if not np.allclose(kernel_grid_ref, kernel_grid):
                    raise ValueError(f"kernel_grid mismatch in {path}")
                if not np.allclose(z1_grid_ref, z1_grid):
                    raise ValueError(f"z1_grid mismatch in {path}")

                if z2_grid_ref is None and z2_grid is not None:
                    raise ValueError(f"z2_grid mismatch in {path}")
                if z2_grid_ref is not None:
                    if z2_grid is None or not np.allclose(z2_grid_ref, z2_grid):
                        raise ValueError(f"z2_grid mismatch in {path}")

            phi_list.append(data["phi_values"])
            f_list.append(data["f_values"])

        phi_stack = np.stack(phi_list, axis=0)
        f_stack = np.stack(f_list, axis=0)

        phi_mean = np.mean(phi_stack, axis=0)
        phi_std = np.std(phi_stack, axis=0, ddof=1) if phi_stack.shape[0] > 1 else np.zeros_like(phi_mean)

        f_mean = np.mean(f_stack, axis=0)
        f_std = np.std(f_stack, axis=0, ddof=1) if f_stack.shape[0] > 1 else np.zeros_like(f_mean)

        output_path = self.function_value_dir / "function_value_summary.npz"

        save_kwargs = {
            "kernel_grid": kernel_grid_ref,
            "phi_mean": phi_mean,
            "phi_std": phi_std,
            "z1_grid": z1_grid_ref,
            "f_mean": f_mean,
            "f_std": f_std,
            "num_trials": np.array(phi_stack.shape[0], dtype=int),
        }

        if z2_grid_ref is not None:
            save_kwargs["z2_grid"] = z2_grid_ref

        np.savez(output_path, **save_kwargs)

        return {
            "kernel_grid": kernel_grid_ref,
            "phi_mean": phi_mean,
            "phi_std": phi_std,
            "z1_grid": z1_grid_ref,
            "z2_grid": z2_grid_ref,
            "f_mean": f_mean,
            "f_std": f_std,
            "num_trials": phi_stack.shape[0],
            "output_path": output_path,
        }
        