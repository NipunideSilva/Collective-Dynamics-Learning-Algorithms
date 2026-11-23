from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

from ips_learning.data.dataset_repository import DatasetRepository, RawTrajectoryDataset
from ips_learning.features.feature_builder import FeatureBuilder
from ips_learning.features.feature_repository import FeatureRepository
from ips_learning.bases.specs import (
    BasisConfig,
    RadialBasisSpec,
    EnvironmentBasisSpec,
)
from ips_learning.bases.builders import BasisBuilder
from ips_learning.bases.repository import BasisRepository
from ips_learning.fitting.solver import LinearSystemSolver
from ips_learning.fitting.assembler import NormalEquationAssembler
from ips_learning.fitting.system_repository import NormalSystemRepository
from ips_learning.models.factories import LearnedModelFactory
from ips_learning.evaluation.run_evaluation import EvaluationRunner
from ips_learning.statistics.density_repository import DensityRepository

from ips_learning.experiments.generate_dataset_collection import (
    run_one_dataset_from_profile,
)

from ips_learning.experiments.generate_levi_data import build_levi_dataset_profile
from ips_learning.experiments.generate_spp_data import build_spp_dataset_profile
from ips_learning.experiments.generate_kuramoto_data import build_kuramoto_dataset_profile

from ips_learning.systems.registry import (
    true_kernel_from_metadata,
    true_env_from_metadata,
)

from ips_learning.experiments.profile_registry import (
    get_profile_common_config,
)

@dataclass(frozen=True)
class MLSweepConfig:
    system_name: str = "spp"
    profile_name: str = "baseline"
    interaction_type: str = "energy"
    artifact_root: str | Path = "artifacts"

    N: int = 10
    D: int = 2
    t0: float = 0.0

    M_values: tuple[int, ...] = (1, 10, 20, 40, 80, 160)
    L_values: tuple[int, ...] = (10, 20, 40, 80, 160) #number of fit-window time samples on [t0, T_fit]

    fit_horizon_end: float = 0.5
    prediction_horizon_end: float = 2.0

    base_seed: int = 909
    ic_generator_name: str | None = None# "clumps"

    # Number of independent training datasets generated for every (M, L) case.
    n_trials: int = 10

    # Separates the random-number streams of consecutive learning trials.
    # Trial q will use base_seed + q * trial_seed_stride.
    trial_seed_stride: int = 100_000_000


    ic_params: dict | None = None
    system_params: dict | None = None

    test_split: str = "test"
    test_dataset_name: str = "dataset_test"

    kde_split: str = "kde"
    kde_dataset_name: str = "dataset_kde"

    radial_degree: int = 1
    radial_n_basis: int = 10
    environment_degree: int = 1
    environment_n_basis_per_dim: int = 10
    environment_grid_size: int = 120
    system_order: int | None = None

    solver_method: str = "lstsq"
    ridge_lambda: float = 1e-8
    svd_tol: float = 1e-12

    rollout_method: str = "Radau"
    rollout_rtol: float = 1e-6
    rollout_atol: float = 1e-6

    kernel_grid_size: int = 300

    overwrite: bool = False #Reuse existing datasets, True - override the dataset




class MLSweepRunner:
    """
    Direct-generation M/L sweep runner for second-order systems.

    For each (M, L), this runner:
    1. Generates or loads a full-horizon dataset with exactly M trajectories
       and L time samples.
    2. Uses the prefix [t0, T_fit] in memory for fitting.
    3. Fits the interaction kernel theta_phi and environment force theta_f.
    4. Rolls out the learned system on:
       - the fitting window,
       - the same ICs over the full horizon,
       - a fixed unseen test dataset.
    5. Computes residual, function, trajectory, and conditioning metrics.
    6. Saves reusable evaluation artifacts under:
       artifacts/{system_name}/evaluations/ML_sweep/.
    """

    def __init__(self, config: MLSweepConfig) -> None:

        """
        Initialize repositories, basis configuration, solver, and evaluators.

        All repositories use explicit roots from config.artifact_root so the runner
        does not depend on the notebook's current working directory.
        """
        self.config = config

        if config.n_trials < 1:
            raise ValueError(
                f"n_trials must be at least 1, got {config.n_trials}."
            )

        if config.trial_seed_stride < 1:
            raise ValueError(
                "trial_seed_stride must be a positive integer, "
                f"got {config.trial_seed_stride}."
            )

        if config.interaction_type not in {"energy", "alignment"}:
            raise ValueError(
                "interaction_type must be 'energy' or 'alignment', "
                f"got {config.interaction_type!r}."
            )
        root = Path(config.artifact_root)

        self.dataset_repo = DatasetRepository(root_dir=root / "datasets")
        self.feature_repo = FeatureRepository(root_dir=root / "features")
        self.basis_repo = BasisRepository(root_dir=root / "bases")
        self.normal_repo = NormalSystemRepository(root_dir=root / "fits")
        self.density_repo = DensityRepository(root=root / "densities")


        self.feature_builder = FeatureBuilder()

        self.basis_config = BasisConfig(
            radial=RadialBasisSpec(
                family="bspline",
                degree=config.radial_degree,
                n_basis=config.radial_n_basis,
            ),
            environment=EnvironmentBasisSpec(
                family="bspline",
                degree=config.environment_degree,
                n_basis_per_dim=config.environment_n_basis_per_dim,
            ),)

        self.basis_builder = BasisBuilder(config=self.basis_config)


        self.solver = LinearSystemSolver(
            method=config.solver_method,
            ridge_lambda=config.ridge_lambda,
            svd_tol=config.svd_tol,
        )

        self.model_factory = LearnedModelFactory()
        self.eval_runner = EvaluationRunner()

        if config.system_order == 1 and config.interaction_type == "alignment":
            raise ValueError("First-order systems currently support only energy interaction.")

    def trial_name(self, trial_index: int) -> str:
        """
        Return the directory-safe name of one independent learning trial.

        Examples
        --------
        trial_index=0 -> "trial_000"
        trial_index=9 -> "trial_009"
        """
        if not 0 <= trial_index < self.config.n_trials:
            raise ValueError(
                f"trial_index must be in [0, {self.config.n_trials - 1}], "
                f"got {trial_index}."
            )

        return f"trial_{trial_index:03d}"


    def trial_base_seed(self, trial_index: int) -> int:
        """
        Return the independent base seed assigned to one learning trial.
        """
        return (
            int(self.config.base_seed)
            + int(trial_index) * int(self.config.trial_seed_stride)
        )

    def dataset_seed(
        self,
        trial_index: int,
        M_target: int,
        L: int,
    ) -> int:
        """
        Return a deterministic but trial-specific seed for one cumulative
        dataset residue.

        The same trial, M_target, and L always produce the same seed.
        Different trials produce independent training datasets.
        """
        return (
            self.trial_base_seed(trial_index)
            + 100_000 * int(M_target)
            + int(L)
        )

    def case_name(self, trial_index: int, M: int, L: int) -> str:
        """
        Return a unique fit/artifact name for one trial and one (M, L) case.

        Example
        -------
        trial_003_M_020_L_051
        """
        return (
            f"{self.trial_name(trial_index)}"
            f"_M_{M:03d}_L_{L:03d}"
        )


    def dataset_name(self, trial_index: int, M: int, L: int) -> str:
        """
        Return a unique dataset name for one trial and one (M, L) case.
        """
        return (
            f"dataset_{self.trial_name(trial_index)}"
            f"_M_{M:03d}_L_{L:03d}"
        )

    def ensure_case_dataset(
        self,
        trial_index: int,
        M: int,
        L: int,
    ) -> None:
        """
    Generate or load one cumulative M/L training dataset for one
    independent learning trial.
    """

        cfg = self.config
        split = "ML_sweep"

        trial_name = self.trial_name(trial_index)
        dataset_name = self.dataset_name(
            trial_index=trial_index,
            M=M,
            L=L,
        )

        try:
            self.dataset_repo.load(
                system_name=cfg.system_name,
                profile_name=cfg.profile_name,
                split=split,
                dataset_name=dataset_name,
            )
            if not cfg.overwrite:
                return
        except FileNotFoundError:
            pass

        profile = self._build_dataset_profile(cfg.system_name, cfg.profile_name)

        M_values_sorted = sorted(cfg.M_values)

        if M not in M_values_sorted:
            raise ValueError(f"M={M} not in config.M_values.")

        # Build cumulative pieces up to current M
        X_parts = []
        V_parts = []
        LHS_parts = []
        U0_parts = []

        prev_M = 0
        L_full = self._full_horizon_L_from_fit_L(L)

        for M_target in M_values_sorted:
            if M_target > M:
                break

            M_residue = M_target - prev_M
            if M_residue <= 0:
                prev_M = M_target
                continue

            residue_name = (
                f"residue_{trial_name}"
                f"_M_{prev_M:03d}_to_{M_target:03d}"
                f"_L_{L:03d}"
            )
            residue_seed = self.dataset_seed(
                trial_index=trial_index,
                M_target=M_target,
                L=L,
            )

            try:
                residue = self.dataset_repo.load(
                    system_name=cfg.system_name,
                    profile_name=cfg.profile_name,
                    split=split,
                    dataset_name=residue_name,
                )
            except FileNotFoundError:
                run_one_dataset_from_profile(
                    profile=profile,
                    split=split,
                    dataset_name=residue_name,
                    M=M_residue,
                    L=L_full,
                    T=cfg.prediction_horizon_end,
                    seed=residue_seed,
                    N_override=cfg.N,
                    split_config={
                        "split": split,
                        "dataset_name": residue_name,
                        "M": M_residue,
                        "L_fit": L,
                        "L_full": L_full,
                        "T_fit": cfg.fit_horizon_end,
                        "T": cfg.prediction_horizon_end,
                        "seed": int(residue_seed),
                    },
                )

                residue = self.dataset_repo.load(
                    system_name=cfg.system_name,
                    profile_name=cfg.profile_name,
                    split=split,
                    dataset_name=residue_name,
                )

            X_parts.append(residue.X)
            LHS_parts.append(residue.LHS)
            U0_parts.append(residue.U0)

            if residue.V is not None:
                V_parts.append(residue.V)

            prev_M = M_target

        X = np.concatenate(X_parts, axis=0)
        LHS = np.concatenate(LHS_parts, axis=0)
        U0 = np.concatenate(U0_parts, axis=0)

        V = None
        if len(V_parts) > 0:
            V = np.concatenate(V_parts, axis=0)

        if X.shape[0] != M:
            raise ValueError(f"Expected cumulative M={M}, got {X.shape[0]}.")

        metadata = {
            **residue.metadata,
            "dataset_name": dataset_name,
            "split": split,
            "trial_index": int(trial_index),
            "trial_name": trial_name,
            "trial_base_seed": int(self.trial_base_seed(trial_index)),
            "M": int(M),
            "L_fit": int(L),
            "L_full": int(L_full),
            "fit_horizon_end": float(cfg.fit_horizon_end),
            "prediction_horizon_end": float(cfg.prediction_horizon_end),
            "dataset_role": "ML_sweep_cumulative_train_full",
            "M_values_cumulative": [m for m in M_values_sorted if m <= M],
        }

        dataset = RawTrajectoryDataset(
            t=residue.t,
            X=X,
            V=V,
            LHS=LHS,
            U0=U0,
            metadata=metadata,
        )

        self.dataset_repo.save(
            dataset=dataset,
            system_name=cfg.system_name,
            profile_name=cfg.profile_name,
            split=split,
            dataset_name=dataset_name,
            overwrite=True,
        )


    def run(self) -> pd.DataFrame:
        """
        Run the complete repeated-trial M/L sweep.

        For each independent trial and each (M, L) pair:

        1. Generate or load a trial-specific training dataset.
        2. Fit the model.
        3. Evaluate the learned model.
        4. Save trial-specific artifacts.
        5. Collect one metrics row.

        Returns
        -------
        pd.DataFrame
            Long-form table containing one row for every
            (trial_index, M, L) combination.
        """
        cfg = self.config
        rows: list[dict] = []

        total_cases = (
            cfg.n_trials
            * len(cfg.M_values)
            * len(cfg.L_values)
        )

        completed_cases = 0

        for trial_index in range(cfg.n_trials):
            trial_name = self.trial_name(trial_index)

            print("=" * 72)
            print(
                f"Starting {trial_name}: "
                f"{trial_index + 1}/{cfg.n_trials}"
            )
            print(
                "Trial base seed:",
                self.trial_base_seed(trial_index),
            )
            print("=" * 72)

            for M in cfg.M_values:
                for L in cfg.L_values:
                    completed_cases += 1

                    print(
                        f"[{completed_cases}/{total_cases}] "
                        f"{trial_name}, M={M}, L={L}"
                    )

                    if (
                        self.case_outputs_exist(
                            trial_index=trial_index,
                            M=M,
                            L=L,
                        )
                        and not cfg.overwrite
                    ):
                        print(
                            "Loading existing trial-specific "
                            f"case: {trial_name}, M={M}, L={L}"
                        )

                        row = self.load_case_metrics(
                            trial_index=trial_index,
                            M=M,
                            L=L,
                        )

                    else:
                        print(
                            "Generating/loading trial-specific "
                            f"dataset: {trial_name}, M={M}, L={L}"
                        )

                        self.ensure_case_dataset(
                            trial_index=trial_index,
                            M=M,
                            L=L,
                        )

                        print(
                            "Running trial-specific case: "
                            f"{trial_name}, M={M}, L={L}"
                        )

                        try:
                            row = self.run_one_case(
                                trial_index=trial_index,
                                M=M,
                                L=L,
                            )

                        except Exception as exc:
                            print(
                                "Case failed:",
                                trial_name,
                                f"M={M}",
                                f"L={L}",
                                type(exc).__name__,
                                str(exc),
                            )

                            row = {
                                "trial_index": int(trial_index),
                                "trial_name": trial_name,
                                "trial_seed": int(
                                    self.trial_base_seed(
                                        trial_index
                                    )
                                ),
                                "M": int(M),
                                "L": int(L),
                                "case_name": self.case_name(
                                    trial_index=trial_index,
                                    M=M,
                                    L=L,
                                ),
                                "system_name": cfg.system_name,
                                "profile_name": cfg.profile_name,
                                "interaction_type": (
                                    cfg.interaction_type
                                ),
                                "rollout_failed": True,
                                "rollout_error_message": (
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            }

                            self._save_case_metrics(
                                row=row,
                                trial_index=trial_index,
                                M=M,
                                L=L,
                            )

                    rows.append(row)

            print(f"Completed {trial_name}")

        trial_df = pd.DataFrame(rows)

        if trial_df.empty:
            raise RuntimeError(
                "The repeated-trial sweep produced no rows."
            )

        sort_columns = [
            column
            for column in [
                "trial_index",
                "M",
                "L",
            ]
            if column in trial_df.columns
        ]

        if sort_columns:
            trial_df = trial_df.sort_values(
                sort_columns
            ).reset_index(drop=True)

        output_dir = (
            Path(cfg.artifact_root)
            / cfg.system_name
            / "evaluations"
            / "ML_sweep"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        trial_metrics_path = (
            output_dir
            / "ML_sweep_trial_metrics.csv"
        )

        trial_df.to_csv(
            trial_metrics_path,
            index=False,
        )

        summary_df = self.aggregate_trial_metrics(
            trial_df
        )

        summary_path = self.save_aggregate_metrics(
            summary_df
        )

        config_path = (
            output_dir
            / "ML_sweep_config.json"
        )

        with open(
            config_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                asdict(cfg),
                file,
                indent=2,
                default=self._json_default,
            )

        print("=" * 72)
        print("Repeated-trial sweep completed.")
        print("Number of rows:", len(trial_df))
        print("Number of summary rows:", len(summary_df))
        print("Trial metrics:", trial_metrics_path)
        print("Mean/std summary:", summary_path)
        print("Configuration:", config_path)
        print("=" * 72)

        return trial_df


    #aggregate  metrics for trials in each M/L case
    def aggregate_trial_metrics(
        self,
        trial_df: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Aggregate repeated-trial metrics by (M, L).

        For every numerical metric, save:

            metric_mean
            metric_std
            metric_count

        The standard deviation is the sample standard deviation
        across successful finite trial values, using ddof=1.

        Parameters
        ----------
        trial_df
            Long-form table with one row per
            (trial_index, M, L) combination.

        Returns
        -------
        pd.DataFrame
            One row per (M, L) pair containing mean, standard
            deviation, valid-value count, and trial-failure counts.
        """
        if trial_df.empty:
            raise ValueError(
                "Cannot aggregate an empty trial metrics table."
            )

        required_columns = {
            "trial_index",
            "M",
            "L",
        }

        missing_columns = (
            required_columns - set(trial_df.columns)
        )

        if missing_columns:
            raise ValueError(
                "Trial metrics table is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        df = trial_df.copy()

        # Ensure the failure flag is available and Boolean.
        if "rollout_failed" not in df.columns:
            df["rollout_failed"] = False

        df["rollout_failed"] = (
            df["rollout_failed"]
            .fillna(False)
            .astype(bool)
        )

        group_columns = ["M", "L"]

        # Columns identifying an experiment rather than numerical metrics.
        excluded_columns = {
            "trial_index",
            "trial_name",
            "trial_seed",
            "M",
            "L",
            "L_fit",
            "L_full",
            "case_name",
            "system_name",
            "profile_name",
            "interaction_type",
            "system_order",
            "environment_variable",
            "rollout_failed",
            "rollout_error_message",
        }

        metric_columns: list[str] = []

        for column in df.columns:
            if column in excluded_columns:
                continue

            converted = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            # Include the column only when at least one numeric
            # value exists.
            if converted.notna().any():
                df[column] = converted
                metric_columns.append(column)

        summary_rows: list[dict] = []

        grouped = df.groupby(
            group_columns,
            sort=True,
            dropna=False,
        )

        for (M, L), group in grouped:
            row: dict = {
                "M": int(M),
                "L": int(L),
                "n_trials_requested": int(
                    self.config.n_trials
                ),
                "n_trials_recorded": int(len(group)),
                "n_successful_trials": int(
                    (~group["rollout_failed"]).sum()
                ),
                "n_failed_trials": int(
                    group["rollout_failed"].sum()
                ),
            }

            for metric in metric_columns:
                values = pd.to_numeric(
                    group[metric],
                    errors="coerce",
                )

                # Exclude inf and -inf before computing statistics.
                values = values.replace(
                    [np.inf, -np.inf],
                    np.nan,
                ).dropna()

                count = int(values.size)

                row[f"{metric}_count"] = count

                if count == 0:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_std"] = np.nan

                elif count == 1:
                    row[f"{metric}_mean"] = float(
                        values.iloc[0]
                    )
                    row[f"{metric}_std"] = np.nan

                else:
                    row[f"{metric}_mean"] = float(
                        values.mean()
                    )
                    row[f"{metric}_std"] = float(
                        values.std(ddof=1)
                    )

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        summary_df = summary_df.sort_values(
            ["M", "L"]
        ).reset_index(drop=True)

        return summary_df

    def save_aggregate_metrics(
        self,
        summary_df: pd.DataFrame,
        ) -> Path:
        """
        Save the repeated-trial mean/std summary table.

        Returns
        -------
        Path
            Path to the saved CSV file.
        """
        output_dir = (
            Path(self.config.artifact_root)
            / self.config.system_name
            / "evaluations"
            / "ML_sweep"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / "ML_sweep_metrics_mean_std.csv"
        )

        summary_df.to_csv(
            output_path,
            index=False,
        )

        return output_path



    def case_artifact_dir(
        self,
        trial_index: int,
        M: int,
        L: int,
    ) -> Path:
        """
        Return the evaluation directory for one trial and one M/L case.

        Path
        ----
        artifacts/{system_name}/evaluations/ML_sweep/
            trials/trial_XXX/cases/{profile}/M_XXX_L_XXX/
        """
        trial_name = self.trial_name(trial_index)
        case_label = f"M_{M:03d}_L_{L:03d}"

        return (
            Path(self.config.artifact_root)
            / self.config.system_name
            / "evaluations"
            / "ML_sweep"
            / "trials"
            / trial_name
            / "cases"
            / self.config.profile_name
            / case_label
        )

    def metrics_path(
        self,
        trial_index: int,
        M: int,
        L: int,
    ) -> Path:
        """
        Return the metrics JSON path for one trial and one M/L case.
        """
        return (
            self.case_artifact_dir(
                trial_index=trial_index,
                M=M,
                L=L,
            )
            / "metrics.json"
        )

    def case_outputs_exist(
        self,
        trial_index: int,
        M: int,
        L: int,
    ) -> bool:
        """
        Check whether all reusable artifacts exist for one trial and
        one M/L case.
        """
        cfg = self.config

        case_name = self.case_name(
            trial_index=trial_index,
            M=M,
            L=L,
        )

        metrics_exists = self.metrics_path(
            trial_index=trial_index,
            M=M,
            L=L,
        ).exists()

        normal_exists = (
            Path(cfg.artifact_root)
            / "fits"
            / cfg.system_name
            / cfg.profile_name
            / case_name
            / "normal_system.npz"
        ).exists()

        feature_exists = (
            Path(cfg.artifact_root)
            / "features"
            / cfg.system_name
            /cfg.profile_name
            / "ML_sweep"
            / case_name
            / "features.npz"
        ).exists()

        basis_exists = (
            Path(cfg.artifact_root)
            / "bases"
            / cfg.system_name
            / cfg.profile_name
            / "ML_sweep"
            / case_name
            / "basis_metadata.json"
        ).exists()

        return (
            metrics_exists
            and normal_exists
            and feature_exists
            and basis_exists
        )

    def load_case_metrics(
        self,
        trial_index: int,
        M: int,
        L: int,
    ) -> dict:
        """
        Load cached metrics for one trial and one M/L case.
        """
        path = self.metrics_path(
            trial_index=trial_index,
            M=M,
            L=L,
        )

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)






    def run_one_case(
        self,
        *,
        trial_index: int,
        M: int,
        L: int,
    ) -> dict:
        """
            Compute all metrics for one M/L case.

            Steps:
            1. Load the direct full-horizon ML-sweep dataset.
            2. Create an in-memory fit-window dataset using [t0, T_fit].
            3. Build features and localized basis.
            4. Assemble and solve normal equations.
            5. Compute condition numbers.
            6. Roll out the learned model.
            7. Compute residual, function, and trajectory metrics.
            8. Save per-case metrics.
            """
        cfg = self.config

        trial_name = self.trial_name(trial_index)

        case_name = self.case_name(
            trial_index=trial_index,
            M=M,
            L=L,
        )

        dataset_name = self.dataset_name(
            trial_index=trial_index,
            M=M,
            L=L,
        )

        # Load the directly generated full-horizon case.

        case_train_full = self.dataset_repo.load(
            system_name=cfg.system_name,
            profile_name=cfg.profile_name,
            split="ML_sweep",
            dataset_name=dataset_name,
        )

        fit_end_idx = self._split_index(case_train_full.t, cfg.fit_horizon_end) + 1
        # Build an in-memory fit-window view; do not save a sliced cache artifact.
        case_train_fit = RawTrajectoryDataset(
            t=case_train_full.t[:fit_end_idx],
            X=case_train_full.X[:, :fit_end_idx],
            V=None if case_train_full.V is None else case_train_full.V[:, :fit_end_idx],
            LHS=case_train_full.LHS[:, :fit_end_idx],
            U0=case_train_full.U0,
            metadata={
                **case_train_full.metadata,
                "dataset_role": "ML_sweep_train_fit_view",
                "fit_horizon_end": float(cfg.fit_horizon_end),
                "fit_end_idx": int(fit_end_idx),
            },
        )
        # Build features on the fit-window data.
        features = self.feature_builder.build(case_train_fit)

        # Infer interaction type from metadata when available.
        interaction_type = features.metadata.get(
            "interaction_type",
            case_train_fit.metadata.get("interaction_type", cfg.interaction_type),
        )
        assembler = NormalEquationAssembler(
            parallel_jobs=1,
            interaction_type=interaction_type,
        )

        self.feature_repo.save(
            features,
            system_name=cfg.system_name,
            profile_name=cfg.profile_name,
            split="ML_sweep",
            feature_name=case_name,
            overwrite=cfg.overwrite,
        )

        basis_bundle = self.basis_builder.build(features)

        self.basis_repo.save(
            basis_bundle=basis_bundle,
            system_name=cfg.system_name,
            profile_name=cfg.profile_name,
            split="ML_sweep",
            basis_name=case_name,
            overwrite=cfg.overwrite,
        )
        # Fit theta = [theta_phi, theta_f].
        fit_result = assembler.fit(
            features=features,
            basis_bundle=basis_bundle,
            solver=self.solver,
            system_name=cfg.system_name,
        )

        if fit_result.A is None or fit_result.b is None:
            raise RuntimeError(
                "fit_result.A and fit_result.b are required for ML Sweep."
            )

        diagnostics_dict = self._diagnostics_to_dict(fit_result.diagnostics)
        # Save normal-system and coefficient artifacts.
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
                "trial_index": int(trial_index),
                "trial_name": trial_name,
                "trial_base_seed": int(
                    self.trial_base_seed(trial_index)
                ),
                "case_name": case_name,
                "M": M,
                "L": L,
                "basis_metadata": basis_bundle.localization_metadata,
                "fit_metadata": fit_result.metadata,
                "system_name": cfg.system_name,
                "interaction_type": interaction_type,
            },
            overwrite=cfg.overwrite,
        )

        normal = self.normal_repo.load(
            system_name=cfg.system_name,
            profile_name=cfg.profile_name,
            fit_case_name=case_name,
        )
        # Compute conditioning diagnostics for the full and block matrices.
        cond_A = self._safe_condition_number(normal["A"])
        cond_A_phi = self._safe_condition_number(normal["A_phi"])
        cond_A_f = self._safe_condition_number(normal["A_f"])
        cond_A_phi_f = self._safe_condition_number(normal["A_phi_f"])
        cond_A_f_phi = self._safe_condition_number(normal["A_f_phi"])

        learned_model = self._build_learned_model(
            fit_result=fit_result,
            N=features.X.shape[2],
            D=features.X.shape[3],
        )


        try:
            # Compute weighted function errors using the KDE/reference density artifacts.
            true_kernel = true_kernel_from_metadata(case_train_fit.metadata)
            true_env = true_env_from_metadata(case_train_fit.metadata)

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

        except Exception as e:
            print("Evaluation failed after successful rollout:")
            print(type(e).__name__, e)
            raise

        system_order = case_train_fit.metadata.get("system_order", cfg.system_order)
        environment_variable = "position" if system_order == 1 else "velocity"
        # Save and return one flat row for this case.
        row = {
            "trial_index": int(trial_index),
            "trial_name": trial_name,
            "trial_base_seed": int(
                self.trial_base_seed(trial_index)
            ),
            "M": M,
            "L": L,
            "system_order": system_order,
            "environment_variable": environment_variable,
            "L_fit": L,
            "L_full": int(case_train_full.t.size),
            "case_name": case_name,
            "system_name": cfg.system_name,
            "profile_name": cfg.profile_name,
            "interaction_type": interaction_type,
            "rollout_failed": False,
            "rollout_error_message": "",

            "E_res": report.residual_metrics.get("E_res", np.nan),
            "E_res_relative": report.residual_metrics.get("E_res_relative", np.nan),
            "E_phi": report.function_metrics.get("E_phi", np.nan),
            "E_phi_relative": report.function_metrics.get("E_phi_relative", np.nan),
            "E_env": report.function_metrics.get("E_env", np.nan),
            "E_env_relative": report.function_metrics.get("E_env_relative", np.nan),
            "cond_A": cond_A,

            "rank": diagnostics_dict.get("rank", np.nan),
            "min_singular": diagnostics_dict.get("min_singular", np.nan),
            "max_singular": diagnostics_dict.get("max_singular", np.nan),
            "ridge_lambda": cfg.ridge_lambda,
        }

        self._save_case_metrics(
            row=row,
            trial_index=trial_index,
            M=M,
            L=L,
        )
        return row

    def _failed_row(
        self,
        *,
        trial_index: int,
        M: int,
        L: int,
        case_name: str,
        error_message: str,
        cond_A: float,
        cond_A_phi: float,
        cond_A_f: float,
        cond_A_phi_f: float,
        cond_A_f_phi: float,
        diagnostics_dict: dict,
    ) -> dict:
        """
        Build and save a metrics row for a failed rollout case.

        This preserves conditioning diagnostics even when trajectory rollout fails.
        """
        trial_name = self.trial_name(trial_index)
        row = {
            "trial_index": int(trial_index),
            "trial_name": trial_name,
            "trial_seed": int(
                self.trial_base_seed(trial_index)
            ),
            "M": M,
            "L": L,
            "case_name": case_name,
            "system_name": self.config.system_name,
            "profile_name": self.config.profile_name,
            "interaction_type": self.config.interaction_type,
            "rollout_failed": True,
            "rollout_error_message": error_message,
            "E_res": np.nan,
            "E_res_relative": np.nan,
            "E_phi": np.nan,
            "E_env": np.nan,
            "E_phi_relative": np.nan,
            "E_env_relative": np.nan,
            "system_order": self.config.system_order,
            "environment_variable": (
                "position" if self.config.system_order == 1 else "velocity"
            ),
            "cond_A": cond_A,
            "rank": diagnostics_dict.get("rank", np.nan),
            "min_singular": diagnostics_dict.get("min_singular", np.nan),
            "max_singular": diagnostics_dict.get("max_singular", np.nan),
            "ridge_lambda": self.config.ridge_lambda,
        }

        self._save_case_metrics(
            row=row,
            trial_index=trial_index,
            M=M,
            L=L,
        )

    def _save_case_metrics(
        self,
        *,
        row: dict,
        trial_index: int,
        M: int,
        L: int,
    ) -> None:
        """
        Save one trial-specific M/L metrics row as JSON.
        """
        metrics_path = self.metrics_path(
            trial_index=trial_index,
            M=M,
            L=L,
        )

        metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                row,
                f,
                indent=2,
                default=self._json_default,
            )


    def _build_learned_model(self, fit_result, N: int, D: int):
        """
            Reconstruct the learned second-order model from a FitResult.
            """
        return self.model_factory.build_from_fit_result(
            fit_result=fit_result,
            N=N,
            D=D,
        )

    def _profile_common_config(self) -> dict:
        """
    Load common system/profile defaults from experiments/profile_registry.py.
    """
        return get_profile_common_config(
            self.config.system_name,
            self.config.profile_name,
        )

    def _full_horizon_L_from_fit_L(self, L_fit: int) -> int:
        cfg = self.config

        if L_fit < 2:
            raise ValueError("L_fit must be at least 2.")

        dt = (cfg.fit_horizon_end - cfg.t0) / (L_fit - 1)

        if dt <= 0:
            raise ValueError("Invalid dt. Check t0, fit_horizon_end, and L_fit.")

        L_full = int(np.floor((cfg.prediction_horizon_end - cfg.t0) / dt)) + 1

        return L_full


    @staticmethod
    def _json_default(value):
        """
        Convert NumPy values to JSON-compatible Python values.
        """
        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, Path):
            return str(value)

        raise TypeError(
            f"Object of type {type(value).__name__} "
            "is not JSON serializable."
        )

    @staticmethod
    def _build_dataset_profile(system_name: str, profile_name: str):
        if system_name == "levi":
            return build_levi_dataset_profile(profile_name)

        if system_name == "spp":
            return build_spp_dataset_profile(profile_name)

        if system_name == "kuramoto":
            return build_kuramoto_dataset_profile(profile_name)

        raise ValueError(f"Unknown system_name={system_name!r}")

    def _get_dataset_generator(self):
        """
    Return the generic second-order dataset generator.
    """
        from ips_learning.experiments.generate_second_order_data import run_one_second_order_dataset
        return run_one_second_order_dataset

    def _default_ic_params(self) -> dict:
        """
        Return IC parameters.

        Priority:
        1. Explicit config.ic_params
        2. Dataset profile common_config["ic_params"]
        """
        if self.config.ic_params is not None:
            return dict(self.config.ic_params)

        common = self._profile_common_config()

        if "ic_params" not in common:
            raise ValueError(
                f"Profile {self.config.system_name}/{self.config.profile_name} "
                "does not define ic_params."
            )

        return dict(common["ic_params"])

    def _default_system_params(self) -> dict:
        """
            Return system parameters.

            Priority:
            1. Explicit config.system_params
            2. Dataset profile common_config["system_params"]
            """
        if self.config.system_params is not None:
            return dict(self.config.system_params)

        common = self._profile_common_config()

        if "system_params" not in common:
            raise ValueError(
                f"Profile {self.config.system_name}/{self.config.profile_name} "
                "does not define system_params."
            )

        return dict(common["system_params"])


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

    @staticmethod
    def _split_index(t: np.ndarray, T_fit: float) -> int:
        """
        Return the last index with t <= T_fit.
        """
        return int(np.searchsorted(t, T_fit, side="right") - 1)

    @staticmethod
    def _safe_condition_number(A: np.ndarray, tol: float = 1e-20) -> float:
        """
        Compute condition number safely.

        Returns inf for numerically singular matrices.
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
    def _diagnostics_to_dict(diag) -> dict:
        """
    Convert solver diagnostics dataclass/object to a JSON-serializable dict.
    """
        out = {}

        for key in [
            "method",
            "condition_number",
            "rank",
            "min_singular",
            "max_singular",
            "ridge_lambda",
            "svd_tol",
        ]:
            if hasattr(diag, key):
                val = getattr(diag, key)
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                out[key] = val

        if hasattr(diag, "singular_values"):
            sv = getattr(diag, "singular_values")
            out["singular_values"] = np.asarray(sv).tolist()

        return out
