from ips_learning.analysis.scalability_trial_runner_parametric import (
    ParametricScalabilityTrialConfig,
    ParametricScalabilityTrialRunner,
)

from ips_learning.analysis.scalability_trial_summary_parametric import ParametricScalabilityTrialSummaryBuilder

from ips_learning.data.dataset_repository import DatasetRepository
from ips_learning.experiments.generate_dataset_collection import (
    run_one_dataset_from_profile,
)
import json
from pathlib import Path


class ParametricScalabilityEvaluationRunner:
    def __init__(self,
                 SYSTEM_NAME ,
                    PROFILE_NAME ,
                    INTERACTION_TYPE ,

                    FIT_END_TIME ,
                    NUM_TRIALS,
                    ARTIFACTS,
                    config:ParametricScalabilityTrialConfig,
                    ):
       
       self.config = config
        
       self.outputs = self._trial_call(SYSTEM_NAME ,
                    PROFILE_NAME ,
                    INTERACTION_TYPE ,

                    FIT_END_TIME ,
                    NUM_TRIALS,
                    ARTIFACTS = Path(ARTIFACTS))
        


    def _trial_call(self,
                    SYSTEM_NAME ,
                    PROFILE_NAME ,
                    INTERACTION_TYPE ,

                    FIT_END_TIME ,
                    NUM_TRIALS,
                    ARTIFACTS):
        
        self._generate_or_load_4N_testdata(
                    SYSTEM_NAME,
                    PROFILE_NAME ,
                    ARTIFACTS)
        
        

        # Run repeated trials
        runner = ParametricScalabilityTrialRunner(self.config)

        trial_df = runner.run_all_trials()

        print(trial_df)

        # Save repeated-trial metric artifact
        algorithm = getattr(self.config, "algorithm", "nonparametric_f")

        eval_dir = (
            ARTIFACTS
            / SYSTEM_NAME
            / PROFILE_NAME
            / "evaluations"
            / "repeated_trials"
            / algorithm
        )
        eval_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = eval_dir / "repeated_trial_metrics.csv"

        trial_df.to_csv(metrics_path, index=False)

        print("Saved repeated-trial metrics:")
        print(metrics_path)


        summary_builder = ParametricScalabilityTrialSummaryBuilder()

        # all_summary = summary_builder.build_all_summaries(trial_df)
        summaries = summary_builder.build_all_summaries(trial_df)

        fit_diagnostic_summary = summaries["fit_diagnostic_summary"]
        trajectory_summary = summaries["trajectory_summary"]
        failure_summary = summaries["failure_summary"]
        theta_f_summary = summaries["parametric_theta_f_summary"]

        print("Fit diagnostic summary:")
        print(fit_diagnostic_summary)

        print("Trajectory summary:")
        print(trajectory_summary)

        print("Failure summary:")
        print(failure_summary)

        # Save summaries

        fit_summary_path = eval_dir / "fit_diagnostic_summary.csv"
        traj_summary_path = eval_dir / "trajectory_summary.csv"
        failure_summary_path = eval_dir / "failure_summary.csv"
        theta_f_summary_path = eval_dir / "parametric_theta_f_summary.csv"


        fit_diagnostic_summary.to_csv(fit_summary_path, index=False)
        trajectory_summary.to_csv(traj_summary_path, index=False)
        failure_summary.to_csv(failure_summary_path, index=False)
        theta_f_summary.to_csv(theta_f_summary_path, index=False)

        print("Saved:")
        print(fit_summary_path)
        print(traj_summary_path)
        print(failure_summary_path)


        # save function value summaries .npz file
        function_summary = runner.summarize_function_values()

        print("Saved function summary to:")
        print(function_summary["output_path"])

        print("phi_mean shape:", function_summary["phi_mean"].shape)
        print("phi_std shape:", function_summary["phi_std"].shape)
        print("f_mean shape:", function_summary["f_mean"].shape)
        print("f_std shape:", function_summary["f_std"].shape)

        system_order = getattr(self.config, "system_order", None)

        if system_order not in (1, 2):
            raise ValueError(f"config.system_order must be 1 or 2, got {system_order}.")

        environment_variable = "position" if system_order == 1 else "velocity"

        # Save config metadata
        config_metadata = {
            "artifact_type": "repeated_trials_evaluation",
            "algorithm": algorithm,
            "parametric_theta_f_summary": str(theta_f_summary_path),
            "system_name": SYSTEM_NAME,
            "profile_name": PROFILE_NAME,
            "interaction_type": INTERACTION_TYPE,
            "num_trials": NUM_TRIALS,
            "train_horizon_end": FIT_END_TIME,
            "stage": "step_6B",
            "system_order": system_order,
            "environment_variable": environment_variable,

            "splits": {
                "train_fit_split": self.config.train_fit_split,
                "train_full_split": self.config.train_full_split,
                "test_split": self.config.test_split,
                "test_4N_split": self.config.test_4N_split,
            },

            "dataset_names": {
                "train_fit_name_template": self.config.train_fit_name_template,
                "train_full_name_template": self.config.train_full_name_template,
                "test_dataset_name": self.config.test_dataset_name,
                "test_4N_dataset_name": self.config.test_4N_dataset_name,
            },

            "basis": {
                "radial_basis_family": self.config.radial_basis_family,
                "radial_degree": self.config.radial_degree,
                "radial_n_basis": self.config.radial_n_basis,
                "environment_basis_family": self.config.environment_basis_family,
                "environment_degree": self.config.environment_degree,
                "environment_n_basis_per_dim": self.config.environment_n_basis_per_dim,
            },

            "solver": {
                "solver_method": self.config.solver_method,
                "ridge_lambda": self.config.ridge_lambda,
                "svd_tol": self.config.svd_tol,
            },

            "rollout": {
                "rollout_method": self.config.rollout_method,
                "rollout_rtol": self.config.rollout_rtol,
                "rollout_atol": self.config.rollout_atol,
                "save_rollouts": self.config.save_rollouts,
            },

            "outputs": {
                "repeated_trial_metrics": str(metrics_path),
                "fit_diagnostic_summary": str(fit_summary_path),
                "trajectory_summary": str(traj_summary_path),
                "failure_summary": str(failure_summary_path),
                "parametric_theta_f_summary": str(theta_f_summary_path),
            },
        }

        config_path = eval_dir / "repeated_trial_config.json"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_metadata, f, indent=2, sort_keys=True)

        print("Saved config metadata:")
        print(config_path)
        
        # Quick validation

        expected_files = [
            metrics_path,
            fit_summary_path,
            traj_summary_path,
            failure_summary_path,
            theta_f_summary_path,
            config_path,
        ]

        for path in expected_files:
            print("FOUND" if path.exists() else "MISSING", path)

        assert all(path.exists() for path in expected_files)

        print("Stage 6B repeated-trial artifacts saved successfully.")

        return {
            "trial_df": trial_df,
            "fit_diagnostic_summary": fit_diagnostic_summary,
            "trajectory_summary": trajectory_summary,
            "failure_summary": failure_summary,
            "eval_dir": eval_dir,
            "parametric_theta_f_summary": theta_f_summary,
        }


    def _generate_or_load_4N_testdata(self,
                    SYSTEM_NAME,
                    PROFILE_NAME ,

                    ARTIFACTS):
        
        # generating or loading 4N testdata sets
        
        profile_obj = self.build_dataset_profile(SYSTEM_NAME, PROFILE_NAME)
        cfg = profile_obj.common_config

        N_train = cfg["N"]

        dataset_repo = DatasetRepository(root_dir=ARTIFACTS / "datasets")

        try:
            dataset_repo.load(
                system_name=SYSTEM_NAME,
                profile_name=PROFILE_NAME,
                split="test_4N",
                dataset_name="dataset_test_4N",
            )
            already_exists = True

        except FileNotFoundError:
            already_exists = False

        test_4N_dir = (
            ARTIFACTS / "datasets" / SYSTEM_NAME / PROFILE_NAME / "test_4N" / "dataset_test_4N"
        )

        if already_exists:
            print("Existing 4N transfer dataset found:")
            print(test_4N_dir)
            return
        print("Generating 4N transfer dataset...")

        if not hasattr(profile_obj, "test"):
            raise RuntimeError("Profile missing test split specification.")

        test_spec = profile_obj.test

        run_one_dataset_from_profile(
            profile=profile_obj,
            split="test_4N",
            dataset_name="dataset_test_4N",
            M=test_spec.M,
            L=test_spec.L,
            T=test_spec.T,
            seed=4404,
            N_override=4 * N_train,
            split_config={
                "split": "test_4N",
                "dataset_name": "dataset_test_4N",
                "M": test_spec.M,
                "L": test_spec.L,
                "T": test_spec.T,
                "seed": 4404,
                "N": 4 * N_train,
            },
        )

        print("Saved 4N transfer dataset:")
        print(test_4N_dir)



    @staticmethod
    def build_dataset_profile(system_name: str, profile_name: str):
        if system_name == "levi":
            from ips_learning.experiments.generate_levi_data import build_levi_dataset_profile
            return build_levi_dataset_profile(profile_name)

        if system_name == "spp":
            from ips_learning.experiments.generate_spp_data import build_spp_dataset_profile
            return build_spp_dataset_profile(profile_name)

        if system_name == "kuramoto":
            from ips_learning.experiments.generate_kuramoto_data import build_kuramoto_dataset_profile
            return build_kuramoto_dataset_profile(profile_name)

        raise ValueError(f"Unknown system_name={system_name!r}")