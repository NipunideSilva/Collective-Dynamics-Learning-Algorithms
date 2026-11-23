from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class ScalabilityTrialSummaryBuilder:
    """
    Summaries for general repeated-trial analysis.

    Expected input:
        output DataFrame from ScalabilityTrialRunner.run_all_trials().

    """
    def build_all_summaries(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {
            "fit_diagnostic_summary": self.summarize_fit_diagnostics(df),
            "trajectory_summary": self.summarize_trajectory_scenarios(df),
            "failure_summary": self.summarize_failures(df),
        }


    def summarize_fit_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "E_res",
            "E_res_relative",
            "E_phi",
            "E_phi_relative",
            "E_env",
            "E_env_relative",
            "condition_number",
            "rank",
            "min_singular",
            "max_singular",
            "ridge_lambda",
            "radial_n_basis",
            "environment_n_basis_per_dim",
            "n_radial",
            "n_environment",
        ]

        rows = []

        valid_df = self._valid_rows(df)
        context = self._context_from_df(valid_df)

        for col in columns:
            if col not in valid_df.columns:
                continue

            values = pd.to_numeric(valid_df[col], errors="coerce").dropna()

            rows.append(
                self._summarize_column(
                    values,
                    metric=col,
                    context=context,
                )
            )

        return pd.DataFrame(rows)

    def summarize_trajectory_scenarios(self, df: pd.DataFrame) -> pd.DataFrame:
        scenario_specs = [
            {
                "scenario": "train_fit [t0,T]",
                "role": "fit-window reconstruction",
                "mean_col": "train_fit_mean_traj_0T",
                "std_col": "train_fit_std_traj_0T",
            },
            {
                "scenario": "train full [t0,T]",
                "role": "seen-IC pre-prediction interval",
                "mean_col": "train_seen_mean_traj_0T",
                "std_col": "train_seen_std_traj_0T",
            },
            {
                "scenario": "train full [T,Tf]",
                "role": "seen-IC prediction",
                "mean_col": "train_seen_mean_traj_TTf",
                "std_col": "train_seen_std_traj_TTf",
            },
            {
                "scenario": "test N [t0,T]",
                "role": "unseen-IC pre-prediction interval",
                "mean_col": "test_unseen_mean_traj_0T",
                "std_col": "test_unseen_std_traj_0T",
            },
            {
                "scenario": "test N [T,Tf]",
                "role": "unseen-IC prediction",
                "mean_col": "test_unseen_mean_traj_TTf",
                "std_col": "test_unseen_std_traj_TTf",
            },
            {
                "scenario": "test 4N [t0,T]",
                "role": "4N transfer pre-prediction interval",
                "mean_col": "test_4N_mean_traj_0T",
                "std_col": "test_4N_std_traj_0T",
            },
            {
                "scenario": "test 4N [T,Tf]",
                "role": "4N transfer prediction",
                "mean_col": "test_4N_mean_traj_TTf",
                "std_col": "test_4N_std_traj_TTf",
            },
        ]

        valid_df = self._valid_rows(df)
        rows = []

        for spec in scenario_specs:
            if spec["mean_col"] not in valid_df.columns:
                continue
            if spec["std_col"] not in valid_df.columns:
                continue

            mean_values = pd.to_numeric(
                valid_df[spec["mean_col"]],
                errors="coerce",
            ).dropna()

            std_values = pd.to_numeric(
                valid_df[spec["std_col"]],
                errors="coerce",
            ).dropna()

            mean_of_mean = self._mean(mean_values)
            trial_std_of_mean = self._std(mean_values)

            mean_of_ic_std = self._mean(std_values)
            trial_std_of_ic_std = self._std(std_values)

            rows.append(
                {
                    "scenario": spec["scenario"],
                    "role": spec["role"],

                    "mean_traj_mean": mean_of_mean,
                    "mean_traj_trial_std": trial_std_of_mean,
                    "mean_traj_pm_trial_std": self._pm(
                        mean_of_mean,
                        trial_std_of_mean,
                    ),

                    "std_traj_mean": mean_of_ic_std,
                    "std_traj_trial_std": trial_std_of_ic_std,
                    "std_traj_pm_trial_std": self._pm(
                        mean_of_ic_std,
                        trial_std_of_ic_std,
                    ),

                    "num_trials_mean": int(mean_values.shape[0]),
                    "num_trials_std": int(std_values.shape[0]),
                }
            )

        return pd.DataFrame(rows)

    def summarize_failures(self, df: pd.DataFrame) -> pd.DataFrame:
        if "failed" not in df.columns:
            return pd.DataFrame(
                columns=["num_trials", "num_failed", "failure_rate"]
            )

        num_trials = int(df.shape[0])
        failed = df[df["failed"].astype(bool)].copy()
        num_failed = int(failed.shape[0])

        rows = [
            {
                "num_trials": num_trials,
                "num_failed": num_failed,
                "failure_rate": (
                    float(num_failed / num_trials) if num_trials > 0 else np.nan
                ),
            }
        ]

        if num_failed > 0 and "failure_message" in failed.columns:
            for msg, group in failed.groupby("failure_message"):
                rows.append(
                    {
                        "num_trials": num_trials,
                        "num_failed": int(group.shape[0]),
                        "failure_rate": float(group.shape[0] / num_trials),
                        "failure_message": msg,
                    }
                )

        return pd.DataFrame(rows)

    def save_fit_diagnostic_summary(
        self,
        summary_df: pd.DataFrame,
        output_dir: str | Path,
        file_name: str = "fit_diagnostic_summary.csv",
    ) -> Path:
        return self._save(summary_df, output_dir, file_name)

    def save_trajectory_summary(
        self,
        summary_df: pd.DataFrame,
        output_dir: str | Path,
        file_name: str = "trajectory_summary.csv",
    ) -> Path:
        return self._save(summary_df, output_dir, file_name)

    def save_failure_summary(
        self,
        summary_df: pd.DataFrame,
        output_dir: str | Path,
        file_name: str = "failure_summary.csv",
    ) -> Path:
        return self._save(summary_df, output_dir, file_name)

    @staticmethod
    def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
        if "failed" not in df.columns:
            return df.copy()

        return df[~df["failed"].astype(bool)].copy()

    @staticmethod
    def _summarize_column(values: pd.Series, metric: str,context: dict | None = None,) -> dict:
        mean_val = ScalabilityTrialSummaryBuilder._mean(values)
        std_val = ScalabilityTrialSummaryBuilder._std(values)

        row = {
            "metric": metric,
            "mean": mean_val,
            "trial_std": std_val,
            "mean_pm_trial_std": ScalabilityTrialSummaryBuilder._pm(
                mean_val,
                std_val,
            ),
            "num_trials": int(values.shape[0]),
        }

        if context is not None:
            row.update(context)

        return row
    
    @staticmethod
    def _context_from_df(df: pd.DataFrame) -> dict:
        context = {}

        for col in [
            "system_name",
            "profile_name",
            "system_order",
            "environment_variable",
            "interaction_type",
        ]:
            if col in df.columns:
                vals = df[col].dropna().unique()
                if len(vals) == 1:
                    context[col] = vals[0]

        return context

    @staticmethod
    def _mean(values: pd.Series) -> float:
        if values.empty:
            return np.nan
        return float(values.mean())

    @staticmethod
    def _std(values: pd.Series) -> float:
        if values.empty:
            return np.nan
        if values.shape[0] == 1:
            return 0.0
        return float(values.std(ddof=1))

    @staticmethod
    def _pm(mean_val: float, std_val: float) -> str:
        if not np.isfinite(mean_val) or not np.isfinite(std_val):
            return "nan ± nan"
        return f"{mean_val:.4e} ± {std_val:.4e}"

    @staticmethod
    def _save(
        df: pd.DataFrame,
        output_dir: str | Path,
        file_name: str,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / file_name
        df.to_csv(path, index=False)
        return path