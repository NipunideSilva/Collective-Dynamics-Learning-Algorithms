from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ModelSelectionRepeatedTrialSummaryBuilder:
    """
    Summaries for repeated-trial model-selection results.

    Expected input:
        output DataFrame from ModelSelectionRepeatedTrialRunner.run_all_trials().
    """

    def build_all_summaries(
        self,
        df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        return {
            # Proposed method
            "selection_frequency": (
                self.summarize_selection_frequency(df)
            ),

            # Information-criterion baselines
            "aic_selection_frequency": (
                self.summarize_aic_selection_frequency(df)
            ),
            "bic_selection_frequency": (
                self.summarize_bic_selection_frequency(df)
            ),

            # Proposed-method mechanism summaries
            "order_frequency": (
                self.summarize_order_frequency(df)
            ),
            "interaction_frequency": (
                self.summarize_interaction_frequency(df)
            ),
            "environment_frequency": (
                self.summarize_environment_frequency(df)
            ),

            # Metric summaries
            "selected_metric_summary": (
                self.summarize_selected_metrics(df)
            ),
            "candidate_metric_summary": (
                self.summarize_candidate_metrics(df)
            ),

            # Method agreement
            "method_agreement_summary": (
                self.summarize_method_agreement(df)
            ),

            "failure_summary": (
                self.summarize_failures(df)
            ),
        }
    
    def summarize_order_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._summarize_frequency(
            df,
            column="selected_system_order",
            output_name="selected_system_order",
        )


    def summarize_interaction_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._summarize_frequency(
            df,
            column="selected_interaction_type",
            output_name="selected_interaction_type",
        )


    def summarize_environment_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._summarize_frequency(
            df,
            column="selected_include_environment",
            output_name="selected_include_environment",
        )

    def summarize_selection_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_df = self._valid_rows(df)

        if "selected_candidate" not in valid_df.columns:
            return pd.DataFrame(
                columns=[
                    "selected_candidate",
                    "selected_count",
                    "selected_fraction",
                    "num_valid_trials",
                ]
            )

        counts = (
            valid_df.groupby("selected_candidate")
            .size()
            .reset_index(name="selected_count")
        )

        num_valid_trials = int(valid_df.shape[0])

        if num_valid_trials == 0:
            counts["selected_fraction"] = np.nan
        else:
            counts["selected_fraction"] = counts["selected_count"] / float(num_valid_trials)

        counts["num_valid_trials"] = num_valid_trials

        return counts.sort_values(
            ["selected_count", "selected_candidate"],
            ascending=[False, True],
        )

    
    def summarize_aic_selection_frequency(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        return self._summarize_frequency(
            df,
            column="aic_selected_candidate",
            output_name="aic_selected_candidate",
        )


    def summarize_bic_selection_frequency(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        return self._summarize_frequency(
            df,
            column="bic_selected_candidate",
            output_name="bic_selected_candidate",
        )

    def summarize_method_agreement(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        valid_df = self._valid_rows(df)

        agreement_columns = [
            "proposed_aic_agree",
            "proposed_bic_agree",
            "aic_bic_agree",
        ]

        rows: list[dict[str, Any]] = []

        context = self._context_from_df(valid_df)

        for column in agreement_columns:
            if column not in valid_df.columns:
                continue

            values = valid_df[column].dropna().astype(bool)

            num_trials = int(values.shape[0])
            agreement_count = int(values.sum())

            row = {
                "comparison": column,
                "agreement_count": agreement_count,
                "agreement_fraction": (
                    agreement_count / float(num_trials)
                    if num_trials > 0
                    else np.nan
                ),
                "num_valid_trials": num_trials,
            }

            row.update(context)
            rows.append(row)

        return pd.DataFrame(rows)

    def summarize_selected_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_df = self._valid_rows(df)

        metric_cols = [
            "selected_E_traj_full",
            "selected_E_traj_full_std",

            "selected_E_traj_reconstruction",
            "selected_E_traj_reconstruction_rel",

            "selected_E_traj_prediction",

            "selected_E_res",
            "selected_E_res_rel",

            "selected_condition_number",

            "selected_complexity",

            "selected_phi_E_norm",
            "selected_phi_A_norm",
            "selected_f_norm",

            "selected_trajectory_reconstruction_rss",
            "selected_trajectory_reconstruction_n_obs",
            "selected_n_model_coefficients",
            "selected_n_likelihood_parameters",
            "selected_AIC",
            "selected_BIC",
        ]
        rows = []

        context = self._context_from_df(valid_df)

        for col in metric_cols:
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

    def summarize_candidate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_df = self._valid_rows(df)

        candidate_names = self._candidate_names_from_columns(valid_df)
        rows: list[dict[str, Any]] = []
        metrics = [
            "primary_value",

            "E_traj_full",
            "E_traj_full_std",

            "E_traj_reconstruction",
            "E_traj_reconstruction_std",
            "E_traj_reconstruction_rel",

            "E_traj_prediction",
            "E_traj_prediction_std",

            "E_res",
            "E_res_rel",

            "condition_number",
            "complexity",

            "phi_E_norm",
            "phi_A_norm",
            "f_norm",

            # AIC/BIC
            "trajectory_reconstruction_rss",
            "trajectory_reconstruction_n_obs",
            "n_model_coefficients",
            "n_likelihood_parameters",
            "AIC",
            "BIC",
            "aic_rank",
            "bic_rank",
        ]

        context = self._context_from_df(valid_df)

        for candidate in candidate_names:
            item: dict[str, Any] = {
                "candidate": candidate,
                "num_trials": int(valid_df.shape[0]),
            }
            item.update(context)

            for metric in metrics:
                col = f"{candidate}__{metric}"

                if col not in valid_df.columns:
                    continue

                values = pd.to_numeric(valid_df[col], errors="coerce").dropna()

                item[f"{metric}_mean"] = self._mean(values)
                item[f"{metric}_trial_std"] = self._std(values)
                item[f"{metric}_mean_pm_trial_std"] = self._pm(
                    item[f"{metric}_mean"],
                    item[f"{metric}_trial_std"],
                )
                item[f"{metric}_num_trials"] = int(values.shape[0])

            rows.append(item)

        return pd.DataFrame(rows).sort_values("candidate")

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

    def save_all_summaries(
        self,
        summaries: dict[str, pd.DataFrame],
        output_dir: str | Path,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}

        for name, summary_df in summaries.items():
            csv_path = output_dir / f"{name}.csv"
            tex_path = output_dir / f"{name}.tex"

            summary_df.to_csv(csv_path, index=False)
            summary_df.to_latex(tex_path, index=False)

            paths[f"{name}_csv"] = csv_path
            paths[f"{name}_tex"] = tex_path

        return paths


    
    def _summarize_frequency(
        self,
        df: pd.DataFrame,
        *,
        column: str,
        output_name: str,
    ) -> pd.DataFrame:
        valid_df = self._valid_rows(df)

        if column not in valid_df.columns:
            return pd.DataFrame(
                columns=[
                    output_name,
                    "selected_count",
                    "selected_fraction",
                    "num_valid_trials",
                ]
            )

        counts = (
            valid_df.groupby(column)
            .size()
            .reset_index(name="selected_count")
            .rename(columns={column: output_name})
        )

        num_valid_trials = int(valid_df.shape[0])

        if num_valid_trials == 0:
            counts["selected_fraction"] = np.nan
        else:
            counts["selected_fraction"] = (
                counts["selected_count"] / float(num_valid_trials)
            )

        counts["num_valid_trials"] = num_valid_trials

        return counts.sort_values(
            ["selected_count", output_name],
            ascending=[False, True],
        )
    
    @staticmethod
    def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
        if "failed" not in df.columns:
            return df.copy()

        return df[~df["failed"].astype(bool)].copy()

    @staticmethod
    def _candidate_names_from_columns(df: pd.DataFrame) -> list[str]:
        candidates = set()

        for col in df.columns:
            if "__primary_value" in col:
                candidates.add(col.split("__primary_value")[0])

        return sorted(candidates)

    @staticmethod
    def _context_from_df(df: pd.DataFrame) -> dict[str, Any]:
        context: dict[str, Any] = {}

        for col in [
            "system_name",
            "profile_name",
            "model_selection_mode",
        ]:
            if col in df.columns:
                vals = df[col].dropna().unique()
                if len(vals) == 1:
                    context[col] = vals[0]

        return context

    @staticmethod
    def _summarize_column(
        values: pd.Series,
        *,
        metric: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mean_val = ModelSelectionRepeatedTrialSummaryBuilder._mean(values)
        std_val = ModelSelectionRepeatedTrialSummaryBuilder._std(values)

        row = {
            "metric": metric,
            "mean": mean_val,
            "trial_std": std_val,
            "mean_pm_trial_std": ModelSelectionRepeatedTrialSummaryBuilder._pm(
                mean_val,
                std_val,
            ),
            "num_trials": int(values.shape[0]),
        }

        if context is not None:
            row.update(context)

        return row

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