from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection


from ips_learning.visualization.utils import (
    FigureRepository,
    TableRepository,
    
)
from ips_learning.visualization.style import apply_paper_style, OKABE_ITO


class NoiseFilterSweepVisualizer:
    """
        Stage 6E plotting/reporting from saved noise-filter-sweep CSV files.

        This visualizer compares preprocessing methods:
            raw, sliding_window, kalman_filter
        across each noise level.
    """
    PREPROCESSING_ORDER = (
        "raw",
        "sliding_window",
        "kalman_filter",
    )

    PREPROCESSING_COLORS = {
        "raw": OKABE_ITO[1],
        "sliding_window": OKABE_ITO[2],
        "kalman_filter": OKABE_ITO[3],
    }

    TRUE_COLOR = "black"

    def __init__(
        self,
        *,
        system_name: str,
        artifact_root: str | Path,
        stage_name: str = "step_6E",
    ) -> None:
        self.system_name = system_name
        self.artifact_root = Path(artifact_root)
        self.stage_name = stage_name

        self.fig_repo = FigureRepository(root_dir=self.artifact_root)
        self.table_repo = TableRepository(root_dir=self.artifact_root)
        apply_paper_style()

    @classmethod
    def _preprocessing_color(cls, preprocessing: str):
        return cls.PREPROCESSING_COLORS.get(
            preprocessing,
            OKABE_ITO[0],
        )

    @staticmethod
    def _labels(user_labels: dict | None, defaults: dict) -> dict:
        out = dict(defaults)
        if user_labels is not None:
            out.update(user_labels)
        return out
    

    def load_metrics(self, csv_path: str | Path) -> pd.DataFrame:
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing noise-sweep CSV: {csv_path}")

        df = pd.read_csv(csv_path)

        if "noise_level" not in df.columns:
            raise ValueError("CSV must contain a 'noise_level' column.")

        required = {"noise_level", "preprocessing"} 

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV must contain columns: {missing}")

        return (
            df
            .sort_values(["noise_level", "preprocessing"])
            .reset_index(drop=True)
        )

    def save_clean_table(
        self,
        df: pd.DataFrame,
        file_name: str = "noise_filter_sweep_clean_table",
    ) -> dict:
        preferred_cols = [
            "noise_level",
            "preprocessing",
            "case_name",
            "E_res",
            "E_phi",
            "E_env",
            "E_recon_train_fit",
            "E_pred_train_same_ic",
            "E_test_preT_unseen_ic",
            "E_test_pred_unseen_ic",
            "cond_A",
            "cond_A_phi",
            "cond_A_f",
            "cond_A_phi_f",
            "cond_A_f_phi",
            "rollout_failed",
            "failure_message",
        ]

        cols = [c for c in preferred_cols if c in df.columns]
        clean = df[cols].copy()

        return self.table_repo.save(
            clean,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=file_name,
            index=False,
            escape=False,
            float_format="%.4e",
            caption=f"{self.system_name.upper()} noise-filter robustness metrics.",
            label=f"tab:{self.system_name}_noise_filter_sweep",
        )

    @staticmethod
    def _preprocessing_label(name: str) -> str:
        labels = {
            "raw": "Noised data",
            "sliding_window": "Sliding window",
            "kalman_filter": "Kalman/RTS",
        }
        return labels.get(name, name)

    def plot_metric_vs_noise_by_preprocessing(
        self,
        df: pd.DataFrame,
        *,
        metric: str,
        ylabel: str | None = None,
        figure_name: str | None = None,
    ) -> Path:
        if metric not in df.columns:
            raise ValueError(f"Missing metric column: {metric}")

        if "preprocessing" not in df.columns:
            raise ValueError("df must contain a 'preprocessing' column.")

        if figure_name is None:
            figure_name = f"noise_filter_sweep_{metric}_by_preprocessing"

        fig, ax = plt.subplots(figsize=(7.5, 5.2))

        for preprocessing, group in df.groupby("preprocessing"):
            group = group.sort_values("noise_level")

            ax.plot(
                group["noise_level"],
                group[metric],
                marker="o",
                linewidth=2,
                label=self._preprocessing_label(preprocessing),
            )

        ax.set_xlabel(r"Noise level $\zeta$")
        ax.set_ylabel(ylabel or metric)

        values = np.asarray(df[metric], dtype=float)
        finite_values = values[np.isfinite(values)]

        if finite_values.size > 0 and np.nanmin(finite_values) > 0:
            ax.set_yscale("log")

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=True, loc="best")

        return self.fig_repo.save(
            fig,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=figure_name,
            close=False,
        )
    
    def plot_feature_errors_vs_noise(
        self,
        df: pd.DataFrame,
        figure_name: str = "noise_sweep_function_errors_vs_noise",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        labels = {
            "E_phi": r"$E_\phi$",
            "E_env": r"$E_f$",
        }
        for metric in ["E_phi", "E_env"]:
            if metric in df.columns:
                ax.plot(
                    df["noise_level"],
                    df[metric],
                    marker="o",
                    linewidth=2,
                    label=labels[metric],
                )

        ax.set_xlabel(r"Noise level $\zeta$")
        # ax.set_ylabel("Error")
        ax.set_yscale("log")
        # ax.set_title("Feature recovery under observation noise")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        return self.fig_repo.save(
            fig,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=figure_name,
            close=False,
        )

    def plot_conditioning_vs_noise(
        self,
        df: pd.DataFrame,
        figure_name: str = "noise_sweep_conditioning_vs_noise",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))

        metrics = [
            "condition_number",
            "cond_A",
            "cond_A_phi",
            "cond_A_f",
            "cond_A_phi_f",
            "cond_A_f_phi",
        ]

        for metric in metrics:
            if metric in df.columns:
                ax.plot(
                    df["noise_level"],
                    df[metric],
                    marker="o",
                    linewidth=2,
                    label=metric,
                )

        ax.set_xlabel(r"Noise level $\zeta$")
        ax.set_ylabel("Condition number")
        ax.set_yscale("log")
        ax.set_title("Conditioning under observation noise")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        return self.fig_repo.save(
            fig,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=figure_name,
            close=False,
        )
    
    def plot_phi_vs_r_filter_panel(
        self,
        r_grid,
        true_phi,
        learned_phi_by_preprocessing: dict[str, callable],
        noise_level: float,
        dist_density_ref=None,
        figure_name: str | None = None,
        labels: dict | None = None,
    ) -> Path:
        """
        for one ζ, overlay φ, φ̂_raw, φ̂_sliding, φ̂_kalman
        """
        noise_tag = str(noise_level).replace(".", "p")

        labels = self._labels(
        labels,
        {
            "phi_true": r"$\phi$",
            "phi_noise": r"$\zeta={noise}$",
            "xlabel": r"$r$",
            "ylabel_left": r"$\phi(r)$",
            "density": r"$\rho_R(r)$",
            "ylabel_right": r"$\rho_R(r)$",
        },
    )
        fig, ax1 = plt.subplots(figsize=(7.8, 5.2))

        r_grid = np.asarray(r_grid, dtype=float).reshape(-1)

        phi_true = np.asarray(true_phi(r_grid.reshape(-1, 1))).reshape(-1)

        ax1.plot(
            r_grid,
            phi_true,
            color=self.TRUE_COLOR,
            linewidth=2.5,
            label=labels["phi_true"],
        )
        ledgend_labels = {
        "raw": "",
        "sliding_window": "SW",
        "kalman_filter": "KF-RTS",
        }

        # for preprocessing, learned_phi in learned_phi_by_preprocessing.items():
        #     phi_hat = np.asarray(
        #         learned_phi(r_grid.reshape(-1, 1))
        #     ).reshape(-1)

        #     ax1.plot(
        #         r_grid,
        #         phi_hat,
        #         linestyle="--",
        #         linewidth=2,
        #         label=rf"$\hat{{\phi}}$ {ledgend_labels.get(preprocessing, preprocessing)}",
        #     )
        for preprocessing in self.PREPROCESSING_ORDER:
            if preprocessing not in learned_phi_by_preprocessing:
                continue

            learned_phi = learned_phi_by_preprocessing[preprocessing]

            phi_hat = np.asarray(
                learned_phi(r_grid.reshape(-1, 1))
            ).reshape(-1)

            ax1.plot(
                r_grid,
                phi_hat,
                color=self._preprocessing_color(preprocessing),
                linestyle="--",
                linewidth=2,
                label=(
                    rf"$\hat{{\phi}}$ "
                    f"{ledgend_labels.get(preprocessing, preprocessing)}"
                ),
            )

        ax1.set_xlabel(labels["xlabel"])
        ax1.set_ylabel(labels["ylabel_left"])
        # ax1.set_title(rf"$\zeta={noise_level}$")
        ax1.grid(True, linestyle="--", alpha=0.4)

        # --------------------------------------------------
        # KDE density from M=2000 reference distribution
        # --------------------------------------------------
        if dist_density_ref is not None:
            ax2 = ax1.twinx()

            # rho_r = dist_density_ref.evaluate(r_grid)
            rho_r = dist_density_ref
            rho_r = np.asarray(rho_r, dtype=float).reshape(-1)

            ax2.fill_between(
                r_grid,
                rho_r,
                alpha=0.08,
                color=OKABE_ITO[5],
                label=labels["density"],
            )

            ax2.set_ylabel(
                labels["ylabel_right"],
                color=OKABE_ITO[5],
                fontweight="bold",
            )

            ax2.tick_params(
                axis="y",
                colors=OKABE_ITO[5],
            )

            ax2.spines["right"].set_color(
                OKABE_ITO[5]
            )

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                frameon=True,
                loc="best",
            )
        else:
            ax1.legend(frameon=True, loc="best")

        fig.tight_layout()
        if figure_name is None:
            noise_tag = str(noise_level).replace(".", "p")
            figure_name = f"noise_filter_sweep_phi_zeta_{noise_tag}"

        return self.fig_repo.save(
            fig,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=figure_name,
            close=False,
        )
        
   
    def plot_env_curves_by_filter(
        self,
        z_grid,
        true_env,
        learned_env_by_preprocessing: dict[str, callable],
        noise_level: float,
        env_density_ref=None,
        figure_name: str | None = None,
        labels: dict | None = None,
    ) -> Path:
        """
        Plot true and learned 1D environment/drift functions under noise.

        Used for Kuramoto:
            z = x, f: R -> R
        """
        labels = self._labels(
            labels,
            {
                "f_true": r"$f$",
                "f_noise": r"$\zeta={noise}$",
                "xlabel": r"$z$",
                "ylabel_left": r"$f(z)$",
                "density": r"$\rho_Z(z)$",
                "ylabel_right": r"$\rho_Z(z)$",
            },
        )

        z_grid = np.asarray(z_grid, dtype=float).reshape(-1)
        Z_eval = z_grid.reshape(-1, 1)

        f_true = np.asarray(true_env(Z_eval), dtype=float)

        if f_true.ndim == 1:
            f_true = f_true.reshape(-1, 1)

        if f_true.shape != (z_grid.size, 1):
            raise ValueError(
                f"Expected true_env output shape {(z_grid.size, 1)}, got {f_true.shape}."
            )

        learned_values = {}

        for preprocessing, learned_env in learned_env_by_preprocessing.items():
            f_hat = np.asarray(learned_env(Z_eval), dtype=float)

            if f_hat.ndim == 1:
                f_hat = f_hat.reshape(-1, 1)

            if f_hat.shape != f_true.shape:
                raise ValueError(
                    f"Learned env for noise={noise_level} has shape {f_hat.shape}, "
                    f"expected {f_true.shape}."
                )

            learned_values[preprocessing] = f_hat

        fig, ax1 = plt.subplots(figsize=(7.8, 5.2))

        ax1.plot(
            z_grid,
            f_true[:, 0],
            color=self.TRUE_COLOR,
            linewidth=2.5,
            label=labels["f_true"],
        )
        ledgend_labels = {
        "raw": "",
        "sliding_window": "SW",
        "kalman_filter": "KF-RTS",
        }

        # for preprocessing, f_hat in learned_values.items():
        #     ax1.plot(
        #         z_grid,
        #         f_hat[:, 0],
        #         linestyle="--",
        #         linewidth=2,
        #         label=rf"$\hat{{f}}$ {ledgend_labels.get(preprocessing, preprocessing)}",
        #     )

        for preprocessing in self.PREPROCESSING_ORDER:
            if preprocessing not in learned_values:
                continue

            f_hat = learned_values[preprocessing]

            ax1.plot(
                z_grid,
                f_hat[:, 0],
                color=self._preprocessing_color(preprocessing),
                linestyle="--",
                linewidth=2,
                label=(
                    rf"$\hat{{f}}$ "
                    f"{ledgend_labels.get(preprocessing, preprocessing)}"
                ),
            )


        ax1.set_xlabel(labels["xlabel"])
        ax1.set_ylabel(labels["ylabel_left"])
        ax1.grid(True, linestyle="--", alpha=0.4)

        if env_density_ref is not None:
            rho_z = np.asarray(env_density_ref, dtype=float).reshape(-1)

            if rho_z.shape[0] != z_grid.shape[0]:
                raise ValueError(
                    f"env_density_ref and z_grid must have same length, got "
                    f"{rho_z.shape[0]} and {z_grid.shape[0]}."
                )

            ax2 = ax1.twinx()

            ax2.fill_between(
                z_grid,
                rho_z,
                alpha=0.20,
                color=OKABE_ITO[-1],
                label=labels["density"],
            )
            ax2.set_ylabel(
                labels["ylabel_right"],
                color=OKABE_ITO[-1],
                fontweight="bold",
            )

            ax2.tick_params(
                axis="y",
                colors=OKABE_ITO[-1],
            )

            ax2.spines["right"].set_color(
                OKABE_ITO[-1]
            )


            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()

            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                frameon=True,
                loc="best",
            )
        else:
            ax1.legend(frameon=True, loc="best")

        fig.tight_layout()
        # ax1.set_title(rf"$\zeta={noise_level}$")
        if figure_name is None:
            noise_tag = str(noise_level).replace(".", "p")
            figure_name = f"noise_filter_sweep_env_curve_zeta_{noise_tag}"

        return self.fig_repo.save(
            fig,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=figure_name,
            close=False,
        )
    


    
    def generate_all(self, csv_path: str | Path) -> dict[str, Path]:
        df = self.load_metrics(csv_path)

        outputs = {
            "clean_table": self.save_clean_table(df),
            "E_res": self.plot_metric_vs_noise_by_preprocessing(
                df,
                metric="E_res",
                ylabel=r"$E_{\mathrm{res}}$",
            ),
            "E_phi": self.plot_metric_vs_noise_by_preprocessing(
                df,
                metric="E_phi",
                ylabel=r"$E_{\phi}$",
            ),
            "E_env": self.plot_metric_vs_noise_by_preprocessing(
                df,
                metric="E_env",
                ylabel=r"$E_{f}$",
            ),
            "E_recon_train_fit": self.plot_metric_vs_noise_by_preprocessing(
                df,
                metric="E_recon_train_fit",
                ylabel=r"$E_{\mathrm{recon}}^{\mathrm{train}}$",
            ),
            "E_test_pred_unseen_ic": self.plot_metric_vs_noise_by_preprocessing(
                df,
                metric="E_test_pred_unseen_ic",
                ylabel=r"$E_{\mathrm{pred}}^{\mathrm{test}}$",
            ),
            "cond_A": self.plot_metric_vs_noise_by_preprocessing(
                df,
                metric="cond_A",
                ylabel=r"$\kappa(A)$",
            ),
        }

        manifest_path = (
            self.artifact_root
            / self.system_name
            / "tables"
            / self.stage_name
            / "noise_filter_sweep_visualization_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        def stringify_output(value):
            if isinstance(value, dict):
                return {k: str(v) for k, v in value.items()}
            return str(value)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({k: stringify_output(v) for k, v in outputs.items()}, f, indent=2)

        return outputs
    
    def plot_seen_ic_coordinate_all_preprocessing(
        self,
        *,
        t_full,
        X_true_full,
        learned_by_preprocessing: dict[str, np.ndarray],
        t_noisy,
        noisy_fit_by_preprocessing: dict[str, np.ndarray],
        noise_level: float,
        T: float = 0.5,
        component: int = 0,
        agent_index: int = 0,
        figure_name: str | None = None,
        labels: dict | None = None,
    ) -> Path:
        """
        Plot one fixed IC/agent/component and compare all preprocessing methods
        for one fixed noise level.

        Used for first-order systems such as Kuramoto.

        Expected shapes:
            X_true_full: (N, D, L_full)
            learned_by_preprocessing[name]: (N, D, L_full)
            noisy_fit_by_preprocessing[name]: (N, D, L_fit)
        """
        labels = self._labels(
            labels,
            {
                "sys_var": "x",
            },  
        )


        if figure_name is None:
            noise_tag = str(noise_level).replace(".", "p")
            figure_name = f"noise_filter_sweep_seen_ic_all_preprocessing_zeta_{noise_tag}"

        t_full = np.asarray(t_full, dtype=float).reshape(-1)
        t_noisy = np.asarray(t_noisy, dtype=float).reshape(-1)
        X_true_full = np.asarray(X_true_full, dtype=float)

        fig, ax = plt.subplots(figsize=(8.8, 5.2))

        # --------------------------------------------------
        # True clean trajectory
        # --------------------------------------------------
        ax.plot(
            t_full,
            X_true_full[agent_index, component, :],
            color=self.TRUE_COLOR,
            linewidth=2.5,
            label=rf"${labels['sys_var']}_{{i}}$",
            zorder=5,
        )

        # --------------------------------------------------
        # Learned trajectories: one line per preprocessing
        # --------------------------------------------------
        linestyles = {
            "raw": "--",
            "sliding_window": "-.",
            "kalman_filter": ":",
        }
        ledgend_labels = {
        "raw": "",
        "sliding_window": "SW",
        "kalman_filter": "KF-RTS",
        }

        # for preprocessing, X_hat_full in learned_by_preprocessing.items():
        #     X_hat_full = np.asarray(X_hat_full, dtype=float)

        #     ax.plot(
        #         t_full,
        #         X_hat_full[agent_index, component, :],
        #         linestyle=linestyles.get(preprocessing, "--"),
        #         linewidth=2.0,
        #         label = ( rf"$\hat{{{labels['sys_var']}}}_{{i}}$ "+ f" {ledgend_labels.get(preprocessing, preprocessing)}"),
        #     )
        for preprocessing in self.PREPROCESSING_ORDER:
            if preprocessing not in learned_by_preprocessing:
                continue

            X_hat_full = np.asarray(
                learned_by_preprocessing[preprocessing],
                dtype=float,
            )

            ax.plot(
                t_full,
                X_hat_full[agent_index, component, :],
                color=self._preprocessing_color(preprocessing),
                linestyle=linestyles.get(preprocessing, "--"),
                linewidth=2.0,
                label=(
                    rf"$\hat{{{labels['sys_var']}}}_{{i}}$ "
                    f"{ledgend_labels.get(preprocessing, preprocessing)}"
                ),
            )

        # -------------------------------------------------
        # Noisy / filtered observations on fit window
        # --------------------------------------------------
        markers = {
            "raw": "o",
            "sliding_window": "s",
            "kalman_filter": "^",
        }

        # for preprocessing, X_noisy_fit in noisy_fit_by_preprocessing.items():
        #     X_noisy_fit = np.asarray(X_noisy_fit, dtype=float)

        #     ax.scatter(
        #         t_noisy,
        #         X_noisy_fit[agent_index, component, :],
        #         s=18,
        #         marker=markers.get(preprocessing, "o"),
        #         alpha=0.45,
        #         label = (rf"${labels['sys_var']}^{{\varepsilon}}_{{i}}$"+ f" {ledgend_labels.get(preprocessing, preprocessing)}"),
        #     )
        for preprocessing in self.PREPROCESSING_ORDER:
            if preprocessing not in noisy_fit_by_preprocessing:
                continue

            X_noisy_fit = np.asarray(
                noisy_fit_by_preprocessing[preprocessing],
                dtype=float,
            )

            ax.scatter(
                t_noisy,
                X_noisy_fit[agent_index, component, :],
                color=self._preprocessing_color(preprocessing),
                s=18,
                marker=markers.get(preprocessing, "o"),
                alpha=0.45,
                label=(
                    rf"${labels['sys_var']}^{{noisy}}_{{i}}$ "
                    f"{ledgend_labels.get(preprocessing, preprocessing)}"
                ),
            )

        ax.axvline(T, color="gray", linestyle=":", linewidth=1.4)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(rf"${labels['sys_var']}_{{i}}$")
        # ax.set_title(rf"$\zeta={noise_level}$")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", frameon=True, ncols=1)

        fig.tight_layout()

        return self.fig_repo.save(
            fig,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=figure_name,
            close=False,
        )