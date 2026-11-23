from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ips_learning.visualization.utils import (
    FigureRepository,
    TableRepository,
    
)
from ips_learning.visualization.style import apply_paper_style, OKABE_ITO


class NoiseSweepVisualizer:
    """
    Stage 2.7D plotting/reporting from saved noise-sweep CSV files.

    This file does not rerun fitting or rollout.
    It only reads saved tables and generates figures/tables.
    """

    def __init__(
        self,
        *,
        system_name: str,
        artifact_root: str | Path,
        stage_name: str = "step_6D",
    ) -> None:
        self.system_name = system_name
        self.artifact_root = Path(artifact_root)
        self.stage_name = stage_name

        self.fig_repo = FigureRepository(root_dir=self.artifact_root)
        self.table_repo = TableRepository(root_dir=self.artifact_root)
        apply_paper_style()


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

        return df.sort_values("noise_level").reset_index(drop=True)

    def save_clean_table(
        self,
        df: pd.DataFrame,
        file_name: str = "noise_sweep_clean_table",
    ) -> dict:
        preferred_cols = [
            "noise_level",
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
            caption=f"{self.system_name.upper()} noise robustness metrics.",
            label=f"tab:{self.system_name}_noise_sweep",
        )

    

    def plot_error_vs_noise(
        self,
        df: pd.DataFrame,
        figure_name: str = "noise_sweep_traj_error_vs_noise",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))

        metrics = [
            "E_res",
            "E_recon_train_fit",
            "E_pred_train_same_ic",
            "E_test_pred_unseen_ic",
        ]
        labels = {
            "E_res": r"$E_{res}$",
            "E_recon_train_fit": r"$E_{\mathrm{recon}}^{\mathrm{train}}:[t_0,T]$",
            "E_pred_train_same_ic": r"$E_{\mathrm{pred}}^{\mathrm{train}}:[T,T_f]$",
            "E_test_pred_unseen_ic": r"$E_{\mathrm{pred}}^{\mathrm{test}}:[T,T_f]$",
        }

        for metric in metrics:
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
        # ax.set_title("Stage 2.7D: error vs noise")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

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
    
    def plot_phi_vs_r_noise_panel(
    self,
    r_grid,
    true_phi,
    learned_phi_by_noise,
    dist_density_ref=None,
    figure_name="noise_sweep_phi_by_noise",
    labels: dict | None = None,
) -> Path:
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
            linewidth=2.5,
            label=labels["phi_true"],
        )

        for noise_level, learned_phi in learned_phi_by_noise.items():
            phi_hat = np.asarray(
                learned_phi(r_grid.reshape(-1, 1))
            ).reshape(-1)

            ax1.plot(
                r_grid,
                phi_hat,
                linestyle="--",
                linewidth=2,
                label=r"$\hat{\phi}$"+f"({labels["phi_noise"].format(noise=noise_level)})",
            )

        ax1.set_xlabel(labels["xlabel"])
        ax1.set_ylabel(labels["ylabel_left"])
        # ax1.set_title(r"Learned interaction kernel under noise")
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
                loc=1,
            )
        else:
            ax1.legend(frameon=True, loc=1)

        fig.tight_layout()

        return self.fig_repo.save(
            fig,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=figure_name,
            close=False,
        )
        
    def plot_env_surfaces_by_noise(
    self,
    v1_grid,
    v2_grid,
    true_env,
    learned_env_by_noise,
    vel_density_ref=None,
    figure_name="noise_sweep_env_surfaces_by_noise",
    labels: dict | None = None,
)-> Path:
        """
        Plot true and learned drift/environment surfaces under noise.

        Rows:
            component 1, component 2

        Columns:
            true f, learned f for each noise level

        Surface color:
            KDE density from M=2000 reference distribution, if provided.
        """
        labels = self._labels(
        labels,
        {
            "xlabel": r"$z_1$",
            "ylabel": r"$z_2$",
            "zlabel_template": r"$f_{{{component}}}$",
            "true_title_template": r"$f_{{{component}}}$",
            "noise_title_template": r" $\hat f_{{{component}}}$ ($\zeta={noise}$)",
            "density_colorbar": r"$\rho_Z(z)$",
        },
    )
        V1, V2 = np.meshgrid(v1_grid, v2_grid, indexing="ij")
        V_grid = np.column_stack([V1.ravel(), V2.ravel()])

        # --------------------------------------------------
        # Evaluate true and learned fields
        # --------------------------------------------------
        f_true = np.asarray(true_env(V_grid)).reshape(V1.shape + (2,))

        learned_values = {}
        for noise_level, learned_env in learned_env_by_noise.items():
            f_hat = np.asarray(learned_env(V_grid)).reshape(V1.shape + (2,))
            learned_values[noise_level] = f_hat

        # --------------------------------------------------
        # KDE density facecolors from M=2000 reference KDE
        # --------------------------------------------------
        if vel_density_ref is not None:
            # density = vel_density_ref.evaluate_on_grid(v1_grid, v2_grid)
            density = vel_density_ref

            density = np.asarray(density).reshape(V1.shape)
            density_min = float(np.nanmin(density))
            density_max = float(np.nanmax(density))

            if density_max > density_min:
                norm = plt.Normalize(vmin=density_min, vmax=density_max)
            else:
                norm = plt.Normalize(vmin=0.0, vmax=1.0)

            cmap = plt.cm.plasma
            facecolors = cmap(norm(density))
        else:
            density = None
            norm = None
            cmap = None
            facecolors = None

        # --------------------------------------------------
        # Layout
        # --------------------------------------------------
        noise_items = list(learned_values.items())
        ncols = len(noise_items) + 1
        nrows = 2

        fig = plt.figure(figsize=(4.4 * ncols, 8.8))

        axes = np.empty((nrows, ncols), dtype=object)

        # --------------------------------------------------
        # Shared z-limits per component
        # --------------------------------------------------
        zlims = []

        for component in range(2):
            Z_values = [f_true[:, :, component]]
            for _, f_hat in noise_items:
                Z_values.append(f_hat[:, :, component])

            zmin = min(float(np.nanmin(Z)) for Z in Z_values)
            zmax = max(float(np.nanmax(Z)) for Z in Z_values)
            zlims.append((zmin, zmax))

        # --------------------------------------------------
        # Plot helper
        # --------------------------------------------------
        def _plot_surface(ax, Z, title, component):
            if facecolors is not None:
                ax.plot_surface(
                    V1,
                    V2,
                    Z,
                    facecolors=facecolors,
                    linewidth=0,
                    antialiased=False,
                )
            else:
                ax.plot_surface(
                    V1,
                    V2,
                    Z,
                    linewidth=0,
                    antialiased=False,
                )

            ax.set_title(title)
            ax.set_xlabel(labels["xlabel"])
            ax.set_ylabel(labels["ylabel"])
            ax.set_zlabel(
                labels["zlabel_template"].format(component=component + 1)
            )
            ax.set_zlim(*zlims[component])
            ax.view_init(elev=25, azim=45)

        # --------------------------------------------------
        # First column: true field
        # --------------------------------------------------
        for component in range(2):
            ax = fig.add_subplot(nrows, ncols, component * ncols + 1, projection="3d")
            axes[component, 0] = ax

            _plot_surface(
                ax=ax,
                Z=f_true[:, :, component],
                title=labels["true_title_template"].format(component=component + 1),
                component=component,
            )

        # --------------------------------------------------
        # Remaining columns: learned fields by noise
        # --------------------------------------------------
        for col, (noise_level, f_hat) in enumerate(noise_items, start=1):
            for component in range(2):
                ax = fig.add_subplot(
                    nrows,
                    ncols,
                    component * ncols + col + 1,
                    projection="3d",
                )
                axes[component, col] = ax

                _plot_surface(
                    ax=ax,
                    Z=f_hat[:, :, component],
                    title=labels["noise_title_template"].format(
                        noise=noise_level,
                        component=component + 1,
                    ),
                    component=component,
                )

        # --------------------------------------------------
        # Shared KDE colorbar
        # --------------------------------------------------
        if vel_density_ref is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            cbar = fig.colorbar(
                sm,
                ax=axes.ravel().tolist(),
                orientation="horizontal",
                fraction=0.04,   # thickness
                pad=0.08,        # distance below plots
                aspect=40,       # long horizontal shape
            )

            cbar.set_label(labels["density_colorbar"])

        # fig.suptitle(
        #     r"True and learned drift surfaces under observation noise",
        #     y=0.98,
        # )

        # fig.tight_layout(rect=[0, 0, 0.96, 0.95])

        return self.fig_repo.save(
            fig,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=figure_name,
            close=False,
        )
    
    def plot_env_curves_by_noise(
        self,
        z_grid,
        true_env,
        learned_env_by_noise: dict[float, callable],
        env_density_ref=None,
        figure_name: str = "noise_sweep_env_curves_by_noise",
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

        for noise_level, learned_env in learned_env_by_noise.items():
            f_hat = np.asarray(learned_env(Z_eval), dtype=float)

            if f_hat.ndim == 1:
                f_hat = f_hat.reshape(-1, 1)

            if f_hat.shape != f_true.shape:
                raise ValueError(
                    f"Learned env for noise={noise_level} has shape {f_hat.shape}, "
                    f"expected {f_true.shape}."
                )

            learned_values[noise_level] = f_hat

        fig, ax1 = plt.subplots(figsize=(7.8, 5.2))

        ax1.plot(
            z_grid,
            f_true[:, 0],
            linewidth=2.5,
            label=labels["f_true"],
        )

        for noise_level, f_hat in learned_values.items():
            ax1.plot(
                z_grid,
                f_hat[:, 0],
                linestyle="--",
                linewidth=2,
                label=r"$f$"+f"({labels["f_noise"].format(noise=noise_level)})",
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

        return self.fig_repo.save(
            fig,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=figure_name,
            close=False,
        )
    


    def plot_seen_ic_phase_portrait_by_noise(
        self,
        *,
        t_full,
        X_true_full,
        X_learned_full,
        t_noisy,
        X_noisy_fit,
        noise_level: float,
        T: float = 0.5,
        Tf: float = 2.0,
        ic_index: int = 0,
        figure_name: str | None = None,
    ) -> Path:
        """
        One phase portrait for one IC:
        - clean true trajectory: thin black line
        - learned rollout: thick time-colored line
        - noisy training observations: scattered black points

        Time is shown by one shared colorbar with ticks [0, T, Tf].
        """

        if figure_name is None:
            noise_tag = str(noise_level).replace(".", "p")
            figure_name = f"noise_sweep_seen_ic_phase_noise_e_{noise_tag}"

        fig, ax = plt.subplots(figsize=(6.8, 6.2))

        norm = Normalize(vmin=0.0, vmax=Tf)
        cmap = plt.cm.Wistia

        N = X_true_full.shape[0]

        # --- plot true and learned first ---
        for i in range(N):
            alpha_i = 1.0 if i == 0 else 0.35

            # TRUE: thin clean black line
            ax.plot(
                X_true_full[i, 0, :],
                X_true_full[i, 1, :],
                color="blue",
                linewidth=0.9,
                linestyle="-",
                zorder=1,
            )

            # # LEARNED: thick colored line by time, slightly transparent
            # x = X_learned_full[i, 0, :]
            # y = X_learned_full[i, 1, :]

            # points = np.array([x, y]).T.reshape(-1, 1, 2)
            # segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # lc = LineCollection(
            #     segments,
            #     cmap=cmap,
            #     norm=norm,
            #     linewidth=3.0 if i == 0 else 2.0,
            #     alpha=0.75 if i == 0 else 0.35,
            #     zorder=2,
            # )
            # lc.set_array(np.asarray(t_full[:-1]))
            # ax.add_collection(lc)

            # LEARNED: thin colored trajectory + visible markers every 10th point
            x = X_learned_full[i, 0, :]
            y = X_learned_full[i, 1, :]

            # thin time-colored line
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidth=1.2 if i == 0 else 0.9,
                alpha=0.75 if i == 0 else 0.35,
                zorder=2,
            )
            lc.set_array(np.asarray(t_full[:-1]))
            ax.add_collection(lc)

            # markers every 10th point
            x_mark = X_learned_full[i, 0, ::10]
            y_mark = X_learned_full[i, 1, ::10]
            t_mark = np.asarray(t_full[::10])

            ax.scatter(
                x_mark,
                y_mark,
                c=t_mark,
                cmap=cmap,
                norm=norm,
                s=20 if i == 0 else 10,
                alpha=0.9 if i == 0 else 0.45,
                edgecolors="none",
                zorder=3,
            )

        # --- noisy data last, so scatter remains visible ---
        for i in range(N):
            ax.scatter(
                X_noisy_fit[i, 0, :],
                X_noisy_fit[i, 1, :],
                color="gray",
                s=8 if i == 0 else 5,
                alpha=0.75 if i == 0 else 0.35,
                marker="o",
                linewidths=0,
                zorder=4,
            )

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title(rf"$\zeta={noise_level}$")
        ax.grid(True, linestyle="--", alpha=0.35)

        # Legend proxies
        true_line = mlines.Line2D(
            [], [], color="blue", linestyle="-", linewidth=0.9,
            label=r"$x_i(t)$"
        )

        learned_line = mlines.Line2D(
            [], [], color=cmap(0.75), linestyle="-", linewidth=3.0,
            label=r"$\hat{x}_i(t)$"
        )

        noisy_points = mlines.Line2D(
            [], [], color="gray", marker="o", linestyle="None",
            markersize=4, label=r" $x_i^{noisy}(t)$"
        )

        ax.legend(
            handles=[true_line, learned_line, noisy_points],
            loc="upper right",
            frameon=True,
        )

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_ticks([0.0, T, Tf])
        cbar.set_ticklabels([r"$0$", rf"$T$", rf"$T_f$"])

        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()

        return self.fig_repo.save(
            fig,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=figure_name,
            close=False,
        )
    
    
        
        
    def generate_all(
        self,
        csv_path: str | Path,
    ) -> dict[str, Path]:
        """
        Generate all Stage 2.7D tables and figures from saved CSV data.
        """
        df = self.load_metrics(csv_path)

        outputs = {
            "clean_table": self.save_clean_table(df),
            "error_vs_noise": self.plot_error_vs_noise(df),
            "feature_errors_vs_noise": self.plot_feature_errors_vs_noise(df),
            # "conditioning_vs_noise": self.plot_conditioning_vs_noise(df),
        }

        manifest_path = (
            self.artifact_root
            / self.system_name
            / "tables"
            / self.stage_name
            / "noise_sweep_visualization_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        def stringify_output(value):
            if isinstance(value, dict):
                return {k: str(v) for k, v in value.items()}
            return str(value)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({k: stringify_output(v) for k, v in outputs.items()}, f, indent=2)

        outputs = {
            "clean_table": self.save_clean_table(df),
            "error_vs_noise": self.plot_error_vs_noise(df),
            "function_errors_vs_noise": self.plot_feature_errors_vs_noise(df),
            # "conditioning_vs_noise": self.plot_conditioning_vs_noise(df),
        }
        return outputs
    
    def plot_seen_ic_coordinate_by_noise(
        self,
        *,
        t_full,
        X_true_full,
        X_learned_full,
        t_noisy,
        X_noisy_fit,
        noise_level: float,
        T: float = 0.5,
        Tf: float = 2.0,
        ic_index: int = 0,
        component: int = 0,
        figure_name: str | None = None,
        labels: dict | None = None,
    ) -> Path:
        labels = self._labels(
            labels,
            {
                "xlabel": r"$t$",
                "ylabel": r"$x_i(t)$",
                "true": r"$x_i(t)$",
                "learned": r"$\hat{x}_i(t)$",
                "noisy": r"$x_i^\varepsilon(t)$",
            },
        )

        if figure_name is None:
            noise_tag = str(noise_level).replace(".", "p")
            figure_name = f"noise_sweep_seen_ic_coordinate_noise_e_{noise_tag}"

        t_full = np.asarray(t_full, dtype=float).reshape(-1)
        t_noisy = np.asarray(t_noisy, dtype=float).reshape(-1)

        X_true_full = np.asarray(X_true_full, dtype=float)
        X_learned_full = np.asarray(X_learned_full, dtype=float)
        X_noisy_fit = np.asarray(X_noisy_fit, dtype=float)

        # Expected shape after notebook transpose: (N, D, L)
        if X_true_full.ndim != 3:
            raise ValueError(f"Expected X_true_full shape (N,D,L), got {X_true_full.shape}.")

        N, D, L_full = X_true_full.shape

        if component >= D:
            raise ValueError(f"component={component} is invalid for D={D}.")

        fig, ax = plt.subplots(figsize=(7.6, 4.8))

        for i in range(N):
            alpha_i = 1.0 if i == 0 else 0.35
            lw_i = 1.6 if i == 0 else 0.9

            ax.plot(
                t_full,
                X_true_full[i, component, :],
                color="blue",
                linewidth=lw_i,
                alpha=alpha_i,
                label=labels["true"] if i == 0 else None,
            )

            ax.plot(
                t_full,
                X_learned_full[i, component, :],
                linestyle="--",
                linewidth=lw_i,
                alpha=alpha_i,
                label=labels["learned"] if i == 0 else None,
            )

            ax.scatter(
                t_noisy,
                X_noisy_fit[i, component, :],
                color="gray",
                s=10 if i == 0 else 5,
                alpha=0.75 if i == 0 else 0.35,
                label=labels["noisy"] if i == 0 else None,
            )

        ax.axvline(T, color="black", linestyle=":", linewidth=1.2)
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.set_title(rf"$\zeta={noise_level}$")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", frameon=True)

        fig.tight_layout()

        return self.fig_repo.save(
            fig,
            system_name=self.system_name,
            stage_name=self.stage_name,
            file_name=figure_name,
            close=False,
        )