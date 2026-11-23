from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ips_learning.visualization.style import apply_paper_style, OKABE_ITO
from ips_learning.model_selection.repository import ModelSelectionRepository

from ips_learning.fitting.system_repository import NormalSystemRepository
from ips_learning.models.factories import LearnedModelFactory
from ips_learning.visualization.learned_functions import LearnedFunctionVisualizer


class ModelSelectionVisualizer:
    """
    Visualize saved model-selection artifacts.

    This class does not fit models and does not run rollouts.
    It only loads saved artifacts and creates figures.
    """

    def __init__(
        self,
        repo: ModelSelectionRepository | None = None,
    ) -> None:
        apply_paper_style()
        self.repo = repo or ModelSelectionRepository()

    def plot_energy_alignment_kernels(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str = "second_order_energy_alignment",
        true_energy_kernel: Callable | None = None,
        true_alignment_kernel: Callable | None = None,
        energy_density: np.ndarray | None = None,
        alignment_density: np.ndarray | None = None,
        labels: dict | None = None,
    ):
        """
        Plot energy and alignment kernels separately from saved diagnostics.
        """

        labels = labels or {}

        diag = self.repo.load_kernel_diagnostics(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
            candidate_name=candidate_name,
        )

        r_grid = np.asarray(diag["r_grid"], dtype=float).reshape(-1, 1)

        phi_E = np.asarray(diag["phi_E"], dtype=float).reshape(-1)
        phi_A = np.asarray(diag["phi_A"], dtype=float).reshape(-1)

        fig_E = self._plot_single_kernel(
            r_grid=r_grid,
            phi_hat=phi_E,
            true_kernel=true_energy_kernel,
            density=energy_density,
            true_label=labels.get("true_energy_label", r"$\phi^E(r)$"),
            learned_label=labels.get("learned_energy_label", r"$\hat{\phi}^E(r)$"),
            density_label=labels.get("energy_density_label", r"$\hat{\rho}_R(r)$"),
            xlabel=labels.get("xlabel", r"$r$"),
            ylabel=labels.get("energy_ylabel", "kernel"),
            density_ylabel=labels.get("density_ylabel", "density"),
        )

        fig_A = self._plot_single_kernel(
            r_grid=r_grid,
            phi_hat=phi_A,
            true_kernel=true_alignment_kernel,
            density=alignment_density,
            true_label=labels.get("true_alignment_label", r"$\phi^A(r)$"),
            learned_label=labels.get("learned_alignment_label", r"$\hat{\phi}^A(r)$"),
            density_label=labels.get("alignment_density_label", r"$\hat{\rho}_R(r)$"),
            xlabel=labels.get("xlabel", r"$r$"),
            ylabel=labels.get("alignment_ylabel", "kernel"),
            density_ylabel=labels.get("density_ylabel", "density"),
        )

        return fig_E, fig_A
    
    def plot_1st_order_kernel(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str = "second_order_energy_alignment",
        true_kernel: Callable | None = None,
        kernel_density: np.ndarray | None = None,
        labels: dict | None = None,
    ):
        """
        Plot energy and alignment kernels separately from saved diagnostics.
        """

        labels = labels or {}

        diag = self.repo.load_kernel_diagnostics(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
            candidate_name=candidate_name,
        )

        r_grid = np.asarray(diag["r_grid"], dtype=float).reshape(-1, 1)

        phi_E = np.asarray(diag["phi_E"], dtype=float).reshape(-1)

        fig_E = self._plot_single_kernel(
            r_grid=r_grid,
            phi_hat=phi_E,
            true_kernel=true_kernel,
            density=kernel_density,
            true_label=labels.get("true_kernel_label", r"$\phi^E(r)$"),
            learned_label=labels.get("learned_kernel_label", r"$\hat{\phi}^E(r)$"),
            density_label=labels.get("kernel_density_label", r"$\hat{\rho}_R(r)$"),
            xlabel=labels.get("xlabel", r"$r$"),
            ylabel=labels.get("energy_ylabel", "kernel"),
            density_ylabel=labels.get("density_ylabel", "density"),
        )

        return fig_E

    def _plot_single_kernel(
        self,
        *,
        r_grid: np.ndarray,
        phi_hat: np.ndarray,
        true_kernel: Callable | None,
        density: np.ndarray | None,
        true_label: str,
        learned_label: str,
        density_label: str,
        xlabel: str,
        ylabel: str,
        density_ylabel: str,
    ):
        r_grid = np.asarray(r_grid, dtype=float).reshape(-1, 1)
        phi_hat = np.asarray(phi_hat, dtype=float).reshape(-1)

        fig, ax1 = plt.subplots(figsize=(8.6, 4.9))

        if true_kernel is not None:
            phi_true = np.asarray(true_kernel(r_grid), dtype=float).reshape(-1)
            ax1.plot(
                r_grid[:, 0],
                phi_true,
                color=OKABE_ITO[4],
                linewidth=2.2,
                label=true_label,
            )

        ax1.plot(
            r_grid[:, 0],
            phi_hat,
            color=OKABE_ITO[1],
            linestyle="--",
            linewidth=2.2,
            label=learned_label,
        )

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_xlim(float(np.min(r_grid)), float(np.max(r_grid)))

        if density is not None:
            density = np.asarray(density, dtype=float).reshape(-1)
            if density.shape[0] != r_grid.shape[0]:
                raise ValueError(
                    "density and r_grid must have the same length, got "
                    f"{density.shape[0]} and {r_grid.shape[0]}."
                )

            ax2 = ax1.twinx()

            ax2.plot(
                r_grid[:, 0],
                density,
                color=OKABE_ITO[5],
                linewidth=2.0,
                label=density_label,
            )

            ax2.fill_between(
                r_grid[:, 0],
                density,
                alpha=0.08,
                color=OKABE_ITO[5],
            )

            ax2.set_ylabel(
                density_ylabel,
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
            ax1.legend(lines1 + lines2, labels1 + labels2, loc=1)
        else:
            ax1.legend(loc="best")

        return fig

    def plot_candidate_kernel_norms(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_labels: dict | None = None,
    ):
        """
        Bar plot of phi_E_norm and phi_A_norm from the saved summary table.
        """

        df = self.repo.load_summary_table(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
        )

        needed = {"candidate", "phi_E_norm", "phi_A_norm"}
        missing = needed.difference(df.columns)
        if missing:
            raise ValueError(f"Summary table is missing columns: {sorted(missing)}")

        x = np.arange(len(df))

        fig, ax = plt.subplots(figsize=(9.5, 4.9))

        width = 0.38
        ax.bar(
            x - width / 2,
            df["phi_E_norm"].astype(float),
            width,
            label=r"$\|\hat{\phi}^E\|$",
        )
        ax.bar(
            x + width / 2,
            df["phi_A_norm"].astype(float),
            width,
            label=r"$\|\hat{\phi}^A\|$",
        )

        ax.set_xticks(x)
        ax.set_ylabel("Kernel norm")
        ax.legend(loc="best")

        if candidate_labels:
            xticklabels = [
                candidate_labels.get(c, c)
                for c in df["candidate"]
            ]

            ax.set_xticklabels(
                xticklabels,
                rotation=25,
                ha="right",
            )   
        else:
            ax.set_xticklabels(df["candidate"], rotation=25, ha="right")

        return fig

    def plot_selection_summary(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        metric: str = "primary_value",
        candidate_labels: dict | None = None,
    ):
        """
        Plot candidate validation scores from the saved summary table.
        """

        df = self.repo.load_summary_table(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
        )

        if metric not in df.columns:
            raise ValueError(f"Metric {metric!r} is not in summary table.")

        fig, ax = plt.subplots(figsize=(9.5, 4.9))

        x = np.arange(len(df))
        ax.bar(x, df[metric].astype(float))

        ax.set_xticks(x)
        ax.set_ylabel(r"Primary metric:$E_{\mathrm{full}}$")

        if candidate_labels:
            xticklabels = [
                candidate_labels.get(c, c)
                for c in df["candidate"]
            ]

            ax.set_xticklabels(
                xticklabels,
                rotation=25,
                ha="right",
            )
        else:
            ax.set_xticklabels(df["candidate"], rotation=25, ha="right")

        return fig

    def plot_rollout_coordinate(
        self,
        *,
        rollout_path: str | Path,
        trajectory_index: int = 0,
        agent_index: int = 0,
        coordinate_index: int = 0,
        labels: dict | None = None,
    ):
        """
        Plot true vs learned rollout coordinate from saved rollout.npz.
        """

        labels = labels or {}

        data = np.load(rollout_path)

        t = np.asarray(data["t"], dtype=float)
        X_true = np.asarray(data["X_true"], dtype=float)
        X_hat = np.asarray(data["X_hat"], dtype=float)

        fig, ax = plt.subplots(figsize=(8.6, 4.9))

        ax.plot(
            t,
            X_true[trajectory_index, :, agent_index, coordinate_index],
            color=OKABE_ITO[4],
            linewidth=2.2,
            label=labels.get("true_label", r"$x(t)$"),
        )

        ax.plot(
            t,
            X_hat[trajectory_index, :, agent_index, coordinate_index],
            color=OKABE_ITO[1],
            linestyle="--",
            linewidth=2.2,
            label=labels.get("learned_label", r"$\hat{x}(t)$"),
        )

        ax.set_xlabel(labels.get("xlabel", r"$t$"))
        ax.set_ylabel(labels.get("ylabel", rf"$x_{coordinate_index + 1}(t)$"))
        ax.legend(loc=labels.get("legend_loc", "best"))

        return fig
    

    def plot_candidate_environment_from_fit(
        self,
        *,
        fit_repo: NormalSystemRepository,
        system_name: str,
        profile_name: str,
        fit_case_name: str,
        z1_grid: np.ndarray,
        reference_density: np.ndarray | None = None,
        z2_grid: np.ndarray | None = None,
        true_env=None,
        environment_variable: str = "v",
        algorithm: str = "nonparametric_f",
        zlim_mode: str = "true_if_available",
        labels: dict | None = None,
    ):
        fit_result = fit_repo.load_fit_result(
            system_name=system_name,
            profile_name=profile_name,
            fit_case_name=fit_case_name,
            algorithm=algorithm,
        )

        learned_model = LearnedModelFactory().build_from_fit_result(fit_result)

        viz = LearnedFunctionVisualizer()

        return viz.plot_environment_with_support(
            learned_model=learned_model,
            z1_grid=z1_grid,
            z2_grid=z2_grid,
            reference_density=reference_density,
            true_env=true_env,
            environment_variable=environment_variable,
            zlim_mode=zlim_mode,
            labels=labels,
        )
    
    def plot_candidate_environment(
        self,
        *,
        system_name: str,
        profile_name: str,
        dataset_name: str,
        candidate_name: str,
        fit_repo: NormalSystemRepository,
        z1_grid: np.ndarray,
        reference_density: np.ndarray | None = None,
        z2_grid: np.ndarray | None = None,
        true_env=None,
        environment_variable: str = "v",
        algorithm: str = "nonparametric_f",
        zlim_mode: str = "true_if_available",
        labels: dict | None = None,
    ):
        refs = self.repo.load_candidate_references(
            system_name=system_name,
            profile_name=profile_name,
            dataset_name=dataset_name,
            candidate_name=candidate_name,
        )

        fit_path = Path(refs["fit_result"])
        fit_case_name = fit_path.parent.name

        return self.plot_candidate_environment_from_fit(
            fit_repo=fit_repo,
            system_name=system_name,
            profile_name=profile_name,
            fit_case_name=fit_case_name,
            z1_grid=z1_grid,
            reference_density=reference_density,
            z2_grid=z2_grid,
            true_env=true_env,
            environment_variable=environment_variable,
            algorithm=algorithm,
            zlim_mode=zlim_mode,
            labels=labels,
        )
    
    def plot_second_order_phase_portrait_from_rollout(
        self,
        *,
        rollout_path: str | Path,
        trajectory_index: int = 0,
        train_fit_end_pt: float | None = None,
        train_fit_mark: bool = False,
        title_true: str = r"$x$",
        title_hat: str = r"$\hat{x}$",
    ):
        """
        Plot true vs learned second-order phase portraits from saved rollout.npz.

        Expects rollout.npz to contain:
            t, X_true, X_hat

        Shapes:
            X_true, X_hat: (M, L, N, D)
        """
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        data = np.load(rollout_path)

        t = np.asarray(data["t"], dtype=float)
        X_true = np.asarray(data["X_true"], dtype=float)
        X_hat = np.asarray(data["X_hat"], dtype=float)

        if X_true.shape[-1] < 2:
            raise ValueError("Phase portrait requires D >= 2.")

        X_true_single = X_true[trajectory_index].transpose(1, 2, 0)  # (N, D, L)
        X_hat_single = X_hat[trajectory_index].transpose(1, 2, 0)    # (N, D, L)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11.8, 5.3),
            sharex=True,
            sharey=True,
        )

        norm = Normalize(vmin=t[0], vmax=t[-1])
        cmap = plt.cm.Wistia

        if train_fit_end_pt is not None:
            mask = t <= train_fit_end_pt
        else:
            mask = np.ones_like(t, dtype=bool)

        for i in range(X_true_single.shape[0]):
            axes[0].scatter(
                X_true_single[i, 0, :],
                X_true_single[i, 1, :],
                c=t,
                cmap=cmap,
                norm=norm,
                s=8,
                alpha=0.8,
            )

            axes[1].scatter(
                X_hat_single[i, 0, :],
                X_hat_single[i, 1, :],
                c=t,
                cmap=cmap,
                norm=norm,
                s=8,
                alpha=0.8,
            )

            if train_fit_mark:
                axes[0].plot(
                    X_true_single[i, 0, mask],
                    X_true_single[i, 1, mask],
                    color="black",
                    linewidth=0.8,
                    alpha=0.45,
                )
                axes[1].plot(
                    X_hat_single[i, 0, mask],
                    X_hat_single[i, 1, mask],
                    color="black",
                    linewidth=0.8,
                    alpha=0.45,
                )

        axes[0].set_title(title_true)
        axes[1].set_title(title_hat)

        for ax in axes:
            ax.set_xlabel(r"$x_1$")
        axes[0].set_ylabel(r"$x_2$")

        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(
            sm,
            ax=axes,
            orientation="vertical",
            shrink=0.8,
            pad=0.03,
        )

        if train_fit_end_pt is not None:
            cbar.set_ticks([t[0], train_fit_end_pt, t[-1]])
            cbar.set_ticklabels([r"$t_0$", r"$T$", r"$T_f$"])
        else:
            cbar.set_label(r"$t$")

        return fig
    

    def plot_first_order_coordinate_from_rollout(
        self,
        *,
        rollout_path: str | Path,
        trajectory_index: int = 0,
        coordinate_index: int = 0,
        train_fit_end_pt: float | None = None,
        title: str = "True vs learned trajectories",
        labels: dict | None = None,
    ):
        """
        Plot first-order true vs learned coordinate trajectories from saved rollout.npz.

        Expects rollout.npz to contain:
            t, X_true, X_hat

        Shapes:
            X_true, X_hat: (M, L, N, D)

        All agents are plotted.
        """
        labels = labels or {}

        data = np.load(rollout_path)

        t = np.asarray(data["t"], dtype=float)
        X_true = np.asarray(data["X_true"], dtype=float)
        X_hat = np.asarray(data["X_hat"], dtype=float)

        X_true_single = X_true[trajectory_index]  # (L, N, D)
        X_hat_single = X_hat[trajectory_index]    # (L, N, D)

        fig, ax = plt.subplots(figsize=(8.6, 4.9))

        N = X_true_single.shape[1]

        for i in range(N):
            ax.plot(
                t,
                X_true_single[:, i, coordinate_index],
                color=OKABE_ITO[4],
                linewidth=1.3,
                alpha=0.45,
            )
            ax.plot(
                t,
                X_hat_single[:, i, coordinate_index],
                color=OKABE_ITO[1],
                linestyle="--",
                linewidth=1.3,
                alpha=0.45,
            )

        ax.plot([], [], color=OKABE_ITO[4], linewidth=2.2, label=labels.get("true_label", r"$X$"))
        ax.plot([], [], color=OKABE_ITO[1], linestyle="--", linewidth=2.2, label=labels.get("learned_label", r"$\widehat{X}$"))

        if train_fit_end_pt is not None:
            ax.axvline(
                train_fit_end_pt,
                color="black",
                linewidth=1.0,
                linestyle=":",
                alpha=0.8,
            )

            xticks = list(ax.get_xticks())
            if not any(abs(x - train_fit_end_pt) < 1e-8 for x in xticks):
                xticks.append(train_fit_end_pt)

            xticks = sorted(xticks)

            tick_labels = []
            for x in xticks:
                if abs(x - train_fit_end_pt) < 1e-8:
                    tick_labels.append(r"$T$")
                else:
                    tick_labels.append(f"{x:.2f}")

            ax.set_xticks(xticks)
            ax.set_xticklabels(tick_labels)

        ax.set_xlabel(labels.get("xlabel", r"$t$"))
        ax.set_ylabel(labels.get("ylabel", rf"$x_{coordinate_index + 1}(t)$"))
        ax.set_title(title)
        ax.legend(loc=labels.get("legend_loc", "best"))

        return fig
    
    def plot_rollout_by_order(
        self,
        *,
        rollout_path: str | Path,
        system_order: int,
        trajectory_index: int = 0,
        coordinate_index: int = 0,
        train_fit_end_pt: float | None = None,
    ):
        """
        Dispatch trajectory visualization by system order.

        Second order:
            phase portrait.

        First order:
            coordinate trajectory plot.
        """
        if system_order == 2:
            return self.plot_second_order_phase_portrait_from_rollout(
                rollout_path=rollout_path,
                trajectory_index=trajectory_index,
                train_fit_end_pt=train_fit_end_pt,
            )

        if system_order == 1:
            return self.plot_first_order_coordinate_from_rollout(
                rollout_path=rollout_path,
                trajectory_index=trajectory_index,
                coordinate_index=coordinate_index,
                train_fit_end_pt=train_fit_end_pt,
            )

        raise ValueError(f"Unsupported system_order={system_order}.")