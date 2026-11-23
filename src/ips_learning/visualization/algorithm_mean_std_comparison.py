from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ips_learning.visualization.style import apply_paper_style, OKABE_ITO


class AlgorithmMeanStdComparison:
    """
    Compare true functions against repeated-trial mean +/- standard deviation
    for nonparametric and parametric environment-force models.

    Expected summary .npz keys from Stage 6B repeated-trial summaries:
        kernel_grid : (K,)
        phi_mean    : (K,)
        phi_std     : (K,)
        z1_grid     : (G1,)
        z2_grid     : (G2,) for 2D systems
        f_mean      : (G1, G2, D) for 2D, or (G1, D) for 1D
        f_std       : same shape as f_mean

    Typical summary input:
        summaries = {
            "nonparametric_f": ".../nonparametric_f/function_value_summary.npz",
            "parametric_f": ".../parametric_f/function_value_summary.npz",
        }
    """

    def __init__(self):
        apply_paper_style()

    # ------------------------------------------------------------------
    # Loading / validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_summary(summary) -> dict:
        """
        Accept either an .npz path, an np.lib.npyio.NpzFile, or a dictionary.
        Return a plain dictionary of NumPy arrays.
        """
        if isinstance(summary, (str, Path)):
            data = np.load(summary)
            return {key: data[key] for key in data.files}

        if isinstance(summary, np.lib.npyio.NpzFile):
            return {key: summary[key] for key in summary.files}

        if isinstance(summary, Mapping):
            return {key: np.asarray(value) for key, value in summary.items()}

        raise TypeError(
            "summary must be a path, npz file, or mapping with summary arrays."
        )

    @staticmethod
    def _summary_dict(summaries: Mapping) -> dict[str, dict]:
        return {
            str(name): AlgorithmMeanStdComparison._load_summary(summary)
            for name, summary in summaries.items()
        }

    @staticmethod
    def _algorithm_style(algorithm: str) -> tuple[str, str]:
        if algorithm == "nonparametric_f":
            return OKABE_ITO[1], "-"
        if algorithm == "parametric_f":
            return OKABE_ITO[2], "-"
        return OKABE_ITO[0], "-"

    @staticmethod
    def _default_kernel_label(algorithm: str) -> str:
        if algorithm == "nonparametric_f":
            return r"$\overline{\hat{\phi}}_{\mathrm{NP}}$"
        if algorithm == "parametric_f":
            return r"$\overline{\hat{\phi}}_{\mathrm{SP}}$"
        return rf"$\overline{{\hat{{\phi}}}}_({algorithm})$"

    @staticmethod
    def _default_env1D_label(algorithm: str) -> str:
        if algorithm == "nonparametric_f":
            return r"$\overline{\hat{f}}_{\mathrm{NP}}$"
        if algorithm == "parametric_f":
            return r"$\overline{\hat{f}}_{\mathrm{SP}}$"
        return rf"$\overline{{\hat{{f}}}}_({algorithm})$"
    
    @staticmethod
    def _default_env_label(algorithm: str, comp: int) -> str:
        idx = comp + 1
        if algorithm == "nonparametric_f":
            return rf"$\hat{{f}}_{{{idx},\mathrm{{NP}}}}$"
        if algorithm == "parametric_f":
            return rf"$\hat{{f}}_{{{idx},\mathrm{{SP}}}}$"
        return rf"$\hat{{f}}_{idx}$ ({algorithm})"

    @staticmethod
    def _validate_common_grid(summary_data: dict[str, dict], key: str) -> np.ndarray:
        ref = None
        for name, data in summary_data.items():
            if key not in data:
                raise KeyError(f"summary for {name!r} is missing key {key!r}.")
            arr = np.asarray(data[key], dtype=float).reshape(-1)
            if ref is None:
                ref = arr
            elif not np.allclose(ref, arr):
                raise ValueError(
                    f"{key} is not identical across summaries. "
                    "Use a common evaluation grid before computing mean/std."
                )
        return ref

    # ------------------------------------------------------------------
    # Kernel: true vs mean +/- std for NP and P
    # ------------------------------------------------------------------
    def plot_kernel_mean_std_algorithm_comparison(
        self,
        *,
        algorithm_summaries: Mapping,
        true_kernel,
        train_density=None,
        labels: dict | None = None,
    ):
        """
        Plot true kernel and repeated-trial mean +/- std for each algorithm.

        Std margins are drawn as dashed lines and the interval is filled with
        alpha=0.2, as requested.
        """
        labels = labels or {}
        summary_data = self._summary_dict(algorithm_summaries)

        r_grid = self._validate_common_grid(summary_data, "kernel_grid")
        r_eval = r_grid.reshape(-1, 1)

        x_label = labels.get("xlabel", labels.get("x_label", r"$r$"))
        y1_label = labels.get("y1label", labels.get("y_label", "kernel"))
        y2_label = labels.get("y2label", labels.get("y2_label", "density"))

        true_label = labels.get(
            "true_kernel_label",
            labels.get("true_label", r"$\phi$"),
        )
        density_label = labels.get(
            "train_density_label",
            labels.get("density_label", r"$\hat{\rho}_R(r)$"),
        )

        mean_labels = labels.get("mean_labels", {})
        std_suffix = labels.get("std_suffix", r"$ \sigma_{\phi}$")

        legend_loc = labels.get("legend_loc", 1)
        legend_bbox_to_anchor = labels.get("legend_bbox_to_anchor", None)

        fig, ax1 = plt.subplots(figsize=labels.get("figsize", (8.6, 4.9)))

        phi_true = np.asarray(true_kernel(r_eval), dtype=float).reshape(-1)
        ax1.plot(
            r_grid,
            phi_true,
            color=OKABE_ITO[4],
            linewidth=2.4,
            linestyle="-",
            label=true_label,
        )

        for algorithm, data in summary_data.items():
            color, linestyle = self._algorithm_style(algorithm)

            phi_mean = np.asarray(data["phi_mean"], dtype=float).reshape(-1)
            phi_std = np.asarray(data["phi_std"], dtype=float).reshape(-1)

            if phi_mean.shape != r_grid.shape or phi_std.shape != r_grid.shape:
                raise ValueError(
                    f"phi_mean and phi_std for {algorithm!r} must have shape "
                    f"{r_grid.shape}; got {phi_mean.shape}, {phi_std.shape}."
                )

            lower = phi_mean - phi_std
            upper = phi_mean + phi_std

            mean_label = mean_labels.get(
                algorithm,
                self._default_kernel_label(algorithm),
            )


            ax1.plot(
                r_grid,
                phi_mean,
                color=color,
                linestyle=linestyle,
                linewidth=2.4,
                label=mean_label,
            )
            ax1.plot(
                r_grid,
                upper,
                color=color,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label=rf"{mean_label} + {std_suffix}",
            )
            ax1.plot(
                r_grid,
                lower,
                color=color,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label=rf"{mean_label} - {std_suffix}",
            )
            ax1.fill_between(
                r_grid,
                lower,
                upper,
                color=color,
                alpha=0.2,
            )

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y1_label)

        if labels.get("title") is not None:
            ax1.set_title(labels["title"])

        if train_density is not None:
            train_density = np.asarray(train_density, dtype=float).reshape(-1)
            if train_density.shape != r_grid.shape:
                raise ValueError(
                    f"train_density must have shape {r_grid.shape}, "
                    f"got {train_density.shape}."
                )

            ax2 = ax1.twinx()
            ax2.plot(
                r_grid,
                train_density,
                color=OKABE_ITO[5],
                linewidth=2.0,
                label=density_label,
            )
            ax2.fill_between(
                r_grid,
                0.0,
                train_density,
                color=OKABE_ITO[5],
                alpha=0.08,
            )
            ax2.set_ylabel(y2_label, color=OKABE_ITO[5], fontweight="bold")
            ax2.tick_params(axis="y", colors=OKABE_ITO[5])
            ax2.spines["right"].set_color(OKABE_ITO[5])

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            handles = lines1 + lines2
            leg_labels = labels1 + labels2
        else:
            handles, leg_labels = ax1.get_legend_handles_labels()

        if legend_bbox_to_anchor is None:
            ax1.legend(handles, leg_labels, loc=legend_loc)
        else:
            ax1.legend(
                handles,
                leg_labels,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
            )

        return fig

    # ------------------------------------------------------------------
    # 2D environment: true vs mean with +/- std surfaces for NP and P
    # ------------------------------------------------------------------
    def plot_environment_surfaces_2d_mean_std(
        self,
        *,
        algorithm_summaries: Mapping,
        reference_density: np.ndarray,
        true_env=None,
        environment_variable: str = "z",
        zlim_mode: str = "joint",
        labels: dict | None = None,
    ):
        """
        Plot true 2D environment force and mean +/- std surfaces.

        Layout for two algorithms:
            row 1: component f_1
            row 2: component f_2
            columns: true, nonparametric mean/std, parametric mean/std

        For learned surfaces:
            - mean is drawn as a solid surface,
            - mean + std and mean - std are drawn as lighter wireframes,
            - a small legend marks mean and +/- std.
        """
        labels = labels or {}
        summary_data = self._summary_dict(algorithm_summaries)

        z1_grid = self._validate_common_grid(summary_data, "z1_grid")
        z2_grid = self._validate_common_grid(summary_data, "z2_grid")

        reference_density = np.asarray(reference_density, dtype=float)
        Z1, Z2 = np.meshgrid(z1_grid, z2_grid, indexing="ij")

        if reference_density.shape != Z1.shape:
            raise ValueError(
                f"reference_density must have shape {Z1.shape}, "
                f"got {reference_density.shape}."
            )

        Z_grid = np.column_stack((Z1.ravel(), Z2.ravel()))

        x1_label = labels.get("x1_label", rf"${environment_variable}_1$")
        x2_label = labels.get("x2_label", rf"${environment_variable}_2$")
        colorbar_label = labels.get(
            "colorbar_label",
            "density",#rf"$\hat{{\rho}}_{{{environment_variable}}}$",
        )

        true_comp_labels = labels.get(
            "true_comp_labels",
            [r"$f_1$", r"$f_2$"],
        )
        mean_comp_labels = labels.get("mean_comp_labels", {})

        def evaluate_true(env_fn):
            values = np.asarray(env_fn(Z_grid), dtype=float)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if values.shape[-1] != 2:
                raise ValueError(
                    "This 2D surface plot expects true_env to return "
                    f"shape (n_points, 2), got {values.shape}."
                )
            return values.reshape(*Z1.shape, 2)

        f_true = evaluate_true(true_env) if true_env is not None else None

        f_means: dict[str, np.ndarray] = {}
        f_stds: dict[str, np.ndarray] = {}

        for algorithm, data in summary_data.items():
            f_mean = np.asarray(data["f_mean"], dtype=float)
            f_std = np.asarray(data["f_std"], dtype=float)

            expected_shape = (*Z1.shape, 2)
            if f_mean.shape != expected_shape:
                raise ValueError(
                    f"f_mean for {algorithm!r} must have shape "
                    f"{expected_shape}; got {f_mean.shape}."
                )
            if f_std.shape != expected_shape:
                raise ValueError(
                    f"f_std for {algorithm!r} must have shape "
                    f"{expected_shape}; got {f_std.shape}."
                )

            f_means[algorithm] = f_mean
            f_stds[algorithm] = f_std

        cmap = plt.cm.plasma
        norm = plt.Normalize(
            float(np.nanmin(reference_density)),
            float(np.nanmax(reference_density)),
        )
        facecolors = cmap(norm(reference_density))

        xlim = (float(z1_grid.min()), float(z1_grid.max()))
        ylim = (float(z2_grid.min()), float(z2_grid.max()))

        def compute_zlims():
            zlims = {}
            for comp in range(2):
                surfaces = []

                if zlim_mode in {"true_if_available", "joint"} and f_true is not None:
                    surfaces.append(f_true[:, :, comp])

                if zlim_mode in {"learned_only", "joint"} or (
                    zlim_mode == "true_if_available" and f_true is None
                ):
                    for algorithm in f_means:
                        mean = f_means[algorithm][:, :, comp]
                        std = f_stds[algorithm][:, :, comp]
                        surfaces.extend([mean - std, mean, mean + std])

                if not surfaces:
                    raise ValueError(
                        f"Unknown zlim_mode={zlim_mode!r}. Use "
                        "'true_if_available', 'joint', or 'learned_only'."
                    )

                z_source = np.concatenate([np.asarray(s).ravel() for s in surfaces])
                zmin = float(np.nanmin(z_source))
                zmax = float(np.nanmax(z_source))

                if np.isclose(zmin, zmax):
                    pad = 1e-6 if zmin == 0.0 else 0.05 * abs(zmin)
                else:
                    pad = 0.05 * (zmax - zmin)

                zlims[comp] = (zmin - pad, zmax + pad)

            return zlims

        zlims = compute_zlims()

        n_cols = (1 if f_true is not None else 0) + len(f_means)

        fig, axes = plt.subplots(
            2,
            n_cols,
            figsize=labels.get("figsize", (5.4 * n_cols, 9.5)),
            subplot_kw={"projection": "3d"},
            squeeze=False,
        )

        def format_axis(ax, comp_idx: int, title: str):
            ax.set_xlabel(x1_label, labelpad=6)
            ax.set_ylabel(x2_label, labelpad=6)
            if labels.get("show_zlabel", False):
                ax.set_zlabel(title, labelpad=8)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlims[comp_idx])
            ax.set_title(title)
            ax.view_init(
                elev=labels.get("elev", 25),
                azim=labels.get("azim", 45),
            )
            ax.tick_params(pad=2)

        def plot_true_surface(ax, Z, comp_idx: int, title: str):
            ax.plot_surface(
                Z1,
                Z2,
                Z,
                facecolors=facecolors,
                linewidth=0,
                antialiased=False,
                shade=False,
            )
            format_axis(ax, comp_idx, title)

        def plot_mean_std_surface(
            ax,
            mean,
            std,
            algorithm: str,
            comp_idx: int,
            title: str,
        ):
            color, _ = self._algorithm_style(algorithm)
            stride = labels.get("std_wire_stride", 6)

            ax.plot_surface(
                Z1,
                Z2,
                mean,
                color=color,
                linewidth=0,
                antialiased=False,
                alpha=labels.get("mean_surface_alpha", 0.78),
                shade=True,
            )

            ax.plot_wireframe(
                Z1,
                Z2,
                mean + std,
                rstride=stride,
                cstride=stride,
                color=color,
                linewidth=labels.get("std_wire_linewidth", 0.7),
                alpha=labels.get("std_wire_alpha", 0.35),
            )

            ax.plot_wireframe(
                Z1,
                Z2,
                mean - std,
                rstride=stride,
                cstride=stride,
                color=color,
                linewidth=labels.get("std_wire_linewidth", 0.7),
                alpha=labels.get("std_wire_alpha", 0.35),
            )

            format_axis(ax, comp_idx, title)

            if labels.get("show_surface_legend", True):
                legend_handles = [
                    Patch(
                        facecolor=color,
                        edgecolor=color,
                        alpha=labels.get("mean_surface_alpha", 0.78),
                        label=labels.get("mean_label", r"$\overline{\hat{f}}$"),
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="-",
                        alpha=labels.get("std_wire_alpha", 0.35),
                        label=labels.get("std_label", r"$\overline{\hat{f}} \pm \sigma_{\hat{f}}$"),
                    ),
                ]
                ax.legend(handles=legend_handles, loc=labels.get("surface_legend_loc", "upper left"))

        col = 0

        if f_true is not None:
            plot_true_surface(
                axes[0, col],
                f_true[:, :, 0],
                0,
                true_comp_labels[0],
            )
            plot_true_surface(
                axes[1, col],
                f_true[:, :, 1],
                1,
                true_comp_labels[1],
            )
            col += 1

        for algorithm in f_means:
            f_mean = f_means[algorithm]
            f_std = f_stds[algorithm]

            comp_labels = mean_comp_labels.get(
                algorithm,
                [
                    self._default_env_label(algorithm, 0),
                    self._default_env_label(algorithm, 1),
                ],
            )

            plot_mean_std_surface(
                axes[0, col],
                f_mean[:, :, 0],
                f_std[:, :, 0],
                algorithm,
                0,
                comp_labels[0],
            )
            plot_mean_std_surface(
                axes[1, col],
                f_mean[:, :, 1],
                f_std[:, :, 1],
                algorithm,
                1,
                comp_labels[1],
            )
            col += 1

        if labels.get("show_density_colorbar", True) and f_true is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(
                sm,
                ax=axes[:, 0].ravel().tolist(),
                shrink=labels.get("colorbar_shrink", 0.75),
                pad=labels.get("colorbar_pad", 0.05),
            )
            cbar.set_label(colorbar_label)

        return fig

    # ------------------------------------------------------------------
    # 1D environment: true vs mean +/- std for NP and P
    # ------------------------------------------------------------------
    def plot_environment_1d_mean_std_algorithm_comparison(
        self,
        *,
        algorithm_summaries: Mapping,
        true_env=None,
        reference_density=None,
        environment_variable: str = "z",
        labels: dict | None = None,
    ):
        """
        Plot true 1D environment force and repeated-trial mean +/- std.
        """
        labels = labels or {}
        summary_data = self._summary_dict(algorithm_summaries)

        z_grid = self._validate_common_grid(summary_data, "z1_grid")
        Z_eval = z_grid.reshape(-1, 1)

        xlabel = labels.get("xlabel", rf"${environment_variable}$")
        ylabel = labels.get("ylabel", rf"$f({environment_variable})$")
        density_ylabel = labels.get("density_ylabel", "density")
        density_label = labels.get("density_label", rf"$\hat{{\rho}}_{{{environment_variable}}}$")
        true_label = labels.get("true_label", rf"$f$")

        legend_loc = labels.get("legend_loc", 1)
        legend_bbox_to_anchor = labels.get("legend_bbox_to_anchor", None)

        fig, ax1 = plt.subplots(figsize=labels.get("figsize", (8.6, 4.9)))

        if true_env is not None:
            f_true = np.asarray(true_env(Z_eval), dtype=float)
            if f_true.ndim == 1:
                f_true = f_true.reshape(-1, 1)
            ax1.plot(
                z_grid,
                f_true[:, 0],
                color=OKABE_ITO[4],
                linewidth=2.4,
                label=true_label,
            )

        for algorithm, data in summary_data.items():
            color, linestyle = self._algorithm_style(algorithm)
            f_mean = np.asarray(data["f_mean"], dtype=float)
            f_std = np.asarray(data["f_std"], dtype=float)

            if f_mean.ndim == 1:
                f_mean = f_mean.reshape(-1, 1)
            if f_std.ndim == 1:
                f_std = f_std.reshape(-1, 1)

            if f_mean.shape[0] != z_grid.shape[0] or f_mean.shape[-1] != 1:
                raise ValueError(
                    f"Expected f_mean for {algorithm!r} to have shape (G,1), "
                    f"got {f_mean.shape}."
                )

            mean = f_mean[:, 0]
            std = f_std[:, 0]
            lower = mean - std
            upper = mean + std

            mean_label = labels.get("mean_labels", {}).get(
                algorithm,
                self._default_env1D_label(algorithm),
            )
            std_suffix = labels.get("std_suffix", r"$ \sigma_{f}$")

            ax1.plot(
                z_grid,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=2.4,
                label=f"{mean_label}",
            )
            ax1.plot(
                z_grid,
                upper,
                color=color,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label=rf"{mean_label} + {std_suffix}",
            )
            ax1.plot(
                z_grid,
                lower,
                color=color,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label=rf"{mean_label} - {std_suffix}",
            )
            ax1.fill_between(
                z_grid,
                lower,
                upper,
                color=color,
                alpha=0.2,
            )

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)

        if labels.get("title") is not None:
            ax1.set_title(labels["title"])

        if reference_density is not None:
            reference_density = np.asarray(reference_density, dtype=float).reshape(-1)
            if reference_density.shape != z_grid.shape:
                raise ValueError(
                    f"reference_density must have shape {z_grid.shape}, "
                    f"got {reference_density.shape}."
                )
            ax2 = ax1.twinx()
            ax2.plot(
                z_grid,
                reference_density,
                color=OKABE_ITO[-1],
                linewidth=2.0,
                label=density_label,
            )
            ax2.fill_between(z_grid, 0.0, reference_density, color=OKABE_ITO[-1], alpha=0.08)
            ax2.set_ylabel(density_ylabel, color=OKABE_ITO[-1], fontweight="bold")
            ax2.tick_params(axis="y", colors=OKABE_ITO[-1])
            ax2.spines["right"].set_color(OKABE_ITO[-1])

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            handles = lines1 + lines2
            leg_labels = labels1 + labels2
        else:
            handles, leg_labels = ax1.get_legend_handles_labels()

        if legend_bbox_to_anchor is None:
            ax1.legend(handles, leg_labels, loc=legend_loc)
        else:
            ax1.legend(
                handles,
                leg_labels,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
            )

        return fig
