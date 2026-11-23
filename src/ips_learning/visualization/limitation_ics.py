from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ips_learning.visualization.style import apply_paper_style, OKABE_ITO

class LimitattionICsVisualization:
    def __init__():
        apply_paper_style()

    def plot_kernel_with_support(
        self,
        learned_model,
        basis_bundle,
        r_grid: np.ndarray,
        train_density: np.ndarray,
        true_kernel=None,
        labels: dict | None = None,
 
    ):
        """
        Plot true and learned interaction kernels with:
        - train observed support (empirical_density)
        """
        labels = labels or {}

        true_kernel_label = labels.get("true_kernel_label", r"$\phi(r)$")
        learned_kernel_label = labels.get("learned_kernel_label", r"$\hat{\phi}(r)$")
        train_density_label = labels.get("train_density_label", r"$\hat{\rho}_R(r)$")

        xlabel = labels.get("xlabel", r"$r$")
        y1label = labels.get("y1label", r"$\phi(r)$")
        y2label = labels.get("y2label", r"$\rho_R(r)$")

        legend_loc = labels.get("legend_loc", 1)
        legend_bbox_to_anchor = labels.get("legend_bbox_to_anchor", (1.0, 1.0))

        r_grid = np.asarray(r_grid, dtype=float).reshape(-1, 1)
        train_density = np.asarray(train_density, dtype=float).reshape(-1)
        phi_hat = np.asarray(learned_model.learned_kernel(r_grid)).reshape(-1)

        fig, ax1 = plt.subplots(figsize=(8.6, 4.9))

        if true_kernel is not None:
            phi_true = np.asarray(true_kernel(r_grid)).reshape(-1)
            ax1.plot(
                r_grid[:, 0],
                phi_true,
                color=OKABE_ITO[4],
                linewidth=2.2,
                label=true_kernel_label,
            )

        ax1.plot(
            r_grid[:, 0],
            phi_hat,
            color=OKABE_ITO[1],
            linestyle="--",
            linewidth=2.2,
            label=learned_kernel_label,
        )

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(y1label)

        ax2 = ax1.twinx()

      

        ax2.plot(
            r_grid[:, 0],
            train_density,
            color=OKABE_ITO[5],
            linewidth=2.0,
            label=train_density_label,
        )

        ax2.fill_between(
            r_grid[:, 0],
            train_density,
            alpha=0.08,
            color=OKABE_ITO[5],
        )
        ax2.set_ylabel(
            y2label,
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
    

        ax1.set_xlim(basis_bundle.localization_metadata["r_min"], basis_bundle.localization_metadata["r_max"])

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

        return fig
    
    def plot_environment_surfaces_2d(
        self,
        *,
        learned_models: dict | None = None,
        z1_grid: np.ndarray,
        z2_grid: np.ndarray,
        reference_density: dict | None = None,#np.ndarray,
        true_env=None,
        environment_variable: str = "z",
        zlim_mode: str = "true_if_available",
        labels: dict | None = None,
    ):
        labels = labels or {}

        x1_label = labels.get("x1_label", rf"${environment_variable}_1$")
        x2_label = labels.get("x2_label", rf"${environment_variable}_2$")

        true_comp_labels = labels.get(
            "true_comp_labels",
            [
                labels.get("true_comp1_label", r"$f_1$"),
                labels.get("true_comp2_label", r"$f_2$"),
            ],
        )

        learned_comp_labels = labels.get(
            "learned_comp_labels",
            {
                "ic1": [
                    r"$\hat f_{1}$:IC1",
                    r"$\hat f_{2}$:IC1",
                ],
                "ic2": [
                    r"$\hat f_{1}$:IC2",
                    r"$\hat f_{2}$:IC2",
                ],
            },
        )

        colorbar_label = labels.get(
            "colorbar_label",
            rf"$\hat{{\rho}}_{{{environment_variable}}}$",
        )

        z1_grid = np.asarray(z1_grid, dtype=float).ravel()
        z2_grid = np.asarray(z2_grid, dtype=float).ravel()
        reference_density = np.asarray(reference_density, dtype=float)

        Z1, Z2 = np.meshgrid(z1_grid, z2_grid, indexing="ij")

        if reference_density.shape != Z1.shape:
            raise ValueError(
                f"reference_density must have shape {Z1.shape}, "
                f"got {reference_density.shape}."
            )

        Z_grid = np.column_stack((Z1.ravel(), Z2.ravel()))

        def evaluate_surface(env_fn):
            return np.asarray(env_fn(Z_grid), dtype=float).reshape(*Z1.shape, 2)

        f_true = evaluate_surface(true_env) if true_env is not None else None


        cmap = plt.cm.plasma
        norm = plt.Normalize(reference_density.min(), reference_density.max())
        facecolors = cmap(norm(reference_density))

        xlim = (float(z1_grid.min()), float(z1_grid.max()))
        ylim = (float(z2_grid.min()), float(z2_grid.max()))

        def compute_zlims():
            zlims = {}

            for comp in range(2):
                if zlim_mode == "learned_only":
                    surfaces = [f_hat[:, :, comp] for f_hat in f_hats.values()]
                elif zlim_mode == "true_if_available" and f_true is not None:
                    surfaces = [f_true[:, :, comp]]
                elif zlim_mode == "joint":
                    surfaces = []
                    if f_true is not None:
                        surfaces.append(f_true[:, :, comp])
                    surfaces.extend([f_hat[:, :, comp] for f_hat in f_hats.values()])
                elif zlim_mode == "true_if_available":
                    surfaces = [f_hat[:, :, comp] for f_hat in f_hats.values()]
                else:
                    raise ValueError(
                        f"Unknown zlim_mode='{zlim_mode}'. "
                        "Use 'true_if_available', 'joint', or 'learned_only'."
                    )

                z_source = np.concatenate([s.ravel() for s in surfaces])
                zmin = float(np.min(z_source))
                zmax = float(np.max(z_source))

                if np.isclose(zmin, zmax):
                    pad = 1e-6 if zmin == 0.0 else 0.05 * abs(zmin)
                else:
                    pad = 0.05 * (zmax - zmin)

                zlims[comp] = (zmin - pad, zmax + pad)

            return zlims

        zlims = compute_zlims()

        n_cols = 1 + len(f_hats) if f_true is not None else len(f_hats)

        fig, axes = plt.subplots(
            2,
            n_cols,
            figsize=labels.get("figsize", (5.0 * n_cols, 9.5)),
            subplot_kw={"projection": "3d"},
            squeeze=False,
        )

        def format_axis(ax, comp_idx, zlabel):
            ax.set_xlabel(x1_label, labelpad=6)
            ax.set_ylabel(x2_label, labelpad=6)

            if labels.get("show_zlabel", False):
                ax.set_zlabel(zlabel, labelpad=8)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlims[comp_idx])
            ax.set_title(zlabel)

            ax.view_init(
                elev=labels.get("elev", 25),
                azim=labels.get("azim", 45),
            )
            ax.tick_params(pad=2)

        def plot_surface(ax, Z, comp_idx, zlabel):
            ax.plot_surface(
                Z1,
                Z2,
                Z,
                facecolors=facecolors,
                linewidth=0,
                antialiased=False,
            )
            format_axis(ax, comp_idx, zlabel)

        col = 0

        if f_true is not None:
            plot_surface(axes[0, col], f_true[:, :, 0], 0, true_comp_labels[0])
            plot_surface(axes[1, col], f_true[:, :, 1], 1, true_comp_labels[1])
            col += 1

        for algorithm, f_hat in f_hats.items():
            comp_labels = learned_comp_labels.get(
                algorithm,
                [
                    rf"$\hat f_1$ ({algorithm})",
                    rf"$\hat f_2$ ({algorithm})",
                ],
            )

            plot_surface(axes[0, col], f_hat[:, :, 0], 0, comp_labels[0])
            plot_surface(axes[1, col], f_hat[:, :, 1], 1, comp_labels[1])
            col += 1

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=axes.ravel().tolist(),
            shrink=labels.get("colorbar_shrink", 0.75),
            pad=labels.get("colorbar_pad", 0.05),
        )
        cbar.set_label(colorbar_label)

        return fig