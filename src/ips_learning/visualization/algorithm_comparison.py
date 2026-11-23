import numpy as np
import matplotlib.pyplot as plt

from ips_learning.visualization.style import apply_paper_style, OKABE_ITO


class AlgorithmComparison:
    """
    Stage 4c: compare learned functions across algorithms.

    Colors match LearnedFunctionVisualizer:
        true        -> OKABE_ITO[4]
        learned NP  -> OKABE_ITO[1]
        learned P   -> OKABE_ITO[2]
        density     -> OKABE_ITO[5]
        surface KDE -> plasma
    """

    def __init__(self):
        apply_paper_style()

    def plot_kernel_algorithm_comparison(
        self,
        *,
        r_grid,
        true_kernel,
        learned_models: dict,
        train_density=None,
        labels: dict | None = None,
    ):
        labels = labels or {}

        x_label = labels.get("xlabel", labels.get("x_label", r"$r$"))
        y1_label = labels.get("y1label", labels.get("y_label", r"$\phi(r)$"))
        y2_label = labels.get("y2label", labels.get("y2_label", r"$\rho_R(r)$"))

        true_label = labels.get("true_kernel_label", labels.get("true_label", r"$\phi(r)$"))
        train_density_label = labels.get(
            "train_density_label",
            labels.get("density_label", r"$\hat{\rho}_R(r)$"),
        )

        learned_labels = labels.get(
            "learned_labels",
            {
                "nonparametric_f": r"$\hat{\phi}_{\mathrm{NP}}(r)$",
                "parametric_f": r"$\hat{\phi}_{\mathrm{P}}(r)$",
            },
        )

        legend_loc = labels.get("legend_loc", 1)
        legend_bbox_to_anchor = labels.get("legend_bbox_to_anchor", (1.0, 1.0))

        r_grid = np.asarray(r_grid, dtype=float).reshape(-1, 1)
        r_flat = r_grid[:, 0]

        fig, ax1 = plt.subplots(figsize=labels.get("figsize", (8.6, 4.9)))

        ax1.plot(
            r_flat,
            np.asarray(true_kernel(r_grid)).reshape(-1),
            color=OKABE_ITO[4],
            linewidth=2.2,
            label=true_label,
        )

        for algorithm, model in learned_models.items():
            if algorithm == "nonparametric_f":
                color = OKABE_ITO[1]
                linestyle = "--"
            elif algorithm == "parametric_f":
                color = OKABE_ITO[2]
                linestyle = "-."
            else:
                color = OKABE_ITO[0]
                linestyle = "-"

            ax1.plot(
                r_flat,
                np.asarray(model.learned_kernel(r_grid)).reshape(-1),
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=learned_labels.get(algorithm, algorithm),
            )

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y1_label)

        if labels.get("title") is not None:
            ax1.set_title(labels["title"])

        if train_density is not None:
            train_density = np.asarray(train_density, dtype=float).reshape(-1)

            if train_density.shape[0] != r_flat.shape[0]:
                raise ValueError(
                    f"train_density and r_grid must have same length, got "
                    f"{train_density.shape[0]} and {r_flat.shape[0]}."
                )

            ax2 = ax1.twinx()

            ax2.plot(
                r_flat,
                train_density,
                color=OKABE_ITO[5],
                linewidth=2.0,
                label=train_density_label,
            )

            ax2.fill_between(
                r_flat,
                train_density,
                alpha=0.08,
                color=OKABE_ITO[5],
            )

            ax2.set_ylabel(
                y2_label,
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
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
            )
        else:
            ax1.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

        return fig

    def plot_environment_surfaces_2d(
        self,
        *,
        learned_model=None,
        learned_models: dict | None = None,
        z1_grid: np.ndarray,
        z2_grid: np.ndarray,
        reference_density: np.ndarray,
        true_env=None,
        environment_variable: str = "z",
        zlim_mode: str = "true_if_available",
        labels: dict | None = None,
    ):
        labels = labels or {}

        if learned_models is None:
            if learned_model is None:
                raise ValueError("Provide either learned_model or learned_models.")
            learned_models = {"learned": learned_model}

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
                "nonparametric_f": [
                    r"$\hat f_{1,\mathrm{NP}}$",
                    r"$\hat f_{2,\mathrm{NP}}$",
                ],
                "parametric_f": [
                    r"$\hat f_{1,\mathrm{P}}$",
                    r"$\hat f_{2,\mathrm{P}}$",
                ],
                "learned": [
                    labels.get("learned_comp1_label", r"$\hat f_1$"),
                    labels.get("learned_comp2_label", r"$\hat f_2$"),
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

        f_hats = {
            algorithm: evaluate_surface(model.learned_environment)
            for algorithm, model in learned_models.items()
        }

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
    
    def plot_environment_surfaces_1d(
        self,
        *,
        learned_model=None,
        learned_models: dict | None = None,
        z_grid: np.ndarray,
        reference_density: np.ndarray | None = None,
        true_env=None,
        environment_variable: str = "z",
        labels: dict | None = None,
    ):
        """
        Plot true and learned scalar environment functions in 1D.

        Used for first-order systems such as Kuramoto:
            z = x
            f: R -> R

        Supports:
            learned_model  : single learned model
            learned_models : algorithm -> learned model dictionary
        """

        labels = labels or {}

        if learned_models is None:
            if learned_model is None:
                raise ValueError("Provide either learned_model or learned_models.")
            learned_models = {"learned": learned_model}

        true_label = labels.get(
            "true_label",
            rf"$f({environment_variable})$",
        )

        learned_labels = labels.get(
            "learned_labels",
            {
                "nonparametric_f": rf"$\hat f_{{\mathrm{{NP}}}}({environment_variable})$",
                "parametric_f": rf"$\hat f_{{\mathrm{{P}}}}({environment_variable})$",
                "learned": rf"$\hat f({environment_variable})$",
            },
        )

        density_label = labels.get(
            "density_label",
            rf"$\hat{{\rho}}_{{{environment_variable}}}$",
        )

        xlabel = labels.get("xlabel", rf"${environment_variable}$")
        ylabel = labels.get("ylabel", rf"$f({environment_variable})$")
        density_ylabel = labels.get(
            "density_ylabel",
            rf"$\rho_{{{environment_variable}}}$",
        )

        legend_loc = labels.get("legend_loc", "best")
        legend_bbox_to_anchor = labels.get("legend_bbox_to_anchor", None)

        z_grid = np.asarray(z_grid, dtype=float).reshape(-1)
        Z_eval = z_grid.reshape(-1, 1)

        f_true = None
        if true_env is not None:
            f_true = np.asarray(true_env(Z_eval), dtype=float)

            if f_true.ndim != 2 or f_true.shape[1] != 1:
                raise ValueError(
                    f"Expected true_env output shape (n,1), got {f_true.shape}."
                )

        f_hats = {}
        for algorithm, model in learned_models.items():
            f_hat = np.asarray(
                model.learned_environment(Z_eval),
                dtype=float,
            )

            if f_hat.ndim != 2 or f_hat.shape[1] != 1:
                raise ValueError(
                    f"Expected learned environment output shape (n,1) for "
                    f"{algorithm}, got {f_hat.shape}."
                )

            if f_true is not None and f_hat.shape != f_true.shape:
                raise ValueError(
                    f"true_env and learned environment outputs must match for "
                    f"{algorithm}, got {f_true.shape} and {f_hat.shape}."
                )

            f_hats[algorithm] = f_hat

        fig, ax1 = plt.subplots(figsize=labels.get("figsize", (8.6, 4.9)))

        if f_true is not None:
            ax1.plot(
                z_grid,
                f_true[:, 0],
                color=OKABE_ITO[4],
                linewidth=2.2,
                label=true_label,
            )

        for algorithm, f_hat in f_hats.items():
            if algorithm == "nonparametric_f":
                color = OKABE_ITO[1]
                linestyle = "--"
            elif algorithm == "parametric_f":
                color = OKABE_ITO[2]
                linestyle = "-."
            else:
                color = OKABE_ITO[1]
                linestyle = "--"

            ax1.plot(
                z_grid,
                f_hat[:, 0],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=learned_labels.get(algorithm, algorithm),
            )

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)

        if labels.get("title") is not None:
            ax1.set_title(labels["title"])

        if reference_density is not None:
            reference_density = np.asarray(reference_density, dtype=float).reshape(-1)

            if reference_density.shape[0] != z_grid.shape[0]:
                raise ValueError(
                    f"reference_density and z_grid must have same length, got "
                    f"{reference_density.shape[0]} and {z_grid.shape[0]}."
                )

            ax2 = ax1.twinx()

            ax2.plot(
                z_grid,
                reference_density,
                color=OKABE_ITO[-1],
                linewidth=2.0,
                label=density_label,
            )

            ax2.fill_between(
                z_grid,
                reference_density,
                alpha=0.08,
                color=OKABE_ITO[-1],
            )
            ax2.set_ylabel(
                density_ylabel,
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

            if legend_bbox_to_anchor is None:
                ax1.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc=legend_loc,
                )
            else:
                ax1.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc=legend_loc,
                    bbox_to_anchor=legend_bbox_to_anchor,
                )

        else:
            if legend_bbox_to_anchor is None:
                ax1.legend(loc=legend_loc)
            else:
                ax1.legend(
                    loc=legend_loc,
                    bbox_to_anchor=legend_bbox_to_anchor,
                )

        return fig