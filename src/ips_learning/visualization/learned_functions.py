import numpy as np
import matplotlib.pyplot as plt

from ips_learning.visualization.style import apply_paper_style, OKABE_ITO


class LearnedFunctionVisualizer:
    """
    Stage 2.4b: visualize reconstructed learned functions before rollout.

    This module does not fit KDEs.
    It only plots:
    - learned functions
    - optional true functions
    - train observed support
    - precomputed reference KDEs
    """


    def __init__(self):
        apply_paper_style()

    def plot_kernel_with_support(
        self,
        learned_model,
        basis_bundle,
        r_grid: np.ndarray,
        # reference_pdf: np.ndarray,
        train_density: np.ndarray,
        true_kernel=None,
        labels: dict | None = None,
 
    ):
        """
        Plot true and learned interaction kernels with:
        - train observed support (empirical_density)
        - reference KDE from M=2000
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
        # reference_pdf = np.asarray(reference_pdf, dtype=float).reshape(-1)

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
        # ax1.set_title(title)


        train_density = np.asarray(train_density, dtype=float).reshape(-1)
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
    

    def plot_environment_with_support(
        self,
        learned_model,
        z1_grid: np.ndarray,
        reference_density: np.ndarray | None = None,
        z2_grid: np.ndarray | None = None,
        true_env=None,
        environment_variable: str = "z",
        zlim_mode: str = "true_if_available",
        labels: dict | None = None,
    ):
        """
        General environment-function plot.

        D=1:
            line plot of f(z) and learned f_hat(z)

        D=2:
            surface plot of f(z1,z2) and learned f_hat(z1,z2)

        Second-order:
            z = v

        First-order:
            z = x
        """
        labels = labels or {}

        if z2_grid is None:
            return self.plot_environment_curve_1d(
                learned_model=learned_model,
                z_grid=z1_grid,
                reference_density=reference_density,
                true_env=true_env,
                environment_variable=environment_variable,
                labels=labels,
            )

        return self.plot_environment_surfaces_2d(
            learned_model=learned_model,
            z1_grid=z1_grid,
            z2_grid=z2_grid,
            reference_density=reference_density,
            true_env=true_env,
            environment_variable=environment_variable,
            zlim_mode=zlim_mode,
            labels=labels,
        )
    
    def plot_environment_curve_1d(
        self,
        learned_model,
        z_grid: np.ndarray,
        reference_density: np.ndarray | None = None,
        true_env=None,
        environment_variable: str = "z",
        labels: dict | None = None,
    ):
        """
        Plot true and learned scalar/vector environment function in 1D.

        Used for Kuramoto:
            z = x
            f: R -> R
        """

        labels = labels or {}


        true_label = labels.get("true_label", rf"$f({environment_variable})$")
        learned_label = labels.get("learned_label", rf"$\hat f({environment_variable})$")
        density_label = labels.get(
            "density_label",
            rf"$\hat{{\rho}}_{{{environment_variable}}}$",
        )
        xlabel = labels.get("xlabel", rf"${environment_variable}$")
        ylabel = labels.get("y1label", rf"$f({environment_variable})$")
        density_ylabel = labels.get("density_ylabel", rf"$\rho_{{{environment_variable}}}$")
        legend_loc = labels.get("legend_loc", "best")

        z_grid = np.asarray(z_grid, dtype=float).reshape(-1)
        Z_eval = z_grid.reshape(-1, 1)

        f_hat = np.asarray(
            learned_model.learned_environment(Z_eval),
            dtype=float,
        )

        if f_hat.ndim != 2 or f_hat.shape[1] != 1:
            raise ValueError(
                f"Expected learned environment output shape (n,1), got {f_hat.shape}."
            )

        f_true = None
        if true_env is not None:
            f_true = np.asarray(true_env(Z_eval), dtype=float)
            if f_true.shape != f_hat.shape:
                raise ValueError(
                    f"true_env and learned environment outputs must match, got "
                    f"{f_true.shape} and {f_hat.shape}."
                )

        fig, ax1 = plt.subplots(figsize=(8.6, 4.9))

        if f_true is not None:
            ax1.plot(
                z_grid,
                f_true[:, 0],
                color=OKABE_ITO[4],
                linewidth=2.2,
                label=true_label,
            )

        ax1.plot(
            z_grid,
            f_hat[:, 0],
            color=OKABE_ITO[1],
            linestyle="--",
            linewidth=2.2,
            label=learned_label,
        )
        ax1.set_ylabel(
            ylabel)

        ax1.set_xlabel(xlabel)
  

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

            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc=legend_loc,
            )

        else:
            ax1.legend(loc=legend_loc)

        return fig

    def plot_environment_surfaces_2d(
        self,
        learned_model,
        z1_grid: np.ndarray,
        z2_grid: np.ndarray,
        reference_density: np.ndarray,
        true_env=None,
        environment_variable: str = "z",
        zlim_mode: str = "true_if_available",
        labels: dict | None = None,
    ):
        """
        Plot true and learned environment force surfaces in 2D velocity space.

        Parameters
        ----------
        learned_model : object
            Must implement learned_environment(V_grid).
        v1_grid, z2_grid : np.ndarray
            1D velocity grids.
        reference_density : np.ndarray
            Density values matching meshgrid(v1_grid, z2_grid, indexing="ij").
        true_env : callable or None, optional
            True environment function for comparison.
        zlim_mode : {"true_if_available", "joint", "learned_only"}
            Strategy for z-axis scaling:
            - "true_if_available": use true surface limits when available
            - "joint": use combined true + learned limits
            - "learned_only": use learned limits only
        """

        labels = labels or {}

        x1_label = labels.get("x1_label", rf"${environment_variable}_1$")
        x2_label = labels.get("x2_label", rf"${environment_variable}_2$")
        true_comp1_label = labels.get("true_comp1_label", r"$f_1$")
        true_comp2_label = labels.get("true_comp2_label", r"$f_2$")
        learned_comp1_label = labels.get("learned_comp1_label", r"$\hat{f}_1$")
        learned_comp2_label = labels.get("learned_comp2_label", r"$\hat{f}_2$")
        colorbar_label = labels.get(
            "colorbar_label",
            "density",#rf"$\hat{{\rho}}_{{{environment_variable}}}$",
        )


        z1_grid = np.asarray(z1_grid, dtype=float).ravel()
        z2_grid = np.asarray(z2_grid, dtype=float).ravel()
        reference_density = np.asarray(reference_density, dtype=float)

        Z1, Z2 = np.meshgrid(z1_grid, z2_grid, indexing="ij")
        Z_grid = np.column_stack((Z1.ravel(), Z2.ravel()))

        def evaluate_surface(env_fn):
            return np.asarray(env_fn(Z_grid), dtype=float).reshape(*Z1.shape, 2)

        f_hat = evaluate_surface(learned_model.learned_environment)
        f_true = evaluate_surface(true_env) if true_env is not None else None

        cmap = plt.cm.plasma
        norm = plt.Normalize(reference_density.min(), reference_density.max())
        facecolors = cmap(norm(reference_density))

        xlim = (float(z1_grid.min()), float(z1_grid.max()))
        ylim = (float(z2_grid.min()), float(z2_grid.max()))

        def compute_zlims():
            zlims = {}

            for comp in range(2):
                z_hat = f_hat[:, :, comp]

                if zlim_mode == "learned_only" or f_true is None:
                    z_source = z_hat
                elif zlim_mode == "true_if_available":
                    z_source = f_true[:, :, comp]
                elif zlim_mode == "joint":
                    z_source = np.concatenate(
                        [f_true[:, :, comp].ravel(), z_hat.ravel()]
                    )
                else:
                    raise ValueError(
                        f"Unknown zlim_mode='{zlim_mode}'. "
                        "Use 'true_if_available', 'joint', or 'learned_only'."
                    )

                zmin = float(np.min(z_source))
                zmax = float(np.max(z_source))

                if np.isclose(zmin, zmax):
                    pad = 1e-6 if zmin == 0.0 else 0.05 * abs(zmin)
                else:
                    pad = 0.05 * (zmax - zmin)

                zlims[comp] = (zmin - pad, zmax + pad)

            return zlims

        zlims = compute_zlims()

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 10),
            subplot_kw={"projection": "3d"},
        )

        def format_axis(ax, comp_idx, zlabel):
            ax.set_xlabel(x1_label, labelpad=6)
            ax.set_ylabel(x2_label, labelpad=6)
            # ax.set_zlabel(zlabel, labelpad=8)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlims[comp_idx])
            ax.set_title(zlabel)

            ax.view_init(elev=25, azim=45)
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

        # True surfaces
        if f_true is not None:
            plot_surface(axes[0, 0], f_true[:, :, 0], 0, true_comp1_label)
            plot_surface(axes[1, 0], f_true[:, :, 1], 1, true_comp2_label)
        else:
            axes[0, 0].axis("off")
            axes[1, 0].axis("off")

        # Learned surfaces
        plot_surface(axes[0, 1], f_hat[:, :, 0], 0, learned_comp1_label)
        plot_surface(axes[1, 1], f_hat[:, :, 1], 1, learned_comp2_label)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=axes.ravel().tolist(),
            shrink=0.75,
            pad=0.05,
        )
        cbar.set_label(colorbar_label)

        return fig
    
   

    


# # Example usage (to be run in a separate script or notebook):
# train_raw = repo.load("spp", "train", "dataset_001")
# kde_raw = repo.load("spp", "kde_ref", "dataset_M2000")

# features_train = SPPFeatureBuilder().build(train_raw)
# features_kde = SPPFeatureBuilder().build(kde_raw)

# basis_bundle = BasisBuilder(config=basis_config).build_for_spp(features_train)
# fit_result = assembler.fit_spp(features_train, basis_bundle, solver)
# learned_model = LearnedModelFactory().build_spp(
#     fit_result=fit_result,
#     N=features_train.X.shape[2],
#     D=features_train.X.shape[3],
# )

# dist_support = PairwiseDistanceSupport().from_R(features_train.R)
# vel_support = VelocitySupport().from_V(features_train.V)

# dist_density_ref = PairwiseDistanceDensityEstimator().fit_from_R(features_kde.R)
# vel_density_ref = VelocityDensityEstimator().fit_from_V(features_kde.V)

# viz = LearnedFunctionVisualizer()
# fig_repo = FigureRepository()

# fig = viz.plot_kernel_with_support(...)
# fig_repo.save(fig, "stage_2_4b", "kernel_with_support")

# fig = viz.plot_environment_surfaces(...)
# fig_repo.save(fig, "stage_2_4b", "environment_surfaces")