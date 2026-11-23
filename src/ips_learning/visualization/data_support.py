from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ips_learning.visualization.style import apply_paper_style, OKABE_ITO

class DataSupportVisualizer:
    """
    Stage 2.2b visualizer.

    This module visualizes two distinct quantities:
    1. observed support from the training data
    2. reference KDE from the large-M dataset

    It does not fit KDEs internally.

    """

    def __init__(self):
        apply_paper_style()

    def plot_pairwise_distance_support(
        self,
        basis_bundle,
        r_grid: np.ndarray,
        reference_pdf: np.ndarray,
        title: str| None = None,# "Pairwise-distance support",
        reference_label: str = r"$\rho_R:M=2000$",
        train_density: np.ndarray | None = None,  #empirical density from training samples
        train_label: str = r"$\hat{\rho}_{R}$",
        xlabel: str = r"$r$",
        ylabel: str = r"$\rho_R(r)$",
    ):
        
        r_grid = np.asarray(r_grid, dtype=float).reshape(-1)
        reference_pdf = np.asarray(reference_pdf, dtype=float).reshape(-1)

        fig, ax = plt.subplots(figsize=(8.4, 4.9))

        if train_density is not None:
            train_density = np.asarray(train_density, dtype=float).reshape(-1)

            ax.plot(
                r_grid,
                train_density,
                color=OKABE_ITO[5],
                label=train_label,
            )
            ax.fill_between(
                r_grid,
                train_density,
                alpha=0.25,                  # transparency
                color=OKABE_ITO[5],
            )

        ax.plot(
            r_grid,
            reference_pdf,
            color=OKABE_ITO[4],
            linewidth=2.2,
            label=reference_label,
        )

        ax.fill_between(
            r_grid,
            reference_pdf,
            alpha=0.2,
            color=OKABE_ITO[4],
        )

        ax.set_xlim(basis_bundle.localization_metadata["r_min"], basis_bundle.localization_metadata["r_max"])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")
        return fig

    def plot_environment_support_density(
        self,
        z1_grid: np.ndarray,
        z2_grid: np.ndarray | None = None,
        density_grid: np.ndarray | None = None,
        empirical_density_grid: np.ndarray | None = None,
        title: str | None = None,
        density_label: str = r"$\rho_Z(z)$ : $M=2000$",
        empirical_density_label: str = r"$\hat{\rho}_Z(z)$",
        z_label: str = r"$z$",
        ylabel: str = r"$\rho_Z(z)$",
    ):
        """
        Plot environment support and density.

        First-order:
            Z = X, usually D=1 for Kuramoto.

        Second-order:
            Z = V, often D=2 for Levi/SPP.
        """
        z1_grid = np.asarray(z1_grid, dtype=float).reshape(-1)

        if z2_grid is None:
            return self.plot_environment_support_density_1d(
                z_grid=z1_grid,
                density=density_grid,
                empirical_density=empirical_density_grid,
                title=title,
                density_label=density_label,
                empirical_density_label=empirical_density_label,
                xlabel=z_label,
                ylabel=ylabel,
            )

        return self.plot_environment_support_density_2d(
            z1_grid=z1_grid,
            z2_grid=z2_grid,
            density_grid=density_grid,
            empirical_density_grid=empirical_density_grid,
            title=title,
            density_label=density_label,
            empirical_density_label=empirical_density_label,
    )
    
    def plot_environment_support_density_1d(
        self,
        z_grid: np.ndarray,
        density: np.ndarray | None = None,
        empirical_density: np.ndarray | None = None,
        title: str | None = None,
        density_label: str = r"$\rho_Z(z)$ : $M=2000$",
        empirical_density_label: str = r"$\hat{\rho}_Z(z)$",
        xlabel: str = r"$z$",
        ylabel: str = r"$\rho_Z(z)$",
    ):
        z_grid = np.asarray(z_grid, dtype=float).reshape(-1)

        fig, ax = plt.subplots(figsize=(8.4, 4.9))

        if density is not None:
            density = np.asarray(density, dtype=float).reshape(-1)
            if density.shape[0] != z_grid.shape[0]:
                raise ValueError("density and z_grid must have the same length.")

            ax.plot(z_grid, density, color=OKABE_ITO[4], linewidth=2.2, label=density_label)
            ax.fill_between(z_grid, density, alpha=0.20, color=OKABE_ITO[4])

        if empirical_density is not None:
            empirical_density = np.asarray(empirical_density, dtype=float).reshape(-1)
            if empirical_density.shape[0] != z_grid.shape[0]:
                raise ValueError("empirical_density and z_grid must have the same length.")

            ax.plot(
                z_grid,
                empirical_density,
                color=OKABE_ITO[3],
                linewidth=2.0,
                linestyle="--",
                label=empirical_density_label,
            )
            ax.fill_between(z_grid, empirical_density, alpha=0.18, color=OKABE_ITO[3])

        ax.set_xlim(float(np.min(z_grid)), float(np.max(z_grid)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")

        return fig
    
    def plot_environment_support_density_2d(
        self,
        z1_grid: np.ndarray,
        z2_grid: np.ndarray,
        density_grid: np.ndarray | None = None,
        empirical_density_grid: np.ndarray | None = None,
        title: str | None = None,
        density_label: str = r"$\rho_Z(z)$ : $M=2000$",
        empirical_density_label: str = r"$\hat{\rho}_Z(z)$",
        xlabel: str = r"$v_1$",
        ylabel: str = r"$v_2$",
    ):
        z1_grid = np.asarray(z1_grid, dtype=float).reshape(-1)
        z2_grid = np.asarray(z2_grid, dtype=float).reshape(-1)

        def _reshape_density(A, name):
            if A is None:
                return None

            A = np.asarray(A, dtype=float)

            if A.ndim == 1:
                A = A.reshape(len(z1_grid), len(z2_grid))

            expected = (len(z1_grid), len(z2_grid))
            if A.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {A.shape}.")

            return A

        density_grid = _reshape_density(density_grid, "density_grid")
        empirical_density_grid = _reshape_density(
            empirical_density_grid,
            "empirical_density_grid",
        )

        if density_grid is None and empirical_density_grid is None:
            raise ValueError(
                "At least one of density_grid or empirical_density_grid must be provided."
            )

        Z1, Z2 = np.meshgrid(z1_grid, z2_grid, indexing="ij")

        all_arrays = [
            A for A in (density_grid, empirical_density_grid)
            if A is not None
        ]

        vmin = min(np.nanmin(A) for A in all_arrays)
        vmax = max(np.nanmax(A) for A in all_arrays)

        levels = np.linspace(vmin, vmax, 30)
        cmap = "viridis"

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.0, 5.2),
            constrained_layout=True,
        )

        contour_ref = None

        if density_grid is not None:
            contour_ref = axes[0].contourf(
                Z1,
                Z2,
                density_grid,
                levels=levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axes[0].set_title(density_label)

        if empirical_density_grid is not None:
            contour_ref = axes[1].contourf(
                Z1,
                Z2,
                empirical_density_grid,
                levels=levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axes[1].set_title(empirical_density_label)

        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect("equal", adjustable="box")

        if title is not None:
            fig.suptitle(title)

        cbar = fig.colorbar(
            contour_ref,
            ax=axes,
            shrink=0.92,
            pad=0.03,
        )
        cbar.set_label("density")

        return fig
    

    def plot_radial_basis_support(
        self,
        basis_bundle,
    ):
        rb = basis_bundle.radial_basis
        r_grid = np.linspace(rb.r_min, rb.r_max, 500).reshape(-1, 1)
        Phi = rb.evaluate(r_grid)

        fig, ax = plt.subplots(figsize=(5.8 , 4.4))
        for k in range(Phi.shape[1]):
            ax.plot(r_grid[:, 0], Phi[:, k], linewidth=1.5)

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$\psi_k(r)$")
        ax.set_title(r"$(\psi_k)_{k=1}^{n_{\phi}}$")
        return fig
    
    def plot_environment_basis_support(
        self,
        basis_bundle,
        variable_label: str = "z",
    ):
        """
        Plot environment basis support.

        First-order:
            environment variable z = x.

        Second-order:
            environment variable z = v.
        """
        eb = basis_bundle.environment_basis
        D = len(eb.evaluators)

        fig, axes = plt.subplots(1, D, figsize=(5.8 * D, 4.4), squeeze=False)
        axes = axes[0]

        for d in range(D):
            grid = np.linspace(eb.domain_min[d], eb.domain_max[d], 120)
            Phi = eb.evaluators[d](grid)

            ax = axes[d]

            for k in range(Phi.shape[1]):
                ax.plot(grid, Phi[:, k], linewidth=1.5)

            ax.set_xlabel(rf"${variable_label}_{{{d+1}}}$" if D > 1 else rf"${variable_label}$")
            ax.set_ylabel(rf"$h_k({variable_label})$")
            ax.set_title(rf"d={d+1}" if D > 1 else r"$(h_k)_{k=1}^{n_{f}}$")

        if D > 1:
            fig.suptitle(r"$(h_k)_{k=1}^{n_{f}}$")

        return fig
    
    