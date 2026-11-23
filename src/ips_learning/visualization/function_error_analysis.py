import numpy as np
import matplotlib.pyplot as plt

from ips_learning.visualization.style import apply_paper_style, OKABE_ITO


class FunctionErrorVisualizer:
    def __init__(self):
        apply_paper_style()

    @staticmethod
    def _labels(user_labels: dict | None, defaults: dict) -> dict:
        """
        Merge user label overrides with defaults.
        """
        out = dict(defaults)
        if user_labels is not None:
            out.update(user_labels)
        return out

    # def plot_kernel_error(
    #     self,
    #     r_grid,
    #     phi_true,
    #     phi_hat,
    #     density=None,
    #     title: str | None = None,
    #     labels: dict | None = None,
    # ):
    #     labels = self._labels(
    #         labels,
    #         {
    #             "phi_true": r"$\phi$",
    #             "phi_hat": r"$\hat{\phi}$",
    #             "error": r"$|\phi-\hat{\phi}|^2$",
    #             "density": r"$\rho_R$ (scaled)",
    #             "xlabel": r"$r$",
    #             "ylabel_top": r"$\phi(r)$",
    #             "ylabel_bottom": r"$|\phi-\hat{\phi}|^2$",
    #         },
    #     )

    #     r_grid = np.asarray(r_grid, dtype=float).reshape(-1)
    #     phi_true = np.asarray(phi_true, dtype=float).reshape(-1)
    #     phi_hat = np.asarray(phi_hat, dtype=float).reshape(-1)

    #     fig, axes = plt.subplots(
    #         2,
    #         1,
    #         figsize=(8.0, 6.5),
    #         sharex=True,
    #         gridspec_kw={"height_ratios": [3, 1]},
    #     )

    #     axes[0].plot(
    #         r_grid,
    #         phi_true,
    #         color=OKABE_ITO[4],
    #         linewidth=2.2,
    #         label=labels["phi_true"],
    #     )
    #     axes[0].plot(
    #         r_grid,
    #         phi_hat,
    #         color=OKABE_ITO[1],
    #         linewidth=2.2,
    #         linestyle="--",
    #         label=labels["phi_hat"],
    #     )
    #     axes[0].set_ylabel(labels["ylabel_top"])
    #     axes[0].set_title(title)
    #     axes[0].legend(loc="best")

    #     err = (phi_true - phi_hat) ** 2

    #     axes[1].plot(
    #         r_grid,
    #         err,
    #         color=OKABE_ITO[2],
    #         linewidth=1.8,
    #         label=labels["error"],
    #     )

    #     if density is not None:
    #         density = np.asarray(density, dtype=float).reshape(-1)

    #         if density.shape[0] != r_grid.shape[0]:
    #             raise ValueError(
    #                 f"density and r_grid must have same length, got "
    #                 f"{density.shape[0]} and {r_grid.shape[0]}."
    #             )

    #         density_max = np.max(density)

    #         if density_max > 0:
    #             density_scaled = density / density_max
    #             err_scale = np.max(err) if np.max(err) > 0 else 1.0

    #             axes[1].fill_between(
    #                 r_grid,
    #                 0.0,
    #                 density_scaled * err_scale,
    #                 alpha=0.25,
    #                 label=labels["density"],
    #             )

    #     axes[1].set_xlabel(labels["xlabel"])
    #     axes[1].set_ylabel(labels["ylabel_bottom"])
    #     axes[1].legend(loc="best")

    #     return fig

    def plot_kernel_error(
        self,
        r_grid,
        phi_true,
        phi_hat,
        density=None,
        title: str | None = None,
        labels: dict | None = None,
    ):
        labels = self._labels(
            labels,
            {
                "error": r"$|\phi-\hat{\phi}|^2$",
                "density": r"$\rho_R$ (scaled)",
                "xlabel": r"$r$",
                "ylabel": r"$|\phi-\hat{\phi}|^2$",
            },
        )

        r_grid = np.asarray(r_grid, dtype=float).reshape(-1)
        phi_true = np.asarray(phi_true, dtype=float).reshape(-1)
        phi_hat = np.asarray(phi_hat, dtype=float).reshape(-1)

        if phi_true.shape[0] != r_grid.shape[0]:
            raise ValueError("phi_true and r_grid must have the same length.")

        if phi_hat.shape[0] != r_grid.shape[0]:
            raise ValueError("phi_hat and r_grid must have the same length.")

        fig, ax = plt.subplots(figsize=(8.0, 4))

        err = (phi_true - phi_hat) ** 2

        ax.plot(
            r_grid,
            err,
            color=OKABE_ITO[2],
            linewidth=1.8,
            label=labels["error"],
        )

        if density is not None:
            density = np.asarray(density, dtype=float).reshape(-1)

            if density.shape[0] != r_grid.shape[0]:
                raise ValueError(
                    f"density and r_grid must have same length, got "
                    f"{density.shape[0]} and {r_grid.shape[0]}."
                )

            density_max = np.max(density)

            if density_max > 0:
                density_scaled = density / density_max
                err_scale = np.max(err) if np.max(err) > 0 else 1.0

                ax.fill_between(
                    r_grid,
                    0.0,
                    density_scaled * err_scale,
                    alpha=0.25,
                    color=OKABE_ITO[4],
                    label=labels["density"],
                )

        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])

        if title is not None:
            ax.set_title(title)

        ax.legend(loc="best")

        return fig


    def plot_environment_error_1d(
        self,
        z_grid,
        f_true,
        f_hat,
        density=None,
        title: str | None = None,
        labels: dict | None = None,
    ):
        """
        Plot 1D environment-function error.

        Used for first-order Kuramoto:
            z = x, f: R -> R

        Also works for any scalar environment variable.
        """
        labels = self._labels(
            labels,
            {
                "error": r"$|f-\hat f|^2$",
                "density": r"$\rho_Z$ (scaled)",
                "xlabel": r"$z$",
                "ylabel_bottom": r"$|f-\hat f|^2$",
            },
        )

        z_grid = np.asarray(z_grid, dtype=float).reshape(-1)

        f_true = np.asarray(f_true, dtype=float)
        f_hat = np.asarray(f_hat, dtype=float)

        if f_true.ndim == 1:
            f_true = f_true.reshape(-1, 1)
        if f_hat.ndim == 1:
            f_hat = f_hat.reshape(-1, 1)

        if f_true.shape != f_hat.shape:
            raise ValueError(
                f"f_true and f_hat must have same shape, got "
                f"{f_true.shape} and {f_hat.shape}."
            )

        if f_true.shape[0] != z_grid.shape[0]:
            raise ValueError(
                f"f_true length and z_grid length must match, got "
                f"{f_true.shape[0]} and {z_grid.shape[0]}."
            )

        if f_true.shape[1] != 1:
            raise ValueError(
                f"plot_environment_error_1d expects output dimension 1, got "
                f"{f_true.shape[1]}."
            )

        fig, ax = plt.subplots(figsize=(8.0, 4))


        err = (f_true[:, 0] - f_hat[:, 0]) ** 2

        ax.plot(
            z_grid,
            err,
            color=OKABE_ITO[2],
            linewidth=1.8,
            label=labels["error"],
        )

        if density is not None:
            density = np.asarray(density, dtype=float).reshape(-1)

            if density.shape[0] != z_grid.shape[0]:
                raise ValueError(
                    f"density and z_grid must have same length, got "
                    f"{density.shape[0]} and {z_grid.shape[0]}."
                )

            density_max = np.max(density)

            if density_max > 0:
                density_scaled = density / density_max
                err_scale = np.max(err) if np.max(err) > 0 else 1.0

                ax.fill_between(
                    z_grid,
                    0.0,
                    density_scaled * err_scale,
                    alpha=0.25,
                    label=labels["density"],
                )

        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel_bottom"])
        ax.legend(loc="best")

        return fig

    def plot_environment_error(
        self,
        v1_grid,
        v2_grid,
        f_true,
        f_hat,
        density=None,
        title: str | None = None,
        labels: dict | None = None,
    ):
        """
        Plot 2D environment-function error field.

        Backward-compatible method name.

        For generalized notation, pass labels with z/x/v notation.
        """
        labels = self._labels(
            labels,
            {
                "error_colorbar": r"$|f-\hat f|$",
                "density_contours": r"$\rho_Z$",
                "xlabel": r"$z_1$",
                "ylabel": r"$z_2$",
            },
        )

        v1_grid = np.asarray(v1_grid, dtype=float).reshape(-1)
        v2_grid = np.asarray(v2_grid, dtype=float).reshape(-1)

        f_true = np.asarray(f_true, dtype=float)
        f_hat = np.asarray(f_hat, dtype=float)

        if f_true.shape != f_hat.shape:
            raise ValueError(
                f"f_true and f_hat must have same shape, got "
                f"{f_true.shape} and {f_hat.shape}."
            )

        if f_true.shape[:2] != (len(v1_grid), len(v2_grid)):
            raise ValueError(
                f"Expected f_true grid shape starts with "
                f"({len(v1_grid)}, {len(v2_grid)}), got {f_true.shape}."
            )

        Z1, Z2 = np.meshgrid(v1_grid, v2_grid, indexing="ij")

        err = np.sqrt(np.sum((f_true - f_hat) ** 2, axis=-1))

        fig, ax = plt.subplots(figsize=(8.0, 6.5))

        c = ax.contourf(Z1, Z2, err, levels=30)
        cb = fig.colorbar(c, ax=ax)
        cb.set_label(labels["error_colorbar"])

        if density is not None:
            density = np.asarray(density, dtype=float)

            if density.ndim == 1:
                density = density.reshape(len(v1_grid), len(v2_grid))

            ax.contour(
                Z1,
                Z2,
                density,
                levels=8,
                linewidths=0.8,
                alpha=0.7,
            )

        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.set_title(title)

        return fig