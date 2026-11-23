import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from ips_learning.visualization.style import apply_paper_style, OKABE_ITO


class TrajectoryVisualizer:
    """
    Stage 2.5b: visualize true and learned trajectories before formal evaluation.
    """

    def __init__(self):
        apply_paper_style()

    def plot_true_vs_learned_coordinate(
        self,
        t_true: np.ndarray,
        X_true: np.ndarray,
        t_hat: np.ndarray,
        X_hat: np.ndarray,
        component: int = 0,
        train_fit_end_pt: float | None = None,
        highlight_agent: int = 0,
        title: str = "True vs learned trajectories ",
        input_parameter: str = "x"
    ):
        """
        Parameters
        ----------
        t_true, t_hat : np.ndarray of shape (L,)
        X_true, X_hat : np.ndarray of shape (N, D, L)
        """
        fig, ax = plt.subplots(figsize=(8.6, 4.9))

        for i in range(X_true.shape[0]):
            if i == highlight_agent:
                ax.plot(
                    t_true,
                    X_true[i, component, :],
                    color=OKABE_ITO[4],
                    linewidth=2.3,
                    label=rf"${input_parameter}^{{({component+1})}}_{{i={i+1}}}$",
                )
                ax.plot(
                    t_hat,
                    X_hat[i, component, :],
                    color=OKABE_ITO[1],
                    linestyle="--",
                    linewidth=2.3,
                    label=rf"$\hat{{{input_parameter}}}^{{({component+1})}}_{{i={i+1}}}$",
                )
            else:
                ax.plot(
                    t_true,
                    X_true[i, component, :],
                    color=OKABE_ITO[4],
                    alpha=0.3,
                    linewidth=1.1,
                )
                ax.plot(
                    t_hat,
                    X_hat[i, component, :],
                    color=OKABE_ITO[1],
                    alpha=0.3,
                    linestyle="--",
                    linewidth=1.1,
                )

        if train_fit_end_pt is not None:
            xticks = list(ax.get_xticks())
            # avoid duplicates (important for floating point)
            if not any(abs(x - train_fit_end_pt) < 1e-8 for x in xticks):
                xticks.append(train_fit_end_pt)

            xticks = sorted(xticks)

            labels = []
            for x in xticks:
                if abs(x - train_fit_end_pt) < 1e-8:
                    labels.append(r"$T$")
                else:
                    labels.append(f"{x:.2f}")

            ax.set_xticks(xticks)
            ax.set_xticklabels(labels)

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(rf"${input_parameter}$")
        ax.set_title(title)
        ax.legend(loc="best")
        return fig

    

    def plot_true_vs_learned_phase_portrait(
        self,
        t: np.ndarray,
        X_true: np.ndarray,
        X_hat: np.ndarray,
        train_fit_end_pt: float | None = None,
        train_fit_mark: bool = True,
        title_true: str = "True phase portrait",
        title_hat: str = "Learned phase portrait",
    ):
        """
        Parameters
        ----------
        t : np.ndarray of shape (L,)
        X_true, X_hat : np.ndarray of shape (N, 2, L)
        train_time_end_pt : float or None
            If provided, mark the point in time color bar.
        """
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3), sharex=True, sharey=True)

        norm = Normalize(vmin=t[0], vmax=t[-1])
        cmap = plt.cm.Wistia

        # mask for [t0, T]
        if train_fit_end_pt is not None:
            mask = t <= train_fit_end_pt
        else:
            mask = np.ones_like(t, dtype=bool)


        for i in range(X_true.shape[0]):
            axes[0].scatter(
                X_true[i, 0, :],
                X_true[i, 1, :],
                c=t,
                cmap=cmap,
                norm=norm,
                s=8,
                alpha=0.8,
            )
            axes[1].scatter(
                X_hat[i, 0, :],
                X_hat[i, 1, :],
                c=t,
                cmap=cmap,
                norm=norm,
                s=8,
                alpha=0.8,
            )
            if train_fit_mark:
                axes[1].plot(
                    X_true[i, 0, mask],
                    X_true[i, 1, mask],
                    color="black",
                    linewidth=1.0,
                    alpha=0.7,
                )

        axes[0].set_title(title_true)
        axes[1].set_title(title_hat)


        for ax in axes:
            ax.set_xlabel(r"$x_1$")
        axes[0].set_ylabel(r"$x_2$")

        # sm = ScalarMappable(norm=norm, cmap=cmap)
        # cbar = fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.8, pad=0.03)
        # cbar.set_label(r"$t$")
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.8, pad=0.03)

        # Define the three time points
        ticks = [t[0], train_fit_end_pt, t[-1]]

        cbar.set_ticks(ticks)
        cbar.set_ticklabels([r"$t_0$", r"$T$", r"$T_f$"])

        # cbar.set_label(r"$t$")

        return fig
    
    def plot_true_vs_learned_phase_portrait_scaled(
        self,
        t_hat: np.ndarray,
        X_true: np.ndarray,
        X_hat: np.ndarray,
        t_true: np.ndarray | None = None,
        train_fit_end_pt: float | None = None,
        train_fit_mark: bool = True,
        title_true: str = "True phase portrait",
        title_hat: str = "Learned phase portrait",
    ):
        """
        Plot true and learned phase portraits when they have different
        time intervals and potentially different numbers of agents.

        Parameters
        ----------
        t_hat : np.ndarray, shape (L_hat,)
            Time grid corresponding to X_hat, typically [t0, T_f].

        X_true : np.ndarray, shape (N_true, 2, L_true)
            True trajectories, typically available only on [t0, T].

        X_hat : np.ndarray, shape (N_hat, 2, L_hat)
            Learned trajectories, typically available on [t0, T_f].

        t_true : np.ndarray, shape (L_true,), optional
            Time grid corresponding to X_true. If None, t_hat is used,
            which requires X_true and X_hat to have the same time length.

        train_fit_end_pt : float, optional
            Final training time T.

        train_fit_mark : bool
            Whether to mark the learned trajectory portion on [t0, T].
        """
        t_hat = np.asarray(t_hat, dtype=float).reshape(-1)

        if t_true is None:
            t_true = t_hat
        else:
            t_true = np.asarray(t_true, dtype=float).reshape(-1)

        X_true = np.asarray(X_true, dtype=float)
        X_hat = np.asarray(X_hat, dtype=float)

        # --------------------------------------------------
        # Validate array shapes
        # --------------------------------------------------
        if X_true.ndim != 3 or X_true.shape[1] != 2:
            raise ValueError(
                "X_true must have shape (N_true, 2, L_true)."
            )

        if X_hat.ndim != 3 or X_hat.shape[1] != 2:
            raise ValueError(
                "X_hat must have shape (N_hat, 2, L_hat)."
            )

        if X_true.shape[-1] != len(t_true):
            raise ValueError(
                f"X_true has {X_true.shape[-1]} time points, "
                f"but t_true has length {len(t_true)}."
            )

        if X_hat.shape[-1] != len(t_hat):
            raise ValueError(
                f"X_hat has {X_hat.shape[-1]} time points, "
                f"but t_hat has length {len(t_hat)}."
            )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11.8, 5.3),
            sharex=True,
            sharey=True,
        )

        # --------------------------------------------------
        # Common color scale: [t0, T_f]
        # --------------------------------------------------
        global_t_min = min(t_true[0], t_hat[0])
        global_t_max = max(t_true[-1], t_hat[-1])

        norm = Normalize(vmin=global_t_min, vmax=global_t_max)
        cmap = plt.cm.Wistia

        # --------------------------------------------------
        # True trajectories: N_true agents on [t0, T]
        # --------------------------------------------------
        for i in range(X_true.shape[0]):
            axes[0].scatter(
                X_true[i, 0, :],
                X_true[i, 1, :],
                c=t_true,
                cmap=cmap,
                norm=norm,
                s=8,
                alpha=0.8,
            )

        # --------------------------------------------------
        # Learned trajectories: N_hat agents on [t0, T_f]
        # --------------------------------------------------
        for i in range(X_hat.shape[0]):
            axes[1].scatter(
                X_hat[i, 0, :],
                X_hat[i, 1, :],
                c=t_hat,
                cmap=cmap,
                norm=norm,
                s=8,
                alpha=0.8,
            )

        # --------------------------------------------------
        # Mark the learned trajectory portion on [t0, T]
        # --------------------------------------------------
        if train_fit_mark and train_fit_end_pt is not None:
            hat_train_mask = t_hat <= train_fit_end_pt

            for i in range(X_hat.shape[0]):
                axes[1].plot(
                    X_hat[i, 0, hat_train_mask],
                    X_hat[i, 1, hat_train_mask],
                    color="black",
                    linewidth=1.0,
                    alpha=0.7,
                )

        axes[0].set_title(title_true)
        axes[1].set_title(title_hat)

        for ax in axes:
            ax.set_xlabel(r"$x_1$")

        axes[0].set_ylabel(r"$x_2$")

        # --------------------------------------------------
        # Shared time color bar
        # --------------------------------------------------
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=axes,
            orientation="vertical",
            shrink=0.8,
            pad=0.03,
        )

        if train_fit_end_pt is not None:
            ticks = [global_t_min, train_fit_end_pt, global_t_max]
            ticklabels = [r"$t_0$", r"$T$", r"$T_f$"]
        else:
            ticks = [global_t_min, global_t_max]
            ticklabels = [r"$t_0$", r"$T_f$"]

        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)

        return fig
    
    def plot_true_vs_learned_phase_portrait_2x2(
        self,
        t_short: np.ndarray,
        X_true_short: np.ndarray,
        X_hat_short: np.ndarray,
        t_long: np.ndarray,
        X_true_long: np.ndarray,
        X_hat_long: np.ndarray,
        train_fit_end_pt: float | None = None,
        train_fit_mark: bool = True,
        title_true_short: str = r"True: $[t_0,T]$",
        title_hat_short: str = r"Learned: $[t_0,T]$",
        title_true_long: str = r"True: $[t_0,T_f]$",
        title_hat_long: str = r"Learned: $[t_0,T_f]$",
    ):
        """
        Plot true and learned phase portraits in a 2 x 2 layout.

        Layout
        ------
        Top-left     : true trajectories on [t0, T]
        Top-right    : learned trajectories on [t0, T]
        Bottom-left  : true trajectories on [t0, Tf]
        Bottom-right : learned trajectories on [t0, Tf]

        Parameters
        ----------
        t_short : ndarray, shape (L_short,)
            Time grid for the top-row trajectories.

        X_true_short : ndarray, shape (N_true_short, 2, L_short)
            True short-time trajectories.

        X_hat_short : ndarray, shape (N_hat_short, 2, L_short)
            Learned short-time trajectories.

        t_long : ndarray, shape (L_long,)
            Time grid for the bottom-row trajectories.

        X_true_long : ndarray, shape (N_true_long, 2, L_long)
            True long-time trajectories.

        X_hat_long : ndarray, shape (N_hat_long, 2, L_long)
            Learned long-time trajectories.

        train_fit_end_pt : float, optional
            Training endpoint T.

        train_fit_mark : bool
            If True, mark the portions of the long-time trajectories lying
            inside [t0, T] using black curves.
        """

        # --------------------------------------------------
        # Convert inputs to arrays
        # --------------------------------------------------
        t_short = np.asarray(t_short, dtype=float).reshape(-1)
        t_long = np.asarray(t_long, dtype=float).reshape(-1)

        X_true_short = np.asarray(X_true_short, dtype=float)
        X_hat_short = np.asarray(X_hat_short, dtype=float)
        X_true_long = np.asarray(X_true_long, dtype=float)
        X_hat_long = np.asarray(X_hat_long, dtype=float)

        # --------------------------------------------------
        # Validate trajectory shapes
        # --------------------------------------------------
        trajectory_data = {
            "X_true_short": (X_true_short, t_short),
            "X_hat_short": (X_hat_short, t_short),
            "X_true_long": (X_true_long, t_long),
            "X_hat_long": (X_hat_long, t_long),
        }

        for name, (X, t_data) in trajectory_data.items():
            if X.ndim != 3 or X.shape[1] != 2:
                raise ValueError(
                    f"{name} must have shape (N, 2, L), "
                    f"but received {X.shape}."
                )

            if X.shape[-1] != len(t_data):
                raise ValueError(
                    f"{name} has {X.shape[-1]} time points, "
                    f"but its time array has length {len(t_data)}."
                )

        # --------------------------------------------------
        # Create 2 x 2 figure
        # --------------------------------------------------
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11.8, 10.0),
            sharex=True,
            sharey=True,
        )

        ax_true_short = axes[0, 0]
        ax_hat_short = axes[0, 1]
        ax_true_long = axes[1, 0]
        ax_hat_long = axes[1, 1]

        cmap = plt.cm.Wistia

        # Use one common time normalization for all four panels.
        # Therefore, the same color represents the same physical time
        # in every panel.
        t_min = min(t_short[0], t_long[0])
        t_max = max(t_short[-1], t_long[-1])

        norm = Normalize(vmin=t_min, vmax=t_max)

        scatter_kwargs = {
            "cmap": cmap,
            "norm": norm,
            "s": 8,
            "alpha": 0.8,
        }

        # --------------------------------------------------
        # Top-left: true short-time trajectories
        # --------------------------------------------------
        for i in range(X_true_short.shape[0]):
            ax_true_short.scatter(
                X_true_short[i, 0, :],
                X_true_short[i, 1, :],
                c=t_short,
                **scatter_kwargs,
            )

        # --------------------------------------------------
        # Top-right: learned short-time trajectories
        # --------------------------------------------------
        for i in range(X_hat_short.shape[0]):
            ax_hat_short.scatter(
                X_hat_short[i, 0, :],
                X_hat_short[i, 1, :],
                c=t_short,
                **scatter_kwargs,
            )

        # --------------------------------------------------
        # Bottom-left: true long-time trajectories
        # --------------------------------------------------
        for i in range(X_true_long.shape[0]):
            ax_true_long.scatter(
                X_true_long[i, 0, :],
                X_true_long[i, 1, :],
                c=t_long,
                **scatter_kwargs,
            )

        # --------------------------------------------------
        # Bottom-right: learned long-time trajectories
        # --------------------------------------------------
        for i in range(X_hat_long.shape[0]):
            ax_hat_long.scatter(
                X_hat_long[i, 0, :],
                X_hat_long[i, 1, :],
                c=t_long,
                **scatter_kwargs,
            )

        # --------------------------------------------------
        # Mark [t0, T] on the long-time panels
        # --------------------------------------------------
        # if train_fit_mark and train_fit_end_pt is not None:
        #     long_train_mask = t_long <= train_fit_end_pt

        #     for i in range(X_true_long.shape[0]):
        #         ax_true_long.plot(
        #             X_true_long[i, 0, long_train_mask],
        #             X_true_long[i, 1, long_train_mask],
        #             color="black",
        #             linewidth=1.0,
        #             alpha=0.7,
        #         )

        #     for i in range(X_hat_long.shape[0]):
        #         ax_hat_long.plot(
        #             X_hat_long[i, 0, long_train_mask],
        #             X_hat_long[i, 1, long_train_mask],
        #             color="black",
        #             linewidth=1.0,
        #             alpha=0.7,
        #         )

        # --------------------------------------------------
        # Titles and labels
        # --------------------------------------------------
        ax_true_short.set_title(title_true_short)
        ax_hat_short.set_title(title_hat_short)
        ax_true_long.set_title(title_true_long)
        ax_hat_long.set_title(title_hat_long)

        for ax in axes.flat:
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_aspect("equal", adjustable="box")

        # --------------------------------------------------
        # Shared color bar
        # --------------------------------------------------
       # Reserve space on the right for the color bar
        fig.subplots_adjust(
            left=0.08,
            right=0.86,
            bottom=0.08,
            top=0.93,
            wspace=0.20,
            hspace=0.25,
        )

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        # [left, bottom, width, height] in figure coordinates
        cbar_ax = fig.add_axes([0.89, 0.15, 0.025, 0.70])

        cbar = fig.colorbar(
            sm,
            cax=cbar_ax,
            orientation="vertical",
        )

        if train_fit_end_pt is not None:
            cbar.set_ticks([t_min, train_fit_end_pt, t_max])
            cbar.set_ticklabels([r"$t_0$", r"$T$", r"$T_f$"])
        else:
            cbar.set_ticks([t_min, t_max])
            cbar.set_ticklabels([r"$t_0$", r"$T_f$"])

        cbar.set_label(r"$t$")

        return fig

    @staticmethod
    def _animation_frame_indices(
        t: np.ndarray,
        max_frames: int = 500,
    ) -> np.ndarray:
        """
        Return evenly spaced indices containing at most max_frames points.

        The first and last time points are always retained.
        """
        t = np.asarray(t).reshape(-1)

        if t.size == 0:
            raise ValueError("t must contain at least one time point.")

        if max_frames < 2:
            raise ValueError("max_frames must be at least 2.")

        if t.size <= max_frames:
            return np.arange(t.size, dtype=int)

        indices = np.linspace(
            0,
            t.size - 1,
            num=max_frames,
        )

        return np.unique(np.rint(indices).astype(int))


    @staticmethod
    def _add_fading_trajectory(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        color,
        linestyle: str = "solid",
        alpha_min: float = 0.1,
        alpha_max: float = 0.7,
        linewidth: float = 1.0,
        zorder: int = 2,
    ):
        """
        Add a trajectory with opacity increasing toward the current time.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if x.size < 2:
            return

        points = np.column_stack((x, y))
        segments = np.stack(
            (points[:-1], points[1:]),
            axis=1,
        )

        rgba = np.asarray(to_rgba(color), dtype=float)
        colors = np.tile(rgba, (segments.shape[0], 1))
        colors[:, 3] = np.linspace(
            alpha_min,
            alpha_max,
            segments.shape[0],
        )

        collection = LineCollection(
            segments,
            colors=colors,
            linewidths=linewidth,
            linestyles=linestyle,
            zorder=zorder,
        )

        ax.add_collection(collection)


    def animate_true_vs_learned(
        self,
        t: np.ndarray,
        X_true: np.ndarray,
        V_true: np.ndarray,
        X_hat: np.ndarray,
        V_hat: np.ndarray,
        interval: int = 60,
        max_frames: int = 500,
    ):
        """
        Animate true and learned trajectories in the same panel.

        Parameters
        ----------
        t : np.ndarray of shape (L,)
            Original time points.

        X_true, V_true, X_hat, V_hat : np.ndarray of shape (N, 2, L)
            True and learned positions and velocities.

        interval : int, default=60
            Delay between displayed frames in milliseconds.

        max_frames : int, default=500
            Maximum number of frames used in the animation.

        Returns
        -------
        fig, ani
        """
        # --------------------------------------------------
        # Convert and validate inputs
        # --------------------------------------------------
        t = np.asarray(t).reshape(-1)
        X_true = np.asarray(X_true)
        V_true = np.asarray(V_true)
        X_hat = np.asarray(X_hat)
        V_hat = np.asarray(V_hat)

        expected_shape = X_true.shape

        if X_true.ndim != 3 or X_true.shape[1] != 2:
            raise ValueError(
                "X_true must have shape (N, 2, L). "
                f"Received {X_true.shape}."
            )

        for name, array in {
            "V_true": V_true,
            "X_hat": X_hat,
            "V_hat": V_hat,
        }.items():
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}. "
                    f"Received {array.shape}."
                )

        if X_true.shape[2] != t.size:
            raise ValueError(
                "The number of time points must match the last "
                "dimension of the trajectory arrays."
            )

        # --------------------------------------------------
        # Compute limits using the complete trajectories
        # --------------------------------------------------
        x_min = min(
            X_true[:, 0, :].min(),
            X_hat[:, 0, :].min(),
        )
        x_max = max(
            X_true[:, 0, :].max(),
            X_hat[:, 0, :].max(),
        )
        y_min = min(
            X_true[:, 1, :].min(),
            X_hat[:, 1, :].min(),
        )
        y_max = max(
            X_true[:, 1, :].max(),
            X_hat[:, 1, :].max(),
        )

        x_padding = 0.05 * max(x_max - x_min, 1.0)
        y_padding = 0.05 * max(y_max - y_min, 1.0)

        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # --------------------------------------------------
        # Reduce the animation to at most max_frames points
        # --------------------------------------------------
        frame_indices = self._animation_frame_indices(
            t=t,
            max_frames=max_frames,
        )

        t_anim = t[frame_indices]
        X_true_anim = X_true[:, :, frame_indices]
        V_true_anim = V_true[:, :, frame_indices]
        X_hat_anim = X_hat[:, :, frame_indices]
        V_hat_anim = V_hat[:, :, frame_indices]

        print(
            f"Animation frames: {t_anim.size} "
            f"(original time points: {t.size})"
        )

        # --------------------------------------------------
        # Create figure
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.6, 6.6))
        N = X_true_anim.shape[0]

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=OKABE_ITO[4],
                linewidth=2,
                label=r"$X$",
            ),
            Line2D(
                [0],
                [0],
                color=OKABE_ITO[1],
                linewidth=2,
                linestyle="--",
                label=r"$\hat{X}$",
            ),
        ]

        def configure_axis(frame: int):
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_title(rf"$t={t_anim[frame]:.2f}$")
            ax.legend(
                handles=legend_handles,
                loc="upper right",
            )
            ax.grid(alpha=0.2)

        def update(frame: int):
            ax.clear()
            configure_axis(frame)

            for i in range(N):
                # ------------------------------------------
                # True trajectory history
                # ------------------------------------------
                self._add_fading_trajectory(
                    ax=ax,
                    x=X_true_anim[i, 0, :frame + 1],
                    y=X_true_anim[i, 1, :frame + 1],
                    color=OKABE_ITO[4],
                    linestyle="solid",
                )

                # ------------------------------------------
                # Learned trajectory history
                # ------------------------------------------
                self._add_fading_trajectory(
                    ax=ax,
                    x=X_hat_anim[i, 0, :frame + 1],
                    y=X_hat_anim[i, 1, :frame + 1],
                    color=OKABE_ITO[1],
                    linestyle="dashed",
                )

                # Current true position
                ax.scatter(
                    X_true_anim[i, 0, frame],
                    X_true_anim[i, 1, frame],
                    color=OKABE_ITO[4],
                    s=20,
                    zorder=4,
                )

                # Current learned position
                ax.scatter(
                    X_hat_anim[i, 0, frame],
                    X_hat_anim[i, 1, frame],
                    color=OKABE_ITO[1],
                    s=20,
                    zorder=4,
                )

                # Current true velocity
                ax.arrow(
                    X_true_anim[i, 0, frame],
                    X_true_anim[i, 1, frame],
                    0.05 * V_true_anim[i, 0, frame],
                    0.05 * V_true_anim[i, 1, frame],
                    head_width=0.02,
                    head_length=0.03,
                    fc=OKABE_ITO[4],
                    ec=OKABE_ITO[4],
                    alpha=0.8,
                    length_includes_head=True,
                    zorder=5,
                )

                # Current learned velocity
                ax.arrow(
                    X_hat_anim[i, 0, frame],
                    X_hat_anim[i, 1, frame],
                    0.05 * V_hat_anim[i, 0, frame],
                    0.05 * V_hat_anim[i, 1, frame],
                    head_width=0.02,
                    head_length=0.03,
                    fc=OKABE_ITO[1],
                    ec=OKABE_ITO[1],
                    alpha=0.8,
                    length_includes_head=True,
                    zorder=5,
                )

            return (ax,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=t_anim.size,
            interval=interval,
            blit=False,
            repeat=False,
        )

        fig.tight_layout()

        return fig, ani


    def animate_true_vs_learned2x1(
        self,
        t: np.ndarray,
        X_true: np.ndarray,
        V_true: np.ndarray,
        X_hat: np.ndarray,
        V_hat: np.ndarray,
        interval: int = 60,
        max_frames: int = 500,
    ):
        """
        Animate true and learned trajectories side by side.

        Parameters
        ----------
        t : np.ndarray of shape (L,)
            Original time points.

        X_true, V_true, X_hat, V_hat : np.ndarray of shape (N, 2, L)
            True and learned positions and velocities.

        interval : int, default=60
            Delay between displayed frames in milliseconds.

        max_frames : int, default=500
            Maximum number of frames used in the animation.

        Returns
        -------
        fig, ani
        """
        # --------------------------------------------------
        # Convert and validate inputs
        # --------------------------------------------------
        t = np.asarray(t).reshape(-1)
        X_true = np.asarray(X_true)
        V_true = np.asarray(V_true)
        X_hat = np.asarray(X_hat)
        V_hat = np.asarray(V_hat)

        expected_shape = X_true.shape

        if X_true.ndim != 3 or X_true.shape[1] != 2:
            raise ValueError(
                "X_true must have shape (N, 2, L). "
                f"Received {X_true.shape}."
            )

        for name, array in {
            "V_true": V_true,
            "X_hat": X_hat,
            "V_hat": V_hat,
        }.items():
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}. "
                    f"Received {array.shape}."
                )

        if X_true.shape[2] != t.size:
            raise ValueError(
                "The number of time points must match the last "
                "dimension of the trajectory arrays."
            )

        # --------------------------------------------------
        # Compute common limits from complete trajectories
        # --------------------------------------------------
        x_min = min(
            X_true[:, 0, :].min(),
            X_hat[:, 0, :].min(),
        )
        x_max = max(
            X_true[:, 0, :].max(),
            X_hat[:, 0, :].max(),
        )
        y_min = min(
            X_true[:, 1, :].min(),
            X_hat[:, 1, :].min(),
        )
        y_max = max(
            X_true[:, 1, :].max(),
            X_hat[:, 1, :].max(),
        )

        x_padding = 0.05 * max(x_max - x_min, 1.0)
        y_padding = 0.05 * max(y_max - y_min, 1.0)

        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # --------------------------------------------------
        # Reduce animation to at most max_frames
        # --------------------------------------------------
        frame_indices = self._animation_frame_indices(
            t=t,
            max_frames=max_frames,
        )

        t_anim = t[frame_indices]
        X_true_anim = X_true[:, :, frame_indices]
        V_true_anim = V_true[:, :, frame_indices]
        X_hat_anim = X_hat[:, :, frame_indices]
        V_hat_anim = V_hat[:, :, frame_indices]

        print(
            f"Animation frames: {t_anim.size} "
            f"(original time points: {t.size})"
        )

        # --------------------------------------------------
        # Create 1 x 2 layout
        # --------------------------------------------------
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.0, 5.8),
            sharex=True,
            sharey=True,
        )

        ax_true, ax_hat = axes
        N = X_true_anim.shape[0]

        true_handle = Line2D(
            [0],
            [0],
            color=OKABE_ITO[4],
            linewidth=2,
            label=r"$X$",
        )

        learned_handle = Line2D(
            [0],
            [0],
            color=OKABE_ITO[1],
            linewidth=2,
            linestyle="--",
            label=r"$\hat{X}$",
        )

        def configure_axis(ax, title, legend_handle):
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_title(title)
            ax.legend(
                handles=[legend_handle],
                loc="upper right",
            )
            ax.grid(alpha=0.2)

        suptitle = fig.suptitle("", fontsize=14)

        def update(frame: int):
            ax_true.clear()
            ax_hat.clear()

            configure_axis(
                ax_true,
                rf"True trajectory, $t={t_anim[frame]:.2f}$",
                true_handle,
            )

            configure_axis(
                ax_hat,
                rf"Learned trajectory, $t={t_anim[frame]:.2f}$",
                learned_handle,
            )

            for i in range(N):
                # ------------------------------------------
                # True trajectory
                # ------------------------------------------
                self._add_fading_trajectory(
                    ax=ax_true,
                    x=X_true_anim[i, 0, :frame + 1],
                    y=X_true_anim[i, 1, :frame + 1],
                    color=OKABE_ITO[4],
                    linestyle="solid",
                )

                ax_true.scatter(
                    X_true_anim[i, 0, frame],
                    X_true_anim[i, 1, frame],
                    color=OKABE_ITO[4],
                    s=20,
                    zorder=4,
                )

                ax_true.arrow(
                    X_true_anim[i, 0, frame],
                    X_true_anim[i, 1, frame],
                    0.05 * V_true_anim[i, 0, frame],
                    0.05 * V_true_anim[i, 1, frame],
                    head_width=0.02,
                    head_length=0.03,
                    fc=OKABE_ITO[4],
                    ec=OKABE_ITO[4],
                    alpha=0.8,
                    length_includes_head=True,
                    zorder=5,
                )

                # ------------------------------------------
                # Learned trajectory
                # ------------------------------------------
                self._add_fading_trajectory(
                    ax=ax_hat,
                    x=X_hat_anim[i, 0, :frame + 1],
                    y=X_hat_anim[i, 1, :frame + 1],
                    color=OKABE_ITO[1],
                    linestyle="dashed",
                )

                ax_hat.scatter(
                    X_hat_anim[i, 0, frame],
                    X_hat_anim[i, 1, frame],
                    color=OKABE_ITO[1],
                    s=20,
                    zorder=4,
                )

                ax_hat.arrow(
                    X_hat_anim[i, 0, frame],
                    X_hat_anim[i, 1, frame],
                    0.05 * V_hat_anim[i, 0, frame],
                    0.05 * V_hat_anim[i, 1, frame],
                    head_width=0.02,
                    head_length=0.03,
                    fc=OKABE_ITO[1],
                    ec=OKABE_ITO[1],
                    alpha=0.8,
                    length_includes_head=True,
                    zorder=5,
                )

            suptitle.set_text(
                rf"True versus learned dynamics at "
                rf"$t={t_anim[frame]:.2f}$"
            )

            return tuple(axes)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=t_anim.size,
            interval=interval,
            blit=False,
            repeat=False,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        return fig, ani
    
    def plot_true_parametric_nonparametric_phase_portrait(
        self,
        *,
        t: np.ndarray,
        X_true: np.ndarray,
        X_parametric: np.ndarray,
        X_nonparametric: np.ndarray,
        train_fit_end_pt: float | None = None,
        train_fit_mark: bool = True,
        title_true: str = r"True",
        title_parametric: str = r"Parametric",
        title_nonparametric: str = r"Non-parametric",
        component_pair: tuple[int, int] = (0, 1),
        labels: dict | None = None,
    ):
        """
        Plot 1x3 phase portraits:
            True, parametric learned, non-parametric learned.

        Expected input shape:
            X_true, X_parametric, X_nonparametric : (N, D, L)

        This follows the same plotting style as plot_true_vs_learned_phase_portrait.
        """
        labels = labels or {}

        t = np.asarray(t, dtype=float).reshape(-1)

        X_true = np.asarray(X_true, dtype=float)
        X_parametric = np.asarray(X_parametric, dtype=float)
        X_nonparametric = np.asarray(X_nonparametric, dtype=float)

        if X_true.shape != X_parametric.shape:
            raise ValueError(
                f"X_true and X_parametric must have same shape, got "
                f"{X_true.shape} and {X_parametric.shape}."
            )

        if X_true.shape != X_nonparametric.shape:
            raise ValueError(
                f"X_true and X_nonparametric must have same shape, got "
                f"{X_true.shape} and {X_nonparametric.shape}."
            )

        if X_true.ndim != 3:
            raise ValueError(
                f"Expected arrays with shape (N,D,L), got {X_true.shape}."
            )

        if X_true.shape[-1] != t.shape[0]:
            raise ValueError(
                f"Last axis of X must match len(t). Got X.shape[-1]={X_true.shape[-1]} "
                f"and len(t)={t.shape[0]}."
            )

        c1, c2 = component_pair

        fig, axes = plt.subplots(
            1,
            3,
            figsize=labels.get("figsize", (16.5, 5.3)),
            sharex=True,
            sharey=True,
        )

        norm = Normalize(vmin=t[0], vmax=t[-1])
        cmap = plt.cm.Wistia

        if train_fit_end_pt is not None:
            mask = t <= train_fit_end_pt
        else:
            mask = np.ones_like(t, dtype=bool)

        panels = [
            (axes[0], X_true, title_true),
            (axes[1], X_parametric, title_parametric),
            (axes[2], X_nonparametric, title_nonparametric),
        ]

        for ax, X, title in panels:
            for i in range(X.shape[0]):
                ax.scatter(
                    X[i, c1, :],
                    X[i, c2, :],
                    c=t,
                    cmap=cmap,
                    norm=norm,
                    s=labels.get("scatter_size", 8),
                    alpha=labels.get("scatter_alpha", 0.8),
                )

                if train_fit_mark and ax != axes[0]:
                    ax.plot(
                        X[i, c1, mask],
                        X[i, c2, mask],
                        color="black",
                        linewidth=labels.get("train_fit_linewidth", 1.0),
                        alpha=labels.get("train_fit_alpha", 0.7),
                    )

            ax.set_title(title)
            ax.set_xlabel(labels.get("xlabel", rf"$x_{c1+1}$"))
            ax.grid(True)

        axes[0].set_ylabel(labels.get("ylabel", rf"$x_{c2+1}$"))

        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(
            sm,
            ax=axes,
            orientation="vertical",
            shrink=labels.get("colorbar_shrink", 0.8),
            pad=labels.get("colorbar_pad", 0.03),
        )

        if train_fit_end_pt is not None:
            ticks = [t[0], train_fit_end_pt, t[-1]]
            ticklabels = [r"$t_0$", r"$T$", r"$T_f$"]
        else:
            ticks = [t[0], t[-1]]
            ticklabels = [r"$t_0$", r"$T_f$"]

        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)

        if labels.get("colorbar_label") is not None:
            cbar.set_label(labels["colorbar_label"])

        # fig.tight_layout()

        return fig


# # Minimal example usage:
# repo = DatasetRepository()
# train_raw = repo.load("spp", "train", "dataset_001")

# # Assume learned rollout already exists from Stage 2.5
# # recon_artifact = ...

# m = 0  # representative trajectory

# X_true_single = train_raw.X[m].transpose(1, 2, 0)      # (N, D, L)
# V_true_single = train_raw.V[m].transpose(1, 2, 0)      # (N, D, L)
# X_hat_single = recon_artifact.X_hat[m].transpose(1, 2, 0)  # (N, D, L)
# V_hat_single = recon_artifact.V_hat[m].transpose(1, 2, 0)  # (N, D, L)

# viz = TrajectoryVisualizer()
# fig_repo = FigureRepository()

# fig = viz.plot_true_vs_learned_coordinate(
#     t_true=train_raw.t,
#     X_true=X_true_single,
#     t_hat=recon_artifact.t,
#     X_hat=X_hat_single,
#     component=0,
#     highlight_agent=0,
#     title="Training-horizon trajectory reconstruction",
# )
# fig_repo.save(fig, "stage_2_5b", "true_vs_learned_coordinate")

# fig = viz.plot_point_trajectories_in_plane(
#     X_true=X_true_single,
#     X_hat=X_hat_single,
#     highlight_agent=0,
#     title="True vs learned trajectories in configuration space",
# )
# fig_repo.save(fig, "stage_2_5b", "true_vs_learned_plane")

# fig = viz.plot_true_vs_learned_phase_portrait(
#     t=train_raw.t,
#     X_true=X_true_single,
#     X_hat=X_hat_single,
# )
# fig_repo.save(fig, "stage_2_5b", "true_vs_learned_phase_portrait")