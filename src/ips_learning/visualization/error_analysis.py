import numpy as np
import matplotlib.pyplot as plt

from ips_learning.visualization.style import apply_paper_style, OKABE_ITO


class ErrorAnalysisVisualizer:
    """
    Stage 2.6: visualize quantitative error-analysis outputs.
    """

    def __init__(self):
        apply_paper_style()

    def plot_error_over_time(
        self,
        t_recon: np.ndarray,
        recon_mean: np.ndarray,
        recon_std: np.ndarray | None = None,
        t_train_full: np.ndarray | None = None,
        train_full_mean: np.ndarray | None = None,
        train_full_std: np.ndarray | None = None,
        t_test: np.ndarray | None = None,
        test_mean: np.ndarray | None = None,
        test_std: np.ndarray | None = None,
        train_horizon_end: float | None = None,
        title: str | None = None, #r"Trajectory error over time",
    ):
        fig, ax = plt.subplots(figsize=(8.8, 5.2))

        t_recon = np.asarray(t_recon, dtype=float)
        recon_mean = np.asarray(recon_mean, dtype=float)

        ax.plot(
            t_recon,
            recon_mean,
            color=OKABE_ITO[4],
            linewidth=2.2,
            label=r"$\bar{e}_{recon}^{train}$",
        )
        if recon_std is not None:
            recon_std = np.asarray(recon_std, dtype=float)
            ax.fill_between(
                t_recon,
                recon_mean - recon_std,
                recon_mean + recon_std,
                color=OKABE_ITO[4],
                alpha=0.18,
            )

        if t_train_full is not None and train_full_mean is not None:
            t_train_full = np.asarray(t_train_full, dtype=float)
            train_full_mean = np.asarray(train_full_mean, dtype=float)
            ax.plot(
                t_train_full,
                train_full_mean,
                color=OKABE_ITO[1],
                linewidth=2.2,
                linestyle="--",
                label=r"$\bar{e}^{train}$",
            )
            if train_full_std is not None:
                train_full_std = np.asarray(train_full_std, dtype=float)
                ax.fill_between(
                    t_train_full,
                    train_full_mean - train_full_std,
                    train_full_mean + train_full_std,
                    color=OKABE_ITO[1],
                    alpha=0.18,
                )

        if t_test is not None and test_mean is not None:
            t_test = np.asarray(t_test, dtype=float)
            test_mean = np.asarray(test_mean, dtype=float)
            ax.plot(
                t_test,
                test_mean,
                color=OKABE_ITO[2],
                linewidth=2.2,
                linestyle="-.",
                label=r"$\bar{e}^{test}$",
            )
            if test_std is not None:
                test_std = np.asarray(test_std, dtype=float)
                ax.fill_between(
                    t_test,
                    test_mean - test_std,
                    test_mean + test_std,
                    color=OKABE_ITO[2],
                    alpha=0.18,
                )

        # avoid duplicates (important for floating point)
        # ------------------------------------------------------------
        # Custom x-ticks: t0, T, Tf
        # ------------------------------------------------------------
        if train_horizon_end is not None:
            # choose the time array that defines the full plotted horizon
            if t_test is not None:
                t_axis = t_test
            elif t_train_full is not None:
                t_axis = t_train_full
            else:
                t_axis = t_recon

            t0 = float(t_axis[0])
            Tf = float(t_axis[-1])
            T = float(train_horizon_end)

            xticks = [t0, T, Tf]

            # remove duplicates while preserving order
            clean_ticks = []
            clean_labels = []

            for tick, label in zip(xticks, [r"$t_0$", r"$T$", r"$T_f$"]):
                if not any(abs(tick - old_tick) < 1e-8 for old_tick in clean_ticks):
                    clean_ticks.append(tick)
                    clean_labels.append(label)

            ax.set_xticks(clean_ticks)
            ax.set_xticklabels(clean_labels)

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$e(t)$")
        if title is not None:
            ax.set_title(title)
        ax.legend(loc="best")
        return fig