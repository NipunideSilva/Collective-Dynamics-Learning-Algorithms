from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ips_learning.visualization.style import apply_paper_style


class MLHeatmapVisualizer:
    """
    Stage 2.7B: visualize errors and condition numbers over (M,L).
    """

    def __init__(self):
        apply_paper_style()

    def pivot_metric(
        self,
        df: pd.DataFrame,
        metric: str,
        M_col: str = "M",
        L_col: str = "L",
    ) -> pd.DataFrame:
        """
        Return matrix with rows=M and columns=L.
        """
        if metric not in df.columns:
            raise KeyError(f"Metric '{metric}' not found in dataframe.")

        table = df.pivot(index=M_col, columns=L_col, values=metric)
        table = table.sort_index(axis=0).sort_index(axis=1)
        return table

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        metric: str,
        title: str,
        colorbar_label: str,
        log10: bool = True,
        failure_col: str | None = "rollout_failed",
        cmap: str = "viridis",
        missing_color: str = "lightgray",
    ):
        """
        Plot a heatmap over M,L.

        Parameters
        ----------
        df : pd.DataFrame
            Sweep metric table with columns M, L, metric.
        metric : str
            Metric column to visualize.
        title : str
            Plot title.
        colorbar_label : str
            Colorbar label.
        log10 : bool
            If True, plot log10(metric). Nonpositive values become NaN.
        failure_col : str or None
            Optional column marking failed rollout cases.
        cmap : str
            Matplotlib colormap.
        missing_color : str
            Color for NaN/failure entries.
        """
        table = self.pivot_metric(df, metric)
        Z = table.to_numpy(dtype=float)

        if log10:
            Z = np.where(Z > 0, np.log10(Z), np.nan)

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color=missing_color)

        fig, ax = plt.subplots(figsize=(7.5, 5.8))

        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            cmap=cmap_obj,
        )

        M_values = table.index.to_numpy()
        L_values = table.columns.to_numpy()

        ax.set_xticks(np.arange(len(L_values)))
        ax.set_yticks(np.arange(len(M_values)))

        ax.set_xticklabels([str(int(v)) for v in L_values])
        ax.set_yticklabels([str(int(v)) for v in M_values])

        ax.set_xlabel(r"$L$ (number of time samples)")
        ax.set_ylabel(r"$M$ (number of trajectories)")
        if title:
            ax.set_title(title)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)

        # Mark failed rollout cells, if present.
        if failure_col is not None and failure_col in df.columns:
            fail_table = self.pivot_metric(df, failure_col)
            fail_Z = fail_table.to_numpy()
            for i in range(fail_Z.shape[0]):
                for j in range(fail_Z.shape[1]):
                    if bool(fail_Z[i, j]):
                        ax.text(
                            j,
                            i,
                            "×",
                            ha="center",
                            va="center",
                            color="red",
                            fontsize=14,
                            fontweight="bold",
                        )

        return fig

    def plot_heatmap_panel(
        self,
        df: pd.DataFrame,
        metrics: list[str],
        titles: list[str],
        colorbar_labels: list[str],
        log10_flags: list[bool],
        failure_col: str | None = "rollout_failed",
        cmap: str = "viridis",
    ):
        """
        Plot several heatmaps in one row.
        """
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.2), constrained_layout=True)

        if n == 1:
            axes = [axes]

        for ax, metric, title, cbar_label, use_log in zip(
            axes, metrics, titles, colorbar_labels, log10_flags
        ):
            table = self.pivot_metric(df, metric)
            Z = table.to_numpy(dtype=float)

            if use_log:
                Z = np.where(Z > 0, np.log10(Z), np.nan)

            cmap_obj = plt.get_cmap(cmap).copy()
            cmap_obj.set_bad(color="lightgray")

            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                cmap=cmap_obj,
            )

            M_values = table.index.to_numpy()
            L_values = table.columns.to_numpy()

            ax.set_xticks(np.arange(len(L_values)))
            ax.set_yticks(np.arange(len(M_values)))

            ax.set_xticklabels([str(int(v)) for v in L_values])
            ax.set_yticklabels([str(int(v)) for v in M_values])

            ax.set_xlabel(r"$L$")
            ax.set_ylabel(r"$M$")
            if title:
                ax.set_title(title)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)

            if failure_col is not None and failure_col in df.columns:
                fail_table = self.pivot_metric(df, failure_col)
                fail_Z = fail_table.to_numpy()
                for i in range(fail_Z.shape[0]):
                    for j in range(fail_Z.shape[1]):
                        if bool(fail_Z[i, j]):
                            ax.text(
                                j,
                                i,
                                "×",
                                ha="center",
                                va="center",
                                color="red",
                                fontsize=14,
                                fontweight="bold",
                            )

        return fig

    def save_metric_table_as_latex(
        self,
        df: pd.DataFrame,
        metric: str,
        output_path: str | Path,
        log10: bool = True,
        float_format: str = "%.3f",
    ) -> Path:
        """
        Save pivoted M,L table as LaTeX.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        table = self.pivot_metric(df, metric)

        if log10:
            table = table.applymap(lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan)

        latex = table.to_latex(
            float_format=float_format,
            na_rep="--",
            caption=rf"\((M,L)\) sweep table for {metric}.",
            label=f"tab:stage27b_{metric}",
            escape=False,
        )

        output_path.write_text(latex, encoding="utf-8")
        return output_path