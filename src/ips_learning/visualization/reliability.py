from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ips_learning.visualization.style import apply_paper_style, OKABE_ITO


class ReliabilityVisualizer:
    """
    Stage 2.7A: visualize repeated-trial reliability.
    """

    def __init__(self):
        apply_paper_style()

    def plot_metric_boxplots(
        self,
        df: pd.DataFrame,
        metric_columns: list[str],
        title: str = "Repeated-trial reliability",
        log_scale: bool = True,
    ):
        fig, ax = plt.subplots(figsize=(10.5, 5.4))

        data = [
            pd.to_numeric(df[col], errors="coerce").dropna().values
            for col in metric_columns
        ]

        ax.boxplot(
            data,
            # labels=metric_columns,
            labels = [r"E_{res}", r"E_{\phi}", r"E_{f}", r"E_{rcon}: train", r"E_{pred}: train", r"E_{rcon}: test", r"E_{pred}: test"],
            showmeans=True,
            patch_artist=True,
        )

        if log_scale:
            ax.set_yscale("log")

        ax.set_ylabel("Error value")
        # ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)

        return fig

    def plot_mean_std_bars(
        self,
        summary_df: pd.DataFrame,
        title: str = "Mean ± std over repeated trials",
        log_scale: bool = True,
    ):
        fig, ax = plt.subplots(figsize=(10.5, 5.4))

        x = np.arange(summary_df.shape[0])
        mean = summary_df["mean"].values
        std = summary_df["std"].values

        ax.bar(x, mean, yerr=std, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["metric"], rotation=30, ha="right")

        if log_scale:
            ax.set_yscale("log")

        ax.set_ylabel("Metric value")
        ax.set_title(title)

        return fig