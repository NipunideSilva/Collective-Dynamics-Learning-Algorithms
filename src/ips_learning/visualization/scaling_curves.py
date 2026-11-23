from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ips_learning.visualization.style import apply_paper_style


class ScalingCurveVisualizer:

    def __init__(self):
        apply_paper_style()

    def plot_mean_std_curve(
        self,
        df_summary: pd.DataFrame,
        x_col: str,
        mean_col: str,
        std_col: str,
        title: str,
        xlabel: str,
        ylabel: str,
        logx: bool = True,
        logy: bool = True,
    ):
        fig, ax = plt.subplots(figsize=(7.5,5.2))

        x = df_summary[x_col].values
        y = df_summary[mean_col].values
        s = df_summary[std_col].values

        ax.plot(x, y, marker="o", linewidth=2.0)
        ax.fill_between(x, y-s, y+s, alpha=0.25)

        if logx:
            ax.set_xscale("log")

        if logy:
            ax.set_yscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        return fig