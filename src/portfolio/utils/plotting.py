from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_covariance_matrix(cov: pd.DataFrame, title: str = "Covariance matrix") -> None:
    """Simple heatmap for a covariance matrix."""
    fig, ax = plt.subplots()
    im = ax.imshow(cov.values)
    ax.set_title(title)
    ax.set_xticks(range(len(cov.columns)))
    ax.set_yticks(range(len(cov.index)))
    ax.set_xticklabels(cov.columns, rotation=90)
    ax.set_yticklabels(cov.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_weights_pie(weights: pd.Series, title: str = "Weights", min_weight: float = 0.0) -> None:
    """Pie chart for weights."""
    w = weights[weights.abs() > min_weight].sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.pie(w.values, labels=w.index, autopct="%1.1f%%")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
