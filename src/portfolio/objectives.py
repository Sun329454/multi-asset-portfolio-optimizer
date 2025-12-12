from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import cvxpy as cp


def log_barrier(w: cp.Expression, cov_matrix: pd.DataFrame, k: float = 0.1) -> cp.Expression:
    """Log-barrier style objective used as an example in advanced MVO demos.

    This returns an additional objective term; you add it to an EfficientFrontier
    objective via `ef.add_objective(...)`.
    """
    x = w.T @ cov_matrix.values @ w
    return -k * cp.log(x)


def deviation_risk_parity_numpy(
    w: np.ndarray,
    cov_matrix: pd.DataFrame,
) -> float:
    """A simple risk parity deviation objective (numpy), useful for scipy-based nonconvex demos."""
    cov = cov_matrix.values
    port_var = float(w.T @ cov @ w)
    if port_var <= 0:
        return 1e6
    sigma = np.sqrt(port_var)
    mrc = cov @ w / sigma
    rc = w * mrc
    target = np.mean(rc)
    return float(np.sum((rc - target) ** 2))
