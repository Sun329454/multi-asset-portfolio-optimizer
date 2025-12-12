from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models


def split_past_future(prices: pd.DataFrame, n_future: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if n_future <= 0 or n_future >= len(prices):
        raise ValueError("n_future must be >0 and < len(prices)")
    past = prices.iloc[:-n_future]
    future = prices.iloc[-n_future:]
    return past, future


def evaluate_risk_models(
    past_prices: pd.DataFrame,
    future_prices: pd.DataFrame,
    methods: Iterable[str] = ("sample_cov", "ledoit_wolf", "exp_cov"),
) -> pd.DataFrame:
    """Compare risk model covariance estimates against realised future cov."""
    future_cov = risk_models.sample_cov(future_prices)
    rows = []
    for m in methods:
        if m == "sample_cov":
            S = risk_models.sample_cov(past_prices)
        elif m == "ledoit_wolf":
            S = risk_models.CovarianceShrinkage(past_prices).ledoit_wolf()
        elif m == "exp_cov":
            S = risk_models.exp_cov(past_prices)
        else:
            raise ValueError(f"Unknown method: {m}")

        # Frobenius norm error
        err = np.linalg.norm((S - future_cov).values, ord="fro")
        rows.append({"method": m, "fro_error": float(err)})
    return pd.DataFrame(rows).sort_values("fro_error")


def evaluate_return_models(
    past_prices: pd.DataFrame,
    future_prices: pd.DataFrame,
    methods: Iterable[str] = ("mean_historical_return", "ema_historical_return", "capm_return"),
) -> pd.DataFrame:
    """Compare return model estimates to realised future returns (simple)."""
    realised = expected_returns.mean_historical_return(future_prices)
    rows = []
    for m in methods:
        if m == "mean_historical_return":
            mu = expected_returns.mean_historical_return(past_prices)
        elif m == "ema_historical_return":
            mu = expected_returns.ema_historical_return(past_prices)
        elif m == "capm_return":
            mu = expected_returns.capm_return(past_prices)
        else:
            raise ValueError(f"Unknown method: {m}")

        mae = (mu - realised).abs().mean()
        rows.append({"method": m, "mae": float(mae)})
    return pd.DataFrame(rows).sort_values("mae")
