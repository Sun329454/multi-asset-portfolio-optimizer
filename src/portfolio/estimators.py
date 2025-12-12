from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from pypfopt import expected_returns, risk_models


ReturnMethod = Literal["mean_historical_return", "ema_historical_return", "capm_return"]
RiskMethod = Literal[
    "sample_cov",
    "semicovariance",
    "exp_cov",
    "ledoit_wolf",
    "oracle_approximating",
]


def estimate_returns(prices: pd.DataFrame, method: ReturnMethod = "mean_historical_return") -> pd.Series:
    """Estimate expected returns from price history."""
    if method == "mean_historical_return":
        return expected_returns.mean_historical_return(prices)
    if method == "ema_historical_return":
        return expected_returns.ema_historical_return(prices)
    if method == "capm_return":
        return expected_returns.capm_return(prices)
    raise ValueError(f"Unknown return method: {method}")


def estimate_risk(prices: pd.DataFrame, method: RiskMethod = "ledoit_wolf") -> pd.DataFrame:
    """Estimate risk model (covariance-like) from price history."""
    if method == "sample_cov":
        return risk_models.sample_cov(prices)
    if method == "semicovariance":
        return risk_models.semicovariance(prices)
    if method == "exp_cov":
        return risk_models.exp_cov(prices)
    if method == "ledoit_wolf":
        return risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    if method == "oracle_approximating":
        return risk_models.CovarianceShrinkage(prices).oracle_approximating()
    raise ValueError(f"Unknown risk method: {method}")
