from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import pandas as pd
from pypfopt import EfficientFrontier, objective_functions

from ..estimators import estimate_returns, estimate_risk
from ..utils.types import OptimizationResult


def mvo_min_vol(
    prices: pd.DataFrame,
    risk_method: str = "ledoit_wolf",
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> OptimizationResult:
    mu = estimate_returns(prices, method="mean_historical_return")
    S = estimate_risk(prices, method=risk_method)  # type: ignore[arg-type]
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    w = ef.min_volatility()
    perf = _perf(ef)
    return _result(ef, w, perf, meta={"optimizer": "min_vol", "risk_method": risk_method})


def mvo_max_sharpe(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.02,
    risk_method: str = "ledoit_wolf",
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    l2_gamma: Optional[float] = None,
) -> OptimizationResult:
    mu = estimate_returns(prices, method="mean_historical_return")
    S = estimate_risk(prices, method=risk_method)  # type: ignore[arg-type]
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    if l2_gamma is not None:
        ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)
    w = ef.max_sharpe(risk_free_rate=risk_free_rate)
    perf = _perf(ef, risk_free_rate=risk_free_rate)
    return _result(ef, w, perf, meta={"optimizer": "max_sharpe", "risk_method": risk_method, "l2_gamma": l2_gamma})


def mvo_efficient_risk(
    prices: pd.DataFrame,
    target_volatility: float,
    risk_method: str = "ledoit_wolf",
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> OptimizationResult:
    mu = estimate_returns(prices, method="mean_historical_return")
    S = estimate_risk(prices, method=risk_method)  # type: ignore[arg-type]
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    w = ef.efficient_risk(target_volatility)
    perf = _perf(ef)
    return _result(ef, w, perf, meta={"optimizer": "efficient_risk", "target_volatility": target_volatility})


def mvo_market_neutral(
    prices: pd.DataFrame,
    risk_method: str = "ledoit_wolf",
    weight_bounds: tuple[float, float] = (-1.0, 1.0),
) -> OptimizationResult:
    mu = estimate_returns(prices, method="mean_historical_return")
    S = estimate_risk(prices, method=risk_method)  # type: ignore[arg-type]
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    w = ef.efficient_return(target_return=float(mu.mean()), market_neutral=True)
    perf = _perf(ef)
    return _result(ef, w, perf, meta={"optimizer": "market_neutral"})


def _perf(ef: EfficientFrontier, risk_free_rate: float = 0.0) -> Dict[str, float]:
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
    return {"expected_return": float(ret), "volatility": float(vol), "sharpe": float(sharpe)}


def _result(ef: EfficientFrontier, w: Dict[str, float], perf: Dict[str, float], meta: Optional[dict] = None) -> OptimizationResult:
    weights = pd.Series(w).sort_values(ascending=False)
    cleaned = pd.Series(ef.clean_weights()).sort_values(ascending=False)
    return OptimizationResult(weights=weights, cleaned_weights=cleaned, performance=perf, meta=meta)
