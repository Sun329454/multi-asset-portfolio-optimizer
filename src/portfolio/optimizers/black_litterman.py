from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from pypfopt import black_litterman, risk_models, EfficientFrontier
from pypfopt.black_litterman import BlackLittermanModel

from ..utils.types import OptimizationResult


def build_market_prior(
    prices: pd.DataFrame,
    market_prices: pd.Series,
    market_caps: Dict[str, float],
    risk_aversion: Optional[float] = None,
    cov_method: str = "ledoit_wolf",
) -> tuple[pd.Series, pd.DataFrame, pd.Series, float]:
    """Compute (pi, S, market_weights, delta)."""
    if cov_method == "ledoit_wolf":
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    else:
        S = risk_models.sample_cov(prices)

    mcaps = pd.Series(market_caps).reindex(prices.columns)
    if mcaps.isna().any():
        missing = mcaps[mcaps.isna()].index.tolist()
        raise ValueError(f"Missing market cap(s) for: {missing}")
    w_mkt = mcaps / mcaps.sum()

    delta = risk_aversion if risk_aversion is not None else black_litterman.market_implied_risk_aversion(market_prices)
    pi = black_litterman.market_implied_prior_returns(w_mkt, delta, S)
    return pi, S, w_mkt, float(delta)


def posterior_from_views(
    pi: pd.Series,
    S: pd.DataFrame,
    views: Dict[str, float],
    confidences: Optional[Dict[str, float]] = None,
) -> BlackLittermanModel:
    """Create a Black-Litterman model from absolute views.

    views: dict[ticker] -> view return
    confidences: dict[ticker] -> confidence in [0,1] (optional)
    """
    if confidences:
        conf = pd.Series(confidences).reindex(pi.index).dropna()
        omega = BlackLittermanModel.default_omega(S, pi)  # baseline
        # Idzorek omega based on confidence for tickers present in views:
        omega = black_litterman.idzorek_method(S, pd.Series(views), conf)
        bl = BlackLittermanModel(S, pi=pi, absolute_views=pd.Series(views), omega=omega)
    else:
        bl = BlackLittermanModel(S, pi=pi, absolute_views=pd.Series(views))
    return bl


def optimise_posterior(
    bl: BlackLittermanModel,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    risk_free_rate: float = 0.02,
) -> OptimizationResult:
    mu = bl.bl_returns()
    S = bl.bl_cov()
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    w = ef.max_sharpe(risk_free_rate=risk_free_rate)
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
    return OptimizationResult(
        weights=pd.Series(w).sort_values(ascending=False),
        cleaned_weights=pd.Series(ef.clean_weights()).sort_values(ascending=False),
        performance={"expected_return": float(ret), "volatility": float(vol), "sharpe": float(sharpe)},
        meta={"optimizer": "bl_max_sharpe"},
    )
