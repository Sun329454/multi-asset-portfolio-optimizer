from __future__ import annotations

import pandas as pd
from pypfopt import HRPOpt, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from ..utils.types import OptimizationResult


def hrp_optimize(prices: pd.DataFrame) -> OptimizationResult:
    rets = expected_returns.returns_from_prices(prices)
    hrp = HRPOpt(rets)
    w = hrp.optimize()
    perf = hrp.portfolio_performance(verbose=False)
    return OptimizationResult(
        weights=pd.Series(w).sort_values(ascending=False),
        performance={"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe": float(perf[2])},
        meta={"optimizer": "hrp"},
    )


def allocate_discrete(weights: pd.Series, prices: pd.DataFrame, total_portfolio_value: float = 10000.0):
    latest = get_latest_prices(prices)
    da = DiscreteAllocation(weights.to_dict(), latest, total_portfolio_value=total_portfolio_value)
    return da.greedy_portfolio()
