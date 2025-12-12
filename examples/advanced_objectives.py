from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from portfolio.data import download_prices
from portfolio.estimators import estimate_risk, estimate_returns
from portfolio.objectives import deviation_risk_parity_numpy
from portfolio.optimizers.mvo import mvo_max_sharpe


def nonconvex_risk_parity_demo():
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "JPM"]
    prices = download_prices(tickers, start="2020-01-01", end="2025-01-01")
    S = estimate_risk(prices, method="ledoit_wolf")

    n = len(tickers)
    x0 = np.ones(n) / n

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    res = minimize(lambda w: deviation_risk_parity_numpy(w, S), x0=x0, bounds=bounds, constraints=cons)
    w = res.x
    print("== Nonconvex risk parity (scipy) ==")
    for t, ww in sorted(zip(tickers, w), key=lambda x: -x[1]):
        print(f"{t}: {ww:.4f}")
    print("success:", res.success, "fun:", res.fun)


def convex_mvo_demo():
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "JPM"]
    prices = download_prices(tickers, start="2020-01-01", end="2025-01-01")
    res = mvo_max_sharpe(prices, risk_free_rate=0.02, l2_gamma=0.1)
    print("== Convex max-sharpe ==")
    print(res.cleaned_weights)
    print(res.performance)


if __name__ == "__main__":
    convex_mvo_demo()
    nonconvex_risk_parity_demo()
