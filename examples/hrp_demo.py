from __future__ import annotations

from portfolio.data import download_prices
from portfolio.optimizers.hrp import hrp_optimize
from portfolio.utils.plotting import plot_weights_pie


def main():
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "JPM", "XOM", "UNH"]
    prices = download_prices(tickers, start="2020-01-01", end="2025-01-01")

    res = hrp_optimize(prices)
    print("== HRP ==")
    print(res.weights)
    print(res.performance)
    plot_weights_pie(res.weights, "HRP Weights", min_weight=0.01)


if __name__ == "__main__":
    main()
