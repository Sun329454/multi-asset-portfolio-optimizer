from __future__ import annotations

from portfolio.data import download_prices
from portfolio.optimizers.mvo import mvo_max_sharpe, mvo_min_vol
from portfolio.utils.plotting import plot_weights_pie


def main():
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "JPM", "XOM", "UNH"]
    prices = download_prices(tickers, start="2020-01-01", end="2025-01-01")

    res1 = mvo_min_vol(prices)
    print("== Min Vol ==")
    print(res1.cleaned_weights.head(10))
    print(res1.performance)
    plot_weights_pie(res1.cleaned_weights, "Min Vol Weights", min_weight=0.01)

    res2 = mvo_max_sharpe(prices, risk_free_rate=0.02, l2_gamma=0.1)
    print("\n== Max Sharpe (L2) ==")
    print(res2.cleaned_weights.head(10))
    print(res2.performance)
    plot_weights_pie(res2.cleaned_weights, "Max Sharpe Weights", min_weight=0.01)


if __name__ == "__main__":
    main()
