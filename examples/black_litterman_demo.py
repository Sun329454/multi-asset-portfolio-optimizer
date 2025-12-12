from __future__ import annotations

from portfolio.data import download_prices, get_market_caps
from portfolio.optimizers.black_litterman import build_market_prior, posterior_from_views, optimise_posterior


def main():
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "JPM"]
    prices = download_prices(tickers, start="2020-01-01", end="2025-01-01")
    market = download_prices(["SPY"], start="2020-01-01", end="2025-01-01")["SPY"]

    mcaps = get_market_caps(tickers)
    pi, S, w_mkt, delta = build_market_prior(prices, market, mcaps)

    # Example absolute views: "AAPL 12% annualised, NVDA 18%..."
    views = {"AAPL": 0.12, "NVDA": 0.18}
    confidences = {"AAPL": 0.6, "NVDA": 0.7}

    bl = posterior_from_views(pi, S, views=views, confidences=confidences)
    res = optimise_posterior(bl, risk_free_rate=0.02)

    print("== Black-Litterman Max Sharpe ==")
    print(res.cleaned_weights)
    print(res.performance)


if __name__ == "__main__":
    main()
