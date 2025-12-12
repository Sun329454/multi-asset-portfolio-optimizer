from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


def download_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download adjusted close prices via yfinance.

    Returns a DataFrame indexed by date with columns = tickers.
    """
    if yf is None:
        raise ImportError("yfinance is required for download_prices()")

    df = yf.download(list(tickers), start=start, end=end, auto_adjust=auto_adjust, progress=False)
    # yfinance returns multi-index columns when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        else:
            df = df.xs(df.columns.levels[0][0], axis=1, level=0)
    return df.dropna(how="all")


def get_market_caps(tickers: Iterable[str]) -> Dict[str, float]:
    """Fetch market caps via yfinance. Network required.

    Returns dict[ticker] = marketCap (float).
    """
    if yf is None:
        raise ImportError("yfinance is required for get_market_caps()")

    caps: Dict[str, float] = {}
    for t in tickers:
        info = yf.Ticker(t).info
        cap = info.get("marketCap")
        if cap is None:
            raise ValueError(f"Missing marketCap for ticker: {t}")
        caps[t] = float(cap)
    return caps


def load_prices_csv(path: str, date_col: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV of prices into a standard format.

    The CSV is expected to have a date column (auto-detected if possible) and
    one column per ticker.
    """
    df = pd.read_csv(path)
    if date_col is None:
        # common names
        for c in ("date", "Date", "DATE", "timestamp", "Timestamp"):
            if c in df.columns:
                date_col = c
                break
    if date_col is None:
        raise ValueError("Could not detect date column. Pass date_col explicitly.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(how="all")
