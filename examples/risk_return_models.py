from __future__ import annotations

import argparse

from portfolio.data import load_prices_csv
from portfolio.risk_return_eval import split_past_future, evaluate_risk_models, evaluate_return_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-csv", required=True, help="Path to a CSV with date column + price columns")
    ap.add_argument("--n-future", type=int, default=60)
    args = ap.parse_args()

    prices = load_prices_csv(args.prices_csv)
    past, future = split_past_future(prices, n_future=args.n_future)

    risk_df = evaluate_risk_models(past, future)
    ret_df = evaluate_return_models(past, future)

    print("== Risk model comparison (lower is better) ==")
    print(risk_df.to_string(index=False))

    print("\n== Return model comparison (lower is better) ==")
    print(ret_df.to_string(index=False))


if __name__ == "__main__":
    main()
