from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import cvxpy as cp


def sector_constraints(
    w: cp.Variable,
    tickers: List[str],
    sector_mapper: Dict[str, str],
    sector_lower: Dict[str, float] | None = None,
    sector_upper: Dict[str, float] | None = None,
) -> List[cp.Constraint]:
    """Create sector exposure constraints for an EfficientFrontier weight variable.

    Parameters
    ----------
    w:
        cvxpy Variable (n_assets,)
    tickers:
        asset tickers aligned with w
    sector_mapper:
        dict[ticker] -> sector name
    sector_lower/sector_upper:
        dict[sector] -> bound (fraction of portfolio)
    """
    sector_lower = sector_lower or {}
    sector_upper = sector_upper or {}

    # build sector -> indices
    sector_to_idx: Dict[str, List[int]] = {}
    for i, t in enumerate(tickers):
        sec = sector_mapper.get(t)
        if sec is None:
            continue
        sector_to_idx.setdefault(sec, []).append(i)

    cons: List[cp.Constraint] = []
    for sec, idxs in sector_to_idx.items():
        expr = cp.sum(w[idxs])
        if sec in sector_lower:
            cons.append(expr >= sector_lower[sec])
        if sec in sector_upper:
            cons.append(expr <= sector_upper[sec])
    return cons
