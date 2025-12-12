from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class OptimizationResult:
    """Standardised output for all optimisers."""

    weights: pd.Series
    performance: Dict[str, float]
    cleaned_weights: Optional[pd.Series] = None
    meta: Optional[Dict[str, Any]] = None
