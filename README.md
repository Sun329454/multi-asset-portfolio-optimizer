# Portfolio Optimisation Toolkit

A reusable Python toolkit for portfolio construction and optimisation, refactored from
research notebooks into a clean, modular codebase.

This repository covers **classical and advanced portfolio optimisation methods**, including:

- Mean–Variance Optimisation (MVO)
- Black–Litterman model
- Hierarchical Risk Parity (HRP)
- Custom convex and non-convex objectives
- Empirical comparison of risk and return estimators

The project is designed to be:
- **Research-friendly** (easy to prototype new objectives and constraints)
- **Reusable** (library code separated from demos)
- **Extensible** (new models slot into a clear structure)

---

## Repository Structure

```text
.
├── src/portfolio/              # Reusable library code
│   ├── data/                   # Data loading (yfinance, CSV)
│   ├── estimators.py           # Expected return & risk estimators
│   ├── objectives.py           # Custom optimisation objectives
│   ├── constraints.py          # Portfolio constraints (e.g. sector bounds)
│   ├── optimizers/
│   │   ├── mvo.py              # Mean–Variance optimisation
│   │   ├── black_litterman.py  # Black–Litterman model
│   │   └── hrp.py              # Hierarchical Risk Parity
│   └── utils/
│       ├── types.py            # Standard result objects
│       └── plotting.py         # Lightweight visualisation helpers
│
├── examples/                   # Runnable demo scripts
│   ├── mvo_basic.py
│   ├── advanced_objectives.py
│   ├── black_litterman_demo.py
│   ├── hrp_demo.py
│   └── risk_return_models.py
│
├── tests/                      # Minimal tests / smoke checks
├── pyproject.toml              # Dependencies & packaging
└── README.md
```

---

## Core Concepts

### 1. Separation of Concerns

The codebase deliberately separates:

- **Library code** (`src/portfolio/`):
  - No plotting by default
  - No hard-coded data paths
  - Functions return structured results

- **Examples** (`examples/`):
  - Demonstrate how to use the library
  - Perform plotting and printing
  - Equivalent to former research notebooks

This makes it easy to reuse the optimisation logic in new projects, papers, or backtests.

---

### 2. Unified Optimisation Output

All optimisers return a common result object:

```python
OptimizationResult(
    weights: pd.Series,
    cleaned_weights: Optional[pd.Series],
    performance: dict,
    meta: Optional[dict],
)
```

This ensures:
- Consistent downstream analysis
- Easy comparison across models
- Cleaner experiment code

---

## Implemented Methods

### Mean–Variance Optimisation (MVO)

Located in `optimizers/mvo.py`:

- Minimum volatility
- Maximum Sharpe ratio (with optional L2 regularisation)
- Efficient risk (target volatility)
- Market-neutral portfolios
- Flexible bounds and constraints

Risk and return estimation methods include:
- Historical mean / EMA / CAPM returns
- Sample covariance
- Ledoit–Wolf shrinkage
- Exponential covariance
- Semi-covariance

---

### Black–Litterman Model

Located in `optimizers/black_litterman.py`:

- Market-implied priors
- Absolute views with optional confidence (Idzorek method)
- Posterior return and covariance estimation
- Portfolio optimisation on posterior beliefs

The design cleanly separates:
- Market data
- Investor views
- Optimisation step

---

### Hierarchical Risk Parity (HRP)

Located in `optimizers/hrp.py`:

- Hierarchical clustering–based allocation
- Robust to estimation error
- Optional discrete allocation step

---

### Advanced / Custom Objectives

Located in `objectives.py` and `examples/advanced_objectives.py`:

- Custom convex objectives (e.g. log-barrier terms)
- Non-convex objectives solved via `scipy.optimize`
- Risk-parity-style deviation objectives

These are intended for **research and experimentation**, not production guarantees.

---

## Risk & Return Model Evaluation

Located in `risk_return_eval.py`:

- Train/test split of price history
- Empirical comparison of:
  - Risk models (via covariance error)
  - Return models (via forecast error)

Useful for:
- Model selection
- Methodological research
- Understanding estimator behaviour out-of-sample

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

---

## Running Examples

```bash
python examples/mvo_basic.py
python examples/advanced_objectives.py
python examples/black_litterman_demo.py
python examples/hrp_demo.py

# Risk/return estimator comparison from CSV data
python examples/risk_return_models.py --prices-csv data/stock_prices.csv
```

Examples use `yfinance` by default and require network access.

---

## Design Philosophy

This repository intentionally avoids:
- Notebook-specific code (`get_ipython`, inline installs)
- Hard-coded paths or assumptions
- Mixing visualisation with optimisation logic

Instead, it aims to provide:
- Clear abstractions
- Minimal but expressive APIs
- A solid base for further research or production work

---

## Disclaimer

This project is for **research and educational purposes**.
It is not financial advice, nor a production-ready trading system.

---

## Possible Extensions

- Factor-based optimisation
- ESG or regulatory constraints
- Robust optimisation techniques
- Transaction cost and turnover control
- Backtesting and simulation layer
