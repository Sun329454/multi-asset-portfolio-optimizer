# Multi-Asset Portfolio Optimizer

The **multi-asset-portfolio-optimizer** repository provides a collection of notebooks for portfolio optimization using different strategies, including traditional and advanced methods. It contains both a **Mean-Variance Portfolio (MVP)** version and an **Upgraded version** with advanced optimization techniques. This repository helps in optimizing multi-asset portfolios, evaluating various strategies, and backtesting the models.

## Directory Structure
```
multi-asset-portfolio-optimizer/
│
├── README.md
│
├── notebooks (Upgrade for more optimizers)/
│ ├── data/
│ │ └── spy_prices.csv
│ ├── Advanced_Mean_Variance_Optimisation.py
│ ├── Black_Litterman_Model.py
│ ├── Hierarchical_Risk_Parity.py
│ └── Risk_Return_Model.py
│
└── notebooks (for only MVP)/
├── data/
│ ├── multiasset_benchmark.csv
│ ├── multiasset_closing_prices.csv
│ ├── multiasset_cum_returns.csv
│ ├── multiasset_daily_returns.csv
│ └── multiasset_ewc.csv
├── reports/
│ └── README.md
├── Backtesting.py
├── DataAcquisition_&_Preprocessing.py
├── Extensions_Dashboard.py
├── Final_Model.py
└── Portfolio_Optimization.py
```

## Repository Overview

This repository is divided into two main sections based on the complexity and functionality:

### 1. **Mean-Variance Portfolio (MVP)**

The MVP implementation follows Harry Markowitz's mean-variance optimization theory to construct an optimal portfolio based on the expected return and risk (variance). This section contains the following key components:

- **DataAcquisition_&_Preprocessing.ipynb**: A notebook for acquiring and preprocessing financial data.
- **Portfolio_Optimization.ipynb**: The core notebook for applying the mean-variance optimization to build the portfolio.
- **Final_Model.ipynb**: The final implementation of the MVP model.
- **Backtesting.ipynb**: A notebook for backtesting the performance of the portfolio.
- **Extensions_Dashboard.ipynb**: A notebook to extend the dashboard functionality for better visualization and analysis.
- **data/**: Contains multiple CSV files for multi-asset data, including benchmark data, closing prices, cumulative returns, daily returns, and economic weightings (EWC).
- **reports/**: Contains a `README.md` for reporting purposes related to MVP.

### 2. **Upgraded Optimization Models**

This section includes more sophisticated optimization techniques and models. The following notebooks provide advanced strategies for optimizing multi-asset portfolios:

- **Advanced_Mean_Variance_Optimisation.ipynb**: A notebook for advanced mean-variance optimization.
- **Black_Litterman_Model.ipynb**: A notebook implementing the Black-Litterman model.
- **Hierarchical_Risk_Parity.ipynb**: A notebook for implementing the hierarchical risk parity approach.
- **Risk_Return_Model.ipynb**: A notebook focusing on risk-return modeling.
- **data/**: Contains `spy_prices.csv` with SPY price data for the optimization models.

## Usage

1. Navigate to the notebooks (for only MVP) directory to begin with the mean-variance portfolio optimization and backtesting.
2. Explore the advanced optimization models in the notebooks (Upgrade for more optimizers) directory for more complex methods.

## Notes
- Each section includes its own data directory with CSV files tailored for the specific optimization method.
- The repository includes no traditional configuration files like .gitignore, requirements.txt, or config.yaml, so make sure to manually manage dependencies if needed.

## Contributions
Feel free to fork the repository and contribute to improving the optimization models. Pull requests and suggestions are welcome.
