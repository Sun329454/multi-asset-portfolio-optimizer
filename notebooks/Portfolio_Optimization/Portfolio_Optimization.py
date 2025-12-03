import cvxpy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load daily returns
daily_returns = pd.read_csv('multiasset_daily_returns.csv', index_col=0, parse_dates=True)

# compute annualized mean returns
mean_daily_returns = daily_returns.mean()
annual_returns = mean_daily_returns * 252.0  # assumption: 252 trading days

# calculate covariance
cov_matrix = daily_returns.cov()
annual_cov_matrix = cov_matrix * 252.0  # assumption: 252 trading days

# convert to NumPy arrays for optimization
mu = annual_returns.values.astype(float)
sigma = annual_cov_matrix.values.astype(float)

# number of assets
n = len(mu)

# --- Leverage settings (modify this value to change leverage) ---
# Set leverage_ratio here (1.0 == 100% gross exposure).
leverage_ratio = 1.0  # <-- change this single value to adjust leverage
leverage = cp.Parameter(nonneg=True, value=float(leverage_ratio))

# helper: ensure sigma is symmetric PSD (project to nearest PSD)
def make_psd(mat, eps=1e-8):
    mat = np.array(mat, dtype=float)
    mat = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(mat)
    vals_clipped = np.clip(vals, eps, None)
    return (vecs @ np.diag(vals_clipped) @ vecs.T)

# helper: try solving a CVXPY problem with multiple solvers (fallback)
import importlib

def get_available_solvers(preferred_order=("osqp", "ecos", "scs")):
    """Return a list of cvxpy solver classes that are available in the environment in preferred order."""
    mapping = {
        "osqp": getattr(cp, "OSQP", None),
        "ecos": getattr(cp, "ECOS", None),
        "scs": getattr(cp, "SCS", None),
    }
    available = []
    for name in preferred_order:
        try:
            # check if python package exists
            if importlib.util.find_spec(name) is not None and mapping.get(name) is not None:
                available.append(mapping[name])
        except Exception:
            # ignore errors in detection
            continue
    # Fallback: if none detected, still try SCS (CVXPY bundles a python SCS wrapper in some installs)
    if not available:
        if getattr(cp, "SCS", None) is not None:
            available.append(cp.SCS)
    return available


def solve_with_fallback(prob, solvers=None, verbose=False):
    """Try a list of cvxpy solver classes (in order). If solvers is None, auto-detect installed solvers.
    Returns the prob.status after attempting solvers.
    """
    if solvers is None:
        solvers = get_available_solvers()
    if not solvers:
        print("No external solvers detected; trying CVXPY's default solver (may fail).")
        try:
            prob.solve(verbose=verbose)
            return prob.status
        except Exception as e:
            print("Default solve failed:", e)
            return prob.status

    last_status = None
    last_exception = None
    for solver in solvers:
        try:
            prob.solve(solver=solver, verbose=verbose)
            last_status = prob.status
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return prob.status
            # otherwise continue to next solver
        except Exception as e:
            last_exception = e
            # don't spam; record and continue
            continue
    if last_exception is not None:
        print("Solver fallback: last exception:", last_exception)
    return last_status or prob.status

# Basic data checks
if np.isnan(mu).any() or np.isnan(sigma).any():
    raise ValueError("Input returns contain NaN. Clean your data before optimization.")

# Ensure sigma is PSD (required for quad_form stability)
min_eig = np.linalg.eigvalsh(sigma).min()
if min_eig < -1e-10:
    print(f"Covariance matrix min eigenvalue = {min_eig:.3e}; projecting to nearest PSD.")
    sigma = make_psd(sigma)
else:
    # small regularization for numerical stability
    sigma = sigma + 1e-10 * np.eye(n)

# define optimization variables using positive/negative parts
pos = cp.Variable(n, nonneg=True)   # long exposure per asset
neg = cp.Variable(n, nonneg=True)   # short exposure per asset (absolute)
w = pos - neg

# define target return (kept as Parameter to allow interactive change if desired)
ret_target_value = 0.10  # 10% annual return
ret_target = cp.Parameter(value=float(ret_target_value))

# objective -> minimize portfolio variance
portfolio_variance = cp.quad_form(w, sigma)
objective = cp.Minimize(portfolio_variance)

# constraints as requested
constraints = [cp.sum(w) == 0,
               cp.sum(pos) + cp.sum(neg) == leverage,
               mu @ w >= ret_target]

# formulate problem
problem = cp.Problem(objective, constraints)

# try solving with solver fallback
print("Solving main problem (hard constraints) with solver fallback...")
status = solve_with_fallback(problem, solvers=None, verbose=False)
print("Main problem status:", status)

# if infeasible or no solution, try slack relaxation
if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
    print("Hard problem not optimal/feasible. Trying slack relaxation (soften return constraint)...")
    # slack variable to relax the return constraint (allow mu@w + slack >= ret_target)
    slack = cp.Variable(nonneg=True)
    penalty = 1e6  # large penalty to discourage violating ret_target
    # rebuild problem with slack and penalty in objective
    obj_relax = cp.Minimize(portfolio_variance + penalty * slack)
    constraints_relax = [cp.sum(w) == 0,
                         cp.sum(pos) + cp.sum(neg) == leverage,
                         mu @ w + slack >= ret_target]
    prob_relax = cp.Problem(obj_relax, constraints_relax)
    status_relax = solve_with_fallback(prob_relax, solvers=None, verbose=False)
    print("Relaxed problem status:", status_relax)
    if w.value is None and prob_relax.solution is not None:
        # sometimes variable values are stored under different attr; try accessing variables
        pass
    # report slack if available
    try:
        print("Slack value:", float(slack.value))
    except Exception:
        pass
    # use prob_relax result if available
    if w.value is None and prob_relax.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        # assign to main problem's w via values
        optimal_weights = (pos.value - neg.value)
    elif w.value is not None:
        optimal_weights = w.value
    else:
        optimal_weights = None
else:
    optimal_weights = w.value

# output results
if optimal_weights is not None:
    portfolio_return = float(np.dot(mu, optimal_weights))
    portfolio_volatility = float(np.sqrt(np.dot(optimal_weights.T, np.dot(sigma, optimal_weights))))
    gross = float(np.sum(np.abs(optimal_weights)))
    print(f"Expected Portfolio Return: {portfolio_return:.6f}")
    print(f"Expected Portfolio Volatility: {portfolio_volatility:.6f}")
    print(f"Gross exposure (sum abs): {gross:.6f}")

    # plot portfolio weights
    plt.figure(figsize=(10, 6))
    plt.bar(daily_returns.columns, optimal_weights)
    plt.title("Optimal Portfolio Weights")
    plt.ylabel("Weight")
    plt.xticks(rotation=45)
    plt.show()
else:
    print("No feasible solution found even after relaxation. Consider lowering ret_target or changing leverage_ratio.")

# generate efficient frontier (using solver fallback and PSD-corrected sigma)
# We reuse the same leverage settings for each target return
target_returns = np.linspace(0.02, 0.12, 20)

portfolio_vols = []
portfolio_returns = []

for r in target_returns:
    P = cp.Variable(n, nonneg=True)
    N = cp.Variable(n, nonneg=True)
    W = P - N
    obj = cp.Minimize(cp.quad_form(W, sigma))
    constr = [cp.sum(W) == 0,
              cp.sum(P) + cp.sum(N) == leverage,
              mu @ W >= r]
    prob = cp.Problem(obj, constr)
    # solve with fallback
    st = solve_with_fallback(prob, solvers=None, verbose=False)
    if W.value is not None:
        portfolio_vols.append(float(np.sqrt(np.dot(W.value.T, np.dot(sigma, W.value)))))
        portfolio_returns.append(float(np.dot(mu, W.value)))
    else:
        portfolio_vols.append(np.nan)
        portfolio_returns.append(np.nan)

portfolio_vols = np.array(portfolio_vols)
portfolio_returns = np.array(portfolio_returns)
valid = ~np.isnan(portfolio_vols)

# plot efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(portfolio_vols[valid], portfolio_returns[valid], 'o-')
plt.xlabel("Portfolio Volatility")
plt.ylabel("Portfolio Returns")
plt.title("Efficient Frontier")
plt.show()
