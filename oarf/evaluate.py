"""Leak-free walk-forward evaluation, metrics (Sec 6), and significance tests.

The harness streams a model through ``(X, Z, Y)`` honouring the predict-then-
update protocol, records point forecasts (and quantiles / weight paths where
available), then computes the metric battery of Sec 6.  Held-out interventional
metrics freeze the trained learner and replay its past-only standardiser on the
``do(A := nu)`` grid.  Diebold-Mariano and the Model Confidence Set provide the
significance layer (Sec 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
#  Streaming a model                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class StreamResult:
    name: str
    yhat: np.ndarray
    y: np.ndarray
    resid: np.ndarray
    sq_err: np.ndarray
    quantiles: np.ndarray | None = None      # (T, 3) lo/mid/hi if available
    w_path: np.ndarray | None = None         # (T, d+1) read-out coef path
    extras: dict = field(default_factory=dict)


def run_stream(model, X, Z, Y, record_weights=False, eval_start=0,
               standardize_y=True):
    """Run one model over the stream and collect outputs (Sec 1 protocol).

    When ``standardize_y`` is set, the target is standardised past-only before
    being revealed to the model (predicting standardised returns), so a fixed
    step size is scale-robust across draws.  The frozen target standardiser is
    stored in ``res.extras['std_y']`` so the same transform can be applied to
    the held-out ``do(A := nu)`` grid.
    """
    from .online import OnlineStandardizer
    T = len(Y)
    yhat = np.empty(T)
    y_used = np.empty(T)
    std_y = OnlineStandardizer(1) if standardize_y else None
    quants = np.empty((T, 3)) if getattr(model, "produces_quantiles", False) else None
    w_path = [] if record_weights else None
    for t in range(T):
        x, z, y = X[t], Z[t], Y[t]
        if quants is not None:
            quants[t] = model.predict_quantiles(x, z)
            yhat[t] = quants[t, 1]
        else:
            yhat[t] = model.predict(x, z)
        ys = float(std_y.transform([y])[0]) if standardize_y else float(y)
        y_used[t] = ys
        model.update(x, z, ys)
        if standardize_y:
            std_y.update([y])
        if record_weights:
            try:
                b0, coef = model.effective_linear()
                w_path.append(np.concatenate(([b0], coef)))
            except Exception:
                w_path.append(np.full(model.d + 1, np.nan))
    resid = y_used - yhat
    res = StreamResult(
        name=model.name, yhat=yhat, y=y_used, resid=resid, sq_err=resid ** 2,
        quantiles=quants,
        w_path=np.array(w_path) if record_weights else None,
        extras={"std_y": std_y.freeze() if standardize_y else None},
    )
    return res


# --------------------------------------------------------------------------- #
#  Point / regime metrics (Sec 6)                                             #
# --------------------------------------------------------------------------- #
def mse(sq_err, mask=None):
    return float(np.mean(sq_err if mask is None else sq_err[mask]))


def per_regime_mse(sq_err, regimes):
    out = {}
    for r in np.unique(regimes):
        out[int(r)] = float(np.mean(sq_err[regimes == r]))
    return out


def worst_regime_mse(sq_err, regimes):
    return max(per_regime_mse(sq_err, regimes).values())


def post_changepoint_mse(sq_err, changepoints, k=50):
    T = len(sq_err)
    mask = np.zeros(T, dtype=bool)
    for c in changepoints:
        mask[c:min(c + k, T)] = True
    return float(np.mean(sq_err[mask])) if mask.any() else float("nan")


def point_metrics(res: StreamResult, regimes, changepoints, eval_start=0, k=50):
    sl = slice(eval_start, None)
    se = res.sq_err[sl]
    reg = regimes[sl]
    cp = changepoints[changepoints >= eval_start] - eval_start
    return {
        "MSE": mse(se),
        "RMSE": float(np.sqrt(mse(se))),
        "worst_regime_MSE": worst_regime_mse(se, reg),
        "post_cp_MSE": post_changepoint_mse(se, cp, k),
    }


# --------------------------------------------------------------------------- #
#  Interventional metrics (the headline, Sec 6)                               #
# --------------------------------------------------------------------------- #
def frozen_interventional(model, grid, std_y=None):
    """Worst-case interventional MSE and the robustness curve over the grid.

    The learner is frozen (final read-out + past-only standardiser) and scored
    on each held-out ``do(A := nu)`` environment.  ``std_y`` is the frozen target
    standardiser used in training (``res.extras['std_y']``); the grid target is
    standardised with it so the error is on the same scale as training.  Returns
    the per-radius mean MSE (robustness curve) and the worst-case MSE.
    """
    model.freeze_for_eval()
    radii = grid.radii
    curve = np.empty(len(radii))
    worst = 0.0
    per_env = []
    for i, r in enumerate(radii):
        mses = []
        for j in range(grid.X[i].shape[0]):          # directions
            Xb, Yb = grid.X[i][j], grid.Y[i][j]
            Yb = std_y.transform(Yb.reshape(-1, 1)).reshape(-1) if std_y is not None else Yb
            pred = model.predict_frozen(Xb)
            m = float(np.mean((Yb - pred) ** 2))
            mses.append(m)
            worst = max(worst, m)
        curve[i] = float(np.mean(mses))
        per_env.append(mses)
    return {
        "radii": radii,
        "robustness_curve": curve,
        "worst_do_MSE": worst,
        "per_env": per_env,
    }


# --------------------------------------------------------------------------- #
#  Distributional metrics (Sec 6)                                             #
# --------------------------------------------------------------------------- #
def pinball_loss(y, q, tau):
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


def crps_from_quantiles(y, quants, taus=(0.05, 0.5, 0.95)):
    """CRPS approximated as 2/|T| * mean pinball over the quantile grid."""
    total = 0.0
    for i, tau in enumerate(taus):
        total += 2.0 * pinball_loss(y, quants[:, i], tau)
    return total / len(taus)


def distributional_metrics(res: StreamResult, eval_start=0, alpha=0.1):
    if res.quantiles is None:
        return None
    sl = slice(eval_start, None)
    y = res.y[sl]
    q = res.quantiles[sl]
    lo, mid, hi = q[:, 0], q[:, 1], q[:, 2]
    cov = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    # Winkler / interval score at nominal 1-alpha
    winkler = hi - lo
    winkler = winkler + (2.0 / alpha) * (lo - y) * (y < lo)
    winkler = winkler + (2.0 / alpha) * (y - hi) * (y > hi)
    return {
        "CRPS": crps_from_quantiles(y, q),
        "pinball_0.05": pinball_loss(y, lo, 0.05),
        "pinball_0.5": pinball_loss(y, mid, 0.5),
        "pinball_0.95": pinball_loss(y, hi, 0.95),
        "coverage": cov,
        "coverage_gap": abs((1 - alpha) - cov),
        "interval_width": width,
        "winkler": float(np.mean(winkler)),
    }


# --------------------------------------------------------------------------- #
#  Online-learning diagnostics (Sec 6)                                        #
# --------------------------------------------------------------------------- #
def dynamic_regret_proxy(res: StreamResult, window=100, eval_start=0):
    """Cumulative loss minus a rolling-oracle's loss (rolling mean predictor)."""
    y = res.y
    T = len(y)
    oracle = np.empty(T)
    for t in range(T):
        lo = max(0, t - window)
        oracle[t] = np.mean(y[lo:t]) if t > lo else y[t]
    oracle_se = (y - oracle) ** 2
    sl = slice(eval_start, None)
    return float(np.sum(res.sq_err[sl]) - np.sum(oracle_se[sl]))


def recovery_delay(res: StreamResult, changepoints, eval_start=0, eps_ratio=1.5,
                   horizon=80):
    """Mean #steps for post-shift squared error to fall back near baseline."""
    se = res.sq_err
    delays = []
    for c in changepoints:
        if c < eval_start + 20 or c + horizon >= len(se):
            continue
        base = np.mean(se[max(0, c - 40):c]) + 1e-9
        thr = eps_ratio * base
        seg = se[c:c + horizon]
        below = np.where(seg <= thr)[0]
        delays.append(int(below[0]) if len(below) else horizon)
    return float(np.mean(delays)) if delays else float("nan")


def coef_stability(res: StreamResult, eval_start=0):
    if res.w_path is None:
        return float("nan")
    w = res.w_path[eval_start:]
    return float(np.sum(np.linalg.norm(np.diff(w, axis=0), axis=1)))


# --------------------------------------------------------------------------- #
#  Decision / economic metrics (Sec 6) — frictional allocator                 #
# --------------------------------------------------------------------------- #
def volatility_timing(var_hat, realized_ret, eval_start=0, cost_bps=1.0,
                      target_vol=0.01, lev_cap=3.0, ann=252):
    """Economic value of a volatility forecast via risk targeting (Sec 6, Fig. 10).

    A volatility-timing investor (Fleming, Kirby & Ostdiek, 2001) scales exposure
    inversely to the *forecast* variance, ``w_t = clip(target_vol^2 / var_hat_t,
    0, lev_cap)``, and earns ``w_t * ret_{t+1}`` net of a turnover cost.  A more
    accurate, regime-robust variance forecast keeps realised risk closer to
    target, lifting the annualised Sharpe and cutting drawdowns — so this turns
    forecast quality into a decision-relevant economic payoff.  ``var_hat`` is on
    the *raw* return scale.
    """
    T = len(var_hat)
    w = np.clip(target_vol ** 2 / (var_hat + 1e-12), 0.0, lev_cap)
    pnl = np.zeros(T)
    pnl[:-1] = w[:-1] * realized_ret[1:]
    turn = np.zeros(T); turn[1:] = np.abs(w[1:] - w[:-1])
    pnl_net = pnl - cost_bps * 1e-4 * turn
    sl = slice(eval_start, T - 1)
    p = pnl_net[sl]
    mu, sd = p.mean(), p.std() + 1e-12
    eq = np.cumsum(p)
    dd = np.maximum.accumulate(eq) - eq
    es_a = -np.mean(np.sort(p)[:max(1, int(0.05 * len(p)))])
    return {
        "ann_Sharpe": float(np.sqrt(ann) * mu / sd),
        "max_drawdown": float(np.max(dd)) if len(dd) else float("nan"),
        "ES_5pct": float(es_a),
        "turnover": float(np.sum(turn[sl])),
        "realised_vol_ann": float(np.std(p) * np.sqrt(ann)),
        "equity_curve": eq.tolist(),
    }


# --------------------------------------------------------------------------- #
#  Significance: Diebold-Mariano and Model Confidence Set                     #
# --------------------------------------------------------------------------- #
def diebold_mariano(se_a, se_b, h=1):
    """Two-sided DM test on squared-error loss differentials (HLN-corrected)."""
    d = se_a - se_b
    n = len(d)
    dbar = d.mean()
    # Newey-West long-run variance with bandwidth h-1
    gamma0 = np.mean((d - dbar) ** 2)
    lrv = gamma0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
        lrv += 2.0 * (1 - lag / h) * cov
    dm = dbar / np.sqrt(lrv / n + 1e-18)
    # Harvey-Leybourne-Newbold small-sample correction
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    pval = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(pval)


def dm_matrix(results, eval_start=0):
    """Pairwise DM statistics (positive => row worse than column)."""
    names = [r.name for r in results]
    n = len(names)
    stat = np.full((n, n), np.nan)
    pval = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s, pv = diebold_mariano(results[i].sq_err[eval_start:],
                                    results[j].sq_err[eval_start:])
            stat[i, j], pval[i, j] = s, pv
    return names, stat, pval


def model_confidence_set(results, eval_start=0, alpha=0.1, n_boot=2000, seed=0):
    """Hansen et al. Model Confidence Set via the range statistic + bootstrap.

    Returns the set of models not rejected at level ``alpha`` (the surviving
    "best" set) together with each model's MCS p-value.
    """
    rng = np.random.default_rng(seed)
    losses = np.array([r.sq_err[eval_start:] for r in results])   # (M, T)
    names = [r.name for r in results]
    M, T = losses.shape
    alive = list(range(M))
    mcs_p = {nm: 1.0 for nm in names}
    # block bootstrap indices (stationary blocks)
    block = max(10, int(T ** (1 / 3)))
    boot_idx = []
    for _ in range(n_boot):
        idx = np.empty(T, dtype=int)
        i = 0
        while i < T:
            start = rng.integers(0, T)
            L = min(block, T - i)
            idx[i:i + L] = (start + np.arange(L)) % T
            i += L
        boot_idx.append(idx)
    boot_idx = np.array(boot_idx)

    eliminated_order = []
    while len(alive) > 1:
        sub = losses[alive]                       # (m, T)
        m = len(alive)
        dbar = sub.mean(axis=1)                   # (m,)
        grand = dbar.mean()
        # t-stats of each model vs the set average
        d_i = dbar - grand
        # bootstrap variance of d_i
        boot_means = sub[:, boot_idx].mean(axis=2)        # (m, n_boot)
        var_i = boot_means.var(axis=1) + 1e-12
        t_i = d_i / np.sqrt(var_i)
        T_range = t_i.max()
        # bootstrap null distribution of the range statistic
        bm_centered = boot_means - dbar[:, None]
        t_boot = bm_centered / np.sqrt(var_i)[:, None]
        T_range_boot = t_boot.max(axis=0)
        p = float(np.mean(T_range_boot >= T_range))
        worst_local = int(np.argmax(t_i))
        worst_global = alive[worst_local]
        mcs_p[names[worst_global]] = max(p, mcs_p[names[worst_global]]
                                         if names[worst_global] in
                                         eliminated_order else 0.0)
        mcs_p[names[worst_global]] = p
        if p >= alpha:
            break
        eliminated_order.append(worst_global)
        alive.remove(worst_global)
    survivors = [names[i] for i in alive]
    return {"survivors": survivors, "mcs_pvalues": mcs_p}


# --------------------------------------------------------------------------- #
#  Channel-recovery metric (Sec 2.3 ground truth)                             #
# --------------------------------------------------------------------------- #
def subspace_alignment(B_hat, B_true):
    """Principal-angle alignment in [0,1] between two column subspaces."""
    Qh, _ = np.linalg.qr(B_hat)
    Qt, _ = np.linalg.qr(B_true)
    s = np.linalg.svd(Qh.T @ Qt, compute_uv=False)
    s = np.clip(s, 0, 1)
    return float(np.mean(s))           # mean cosine of principal angles
