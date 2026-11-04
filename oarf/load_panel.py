"""Real-data panel loader and feature construction (Sec 3-4).

Aligns the public sources (FRED targets/drivers, GPR, EPU, optional EUA) to a
business-day calendar, builds one-step-ahead **base-expert forecasts** for the
forecast-combination experiment (Sec 4), assembles the observable
context/regime-driver matrix ``Z``, and derives regime labels and changepoints
from an observable macro-financial stress index (so the regime-aware metrics of
Sec 6 apply to the real series without any look-ahead).

The loader is robust to missing sources: any series that failed to download is
skipped, and the EUA target is used only if ``data/eua.csv`` exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA = "data"


# --------------------------------------------------------------------------- #
#  Raw series loaders                                                         #
# --------------------------------------------------------------------------- #
def _fred(sid):
    path = os.path.join(DATA, f"{sid}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.rename(sid).dropna()


def _gpr():
    path = os.path.join(DATA, "gpr_daily.xls")
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path)
    s = pd.to_numeric(df["GPRD"], errors="coerce")
    s.index = pd.to_datetime(df["date"])
    return s.rename("GPR").dropna()


def _epu():
    path = os.path.join(DATA, "epu_daily.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    idx = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day),
                         errors="coerce")
    s = pd.to_numeric(df["daily_policy_index"], errors="coerce")
    s.index = idx
    return s.rename("EPU").dropna()


def _eua():
    path = os.path.join(DATA, "eua.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    price_col = next((c for c in df.columns
                      if any(k in c.lower() for k in ("price", "close", "settle"))),
                     df.columns[-1])
    s = pd.to_numeric(df[price_col].astype(str).str.replace(",", ""),
                      errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.rename("EUA").dropna().sort_index()


# --------------------------------------------------------------------------- #
#  Panel assembly                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    dates: pd.DatetimeIndex
    target_name: str
    y: np.ndarray              # (T,)   one-step-ahead target return
    experts: np.ndarray        # (T, K) base-expert one-step forecasts
    expert_names: list
    Z: np.ndarray              # (T, p) observable context / regime drivers
    z_names: list
    regimes: np.ndarray        # (T,)   regime id from the stress index
    changepoints: np.ndarray   # (K,)   stress-regime transition indices
    raw_price: np.ndarray
    ret: np.ndarray            # (T,)   underlying log-returns (for vol-timing)
    target_type: str = "logvar"
    anchor_cols: tuple = ()    # indices of the driver-level anchor channel in Z


def _align(series_list, cap=5):
    """Outer-join on a business-day index; forward-fill with a small cap."""
    df = pd.concat(series_list, axis=1, sort=True)
    bidx = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(bidx).ffill(limit=cap)
    return df


def _ewma_vol(r, lam=0.94):
    v = np.zeros(len(r))
    v[0] = r[0] ** 2
    for t in range(1, len(r)):
        v[t] = lam * v[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(v)


def _garch11_onestep(r):
    """Cheap recursive GARCH(1,1)-style one-step vol forecast (fixed params)."""
    omega, a, b = 1e-6, 0.08, 0.90
    s2 = np.zeros(len(r))
    s2[0] = np.var(r[:20]) if len(r) > 20 else r[0] ** 2 + 1e-8
    for t in range(1, len(r)):
        s2[t] = omega + a * r[t - 1] ** 2 + b * s2[t - 1]
    return np.sqrt(s2)


def _lag(a, k=1):
    out = np.zeros_like(a, dtype=float)
    out[k:] = a[:-k]
    return out


def _build_experts_return(y, Zlevels):
    """One-step-ahead *return* base experts (Sec 4): AR(1), AR(5), EWMA,
    GARCH mean-reversion, momentum, ridge-on-lags+drivers.  All causal."""
    T = len(y)
    experts, names = [], []
    experts.append(_lag(y, 1)); names.append("AR1")
    ar5 = sum(_lag(y, k) for k in range(1, 6))
    experts.append(ar5 / 5.0); names.append("AR5")
    ew = np.zeros(T); lam = 0.8
    for t in range(1, T):
        ew[t] = lam * ew[t - 1] + (1 - lam) * y[t - 1]
    experts.append(ew); names.append("EWMA")
    vol = _garch11_onestep(y)
    experts.append(-0.1 * _lag(y, 1) / (vol + 1e-6)); names.append("GARCHrev")
    mom = sum(_lag(y, k) for k in range(1, 11))
    experts.append(0.05 * np.sign(mom)); names.append("MOM10")
    experts.append(_ridge_expert(y, Zlevels)); names.append("RIDGE")
    return np.column_stack(experts), names


def _build_experts_vol(logrv, ret, vix_log, Zlevels):
    """One-step-ahead *log realised-variance* experts (Sec 4).

    Realised volatility is genuinely predictable (volatility clustering) and
    strongly regime-dependent, which is the natural showcase for regime-robust
    combination.  Experts: the three HAR components (Corsi, 2009) — daily /
    weekly / monthly average log-RV — an EWMA (RiskMetrics) log-vol, a
    GARCH(1,1) log-variance, the (lagged) implied-vol proxy ``log VIX``, and a
    causal ridge on HAR + drivers.  Every expert at ``t`` uses information up to
    ``t-1`` (lagged), so the protocol is leak-free.
    """
    T = len(logrv)
    experts, names = [], []
    experts.append(_lag(logrv, 1)); names.append("HAR-d")          # daily
    har_w = np.array([np.mean(logrv[max(0, t - 5):t]) if t > 0 else logrv[0]
                      for t in range(T)])
    experts.append(har_w); names.append("HAR-w")                   # weekly
    har_m = np.array([np.mean(logrv[max(0, t - 22):t]) if t > 0 else logrv[0]
                      for t in range(T)])
    experts.append(har_m); names.append("HAR-m")                   # monthly
    ew = np.zeros(T); lam = 0.94
    for t in range(1, T):
        ew[t] = lam * ew[t - 1] + (1 - lam) * ret[t - 1] ** 2
    experts.append(np.log(ew + 1e-8)); names.append("EWMA-RM")     # RiskMetrics
    g = _garch11_onestep(ret)
    experts.append(np.log(_lag(g, 1) ** 2 + 1e-8)); names.append("GARCH")
    if vix_log is not None:
        experts.append(_lag(vix_log, 1)); names.append("VIX")      # implied vol
    experts.append(_ridge_expert(logrv, Zlevels)); names.append("RIDGE")
    return np.column_stack(experts), names


def _build_experts_weekly(logrv_w, ewma_w, garch_w, vix_w, Zlevels):
    """One-week-ahead log realised-variance experts on non-overlapping weeks.

    Every expert for week ``w`` uses only data up to the end of week ``w-1``:
    last week's log RV (persistence), its 4- and 13-week means (HAR-style),
    the daily EWMA and GARCH variance at the end of week ``w-1`` scaled to a
    five-day horizon, the log VIX at the end of week ``w-1``, and a rolling
    ridge regression on lags and drivers.
    """
    T = len(logrv_w)
    experts, names = [], []
    experts.append(_lag(logrv_w, 1)); names.append("Persist")
    for k, nm in ((4, "HAR-4w"), (13, "HAR-13w")):
        h = np.array([np.mean(logrv_w[max(0, t - k):t]) if t > 0 else logrv_w[0]
                      for t in range(T)])
        experts.append(h); names.append(nm)
    experts.append(_lag(np.log(5.0 * ewma_w + 1e-10), 1)); names.append("EWMA-RM")
    experts.append(_lag(np.log(5.0 * garch_w + 1e-10), 1)); names.append("GARCH")
    if vix_w is not None:
        experts.append(_lag(vix_w, 1)); names.append("VIX")
    experts.append(_ridge_expert(logrv_w, Zlevels, refit_every=26, window=156,
                                 min_train=26)); names.append("RIDGE")
    return np.column_stack(experts), names


def _ridge_expert(y, Zlevels, refit_every=125, window=750, ridge=10.0,
                  n_lags=5, min_train=60):
    """Rolling-window ridge on target lags + lagged drivers (one-step).

    Features are standardised with the mean and standard deviation of the
    current training window only, so no information after the refit date is
    used.
    """
    T = len(y)
    feats = [np.zeros(T) for _ in range(n_lags)]
    for k in range(1, n_lags + 1):
        feats[k - 1][k:] = y[:-k]
    Zl = np.zeros_like(Zlevels)
    Zl[1:] = Zlevels[:-1]
    F = np.column_stack(feats + [Zl])
    pred = np.zeros(T)
    w = np.zeros(F.shape[1] + 1)
    mu, sd = np.zeros(F.shape[1]), np.ones(F.shape[1])
    for t in range(T):
        pred[t] = w[0] + ((F[t] - mu) / sd) @ w[1:]
        if t > min_train and t % refit_every == 0:
            lo = max(0, t - window)
            Ft = F[lo:t]
            mu = Ft.mean(axis=0)
            sd = Ft.std(axis=0) + 1e-8
            Xt = np.column_stack([np.ones(t - lo), (Ft - mu) / sd])
            G = Xt.T @ Xt + ridge * np.diag(np.r_[0.0, np.ones(F.shape[1])])
            w = np.linalg.solve(G, Xt.T @ y[lo:t])
    return pred


def _stress_regimes(stress, n_levels=3, min_seg=40, smooth=20):
    """Tercile regimes of a smoothed stress index + transition changepoints.

    The tercile thresholds use the full sample. The labels are used only to
    score forecasts by regime; no method sees them.
    """
    sm = pd.Series(stress).rolling(smooth, min_periods=1).mean().to_numpy()
    qs = np.quantile(sm, np.linspace(0, 1, n_levels + 1)[1:-1])
    reg = np.digitize(sm, qs)
    # enforce a minimum segment length to avoid chattering
    out = reg.copy()
    last, start = reg[0], 0
    for t in range(1, len(reg)):
        if reg[t] != last:
            if t - start < min_seg:
                out[start:t] = out[start - 1] if start > 0 else reg[t]
            start, last = t, reg[t]
    cps = np.where(np.diff(out) != 0)[0] + 1
    return out.astype(int), cps


def load_panel(target="DCOILBRENTEU", start="2010-01-01", target_type="logvar",
               end="2026-06-25", freq="weekly"):
    """Assemble the forecast-combination panel for one target (Sec 3-4).

    ``freq="weekly"`` (default) forecasts the log realised variance of
    non-overlapping weeks: the target for week ``w`` is the log of the sum of
    squared daily log returns in that week, and every expert and driver uses
    data up to the end of week ``w-1`` only. ``freq="daily"`` is the earlier
    design, kept for comparison: the target is the log of a trailing five-day
    mean of squared returns, so four of its five terms are known one step
    ahead.

    Driver series enter in raw units (level and first difference); each
    learner standardises them with past-only statistics. ``target_type`` other
    than ``"logvar"`` is supported only for ``freq="daily"``.
    """
    drivers = {sid: _fred(sid) for sid in ["VIXCLS", "DEXUSEU", "DGS10"]}
    drivers = {k: v for k, v in drivers.items() if v is not None}
    gpr, epu = _gpr(), _epu()
    energy = {sid: _fred(sid) for sid in ["DCOILBRENTEU", "DHHNGSP"]}
    energy = {k: v for k, v in energy.items() if v is not None}

    if target == "EUA":
        tgt = _eua()
    else:
        tgt = _fred(target)
    if tgt is None:
        raise FileNotFoundError(f"target series {target} not available in {DATA}")

    series = [tgt] + list(drivers.values())
    if gpr is not None:
        series.append(gpr)
    if epu is not None:
        series.append(epu)
    for sid, s in energy.items():
        if sid != target:
            series.append(s)
    df = _align(series).loc[start:end].dropna(how="all")
    df = df.dropna()

    price = df[tgt.name].to_numpy()
    ret = np.zeros(len(price))
    ret[1:] = np.diff(np.log(price))                     # daily log-returns
    driver_cols = [c for c in df.columns if c != tgt.name]

    if freq == "weekly":
        return _weekly_panel(df, tgt.name, driver_cols, ret, price, target)
    if freq != "daily":
        raise ValueError(f"unknown freq {freq!r}")

    # ---- daily design (overlapping five-day target), kept for comparison ----
    rv5 = pd.Series(ret ** 2).rolling(5, min_periods=1).mean().to_numpy()
    logrv = np.log(rv5 + 1e-8)

    z_cols, z_names = [], []

    def add(name, arr):
        z_cols.append(arr); z_names.append(name)

    for col in driver_cols:
        lv = df[col].to_numpy().astype(float)
        ch = np.zeros_like(lv); ch[1:] = np.diff(lv)
        add(f"{col}_lvl", lv)
        add(f"{col}_chg", ch)
    rv_l = np.zeros_like(logrv); rv_l[1:] = logrv[:-1]
    add("target_RV_lag", rv_l)
    add("policy_event", _policy_event_indicator(df.index))
    Z = np.column_stack(z_cols)

    vix_log = np.log(df["VIXCLS"].to_numpy()) if "VIXCLS" in df.columns else None
    if target_type == "return":
        y = ret
        experts, expert_names = _build_experts_return(y, Z.copy())
    else:
        y = logrv
        experts, expert_names = _build_experts_vol(logrv, ret, vix_log, Z.copy())

    regimes, cps = _stress_regimes(rv_l)
    anchor_cols = _anchor_cols(z_names)
    sl = slice(40, None)
    return Panel(
        dates=df.index[sl], target_name=target, y=y[sl],
        experts=experts[sl], expert_names=expert_names,
        Z=Z[sl], z_names=z_names,
        regimes=regimes[sl] - regimes[sl].min(),
        changepoints=(cps[cps >= 40] - 40),
        raw_price=price[sl], ret=ret[sl], target_type=target_type,
        anchor_cols=anchor_cols,
    )


def _anchor_cols(z_names):
    """Hand-chosen anchor channel: levels of VIX, GPR and EPU (Sec 4)."""
    return tuple(i for i, n in enumerate(z_names)
                 if n.endswith("_lvl") and any(k in n for k in ("VIX", "GPR", "EPU")))


def _weekly_panel(df, tgt_name, driver_cols, ret, price, target, warm=13):
    """Aggregate the daily panel to non-overlapping Friday-ending weeks."""
    daily = pd.DataFrame({"ret": ret, "price": price}, index=df.index)
    # daily variance filters, evaluated at each day's close (use data to t-1)
    ewma = np.zeros(len(ret)); lam = 0.94
    for t in range(1, len(ret)):
        ewma[t] = lam * ewma[t - 1] + (1 - lam) * ret[t - 1] ** 2
    # value available at the close of day t for the next day: include r_t
    ewma_next = lam * ewma + (1 - lam) * ret ** 2
    g = _garch11_onestep(ret) ** 2
    garch_next = 1e-6 + 0.08 * ret ** 2 + 0.90 * g
    daily["ewma_next"] = ewma_next
    daily["garch_next"] = garch_next
    for c in driver_cols:
        daily[c] = df[c].to_numpy().astype(float)
    if "VIXCLS" in driver_cols:
        daily["logvix"] = np.log(df["VIXCLS"].to_numpy())
    daily["policy_event"] = _policy_event_indicator(df.index)

    wk = daily.resample("W-FRI")
    n_obs = wk["ret"].count()
    agg = pd.DataFrame({
        "rv": wk["ret"].apply(lambda r: float(np.sum(r ** 2))),
        "wret": wk["ret"].sum(),
        "price": wk["price"].last(),
        "ewma": wk["ewma_next"].last(),
        "garch": wk["garch_next"].last(),
        "policy_event": wk["policy_event"].max(),
    })
    for c in driver_cols:
        agg[c] = wk[c].last()
    if "logvix" in daily:
        agg["logvix"] = wk["logvix"].last()
    agg = agg[n_obs >= 2].dropna()
    agg = agg[agg["rv"] > 0]

    logrv_w = np.log(agg["rv"].to_numpy())
    z_cols, z_names = [], []
    for c in driver_cols:
        lv = agg[c].to_numpy()
        ch = np.zeros_like(lv); ch[1:] = np.diff(lv)
        z_cols += [lv, ch]; z_names += [f"{c}_lvl", f"{c}_chg"]
    rv_l = np.zeros_like(logrv_w); rv_l[1:] = logrv_w[:-1]
    z_cols += [rv_l, agg["policy_event"].to_numpy()]
    z_names += ["target_RV_lag", "policy_event"]
    Z = np.column_stack(z_cols)

    vix_w = agg["logvix"].to_numpy() if "logvix" in agg else None
    experts, names = _build_experts_weekly(
        logrv_w, agg["ewma"].to_numpy(), agg["garch"].to_numpy(), vix_w, Z.copy())

    regimes, cps = _stress_regimes(rv_l, min_seg=8, smooth=4)
    sl = slice(warm, None)
    return Panel(
        dates=agg.index[sl], target_name=target, y=logrv_w[sl],
        experts=experts[sl], expert_names=names,
        Z=Z[sl], z_names=z_names,
        regimes=regimes[sl] - regimes[sl].min(),
        changepoints=(cps[cps >= warm] - warm),
        raw_price=agg["price"].to_numpy()[sl], ret=agg["wret"].to_numpy()[sl],
        target_type="logvar_weekly", anchor_cols=_anchor_cols(z_names),
    )


def _policy_event_indicator(index, window=5):
    """+/- ``window``-day bumps around major EU-ETS policy milestones (Sec 3.5)."""
    events = ["2013-02-25",  # back-loading vote
              "2015-07-15",  # MSR decision
              "2018-02-27",  # MSR reform / Phase IV
              "2019-01-01",  # MSR activation
              "2021-01-01",  # Phase III -> IV
              "2023-04-25",  # Fit-for-55 ETS reform
              "2024-01-01"]  # MSR intake-rate change / ETS2 milestones
    ev = pd.to_datetime(events)
    ind = np.zeros(len(index))
    for e in ev:
        hits = np.where(np.abs((index - e).days) <= window)[0]
        ind[hits] = 1.0
    return ind


if __name__ == "__main__":
    for tgt in ["DCOILBRENTEU", "DHHNGSP"]:
        try:
            p = load_panel(tgt)
            print(f"{tgt}: T={len(p.y)}, experts={p.expert_names}, "
                  f"p={p.Z.shape[1]}, regimes={len(np.unique(p.regimes))}, "
                  f"changepoints={len(p.changepoints)}, "
                  f"dates {p.dates[0].date()}..{p.dates[-1].date()}")
        except Exception as e:
            print(f"{tgt}: {e}")
