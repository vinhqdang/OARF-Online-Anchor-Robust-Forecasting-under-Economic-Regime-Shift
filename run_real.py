#!/usr/bin/env python3
"""Real-data forecast-combination experiment (Sec 3-4, 8).

The base experts (AR/EWMA/GARCH/momentum/ridge) are combined online by every
method in the combination track (Sec 5), including the 2026 distributionally-
robust combiners that are OARF's headline competitors.  OARF treats the experts
as its predictors ``x_t`` and the macro-financial drivers as the context ``z_t``
through which regime shifts travel.  We report point, regime-aware,
distributional, and economic metrics with DM and MCS significance, over a
rolling-origin set of evaluation start points (Sec 6, 8).

Usage::

    python run_real.py --targets DCOILBRENTEU DHHNGSP
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from oarf import evaluate as ev
from oarf.load_panel import load_panel
from oarf.models import (COMBINATION_TRACK, AnchorRegression, OARF,
                         build_model_suite)

OUT = "results/real"


def _driver_channel(panel):
    """Hand-chosen anchor channel selecting the macro driver levels (Sec 4)."""
    p = panel.Z.shape[1]
    cols = panel.anchor_cols
    B = np.zeros((p, len(cols)))
    for j, i in enumerate(cols):
        B[i, j] = 1.0
    return B, len(cols)


def run_target(target, eval_fracs=(0.2, 0.35, 0.5)):
    panel = load_panel(target)
    d = panel.experts.shape[1]
    p = panel.Z.shape[1]
    T = len(panel.y)
    X, Z, Y = panel.experts, panel.Z, panel.y
    B, q = _driver_channel(panel)

    suite = build_model_suite(d, p, q=q, include=COMBINATION_TRACK)
    # configure the anchor methods with the macro-driver channel + a modest,
    # validation-appropriate robustness weight for the combination task
    suite["OARF"] = OARF(d, p, B=B, xi=1.5, eta=0.1, name="OARF")
    suite["OARF-CD"] = OARF(d, p, q=q, xi=2.0, eta=0.1, learn_channel=True,
                            name="OARF-CD")
    suite["AnchorReg"] = AnchorRegression(d, p, B=B, lam=4.0)
    # keep a stable display order
    order = ["OARF", "OARF-CD", "OGD", "Rolling-OLS", "ACI", "M-FISHER",
             "DR-Combo(Wang26)", "FC-DRO(Liu26)", "FC-DRO-ES(Liu26)",
             "W-DRO-OL(Chen26)", "AnchorReg"]
    suite = {k: suite[k] for k in order if k in suite}

    results, rows = [], {}
    for name, m in suite.items():
        rec = name in ("OARF", "OGD")
        r = ev.run_stream(m, X, Z, Y, record_weights=rec, standardize_y=True)
        results.append(r)

    # primary evaluation start = first 20% for warm-up/HP (Sec 8)
    es0 = int(eval_fracs[0] * T)
    std_y = results[0].extras["std_y"]
    for r in results:
        pm = ev.point_metrics(r, panel.regimes, panel.changepoints, es0)
        dist = ev.distributional_metrics(r, es0)
        # unstandardise the log-variance forecast -> raw variance for vol-timing
        yhat_unstd = r.yhat * std_y._std[0] + std_y.mean[0]
        var_hat = np.exp(yhat_unstd) if panel.target_type == "logvar" \
            else np.full(T, np.var(panel.ret))
        econ = ev.volatility_timing(var_hat, panel.ret, es0)
        # rolling-origin robustness: MSE over several evaluation starts
        ro = {f"MSE@{int(f*100)}pct": ev.mse(r.sq_err[int(f * T):])
              for f in eval_fracs}
        rows[r.name] = {**pm, "rolling_origin": ro,
                        "distributional": dist,
                        "economic": {k: v for k, v in econ.items()
                                     if k != "equity_curve"},
                        "equity_curve": econ["equity_curve"]}

    names, dm_stat, dm_p = ev.dm_matrix(results, es0)
    mcs = ev.model_confidence_set(results, es0, n_boot=1000, seed=0)
    per_regime = {r.name: ev.per_regime_mse(r.sq_err[es0:], panel.regimes[es0:])
                  for r in results}
    reg_ids = sorted(np.unique(panel.regimes[es0:]).tolist())
    detail = {
        "dates": [str(dd.date()) for dd in panel.dates],
        "regimes": panel.regimes.tolist(),
        "changepoints": panel.changepoints.tolist(),
        "eval_start": es0,
        "raw_price": panel.raw_price.tolist(),
        "sq_err": {r.name: r.sq_err.tolist() for r in results
                   if r.name in ("OARF", "OGD", "DR-Combo(Wang26)",
                                 "FC-DRO-ES(Liu26)")},
        "expert_names": panel.expert_names,
        "z_names": panel.z_names,
    }
    return {
        "target": target, "T": T, "n_experts": d, "p": p,
        "rows": rows,
        "dm": {"names": names, "stat": np.asarray(dm_stat).tolist(),
               "pval": np.asarray(dm_p).tolist()},
        "mcs": mcs,
        "per_regime": {k: {str(i): v.get(i) for i in reg_ids}
                       for k, v in per_regime.items()},
        "regime_ids": reg_ids,
    }, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["DCOILBRENTEU", "DHHNGSP"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    summary = {}
    for tgt in args.targets:
        print(f"[real] {tgt} ...", flush=True)
        try:
            res, detail = run_target(tgt)
        except FileNotFoundError as e:
            print(f"[real] skip {tgt}: {e}")
            continue
        summary[tgt] = res
        with open(os.path.join(OUT, f"detail_{tgt}.json"), "w") as f:
            json.dump(detail, f)
        print(f"\n=== {tgt}  (T={res['T']}, {res['n_experts']} experts) ===")
        print(f"{'model':18s} {'MSE':>9s} {'worstReg':>9s} {'postCP':>9s} "
              f"{'Sharpe':>8s} {'cover':>7s}")
        for nm, d in res["rows"].items():
            cov = d["distributional"]["coverage"] if d["distributional"] else float("nan")
            print(f"{nm:18s} {d['MSE']:9.5f} {d['worst_regime_MSE']:9.5f} "
                  f"{d['post_cp_MSE']:9.5f} {d['economic']['ann_Sharpe']:8.2f} "
                  f"{cov:7.3f}")
        print(f"MCS survivors: {res['mcs']['survivors']}")

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[real] wrote {OUT}/metrics.json")


if __name__ == "__main__":
    main()
