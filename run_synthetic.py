#!/usr/bin/env python3
"""End-to-end synthetic causal-robustness experiment (Sec 8).

Runs the full comparison suite (Sec 5) over multiple seeds on the anchor-SCM,
computes the point / regime / interventional / diagnostic metrics (Sec 6),
sweeps the robustness penalty ``xi`` for the adaptation-immunisation Pareto
frontier (Fig. 4), and scores the online channel-discovery layer (Sec 2.3)
against ground truth.  All artefacts are written to ``results/synthetic/``.

Usage::

    python run_synthetic.py --seeds 8 --T 8000
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from oarf import evaluate as ev
from oarf.models import (REGRESSION_TRACK, OARF, build_model_suite)
from oarf.synthetic import SCMConfig, make_intervention_grid, make_scm

OUT = "results/synthetic"


def run_one_seed(seed, T, xi=6.0):
    cfg = SCMConfig(T=T, seed=seed)
    scm = make_scm(cfg)
    grid = make_intervention_grid(scm, n_dir=6, n_batch=400, seed=10_000 + seed)
    eval_start = T // 5                       # first 20% for warm-up/HP (Sec 8)

    suite = build_model_suite(scm.cfg.d, scm.cfg.p, B_fixed=scm.B_true,
                              q=scm.cfg.q, include=REGRESSION_TRACK, seed=seed)
    rows = {}
    results = []
    for name, m in suite.items():
        rec = name in ("OARF", "OGD", "AnchorReg")
        r = ev.run_stream(m, scm.X, scm.Z, scm.Y, record_weights=rec,
                          eval_start=eval_start)
        pm = ev.point_metrics(r, scm.regimes, scm.changepoints, eval_start)
        iv = ev.frozen_interventional(m, grid, std_y=r.extras["std_y"])
        dm = {
            "dyn_regret": ev.dynamic_regret_proxy(r, eval_start=eval_start),
            "recovery_delay": ev.recovery_delay(r, scm.changepoints, eval_start),
            "coef_stability": ev.coef_stability(r, eval_start),
        }
        dist = ev.distributional_metrics(r, eval_start)
        rows[name] = {**pm, **dm, "worst_do_MSE": iv["worst_do_MSE"],
                      "robustness_curve": iv["robustness_curve"].tolist(),
                      "radii": iv["radii"].tolist(),
                      "distributional": dist}
        results.append(r)
        if name == "OARF-CD":
            rows[name]["channel_alignment"] = ev.subspace_alignment(m.B, scm.B_true)
            rows[name]["alignment_traj"] = [
                (s, ev.subspace_alignment(np.array(B), scm.B_true))
                for s, B in m.B_hist[::3]]

    names, dm_stat, dm_p = ev.dm_matrix(results, eval_start)
    mcs = ev.model_confidence_set(results, eval_start, n_boot=1000, seed=seed)
    # per-regime MSE matrix (methods x regimes) for the heatmap
    reg_ids = sorted(np.unique(scm.regimes[eval_start:]).tolist())
    per_regime = {r.name: ev.per_regime_mse(r.sq_err[eval_start:],
                                            scm.regimes[eval_start:])
                  for r in results}
    return {
        "seed": seed, "rows": rows,
        "dm": {"names": names, "stat": np.asarray(dm_stat).tolist(),
               "pval": np.asarray(dm_p).tolist()},
        "mcs": mcs,
        "per_regime": {k: {str(i): v.get(i, None) for i in reg_ids}
                       for k, v in per_regime.items()},
        "regime_ids": reg_ids,
    }, scm, results, grid, eval_start


def pareto_sweep(seed, T, xis):
    """Sweep xi to trace in-regime MSE vs worst-case interventional MSE (Fig. 4)."""
    scm = make_scm(SCMConfig(T=T, seed=seed))
    grid = make_intervention_grid(scm, n_dir=6, n_batch=400, seed=10_000 + seed)
    eval_start = T // 5
    pts = []
    for xi in xis:
        m = OARF(scm.cfg.d, scm.cfg.p, B=scm.B_true, q=scm.cfg.q, xi=xi, eta=0.1)
        r = ev.run_stream(m, scm.X, scm.Z, scm.Y, eval_start=eval_start)
        pm = ev.point_metrics(r, scm.regimes, scm.changepoints, eval_start)
        iv = ev.frozen_interventional(m, grid, std_y=r.extras["std_y"])
        pts.append({"xi": xi, "in_regime_MSE": pm["MSE"],
                    "worst_do_MSE": iv["worst_do_MSE"]})
    return pts


def aggregate(seed_results):
    """Mean +/- sd across seeds for each model and metric (Sec 6)."""
    agg = {}
    metrics = ["MSE", "RMSE", "worst_regime_MSE", "post_cp_MSE", "worst_do_MSE",
               "dyn_regret", "recovery_delay", "coef_stability"]
    names = list(seed_results[0]["rows"].keys())
    for nm in names:
        agg[nm] = {}
        for mt in metrics:
            vals = [sr["rows"][nm].get(mt) for sr in seed_results]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                agg[nm][mt] = {"mean": float(np.mean(vals)),
                               "sd": float(np.std(vals))}
        ca = [sr["rows"][nm].get("channel_alignment") for sr in seed_results
              if "channel_alignment" in sr["rows"].get(nm, {})]
        if ca:
            agg[nm]["channel_alignment"] = {"mean": float(np.mean(ca)),
                                            "sd": float(np.std(ca))}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--T", type=int, default=8000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    seed_results = []
    for s in range(args.seeds):
        print(f"[synthetic] seed {s} ...", flush=True)
        sr, scm, results, grid, eval_start = run_one_seed(s, args.T)
        if s == 0:                            # seed-0 stream detail for figures
            save_seed0_detail(scm, results, eval_start)
        seed_results.append(sr)

    agg = aggregate(seed_results)
    xis = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    pareto = {s: pareto_sweep(s, args.T, xis) for s in range(min(3, args.seeds))}

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump({"aggregate": agg, "per_seed": seed_results,
                   "pareto": pareto, "config": {"T": args.T, "seeds": args.seeds}},
                  f, indent=2)
    print(f"[synthetic] wrote {OUT}/metrics.json")

    # console summary
    print("\n=== Synthetic causal-robustness (mean over seeds) ===")
    print(f"{'model':18s} {'MSE':>9s} {'worstReg':>9s} {'postCP':>9s} "
          f"{'worst_do':>9s}")
    for nm, d in agg.items():
        print(f"{nm:18s} {d['MSE']['mean']:9.4f} {d['worst_regime_MSE']['mean']:9.4f}"
              f" {d['post_cp_MSE']['mean']:9.4f} {d['worst_do_MSE']['mean']:9.4f}")
    if "OARF-CD" in agg and "channel_alignment" in agg["OARF-CD"]:
        ca = agg["OARF-CD"]["channel_alignment"]
        print(f"\nOARF-CD channel alignment: {ca['mean']:.3f} +/- {ca['sd']:.3f}")


def save_seed0_detail(scm, results, eval_start):
    """Persist seed-0 per-step traces used by the time-series figures (Sec 7)."""
    detail = {
        "regimes": scm.regimes.tolist(),
        "changepoints": scm.changepoints.tolist(),
        "eval_start": eval_start,
        "sq_err": {r.name: r.sq_err.tolist() for r in results
                   if r.name in ("OARF", "OGD", "Rolling-OLS", "AnchorReg")},
        "w_path": {r.name: r.w_path.tolist() for r in results
                   if r.w_path is not None},
        "w_invariant": scm.w_invariant.tolist(),
        "parent_idx": scm.parent_idx.tolist(),
    }
    with open(os.path.join(OUT, "seed0_detail.json"), "w") as f:
        json.dump(detail, f)


if __name__ == "__main__":
    main()
