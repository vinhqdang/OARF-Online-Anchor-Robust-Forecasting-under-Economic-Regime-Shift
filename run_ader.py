#!/usr/bin/env python3
"""Ader-wrapped vs AdaGrad OARF: is the deployed algorithm a faithful stand-in?

The interventional-regret theorem assumes the path-length-adaptive Ader step wrapper
\\citep{zhang2018adaptive}; the deployed implementation takes a single AdaGrad step for
simplicity and stability. A reviewer rightly asks whether the algorithm we actually run
behaves like the one we analyse. This script runs both step rules on the controlled
anchor-SCM (fixed true channel, identical xi) and reports in-regime MSE, worst-case
do(A) MSE and the rank correlation of their per-seed worst-do scores. Writes
``results/synthetic/ader.json``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from oarf import evaluate as ev
from oarf.models import OARF, OARFAder
from oarf.synthetic import SCMConfig, make_intervention_grid, make_scm

OUT = "results/synthetic"


def score(model, scm, grid, es):
    r = ev.run_stream(model, scm.X, scm.Z, scm.Y, eval_start=es)
    pm = ev.point_metrics(r, scm.regimes, scm.changepoints, es)
    iv = ev.frozen_interventional(model, grid, std_y=r.extras["std_y"])
    return pm["MSE"], iv["worst_do_MSE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--T", type=int, default=8000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    acc = {"AdaGrad": {"mse": [], "wdo": []},
           "Ader": {"mse": [], "wdo": []},
           "OGD": {"mse": [], "wdo": []}}

    for s in range(args.seeds):
        print(f"[ader] seed {s}", flush=True)
        scm = make_scm(SCMConfig(T=args.T, seed=s))
        grid = make_intervention_grid(scm, n_dir=6, n_batch=400, seed=10_000 + s)
        es = args.T // 5
        d, p, q, B = scm.cfg.d, scm.cfg.p, scm.cfg.q, scm.B_true
        m_ada = OARF(d, p, B=B, q=q, xi=6.0, eta=0.1)
        m_ader = OARFAder(d, p, B=B, q=q, xi=6.0)
        m_ogd = OARF(d, p, B=B, q=q, xi=0.0, eta=0.1)
        for tag, m in (("AdaGrad", m_ada), ("Ader", m_ader), ("OGD", m_ogd)):
            mse, wdo = score(m, scm, grid, es)
            acc[tag]["mse"].append(mse)
            acc[tag]["wdo"].append(wdo)

    def agg(v):
        a = np.asarray(v, float)
        return {"median": float(np.median(a)), "mean": float(a.mean()),
                "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75))}

    out = {"config": {"seeds": args.seeds, "T": args.T},
           "methods": {k: {"in_regime_MSE": agg(v["mse"]),
                           "worst_do_MSE": agg(v["wdo"])} for k, v in acc.items()}}
    # Spearman rank correlation of per-seed worst-do between the two step rules
    aa = np.asarray(acc["AdaGrad"]["wdo"]); ad = np.asarray(acc["Ader"]["wdo"])
    ra = aa.argsort().argsort(); rd = ad.argsort().argsort()
    rho = float(np.corrcoef(ra, rd)[0, 1])
    out["rank_corr_worst_do"] = rho
    with open(os.path.join(OUT, "ader.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'method':10s} {'inMSE(med)':>12s} {'worst_do(med)':>14s} {'worst_do(mean)':>15s}")
    for k in ("AdaGrad", "Ader", "OGD"):
        d = out["methods"][k]
        print(f"{k:10s} {d['in_regime_MSE']['median']:12.4f} "
              f"{d['worst_do_MSE']['median']:14.4f} {d['worst_do_MSE']['mean']:15.4f}")
    print(f"\nSpearman rank corr (worst-do, AdaGrad vs Ader): {rho:.3f}")


if __name__ == "__main__":
    main()
