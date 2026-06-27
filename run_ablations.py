#!/usr/bin/env python3
"""Ablation suite for OARF (Sec 8.x).

Isolates the contribution of each design choice on the controlled anchor-SCM,
over several seeds: the robustness penalty ``xi`` (the adaptation--immunisation
frontier), the EMA decay ``beta`` (moment-tracking horizon), EMA centring,
adaptive (AdaGrad) vs fixed steps, and the channel (true / random / learned, and
the discovered-channel dimension ``q``).  Writes ``results/synthetic/ablations.json``.

Usage::  python run_ablations.py --seeds 6 --T 6000
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from oarf import evaluate as ev
from oarf.models import OARF
from oarf.synthetic import SCMConfig, make_intervention_grid, make_scm

OUT = "results/synthetic"


def _eval(model, scm, grid, eval_start):
    r = ev.run_stream(model, scm.X, scm.Z, scm.Y, eval_start=eval_start)
    pm = ev.point_metrics(r, scm.regimes, scm.changepoints, eval_start)
    iv = ev.frozen_interventional(model, grid, std_y=r.extras["std_y"])
    return pm["MSE"], iv["worst_do_MSE"]


def _agg(rows):
    """rows: list of dicts with numeric fields -> mean/sd per field across seeds."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        out[k] = {"mean": float(v.mean()), "sd": float(v.std())}
    if "label" in rows[0]:
        out["label"] = rows[0]["label"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--T", type=int, default=6000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d_par, d_ch = 3, 3
    d = d_par + d_ch

    XIS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    BETAS = [0.90, 0.95, 0.98, 0.99, 0.995]
    QS = [1, 2, 3, 4]

    res = {"xi": {}, "beta": {}, "components": {}, "channel": {}, "channel_q": {}}
    acc = {k: {} for k in res}

    for s in range(args.seeds):
        print(f"[ablation] seed {s}", flush=True)
        scm = make_scm(SCMConfig(T=args.T, seed=s))
        grid = make_intervention_grid(scm, n_dir=6, n_batch=300, seed=9000 + s)
        es = args.T // 5
        B = scm.B_true

        # 1) xi sweep (frontier)
        for xi in XIS:
            mse, wdo = _eval(OARF(d, scm.cfg.p, B=B, xi=xi, eta=0.1), scm, grid, es)
            acc["xi"].setdefault(xi, []).append(
                {"label": f"xi={xi}", "in_regime_MSE": mse, "worst_do_MSE": wdo})

        # 2) beta sweep (at xi=6)
        for b in BETAS:
            mse, wdo = _eval(OARF(d, scm.cfg.p, B=B, xi=6.0, eta=0.1, beta=b),
                             scm, grid, es)
            acc["beta"].setdefault(b, []).append(
                {"label": f"beta={b}", "in_regime_MSE": mse, "worst_do_MSE": wdo})

        # 3) component ablations (all at xi=6, true channel unless noted)
        comps = {
            "OARF (full)": OARF(d, scm.cfg.p, B=B, xi=6.0, eta=0.1),
            "no centring": OARF(d, scm.cfg.p, B=B, xi=6.0, eta=0.1, center=False),
            "no AdaGrad": OARF(d, scm.cfg.p, B=B, xi=6.0, eta=0.02, adagrad=False),
            "xi=0 (OGD)": OARF(d, scm.cfg.p, B=B, xi=0.0, eta=0.1),
            "random channel": OARF(d, scm.cfg.p, q=scm.cfg.q, xi=6.0, eta=0.1),
        }
        for name, m in comps.items():
            mse, wdo = _eval(m, scm, grid, es)
            acc["components"].setdefault(name, []).append(
                {"label": name, "in_regime_MSE": mse, "worst_do_MSE": wdo})

        # 4) channel: true vs random vs learned (q=2)
        chans = {
            "true channel": OARF(d, scm.cfg.p, B=B, xi=6.0, eta=0.1),
            "random channel": OARF(d, scm.cfg.p, q=scm.cfg.q, xi=6.0, eta=0.1),
            "learned (OARF-CD)": OARF(d, scm.cfg.p, q=scm.cfg.q, xi=6.0, eta=0.1,
                                      learn_channel=True),
        }
        for name, m in chans.items():
            mse, wdo = _eval(m, scm, grid, es)
            row = {"label": name, "in_regime_MSE": mse, "worst_do_MSE": wdo}
            if name == "learned (OARF-CD)":
                row["alignment"] = ev.subspace_alignment(m.B, scm.B_true)
            acc["channel"].setdefault(name, []).append(row)

        # 5) learned-channel dimension q
        for q in QS:
            m = OARF(d, scm.cfg.p, q=q, xi=6.0, eta=0.1, learn_channel=True)
            mse, wdo = _eval(m, scm, grid, es)
            al = ev.subspace_alignment(m.B, scm.B_true) if q >= scm.cfg.q else \
                ev.subspace_alignment(m.B, scm.B_true[:, :q])
            acc["channel_q"].setdefault(q, []).append(
                {"label": f"q={q}", "in_regime_MSE": mse, "worst_do_MSE": wdo,
                 "alignment": al})

    for grp in acc:
        for key, rows in acc[grp].items():
            res[grp][str(key)] = _agg(rows)

    with open(os.path.join(OUT, "ablations.json"), "w") as f:
        json.dump({"config": {"seeds": args.seeds, "T": args.T}, **res}, f, indent=2)
    print(f"[ablation] wrote {OUT}/ablations.json")

    # console summary
    print("\n=== component ablation (worst-do MSE, mean over seeds) ===")
    for name, d_ in res["components"].items():
        print(f"  {name:18s} MSE={d_['in_regime_MSE']['mean']:.4f} "
              f"worst_do={d_['worst_do_MSE']['mean']:.4f}")
    print("\n=== channel ===")
    for name, d_ in res["channel"].items():
        al = d_.get("alignment", {}).get("mean")
        print(f"  {name:18s} worst_do={d_['worst_do_MSE']['mean']:.4f}"
              + (f"  align={al:.3f}" if al is not None else ""))


if __name__ == "__main__":
    main()
