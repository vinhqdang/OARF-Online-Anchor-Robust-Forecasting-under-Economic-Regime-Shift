#!/usr/bin/env python3
"""Stress tests: how OARF degrades when the anchor is noisy or misspecified.

A reviewer's central concern is that the synthetic benchmark is built around a
clean, fully observed anchor, so the proposed method is bound to win.  This
script probes the regime where that assumption fails, along two axes that do not
require regenerating the SCM:

* **Anchor observation noise.** The observable context ``Z`` is corrupted by
  additive Gaussian noise of growing scale before any method sees it, so the
  channel becomes progressively unobservable.

* **Channel misspecification.** The fixed channel handed to OARF is rotated away
  from the truth by an angle ``theta`` (subspace alignment ``cos theta``), from
  the true channel (``theta=0``) to an orthogonal, useless one (``theta=90``).

For each level we report the worst-case interventional MSE of fixed-channel OARF,
the learned-channel OARF-CD, and the no-anchor OGD floor, averaged over seeds.
This quantifies graceful degradation and locates the threshold beyond which
anchoring on the wrong directions is worse than not anchoring at all --- the
practical diagnostic the review asks for.  Writes ``results/synthetic/stress.json``.
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


def _worst_do(model, scm, grid, Zin, eval_start):
    r = ev.run_stream(model, scm.X, Zin, scm.Y, eval_start=eval_start)
    iv = ev.frozen_interventional(model, grid, std_y=r.extras["std_y"])
    return iv["worst_do_MSE"]


def _rotate_channel(B_true, theta, rng):
    """Rotate each column of B_true toward a random orthogonal direction by theta."""
    p, q = B_true.shape
    # orthonormal complement basis of span(B_true)
    Q, _ = np.linalg.qr(np.hstack([B_true, rng.normal(size=(p, p - q))]))
    perp = Q[:, q:]                                   # (p, p-q)
    B = np.empty_like(B_true)
    for j in range(q):
        d = perp[:, j % perp.shape[1]]
        B[:, j] = np.cos(theta) * B_true[:, j] + np.sin(theta) * d
    B, _ = np.linalg.qr(B)
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--T", type=int, default=8000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d_par, d_ch = 3, 3
    d = d_par + d_ch

    NOISE = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    ANGLES_DEG = [0, 15, 30, 45, 60, 90]

    acc_noise = {lv: {"OARF": [], "OARF-CD": [], "OGD": []} for lv in NOISE}
    acc_ang = {a: {"OARF": [], "OGD": []} for a in ANGLES_DEG}

    for s in range(args.seeds):
        print(f"[stress] seed {s}", flush=True)
        scm = make_scm(SCMConfig(T=args.T, seed=s))
        grid = make_intervention_grid(scm, n_dir=6, n_batch=300, seed=7000 + s)
        es = args.T // 5
        rng = np.random.default_rng(1000 + s)
        zscale = scm.Z.std()

        # --- axis 1: anchor observation noise (corrupt the observed context) ---
        for lv in NOISE:
            Zn = scm.Z + lv * zscale * rng.normal(size=scm.Z.shape)
            acc_noise[lv]["OARF"].append(
                _worst_do(OARF(d, scm.cfg.p, B=scm.B_true, xi=6.0, eta=0.1),
                          scm, grid, Zn, es))
            acc_noise[lv]["OARF-CD"].append(
                _worst_do(OARF(d, scm.cfg.p, q=scm.cfg.q, xi=6.0, eta=0.1,
                               learn_channel=True), scm, grid, Zn, es))
            acc_noise[lv]["OGD"].append(
                _worst_do(OARF(d, scm.cfg.p, B=scm.B_true, xi=0.0, eta=0.1),
                          scm, grid, Zn, es))

        # --- axis 2: channel misspecification (rotate the fixed channel) ---
        for a in ANGLES_DEG:
            B = _rotate_channel(scm.B_true, np.deg2rad(a), rng)
            acc_ang[a]["OARF"].append(
                _worst_do(OARF(d, scm.cfg.p, B=B, xi=6.0, eta=0.1),
                          scm, grid, scm.Z, es))
            acc_ang[a]["OGD"].append(
                _worst_do(OARF(d, scm.cfg.p, B=scm.B_true, xi=0.0, eta=0.1),
                          scm, grid, scm.Z, es))

    def agg(d_):
        return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                for k, v in d_.items()}

    out = {
        "config": {"seeds": args.seeds, "T": args.T},
        "noise_levels": NOISE,
        "angles_deg": ANGLES_DEG,
        "noise": {str(lv): agg(acc_noise[lv]) for lv in NOISE},
        "angle": {str(a): agg(acc_ang[a]) for a in ANGLES_DEG},
    }
    with open(os.path.join(OUT, "stress.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[stress] wrote {OUT}/stress.json")

    print("\n=== worst-do(A) MSE vs anchor observation noise ===")
    for lv in NOISE:
        a = out["noise"][str(lv)]
        print(f"  noise={lv:4.2f}x  OARF={a['OARF']['mean']:.3f}  "
              f"OARF-CD={a['OARF-CD']['mean']:.3f}  OGD={a['OGD']['mean']:.3f}")
    print("\n=== worst-do(A) MSE vs channel misspecification angle ===")
    for ang in ANGLES_DEG:
        a = out["angle"][str(ang)]
        print(f"  angle={ang:3d}deg (align={np.cos(np.deg2rad(ang)):.2f})  "
              f"OARF={a['OARF']['mean']:.3f}  OGD={a['OGD']['mean']:.3f}")


if __name__ == "__main__":
    main()
