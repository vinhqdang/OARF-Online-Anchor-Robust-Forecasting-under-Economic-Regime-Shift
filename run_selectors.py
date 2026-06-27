#!/usr/bin/env python3
"""Benchmark the channel-discovery rule against alternative anchor selectors.

A reviewer asks whether OARF-CD's block-mean nonstationary-subspace rule is the best
way to pick the anchor, or merely a fragile one. We compare several past-only
selectors that each return a ``p x q`` channel from the warm-up stream, then score
each by (i) subspace alignment to the ground-truth channel and (ii) the worst-case
do(A) MSE of fixed-channel OARF using that channel. Selectors:

* ``true``         -- the ground-truth channel (upper bound).
* ``block-mean``   -- OARF-CD's rule: top-q eigenvectors of the across-block scatter
                      of the block-mean context (unsupervised nonstationarity).
* ``residual-corr``-- supervised: top-q eigenvectors of the across-block scatter of
                      the block-mean *residual--context coupling* z*r of an OGD
                      reference predictor (directions whose coupling is regime-unstable).
* ``stationary-PCA``-- top-q principal components of Cov(z) (ignores nonstationarity;
                      a deliberately mis-motivated baseline).
* ``random``       -- random orthonormal channel (lower bound).

Writes ``results/synthetic/selectors.json``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from oarf import evaluate as ev
from oarf.models import OARF
from oarf.synthetic import SCMConfig, make_intervention_grid, make_scm


def _topq(S, q):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return V[:, -q:]


def _block_means(M, L):
    n = len(M) // L
    return np.array([M[i * L:(i + 1) * L].mean(0) for i in range(n)])


def select_channel(kind, Z, Y, q, L=40, rng=None):
    """Return a p x q channel estimated from (Z, Y) using a past-only rule."""
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    if kind == "random":
        B, _ = np.linalg.qr(rng.normal(size=(Z.shape[1], q)))
        return B
    if kind == "stationary-PCA":
        return _topq(np.cov(Zs.T), q)
    if kind == "block-mean":
        bm = _block_means(Zs, L)
        return _topq(np.cov(bm.T) if len(bm) > q else np.eye(Z.shape[1])[:, :q] * 0 + np.cov(bm.T), q)
    if kind == "residual-corr":
        # OGD reference residual, then across-block scatter of block-mean (z * r)
        ys = (Y - Y.mean()) / (Y.std() + 1e-9)
        w = np.zeros(Z.shape[1] + 1); G = np.zeros(Z.shape[1] + 1)
        r = np.zeros(len(ys))
        for t in range(len(ys)):
            phi = np.concatenate(([1.0], Zs[t]))
            r[t] = ys[t] - w @ phi
            g = -r[t] * phi; G += g * g
            w -= 0.1 * g / (np.sqrt(G) + 1e-6)
        coup = Zs * r[:, None]
        bm = _block_means(coup, L)
        return _topq(np.cov(bm.T), q)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--T", type=int, default=8000)
    args = ap.parse_args()
    os.makedirs("results/synthetic", exist_ok=True)
    KINDS = ["true", "block-mean", "residual-corr", "stationary-PCA", "random"]
    acc = {k: {"align": [], "worst_do": []} for k in KINDS}

    for s in range(args.seeds):
        print(f"[selector] seed {s}", flush=True)
        scm = make_scm(SCMConfig(T=args.T, seed=s))
        grid = make_intervention_grid(scm, n_dir=6, n_batch=300, seed=5000 + s)
        es = args.T // 5
        rng = np.random.default_rng(2000 + s)
        # estimate channels on the warm-up block only (no look-ahead)
        Zw, Yw = scm.Z[:es], scm.Y[:es]
        for k in KINDS:
            B = scm.B_true if k == "true" else select_channel(k, Zw, Yw, scm.cfg.q, rng=rng)
            acc[k]["align"].append(ev.subspace_alignment(B, scm.B_true))
            m = OARF(scm.cfg.d, scm.cfg.p, B=B, xi=6.0, eta=0.1)
            r = ev.run_stream(m, scm.X, scm.Z, scm.Y, eval_start=es)
            iv = ev.frozen_interventional(m, grid, std_y=r.extras["std_y"])
            acc[k]["worst_do"].append(iv["worst_do_MSE"])

    out = {"config": {"seeds": args.seeds, "T": args.T},
           "selectors": {k: {"align_mean": float(np.mean(v["align"])),
                             "align_sd": float(np.std(v["align"])),
                             "worst_do_median": float(np.median(v["worst_do"])),
                             "worst_do_mean": float(np.mean(v["worst_do"]))}
                         for k, v in acc.items()}}
    with open("results/synthetic/selectors.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{'selector':16s} {'align':>8s} {'worst_do(med)':>14s}")
    for k in KINDS:
        d = out["selectors"][k]
        print(f"{k:16s} {d['align_mean']:8.3f} {d['worst_do_median']:14.3f}")


if __name__ == "__main__":
    main()
