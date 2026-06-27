#!/usr/bin/env python3
"""Failure-mode scenarios: where does anchor robustness help, and where does it fail?

The base synthetic SCM is, by design, ideal for OARF. A reviewer rightly asks what
happens when the anchor-regression assumptions are violated. This script generates
several SCM variants that break a different assumption each, evaluates fixed-channel
OARF (given the observed anchor), the learned-channel OARF-CD, and the no-anchor OGD
floor on a held-out ``do`` grid, and reports where OARF helps and where it does not.

Scenarios (each a controlled violation):
* ``clean``     -- the ideal anchor SCM (reference; OARF should help).
* ``parents``   -- the intervention also hits the parents' levels (still
                   anchor-mediated; the invariant predictor should remain valid).
* ``unobserved``-- a second regime driver contaminates the children but is absent
                   from the observable context, so no channel in ``z`` can capture it.
* ``mechanism`` -- the regime shifts the parent->target coefficient itself, so the
                   target mechanism is non-invariant and no fixed predictor is robust.
* ``proxy``     -- the observed anchor is a noisy proxy correlated with the true
                   latent driver but not equal to it (anchor predictive, not causal).
* ``variance``  -- regimes shift the anchor *variance*, not its mean, so the
                   mean-shift-intervention equivalence of Prop. 1 does not apply.

We expect OARF to help on ``clean`` and ``parents`` and to fail (match or trail OGD)
on the others; demonstrating the failures is the point. Writes
``results/synthetic/scenarios.json``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from oarf import evaluate as ev
from oarf.models import OARF

OUT = "results/synthetic"
SCEN = ["clean", "parents", "unobserved", "mechanism", "proxy", "variance"]


def generate(scenario, seed, T=8000, d_par=3, d_ch=3, p=8, q=2, h=2, n_reg=8):
    """Self-contained anchor-SCM generator with per-scenario assumption violations."""
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(int(0.05 * T), int(0.95 * T)),
                              size=n_reg - 1, replace=False))
    reg = np.zeros(T, dtype=int)
    for c in cuts:
        reg[c:] += 1
    cps = cuts.astype(int)
    means = rng.normal(0, 1.5, size=(n_reg, q))

    W_Ap = rng.normal(0, 1, (q, d_par))
    W_Hp = rng.normal(0, 1, (h, d_par)) * 0.9
    b_par = rng.normal(0, 1, d_par)
    d_H = rng.normal(0, 1, h) * 0.9
    c_ch = rng.normal(0, 1, d_ch) * 1.3
    W_Ac = rng.normal(0, 1, (q, d_ch)) * 1.1
    W_Hc = rng.normal(0, 1, (h, d_ch)) * 0.9
    # unobserved extra driver (only used by 'unobserved')
    W_Uc = rng.normal(0, 1, (1, d_ch)) * 1.1
    mech = rng.normal(0, 1, d_par)              # mechanism-shift direction

    Qrot, _ = np.linalg.qr(rng.normal(size=(p, p)))
    B_sel = np.zeros((p, q)); B_sel[np.arange(q), np.arange(q)] = 1.0
    B_true = Qrot @ B_sel

    def draw(n, A_set=None, var_scale=None):
        """Draw one batch; A_set fixes the anchor (interventions), else regime means."""
        H = rng.normal(0, 1, (n, h))
        if A_set is not None:
            A = np.repeat(A_set[None, :], n, axis=0) + rng.normal(0, 0.6, (n, q))
        else:
            A = means[reg[:n]] + rng.normal(0, 0.6, (n, q))
        if var_scale is not None:                # variance-shift regimes
            A = means[0][None, :] + var_scale[:, None] * rng.normal(0, 0.6, (n, q))
        Xp = A @ W_Ap + H @ W_Hp + rng.normal(0, 0.4, (n, d_par))
        if scenario == "parents":                # intervention also shifts parents more
            Xp = Xp + 0.8 * (A @ W_Ap)
        bp = b_par.copy()
        Y = Xp @ b_par + H @ d_H + rng.normal(0, 0.5, n)
        if scenario == "mechanism":              # anchor modulates the parent->Y map
            Y = Y + 0.9 * (A[:, 0]) * (Xp @ mech)
        Xc = np.outer(Y, c_ch) + A @ W_Ac + H @ W_Hc + rng.normal(0, 0.4, (n, d_ch))
        if scenario == "unobserved":
            U = rng.normal(0, 1.5, (n, 1))       # hidden driver, not in z
            Xc = Xc + U @ W_Uc
        X = np.hstack([Xp, Xc])
        return X, Y, A, H

    # ---- training stream ----
    if scenario == "variance":
        vs = 0.5 + reg / (n_reg - 1) * 3.0       # regime-dependent anchor sd
        X, Y, A, H = draw(T, var_scale=vs)
    else:
        X, Y, A, H = draw(T)

    # observable context z = rotation of [anchor; AR(1) decoys]
    z_raw = np.zeros((T, p)); z_raw[:, :q] = A
    decoy = np.zeros((T, p - q))
    for t in range(1, T):
        decoy[t] = 0.6 * decoy[t - 1] + rng.normal(0, 1, p - q)
    z_raw[:, q:] = decoy
    if scenario == "proxy":                      # observed anchor is a noisy proxy
        z_raw[:, :q] = A + rng.normal(0, 1.2, (T, q))
    Z = z_raw @ Qrot.T

    # ---- held-out do() grid on the observed anchor ----
    radii = np.linspace(0, 3, 8) * float(np.sqrt(np.mean(np.sum(A ** 2, axis=1))))
    Xg, Yg = [], []
    for r in radii:
        dirs = rng.normal(size=(5, q)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
        for j in range(5):
            xb, yb, _, _ = draw(300, A_set=dirs[j] * r)
            Xg.append(xb); Yg.append(yb)
    grid = type("G", (), {})()
    grid.radii = radii
    grid.X = [np.stack(Xg[i * 5:(i + 1) * 5]) for i in range(len(radii))]
    grid.Y = [np.stack(Yg[i * 5:(i + 1) * 5]) for i in range(len(radii))]
    return X, Z, Y, reg, cps, B_true, grid


def worst_do(model, X, Z, Y, grid, es):
    r = ev.run_stream(model, X, Z, Y, eval_start=es)
    iv = ev.frozen_interventional(model, grid, std_y=r.extras["std_y"])
    return iv["worst_do_MSE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--T", type=int, default=8000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = 6
    acc = {sc: {"OARF": [], "OARF-CD": [], "OGD": []} for sc in SCEN}

    for s in range(args.seeds):
        print(f"[scenario] seed {s}", flush=True)
        for sc in SCEN:
            X, Z, Y, reg, cps, B, grid = generate(sc, s, args.T)
            es = args.T // 5
            acc[sc]["OARF"].append(worst_do(OARF(d, Z.shape[1], B=B, xi=6.0, eta=0.1),
                                            X, Z, Y, grid, es))
            acc[sc]["OARF-CD"].append(worst_do(
                OARF(d, Z.shape[1], q=2, xi=6.0, eta=0.1, learn_channel=True),
                X, Z, Y, grid, es))
            acc[sc]["OGD"].append(worst_do(OARF(d, Z.shape[1], B=B, xi=0.0, eta=0.1),
                                           X, Z, Y, grid, es))

    out = {"config": {"seeds": args.seeds, "T": args.T}, "scenarios": {}}
    for sc in SCEN:
        out["scenarios"][sc] = {k: {"mean": float(np.mean(v)),
                                    "median": float(np.median(v)),
                                    "sd": float(np.std(v))}
                                for k, v in acc[sc].items()}
    with open(os.path.join(OUT, "scenarios.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[scenario] wrote {OUT}/scenarios.json\n")
    print(f"{'scenario':12s} {'OARF':>10s} {'OARF-CD':>10s} {'OGD':>10s}  verdict")
    for sc in SCEN:
        o = out["scenarios"][sc]
        helps = o["OARF"]["median"] < 0.9 * o["OGD"]["median"]
        print(f"{sc:12s} {o['OARF']['median']:10.3f} {o['OARF-CD']['median']:10.3f} "
              f"{o['OGD']['median']:10.3f}  {'OARF helps' if helps else 'OARF does NOT help'}")


if __name__ == "__main__":
    main()
