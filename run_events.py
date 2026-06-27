#!/usr/bin/env python3
"""Real-data event study around COVID-19 and the 2022 energy shock.

The regime-tercile analysis in the main text is aggregate. An economics reader also
wants to see behaviour around recognisable events. We run fixed-anchor \\OARF{} and the
no-anchor OGD floor on the real energy combination stream, recording per-step squared
error and the smoothed anchor--residual correlation, and extract two windows: the
COVID-19 crash (2020) and the 2022 European energy shock. We plot cumulative squared
loss (OARF minus OGD; negative = OARF better) and the anchor--residual correlation for
both methods. The point is descriptive --- the MCS shows no significant edge --- but it
makes the mechanism concrete around dated events. Writes ``results/real/events_*.json``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from oarf.load_panel import load_panel
from oarf.models import OARF
from oarf.online import EMA, OnlineStandardizer

OUT = "results/real"
EVENTS = {
    "COVID-19 (2020)": ("2020-01-01", "2020-09-30"),
    "Energy shock (2022)": ("2022-01-01", "2022-12-31"),
}


def _driver_channel(panel):
    p = panel.Z.shape[1]
    cols = panel.anchor_cols
    B = np.zeros((p, len(cols)))
    for j, i in enumerate(cols):
        B[i, j] = 1.0
    return B, len(cols)


def run_with_trace(model, X, Z, Y):
    """Stream the model, returning per-step squared error and EMA anchor--resid corr."""
    sy = OnlineStandardizer(1)
    q = model.B.shape[1]
    ear, eva, evr = EMA(q, 0.97), EMA(q, 0.97), EMA(1, 0.97)
    T = len(Y)
    sq = np.empty(T)
    corr = np.empty(T)
    for t in range(T):
        yhat = model.predict(X[t], Z[t])
        ys = float(sy.transform([Y[t]])[0])
        r = ys - yhat
        sq[t] = r * r
        a = model.B.T @ model.std_z.transform(Z[t])
        ca, va = ear.update(a * r), eva.update(a * a)
        vr = float(evr.update([r * r])[0])
        corr[t] = float(np.mean(np.abs(ca) / (np.sqrt(va * vr) + 1e-8)))
        model.update(X[t], Z[t], ys)
        sy.update([Y[t]])
    return sq, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["DCOILBRENTEU", "DHHNGSP"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for tgt in args.targets:
        print(f"[events] {tgt}", flush=True)
        panel = load_panel(tgt)
        d, p = panel.experts.shape[1], panel.Z.shape[1]
        X, Z, Y = panel.experts, panel.Z, panel.y
        B, q = _driver_channel(panel)
        sq_o, corr_o = run_with_trace(OARF(d, p, B=B, xi=1.5, eta=0.1), X, Z, Y)
        sq_g, corr_g = run_with_trace(OARF(d, p, B=B, xi=0.0, eta=0.1), X, Z, Y)
        dates = np.array([str(dd.date()) for dd in panel.dates])
        dts = panel.dates

        windows = {}
        for label, (lo, hi) in EVENTS.items():
            mask = (dts >= lo) & (dts <= hi)
            idx = np.where(mask)[0]
            if len(idx) < 10:
                continue
            cum_o = np.cumsum(sq_o[idx]); cum_g = np.cumsum(sq_g[idx])
            windows[label] = {
                "dates": dates[idx].tolist(),
                "cum_sq_OARF": cum_o.tolist(),
                "cum_sq_OGD": cum_g.tolist(),
                "cum_diff_OARF_minus_OGD": (cum_o - cum_g).tolist(),
                "corr_OARF": corr_o[idx].tolist(),
                "corr_OGD": corr_g[idx].tolist(),
                "final_cum_diff": float(cum_o[-1] - cum_g[-1]),
            }
            print(f"  {label:22s} final cum sq-loss (OARF-OGD) = "
                  f"{cum_o[-1]-cum_g[-1]:+.3f} over {len(idx)} days")
        with open(os.path.join(OUT, f"events_{tgt}.json"), "w") as f:
            json.dump({"target": tgt, "windows": windows}, f)
    print(f"[events] wrote {OUT}/events_*.json")


if __name__ == "__main__":
    main()
