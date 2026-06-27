"""Generate the paper's figures (Sec 7) from the saved experiment artefacts.

Reads ``results/synthetic/*.json`` and ``results/real/*.json`` and writes
publication-quality PDFs (and PNGs) to ``figures/``.  Run after
``run_synthetic.py`` and ``run_real.py``.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "figures"
SYN = "results/synthetic"
REAL = "results/real"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 220, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.axisbelow": True, "legend.fontsize": 8, "figure.autolayout": True,
})

# colour-blind-safe palette; OARF family highlighted
PAL = {
    "OARF": "#d62728", "OARF-CD": "#9467bd", "OGD": "#1f77b4",
    "Rolling-OLS": "#7f7f7f", "ACI": "#2ca02c", "M-FISHER": "#ff7f0e",
    "WATCH/WCTM": "#8c564b", "W-DRO-OL(Chen26)": "#17becf", "W-DRO-OL": "#17becf",
    "CostACI": "#bcbd22", "AnchorReg": "#e377c2", "DRIG": "#aec7e8",
    "DR-Combo(Wang26)": "#ffbb78", "FC-DRO(Liu26)": "#98df8a",
    "FC-DRO-ES(Liu26)": "#c5b0d5",
}


def _c(name):
    return PAL.get(name, "#333333")


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)
    print(f"[fig] {name}")


def _load(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
def fig1_rolling_error(syn_detail):
    """Fig. 1 — rolling squared error across regimes with changepoints marked."""
    cps = syn_detail["changepoints"]
    es = syn_detail["eval_start"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for nm in ["OGD", "Rolling-OLS", "OARF"]:
        if nm not in syn_detail["sq_err"]:
            continue
        se = np.array(syn_detail["sq_err"][nm])
        roll = np.convolve(se, np.ones(60) / 60, mode="same")
        ax.plot(roll, color=_c(nm), lw=1.2, label=nm, alpha=0.9)
    for c in cps:
        ax.axvline(c, color="k", ls=":", lw=0.6, alpha=0.4)
    ax.axvspan(0, es, color="gray", alpha=0.08)
    ax.set_yscale("log")
    ax.set_xlabel("time step $t$"); ax.set_ylabel("rolling squared error (60-step)")
    ax.set_title("Rolling error across regimes (changepoints dotted)")
    ax.legend(ncol=3)
    _save(fig, "fig1_rolling_error")


def fig2_bar_metrics(syn_agg):
    """Fig. 2 — overall / post-cp / worst-do(A) bar chart."""
    names = list(syn_agg.keys())
    metrics = [("MSE", "overall MSE"), ("post_cp_MSE", "post-changepoint MSE"),
               ("worst_do_MSE", "worst-case do(A) MSE")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    order = sorted(names, key=lambda n: syn_agg[n]["worst_do_MSE"]["mean"])
    for ax, (mt, title) in zip(axes, metrics):
        vals = [syn_agg[n][mt]["mean"] for n in order]
        sds = [syn_agg[n][mt]["sd"] for n in order]
        cols = [_c(n) for n in order]
        ax.barh(range(len(order)), vals, xerr=sds, color=cols, alpha=0.9,
                error_kw=dict(lw=0.6))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if ax is axes[0] else [], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title); ax.set_xlabel("MSE")
    fig.suptitle("Synthetic: accuracy vs interventional robustness (mean $\\pm$ sd, 8 seeds)")
    _save(fig, "fig2_bar_metrics")


def fig3_robustness_curve(syn_perseed):
    """Fig. 3 — robustness curve: MSE vs intervention magnitude ||nu||."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rows0 = syn_perseed[0]["rows"]
    radii = np.array(rows0["OARF"]["radii"])
    show = ["OARF", "OARF-CD", "OGD", "Rolling-OLS", "W-DRO-OL(Chen26)",
            "AnchorReg", "M-FISHER"]
    for nm in show:
        curves = np.array([sr["rows"][nm]["robustness_curve"]
                           for sr in syn_perseed if nm in sr["rows"]])
        mean = curves.mean(0)
        se = curves.std(0) / np.sqrt(curves.shape[0])
        lw = 2.4 if nm in ("OARF", "OARF-CD") else 1.4
        ax.plot(radii, mean, color=_c(nm), lw=lw, marker="o", ms=3, label=nm)
        ax.fill_between(radii, np.maximum(mean - se, 1e-3), mean + se,
                        color=_c(nm), alpha=0.10)
    ax.set_yscale("log")
    ax.set_xlabel(r"intervention magnitude $\|\nu\|$")
    ax.set_ylabel(r"MSE under $\mathrm{do}(A:=\nu)$")
    ax.set_title("Robustness curve (flat = immunised)")
    ax.legend()
    _save(fig, "fig3_robustness_curve")


def fig4_pareto(syn_pareto):
    """Fig. 4 — adaptation-immunisation Pareto frontier as xi sweeps."""
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for s, pts in syn_pareto.items():
        xis = [p["xi"] for p in pts]
        inr = [p["in_regime_MSE"] for p in pts]
        wdo = [p["worst_do_MSE"] for p in pts]
        ax.plot(inr, wdo, "-o", ms=3, lw=1.0, alpha=0.7, label=f"seed {s}")
        for p in pts:
            if p["xi"] in (0.0, 8.0, 128.0):
                ax.annotate(f"$\\xi$={p['xi']:.0f}", (p["in_regime_MSE"],
                            p["worst_do_MSE"]), fontsize=7,
                            textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel("in-regime MSE (adaptation)")
    ax.set_ylabel("worst-case do(A) MSE (immunisation)")
    ax.set_title(r"Adaptation–immunisation frontier ($\xi:0\to\infty$)")
    ax.legend()
    _save(fig, "fig4_pareto")


def fig5_immunization(seed=0, T=8000):
    """Fig. 5 — immunisation: the anchor--residual correlation over time.

    The mechanism of Sec 2.2 is to drive ``E[a r] -> 0``.  We recompute, on the
    seed-0 SCM, the running standardised correlation between the anchor and the
    forecast residual for OGD (xi=0) and OARF (xi>0); OARF holds it near zero
    while the reactive learner lets regime structure leak into its residual.
    """
    from .synthetic import make_scm, SCMConfig
    from .models import OARF
    from .online import EMA, OnlineStandardizer
    scm = make_scm(SCMConfig(T=T, seed=seed))

    def run(xi):
        m = OARF(scm.cfg.d, scm.cfg.p, B=scm.B_true, q=scm.cfg.q, xi=xi, eta=0.1)
        sy = OnlineStandardizer(1)
        ear, eva, evr = EMA(scm.cfg.q, 0.99), EMA(scm.cfg.q, 0.99), EMA(1, 0.99)
        tr = np.empty(T)
        for t in range(T):
            yhat = m.predict(scm.X[t], scm.Z[t])
            ys = float(sy.transform([scm.Y[t]])[0])
            r = ys - yhat
            a = m.B.T @ m.std_z.transform(scm.Z[t])
            ca, va = ear.update(a * r), eva.update(a * a)
            vr = float(evr.update([r * r])[0])
            tr[t] = float(np.mean(np.abs(ca) / (np.sqrt(va * vr) + 1e-8)))
            m.update(scm.X[t], scm.Z[t], ys); sy.update([scm.Y[t]])
        return tr

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    sm = lambda a: np.convolve(a, np.ones(100) / 100, mode="same")
    ax.plot(sm(run(0.0)), color=_c("OGD"), lw=1.4, label=r"OGD ($\xi=0$)")
    ax.plot(sm(run(6.0)), color=_c("OARF"), lw=1.8, label=r"OARF ($\xi=6$)")
    for c in _load(os.path.join(SYN, "seed0_detail.json"))["changepoints"]:
        ax.axvline(c, color="k", ls=":", lw=0.5, alpha=0.3)
    ax.set_xlabel("time step $t$")
    ax.set_ylabel(r"$|\mathrm{corr}(a_t, r_t)|$ (EMA)")
    ax.set_title("Immunisation: OARF drives the anchor--residual correlation to zero")
    ax.legend()
    _save(fig, "fig5_immunization")


def fig6_learned_channel(syn_perseed):
    """Fig. 6 — online channel discovery: subspace alignment to truth over time."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    plotted = False
    for sr in syn_perseed:
        cd = sr["rows"].get("OARF-CD", {})
        traj = cd.get("alignment_traj")
        if not traj:
            continue
        steps = [t[0] for t in traj]
        al = [t[1] for t in traj]
        ax.plot(steps, al, color=_c("OARF-CD"), lw=1.0, alpha=0.45)
        plotted = True
    if plotted:
        # mean trajectory across seeds (interpolated on a common grid)
        grids = [np.array([t[0] for t in sr["rows"]["OARF-CD"]["alignment_traj"]])
                 for sr in syn_perseed if sr["rows"].get("OARF-CD", {}).get("alignment_traj")]
        common = grids[0]
        vals = []
        for sr in syn_perseed:
            traj = sr["rows"].get("OARF-CD", {}).get("alignment_traj")
            if traj:
                g = np.array([t[0] for t in traj]); a = np.array([t[1] for t in traj])
                vals.append(np.interp(common, g, a))
        mean = np.mean(vals, 0)
        ax.plot(common, mean, color=_c("OARF-CD"), lw=2.6, label="mean over seeds")
    ax.axhline(0.42, color="k", ls=":", lw=0.8, label="random-subspace baseline")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("subspace alignment to true channel")
    ax.set_title("Online channel discovery (Sec 2.3): recovery over time")
    ax.legend()
    _save(fig, "fig6_learned_channel")


def fig7_per_regime_heatmap(syn_perseed):
    """Fig. 7 — per-regime MSE heatmap (methods x regimes), seed-averaged."""
    names = list(syn_perseed[0]["per_regime"].keys())
    reg_ids = syn_perseed[0]["regime_ids"]
    M = np.full((len(names), len(reg_ids)), np.nan)
    for i, nm in enumerate(names):
        for j, rid in enumerate(reg_ids):
            vals = [sr["per_regime"][nm].get(str(rid)) for sr in syn_perseed
                    if sr["per_regime"][nm].get(str(rid)) is not None]
            if vals:
                M[i, j] = np.mean(vals)
    # normalise each regime column by the column min for readability
    Mn = M / np.nanmin(M, axis=0, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    im = ax.imshow(Mn, aspect="auto", cmap="RdYlGn_r", vmin=1.0, vmax=2.0)
    ax.set_xticks(range(len(reg_ids)))
    ax.set_xticklabels([f"R{r}" for r in reg_ids])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("regime"); ax.set_title("Per-regime MSE (relative to best method)")
    fig.colorbar(im, ax=ax, label="MSE / best-in-regime")
    _save(fig, "fig7_per_regime_heatmap")


def fig8_dm_heatmap(syn_perseed):
    """Fig. 8 — DM-test significance heatmap (pairwise, seed-0)."""
    dm = syn_perseed[0]["dm"]
    names = dm["names"]
    stat = np.array(dm["stat"])
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(stat, cmap="coolwarm", vmin=-6, vmax=6)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Diebold–Mariano statistic (row vs column; <0 = row better)")
    fig.colorbar(im, ax=ax, label="DM statistic")
    _save(fig, "fig8_dm_heatmap")


def fig9_real_regime(real_detail, real_metrics, target):
    """Fig. 9 — real-data: realised vol regimes + per-regime/post-cp bars."""
    reg = np.array(real_detail["regimes"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    se = real_detail["sq_err"]
    for nm in ["OGD", "OARF"]:
        if nm in se:
            roll = np.convolve(np.array(se[nm]), np.ones(120) / 120, mode="same")
            ax.plot(roll, color=_c(nm), lw=1.0, label=nm)
    ax2 = ax.twinx()
    ax2.fill_between(range(len(reg)), 0, reg, color="gray", alpha=0.12, step="mid")
    ax2.set_yticks(sorted(np.unique(reg))); ax2.set_ylabel("vol regime")
    ax.set_xlabel("trading day"); ax.set_ylabel("rolling sq. error")
    ax.set_title(f"{target}: rolling error & vol regimes"); ax.legend(loc="upper left")

    ax = axes[1]
    rows = real_metrics[target]["rows"]
    order = sorted(rows.keys(), key=lambda n: rows[n]["worst_regime_MSE"])
    vals = [rows[n]["worst_regime_MSE"] for n in order]
    ax.barh(range(len(order)), vals, color=[_c(n) for n in order], alpha=0.9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("worst-regime MSE")
    ax.set_title(f"{target}: worst-regime MSE")
    _save(fig, f"fig9_real_{target}")


def fig10_equity(real_metrics, target):
    """Fig. 10 — volatility-timing equity curves (decision metric, Sec 6)."""
    rows = real_metrics[target]["rows"]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for nm in ["OARF", "OARF-CD", "OGD", "FC-DRO(Liu26)", "DR-Combo(Wang26)"]:
        if nm in rows and rows[nm].get("equity_curve"):
            ax.plot(rows[nm]["equity_curve"], color=_c(nm), lw=1.2, label=nm)
    ax.set_xlabel("trading day"); ax.set_ylabel("cumulative timing P&L")
    ax.set_title(f"{target}: volatility-timing equity curves")
    ax.legend()
    _save(fig, f"fig10_equity_{target}")


def fig11_ablations(abl):
    """Fig. 11 — ablations: beta sensitivity, component knock-outs, channel."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    # (a) beta sensitivity
    ax = axes[0]
    betas = sorted(float(k) for k in abl["beta"])
    wdo = [abl["beta"][f"{b}"]["worst_do_MSE"]["mean"] for b in betas]
    wsd = [abl["beta"][f"{b}"]["worst_do_MSE"]["sd"] for b in betas]
    mse = [abl["beta"][f"{b}"]["in_regime_MSE"]["mean"] for b in betas]
    ax.errorbar(betas, wdo, yerr=wsd, color=_c("OARF"), marker="o", lw=1.6,
                capsize=2, label="worst-$\\mathrm{do}(A)$ MSE")
    ax.plot(betas, mse, color=_c("OGD"), marker="s", lw=1.2, ls="--",
            label="in-regime MSE")
    ax.set_xlabel(r"EMA decay $\beta$"); ax.set_ylabel("MSE")
    ax.set_title("(a) Moment-tracking horizon"); ax.legend()

    # (b) component ablations (worst-do MSE)
    ax = axes[1]
    order = ["OARF (full)", "no centring", "no AdaGrad", "random channel",
             "xi=0 (OGD)"]
    order = [o for o in order if o in abl["components"]]
    vals = [abl["components"][o]["worst_do_MSE"]["mean"] for o in order]
    sds = [abl["components"][o]["worst_do_MSE"]["sd"] for o in order]
    cols = ["#d62728" if o == "OARF (full)" else "#888888" for o in order]
    ax.barh(range(len(order)), vals, xerr=sds, color=cols, alpha=0.9,
            error_kw=dict(lw=0.6))
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("worst-$\\mathrm{do}(A)$ MSE")
    ax.set_title("(b) Component knock-outs")

    # (c) learned-channel dimension q: worst-do MSE + alignment
    ax = axes[2]
    qs = sorted(int(k) for k in abl["channel_q"])
    wq = [abl["channel_q"][str(q)]["worst_do_MSE"]["mean"] for q in qs]
    aq = [abl["channel_q"][str(q)]["alignment"]["mean"] for q in qs]
    ax.axvline(2, color="k", ls=":", lw=0.7, alpha=0.4)   # true q (no legend)
    l1, = ax.plot(qs, wq, color=_c("OARF"), marker="o", lw=1.6,
                  label="worst-$\\mathrm{do}(A)$ MSE")
    ax.set_xlabel(r"discovered channel dim $q$"); ax.set_ylabel("worst-do MSE")
    ax.set_title("(c) Channel dimension"); ax.set_xticks(qs)
    ax2 = ax.twinx()
    l2, = ax2.plot(qs, aq, color="#2ca02c", marker="^", lw=1.2, ls="--",
                   label="subspace alignment")
    ax2.set_ylabel("alignment to truth", color="#2ca02c"); ax2.set_ylim(0, 1)
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=7,
              loc="center right")
    _save(fig, "fig11_ablations")


def main():
    syn = _load(os.path.join(SYN, "metrics.json"))
    syn_detail = _load(os.path.join(SYN, "seed0_detail.json"))
    fig1_rolling_error(syn_detail)
    fig2_bar_metrics(syn["aggregate"])
    fig3_robustness_curve(syn["per_seed"])
    fig4_pareto({int(k): v for k, v in syn["pareto"].items()})
    fig5_immunization()
    fig6_learned_channel(syn["per_seed"])
    fig7_per_regime_heatmap(syn["per_seed"])
    fig8_dm_heatmap(syn["per_seed"])

    abl_path = os.path.join(SYN, "ablations.json")
    if os.path.exists(abl_path):
        fig11_ablations(_load(abl_path))

    real_path = os.path.join(REAL, "metrics.json")
    if os.path.exists(real_path):
        real = _load(real_path)
        for tgt in real:
            dpath = os.path.join(REAL, f"detail_{tgt}.json")
            if os.path.exists(dpath):
                fig9_real_regime(_load(dpath), real, tgt)
            fig10_equity(real, tgt)


if __name__ == "__main__":
    main()
