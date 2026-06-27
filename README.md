# OARF — Online Anchor-Robust Forecasting under Economic Regime Shift

A streaming anchor-regression learner that **immunises** a forecaster against
economic regime shifts *ex ante* by driving the part of its residual correlated
with observable regime drivers (geopolitical risk, policy uncertainty,
volatility) toward zero — and, in its novel layer, **discovers that causal
channel online** instead of being handed it. The target guarantee is a new
object, *interventional regret*: regret against the best regime-conditional
predictor under the worst-case bounded future intervention on the regime driver.

This repository contains the full implementation, the leak-free evaluation
harness, the data-build scripts, the figure code, and a Springer-formatted
manuscript. Every number and figure in the paper is reproducible from source.

> The research plan (algorithm, datasets, baselines, metrics, figures) is in
> [`algorithm.md`](algorithm.md) / [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).

## Quick start

```bash
pip install -r requirements.txt

python run_synthetic.py --seeds 8 --T 8000   # controlled anchor-SCM benchmark
python download_data.py                       # fetch the public real-data panel
python run_real.py                            # energy-volatility combination
python -m oarf.figures                        # regenerate all figures

# build the manuscript (needs a TeX installation)
cd manuscript && pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

## What is implemented

| File | Contents | Status |
|---|---|---|
| `oarf/synthetic.py` | Canonical anchor-SCM (parents/children) + held-out `do(A:=ν)` grid | ✅ |
| `oarf/models.py` | **OARF** (§2.2), **online channel discovery** (§2.3), and all baselines (§5) | ✅ |
| `oarf/evaluate.py` | Leak-free walk-forward, full metric battery (§6), Diebold–Mariano, Model Confidence Set | ✅ |
| `oarf/online.py` | Past-only winsorised standardiser, EMA, AdaGrad step | ✅ |
| `oarf/load_panel.py` | FRED / GPR / EPU loader, HAR/GARCH experts, vol regimes (§3–4) | ✅ |
| `oarf/figures.py` | Figures 1–10 (§7) | ✅ |
| `download_data.py` | Scriptable download of the public panel | ✅ |
| `run_synthetic.py`, `run_real.py` | End-to-end experiments | ✅ |
| `manuscript/` | Springer-formatted paper (`manuscript.pdf`) + `references.bib` | ✅ |

### Baselines (§5)
Floors — **OGD** (= OARF with ξ=0), **Rolling-OLS**, **ACI**.
2025 — **M-FISHER** (martingale-triggered Fisher natural-gradient adaptation),
**WATCH/WCTM** (weighted conformal test martingales).
2026 — **DR-Combo** (Wang, randomised Wasserstein-DRO combination),
**FC-DRO / FC-DRO-ES** (Liu et al., variance-/ES-scaled exponential weights),
**W-DRO-OL** (Chen et al., Wasserstein DRO online), **CostACI** (cost-aware ACI).
Foundational causal-robust ablations — **AnchorReg** (batch, rolling window),
**DRIG**.

## The synthetic-vs-real boundary (read this)

The two experiments answer different questions and the paper is explicit about it.

* **Synthetic (controlled).** A linear anchor-SCM with a *known* causal channel
  and *known* anchor-mediated confounding (anti-causal children of `Y` driven by
  the anchor). Here the held-out `do(A:=ν)` grid lets us measure genuine
  interventional robustness against ground truth. **OARF cuts worst-case
  interventional MSE by 55–78%** versus reactive/DRO online methods, approaching
  the *batch* causal oracle while remaining fully online, and the
  channel-discovery layer recovers the true intervention subspace
  (mean alignment **0.87**, random baseline 0.42). This is where the mechanism is
  demonstrated decisively.

* **Real (deployment).** Sixteen years (2010–2026) of daily public data, framed
  as one-step-ahead **log realised-variance** forecasting (predictable and
  regime-dependent) combined from heterogeneous experts. There is **no** ground-
  truth intervention, so we report regime-aware, distributional and
  decision-economic metrics with DM/MCS significance. OARF / OARF-CD are within
  the Model Confidence Set of the best method and **lead the 2026 DRO combiners
  and reactive online learners on the regime-robustness metrics**, with the
  *learned* channel giving the best worst-regime MSE in the OARF family. Gains are
  modest in absolute terms — on real series the driver–residual coupling is weaker
  than in the controlled SCM — and we say so. We do **not** claim causal
  identification; the estimand is the diluted-causal parameter.

## Data sources (§3)

All public, citation-requested. `download_data.py` fetches them into `data/`:

* **FRED** CSV endpoint — Brent (`DCOILBRENTEU`), Henry Hub (`DHHNGSP`), VIX
  (`VIXCLS`), EUR/USD, 10y Treasury.
* **Geopolitical Risk** (Caldara & Iacoviello) — daily `GPRD`.
* **Economic Policy Uncertainty** (Baker, Bloom & Davis) — US daily index.
* **EUA carbon** has no clean programmatic feed: export it by hand (ICAP /
  Investing.com) to `data/eua.csv` and the loader picks it up. The daily LBMA
  gold series was discontinued at FRED; the two energy targets are used by default.

## Reproducibility

`run_synthetic.py` writes `results/synthetic/metrics.json` (+ per-seed detail);
`run_real.py` writes `results/real/metrics.json` (+ per-target detail);
`oarf/figures.py` reads those and writes `figures/*.{pdf,png}`. Seeds are fixed,
the protocol is strictly leak-free (predict → reveal → update; past-only
standardisation replayed frozen on the held-out grid).

## Licence

See [`LICENSE`](LICENSE).
