# OARF — Online Anchor-Robust Forecasting under Economic Regime Shift
### Full research plan: algorithm, datasets, baselines, metrics, figures

This single document specifies everything needed to implement and evaluate the
method end to end. Notation is consistent throughout. Equations use `$...$`.

---

## 0. One-paragraph thesis

The online-forecasting-under-shift literature *reacts* to regime change (detect,
recalibrate, re-weight) and pays a re-adaptation cost at every break. We instead
*immunize* against shifts ex ante using the **causal channel** they travel
through: a streaming anchor-regression learner that drives the part of its
residual correlated with observable regime drivers (geopolitical risk, policy
uncertainty, volatility) to zero, and—our novel layer—**discovers that channel
online** instead of being handed it. The target guarantee is a new object,
*interventional regret*: regret against the best predictor under the worst-case
bounded future intervention on the regime driver.

---

## 1. Problem setup and notation

Discrete time $t = 1, \dots, T$. At each step we observe:

- $x_t \in \mathbb{R}^d$ — predictors (or base-model forecasts to be combined);
- $z_t \in \mathbb{R}^p$ — candidate **context / regime drivers** (GPR, EPU, VIX, energy shocks, lagged realized vol);
- $y_t \in \mathbb{R}$ — target (one-step-ahead return or volatility of EUA / Brent / Henry Hub / gold).

A linear predictor with feature map $\phi(x)=[1;x]\in\mathbb{R}^{d+1}$ and weights $w$:
$$\hat y_t = w^\top \phi(x_t), \qquad r_t = y_t - \hat y_t \ \text{(residual)}, \qquad \ell_t(w) = r_t^2 .$$

An **anchor** $a_t \in \mathbb{R}^q$ is a (possibly learned) function of context, $a_t = B^\top z_t$ with channel matrix $B \in \mathbb{R}^{p\times q}$. In the baseline OARF, $B$ is fixed (hand-chosen drivers); in the novel layer $B$ is learned online.

**Protocol (strict, leak-free).** At step $t$: predict $\hat y_t$ from $x_t$; incur $\ell_t$; *then* reveal $y_t$ and update. All standardization uses past-only statistics.

---

## 2. The algorithm

### 2.1 Causal-robustness objective (population)

Following anchor regression (Rothenhäusler et al., 2021), trade off predictability against invariance of the residual to the anchor:
$$
b_{\mathrm{AR}}(w;\xi) \;=\; \underbrace{\mathbb{E}\big[(y - w^\top\phi(x))^2\big]}_{\text{predict}}
\;+\; \xi\,\underbrace{\mathbb{E}[a\,r]^\top \,\big(\mathbb{E}[a a^\top]\big)^{-1}\,\mathbb{E}[a\,r]}_{P(w)\,=\,\|\mathrm{proj}_a r\|^2},
\qquad \xi \ge 0 .
$$
$P(w)$ is the squared length of the anchor-projection of the residual; penalizing it forces $\mathbb{E}[a\,r]\to 0$, i.e. the predictor stops exploiting any $a$-correlated (regime-specific) structure. $\xi=0$ recovers ordinary least squares.

**Interventional-robustness equivalence (the reason we do this).** For linear SCMs there exists $\rho(\xi)$ such that
$$
\min_w b_{\mathrm{AR}}(w;\xi)\;=\;\min_w \ \sup_{\nu:\ \|\nu\|\le \rho(\xi)} \ \mathbb{E}_{\,\mathrm{do}(a:=\nu)}\big[(y - w^\top\phi(x))^2\big].
$$
Minimizing the $\xi$-penalized loss is *exactly* minimizing worst-case MSE over bounded interventions $\mathrm{do}(a:=\nu)$ on the anchor. The robustness set is a (possibly off-center) ellipsoid shaped by $\mathbb{E}[aa^\top]$ — the causal channel — **not** an isotropic ball. This is the structural difference from DRO baselines (§5).

### 2.2 Online update (streaming anchor gradient)

Gradient of the population objective w.r.t. the coefficient block:
$$
\nabla_w\, b_{\mathrm{AR}} \;=\; -2\,\mathbb{E}[\phi\, r]\;-\;2\xi\,\mathbb{E}[\phi\, a^\top]\,\big(\mathbb{E}[a a^\top]\big)^{-1}\,\mathbb{E}[a\, r].
$$
Replace expectations by exponential-moving-average (EMA, decay $\beta$) estimates on **centered** variables (centering removes the regime mean-shift in $a$, which is the intervention, leaving covariance):
$$
\begin{aligned}
&m^{Xr}_t = \beta m^{Xr}_{t-1} + (1-\beta)\,\tilde x_t \tilde r_t, \quad
 m^{XA}_t = \beta m^{XA}_{t-1} + (1-\beta)\,\tilde x_t \tilde a_t^\top,\\
&M^{AA}_t = \beta M^{AA}_{t-1} + (1-\beta)\,\tilde a_t \tilde a_t^\top, \quad
 m^{Ar}_t = \beta m^{Ar}_{t-1} + (1-\beta)\,\tilde a_t \tilde r_t,
\end{aligned}
$$
where $\tilde{\cdot}$ denotes EMA-centering. The update (intercept unpenalized):
$$
\boxed{\;w_{t+1} = w_t - \eta\Big[\underbrace{-r_t\,\phi(x_t)}_{\text{prediction}} \;\underbrace{-\,2\xi\,[\,0;\,m^{XA}_t (M^{AA}_t+\epsilon I)^{-1} m^{Ar}_t\,]}_{\text{anchor robustness}} \;+\;\lambda_2 w_t\Big]\;}
$$
with step $\eta$, ridge $\epsilon$ on the anchor Gram, and $\ell_2$ weight $\lambda_2$. The EMA decay $\beta$ gives controlled forgetting so the covariance estimates track within-regime structure. Reference implementation: `models.py::OARF` (already validated; see `README.md`).

**Pseudocode.**
```
init w=0; EMA means/cross-moments=0
for t = 1..T:
    x,z,y <- observe;  a = B^T z
    xs, as <- standardize(x, a) using past-only stats
    yhat = w·[1;xs];  emit yhat;  r = y - yhat       # predict before update
    update EMA means; center xs,as,r; update m_Xr,m_XA,M_AA,m_Ar
    Minv_Ar = solve(M_AA + eps I, m_Ar)
    g = -r·[1;xs] ;  g[1:] += -2ξ·(m_XA · Minv_Ar)
    w <- w - η(g + λ2 w)
    absorb (x,a) into standardizer stats
```

### 2.3 Novel layer — online intervention-channel discovery

Baseline OARF assumes the analyst supplies the anchor. The contribution that lifts this above "known batch method, run online" is to **learn the channel** $B$ (which directions of $z_t$ act as the anchor) from the stream. Intuition: the channel is the subspace of $z$ along which the residual–context relationship is *most regime-unstable* — that is precisely where future shifts will travel and where invariance must be enforced.

Two-timescale objective. With $a_t=B^\top z_t$, update $w$ as in §2.2 (fast), and update $B$ (slow, rate $\eta_B\ll\eta$) to **maximize across-regime dispersion** of the anchor–residual covariance subject to a norm constraint:
$$
\max_{\|B\|_F \le 1}\ \mathcal{D}(B) \;=\; \mathrm{Var}_{\text{regimes}}\!\big(\,\mathbb{E}[\,(B^\top z)\, r \mid \text{regime}\,]\,\big)
\;-\;\gamma_{\!c}\,\big\|\mathbb{E}[(B^\top z)\,u]\big\|^2,
$$
where the first term seeks directions whose residual-coupling changes across regimes (the channel) and the penalty discourages loading on the stable signal $u$ (keeping the anchor exogenous-like). In streaming form, regimes are proxied by a forgetting partition or by an online change indicator; $\mathcal{D}$ is optimized by projected gradient ascent on $B$ with the same EMA machinery. This is where the ORACLE-style streaming-causal-discovery apparatus plugs in.

> Status: §2.2 is implemented and validated. §2.3 is the open, novel layer; the
> theorem below is stated for fixed $B$ and is the thing to confirm before the
> learned-$B$ version is scaled up.

### 2.4 Guarantee — interventional regret

Define, for radius $r\ge 0$,
$$
\mathrm{IntReg}_T(r) \;=\; \sum_{t=1}^T \ell_t(w_t)\;-\;\min_{w\in\mathcal{W}}\ \sum_{t=1}^T \ \sup_{\|\nu_t\|\le r}\ \mathbb{E}_{\,\mathrm{do}(a:=\nu_t)}\big[(y_t - w^\top\phi(x_t))^2\big].
$$

**Target theorem (fixed channel).** Under (A1) bounded features/anchor, (A2) the linear-SCM anchor model, (A3) the comparator class $\mathcal{W}$ is the set of **regime-conditional** predictors with at most $K$ switches, the streaming update of §2.2 attains
$$
\mathrm{IntReg}_T(\rho(\xi)) \;=\; O\!\big(\sqrt{T(1+K)}\big),
$$
without prior knowledge of $K$. *Proof strategy:* (i) the per-step anchor loss is convex in $w$, so OCO/OGD dynamic-regret machinery gives $O(\sqrt{T(1+P_T)})$ against a drifting comparator; (ii) the §2.1 equivalence converts the penalized comparator into the worst-case-intervention comparator; (iii) a concentration lemma bounds the error of the EMA cross-moment estimates under within-regime stationarity. Step (iii) is the genuinely new and hardest piece.

**Honesty caveat (shapes the comparator).** Adversarial no-regret for risk-adjusted objectives is impossible (Uziel & El-Yaniv, 2017), so $\mathcal{W}$ must be regime-conditional/stochastic, **not** fully adversarial. This is exactly why the observable regime drivers (GPR/EPU/VIX) enter the theorem: they define the regimes the comparator conditions on. Anchor methods give *interventional robustness, not causal identification* — do not claim to recover a causal effect (the estimand is the "diluted-causal" parameter).

---

## 3. Datasets — exact, scriptable URLs

All daily, aligned to a business-day calendar, forward-filled with a small cap.
Targets are regime-prone; the regime-driver panel supplies the anchors $z_t$.
Loader: `load_panel.py` (already written).

### 3.1 Targets and volatility-state drivers — FRED (public domain, citation requested)

FRED exposes a direct CSV endpoint, fully scriptable:
`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`

| Role | Series | Series ID | Exact URL |
|---|---|---|---|
| Target | Brent crude | `DCOILBRENTEU` | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU` |
| Target | Henry Hub gas | `DHHNGSP` | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP` |
| Target | Gold (LBMA PM) | `GOLDPMGBD228NLBM` | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=GOLDPMGBD228NLBM` |
| Driver | VIX (CBOE) | `VIXCLS` | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS` |
| Driver (opt) | EUR/USD | `DEXUSEU` | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU` |
| Driver (opt) | 10y Treasury | `DGS10` | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10` |

Missing values are coded `.` in FRED CSVs — coerce to NaN.

### 3.2 Geopolitical Risk — Caldara & Iacoviello (CC-BY)

- Daily (recent) Excel: `https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls`
- Monthly Excel (incl. 44 country indices): `https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls`
- Dated daily vintages follow `data_gpr_daily_recent_YYYYMMDD.xls`; daily file updated every Monday. Use columns `date` and `GPRD` (daily benchmark). Confirm the exact filename against the "Daily data … here" link on `https://www.matteoiacoviello.com/gpr.htm` before scripting.

### 3.3 Economic Policy Uncertainty — Baker, Bloom & Davis (free, citation)

- US daily EPU CSV: `https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv`
  (columns `year, month, day, daily_policy_index`). Confirm via `https://www.policyuncertainty.com/us_daily.html`.

### 3.4 EUA carbon price — **no clean programmatic URL** (manual export)

Daily EUA price is the one series without a one-line download. Use one of:
- **ICAP Allowance Price Explorer** — `https://icapcarbonaction.com/en/ets-prices` → "Download" (CSV; authoritative EEX auction-settlement since 2009). JS-driven, so export by hand once.
- **Investing.com** — `https://www.investing.com/commodities/carbon-emissions-historical-data` → "Download Data" (free account).

Save as a CSV with a date column and a price/close column and pass it to the loader via `--eua-csv`. ICAP/Sandbag note: pre-2010 prices were sourced from Quandl; post-2010 from EEX/ICAP.

### 3.5 Policy-intervention dates (the labeled $\mathrm{do}(a:=\nu)$ events)

Hand-code from the EU ETS calendar (EEA / European Commission): MSR activation (2019), intake-rate change (2024), Phase III→IV transition (2021), ETS2 milestones, major back-loading decisions. These dated events are the *ground-truth interventions* used to (a) define regimes for the comparator and (b) test the channel-discovery layer. Reference dataset (emissions/allowances, for context, CC0):
`https://raw.githubusercontent.com/datasets/eu-emissions-trading-system/main/data/eu-emissions-trading-system.csv` (note: emissions, not price).

### 3.6 Download script sketch

```python
import io, requests, pandas as pd
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
def fred(sid):
    df = pd.read_csv(io.BytesIO(requests.get(FRED.format(sid), timeout=60).content))
    s = pd.to_numeric(df.iloc[:,1], errors="coerce"); s.index = pd.to_datetime(df.iloc[:,0])
    return s.rename(sid).dropna()
for sid in ["DCOILBRENTEU","DHHNGSP","GOLDPMGBD228NLBM","VIXCLS"]:
    fred(sid).to_csv(f"{sid}.csv")
# GPR daily:
open("gpr_daily.xls","wb").write(requests.get(
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls", timeout=60).content)
# EPU daily:
open("epu_daily.csv","wb").write(requests.get(
    "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv", timeout=60).content)
```
(Full version in `load_panel.py`, which aligns everything to a business-day index.)

---

## 4. Feature construction

- **Returns:** $y_t = \log P_t - \log P_{t-1}$ for each target (and a volatility target: $\mathrm{RV}_t$ from squared returns or Garman–Klass if OHLC available).
- **Base experts $x_t$** (forecast-combination mode): AR(1)/AR(5), EWMA, GARCH(1,1) one-step forecast, random-forest on lags+drivers, and a small MLP. Combining heterogeneous experts is where robustness pays off.
- **Context / drivers $z_t$:** levels and 1-day changes of GPR, EPU, VIX; energy returns (Brent, Henry Hub); lagged target realized vol; policy-event indicator (±k-day windows around §3.5 dates).
- All standardized with expanding past-only z-scores (see `evaluate.py::OnlineStandardizer`).

---

## 5. Baselines (re-implementable; 2025 **and** 2026, plus foundational ablations)

Each entry: citation/year, mechanism, the update to re-implement, and the
contrast with OARF. Classic floors (OGD, Rolling-OLS, ACI) are included for
calibration; the *recent* comparison set is the 2025/2026 block.

### Floors (calibration)
- **OGD** (Zinkevich, 2003): $w_{t+1}=w_t+\eta\, r_t\phi(x_t)$. Reactive, anchor-blind. (= OARF with $\xi{=}0$.)
- **Rolling-OLS**: refit OLS on a trailing window $W$ each step.
- **ACI** (Gibbs & Candès, 2021): adaptive miscoverage $\alpha_{t+1}=\alpha_t+\gamma(\alpha-\mathbb{1}\{y_t\notin \hat C_t\})$. Coverage-targeting scalar adapter; for point comparison use its interval midpoint / a quantile-regression head.

### 2025 baselines
- **M-FISHER** (2025): sequential distribution-shift detection + adaptation. *Mechanism:* build an exponential test martingale from non-conformity scores, $W_t=\prod_{s\le t} e_s$, with Ville's inequality controlling false alarms; on evidence of shift, adapt parameters by **Fisher-preconditioned natural-gradient** steps (NGD on the distributional manifold), with detection delay $O(\log(1/\delta)/\Gamma)$. *Re-implement:* (i) non-conformity scores $\to$ e-values $e_s$; (ii) wealth process + Ville threshold; (iii) on trigger, NGD update $\theta\leftarrow\theta-\eta F^{-1}\nabla\mathrm{KL}$. *Contrast:* detect-then-adapt (reactive); OARF never needs a trigger.
- **WATCH / WCTM** (2025, arXiv:2505.04608): weighted conformal test martingales. *Mechanism:* monitor weighted conformal $p$-values; adapt interval sharpness to *benign* covariate shift while still flagging harmful concept shift, reducing false alarms vs standard CTMs. *Re-implement:* weighted-conformal $p$-values $\to$ martingale; adaptation rule on interval width; root-cause split (covariate vs concept). *Contrast:* adapts the *interval*, not the predictor's reliance on the shift channel.

### 2026 baselines
- **Online Randomized Distributionally Robust Forecast Combination** (Wang, 2026, *J. Time Series Analysis*, doi:10.1111/jtsa.70056): **most directly comparable** (a combiner). *Mechanism:* model weights as random draws from a parametric family; update the family's parameters online to minimize worst-case expected loss over a **Wasserstein ambiguity set centered at the empirical joint distribution** of forecasts and realizations. *Re-implement:* parameterize $w\sim q_\theta$ on the simplex (e.g. logistic-normal); sequential update of $\theta$ via stochastic mirror descent on the DR objective; randomization for exploration. *Contrast:* isotropic Wasserstein ball vs OARF's causal (anchor) robustness set.
- **FC-DRO-ES / Adaptive DR Forecast Combination** (Liu et al., 8 Jan 2026, arXiv:2601.04608): explicit Algorithm OA.5. *Update to re-implement:* variance-scaled exponential weights
  $\tilde w_{t,k}=w_{t-1,k}\exp\!\big(-\tfrac12\sum_{u=s}^{t-1} E_{u,k}^2/v_k\big)\,v_k^{-1/2}$, with rolling/EWMA error variance $v_k$; an Expected-Shortfall variant uses $L_k=\mathrm{ES}_\alpha(E_{:,k})$ and tail-penalized weights; mixing parameter $\lambda$, robustness $\eta$, DR-mean-variance radius $\tau$. *Contrast:* moment/ES ambiguity vs causal channel.
- **Wasserstein DRO Online Learning** (Chen, Fattahi, Shafiee, 2026, arXiv:2602.20403): *Mechanism:* online zero-sum game — at each $t$ the dual player picks the worst-case $Q_t$ in a Wasserstein ball centered on recent data, the primal player picks $w_t$; sublinear regret + risk control. *Re-implement:* primal-dual OGD with a per-step Wasserstein-DRO inner solve (closed form for quadratic loss). *Contrast:* agnostic ball; needlessly conservative when the true shift geometry is anisotropic (the economic case).
- **Cost-Aware Adaptive Conformal Inference** (2026, arXiv:2605.24463): *Mechanism:* ACI with a cost-coupled loss linking the miscoverage indicator to violation cost, giving dual control of violation frequency and cumulative cost. *Update:* $\theta_{t+1}=\theta_t+\gamma\,\partial_\theta[\text{cost-weighted miscoverage}]$. *Contrast:* asymmetric *coverage* cost, not regime immunization.

### Foundational ablations (causal-robust parents; offline)
- **Anchor Regression** (Rothenhäusler et al., 2021): closed form $\hat\gamma_{\mathrm{AR}}=[X^\top(I+\lambda P_A)X]^{-1}X^\top(I+\lambda P_A)Y$, refit on a rolling window — isolates the value of doing it *online* and the channel-discovery layer.
- **DRIG** (Shen et al., 2023): distributional robustness via invariant gradients (anchor regression is a special case) — batch invariance baseline.

> The 2025/2026 block is the headline comparison; the floors and anchor/DRIG
> ablations make the contribution legible (online vs batch; causal set vs DRO
> ball; learned vs fixed channel).

---

## 6. Metrics

Let $\hat y_t$ be point forecasts, $\hat F_t$ predictive distributions (if a
quantile/density head is used), $r(t)$ the regime id, $\mathcal{C}$ the
changepoint set.

### Point accuracy
- **MSE / RMSE** overall: $\frac1T\sum (y_t-\hat y_t)^2$.
- **Per-regime MSE** and **worst-regime MSE** $=\max_{r}\mathrm{MSE}_{r}$.
- **Post-changepoint MSE**: mean SE over $\bigcup_{c\in\mathcal C}[c, c{+}k)$ (re-adaptation cost; $k\approx 50$).

### Interventional robustness (the headline)
- **Worst-case interventional MSE**: freeze the learner, evaluate on held-out $\mathrm{do}(a:=\nu)$ environments with $\nu$ outside the training range; report $\max_\nu \mathrm{MSE}(\nu)$. (`evaluate.py::frozen_interventional`.)
- **Robustness curve**: $\mathrm{MSE}(\nu)$ as a function of $\|\nu\|$ (flat = robust).

### Distributional (if quantile/density head)
- **CRPS**, **pinball loss** at $\tau\in\{0.05,0.5,0.95\}$.
- **Empirical coverage** at nominal $1-\alpha$ and **coverage gap** $|(1-\alpha)-\widehat{\text{cov}}|$.
- **Mean interval width** and **Winkler / interval score**.

### Decision / economic (if the frictional allocator head is added)
- Realized **P&L**, **annualized Sharpe**, **max drawdown**, **CVaR/ES$_\alpha$** of returns, **turnover** $\sum\|a_t-a_{t-1}\|_1$.
- **Decision-regret** vs the best regime-conditional allocation policy.

### Online-learning diagnostics
- **Dynamic-regret proxy**: cumulative loss minus a rolling oracle's loss.
- **Detection/recovery delay**: steps for post-shift error to return within $\epsilon$ of pre-shift level.
- **Coefficient stability**: $\sum_t\|w_t-w_{t-1}\|$.

### Significance
- **Diebold–Mariano** pairwise tests on loss differentials (per target).
- **Model Confidence Set** (Hansen et al.) across all methods.
- Report **mean ± sd over ≥ 8 seeds** (synthetic) and rolling-origin folds (real).

---

## 7. Figures / charts

1. **Rolling squared error across regimes** with changepoints marked — reactive baselines spike at breaks, OARF flat. *(have: left panel of `oarf_synthetic_diagnostics.png`)*
2. **Regime-aware + interventional bar chart** (overall / post-cp / worst-do(A)). *(have: right panel)*
3. **Robustness curve**: MSE vs intervention magnitude $\|\nu\|$, all methods — the cleanest single picture of the claim.
4. **Adaptation–immunization Pareto frontier**: in-regime MSE vs worst-case interventional MSE as $\xi$ sweeps $0\to\infty$ (OGD at one end, fully-immunized at the other) — the paper's conceptual figure.
5. **Coefficient paths** $w_t$ on causal vs anchor-driven features over time — visualizes immunization.
6. **Learned channel** $B_t$ (loadings on GPR/EPU/VIX/energy) over time — interpretability of the novel layer; overlay policy-event dates.
7. **Per-regime MSE heatmap** (methods × regimes).
8. **DM-test significance heatmap** (pairwise).
9. **Coverage & CRPS over time** (if distributional head).
10. **Equity curve + drawdown** (if decision head).

---

## 8. Experimental protocol

- **Split:** expanding-window walk-forward; first ~20% for hyperparameter selection via rolling-origin CV, remainder for evaluation. No look-ahead anywhere.
- **Hyperparameters:** $\eta$ (step), $\xi$ (robustness), $\beta$ (EMA decay), $\epsilon$ (anchor ridge), $\lambda_2$; tune on the validation block only; report sensitivity to $\xi$ and $\beta$ as the Pareto frontier (Fig. 4).
- **Ablations:** (i) $\xi=0$ → recovers OGD (sanity); (ii) fixed channel vs learned channel (§2.3); (iii) with/without policy-event drivers; (iv) point-only vs +distributional head vs +decision head.
- **Targets:** EUA (climate-CfP headline), Brent, Henry Hub, gold; shared driver panel.
- **Robustness checks:** multiple seeds (synthetic), multiple rolling origins (real), alternative anchor sets.

---

## 9. Repository (current state)

```
oarf/
  synthetic.py    # SCM fixture + held-out intervention grid        [done]
  models.py       # OARF + OGD + Rolling-OLS                         [done]
  evaluate.py     # leak-free walk-forward, metrics, frozen do(A)    [done]
  run_synthetic.py# end-to-end demo + diagnostic plot               [done]
  load_panel.py   # FRED/GPR/EPU/EUA loader (run with open internet) [done]
  README.md       # status, synthetic-vs-real boundary              [done]
  RESEARCH_PLAN.md# this document
```
To build next: distributional (quantile) head; frictional allocator head;
channel-discovery (§2.3); baseline implementations (§5); figures 3–10;
DM / MCS significance harness.

---

## 10. Reference list (for the paper's related work)

- Rothenhäusler, Meinshausen, Bühlmann, Peters (2021). *Anchor regression*. JRSS-B.
- Shen et al. (2023). *DRIG: distributional robustness via invariant gradients*.
- Gibbs & Candès (2021). *Adaptive conformal inference*.
- Uziel & El-Yaniv (2017). *Growth-optimal portfolio selection under CVaR constraints* (impossibility of adversarial risk-adjusted no-regret).
- M-FISHER (2025). *Sequential test-time adaptation via martingale-driven Fisher prompting*.
- WATCH / WCTM (2025). arXiv:2505.04608.
- Wang (2026). *Online randomized distributionally robust forecast combination for dependent data*. JTSA, doi:10.1111/jtsa.70056.
- Liu et al. (2026). *Forecasting the U.S. Treasury yield curve: a distributionally robust ML approach*. arXiv:2601.04608.
- Chen, Fattahi, Shafiee (2026). *Wasserstein distributionally robust online learning*. arXiv:2602.20403.
- Cost-Aware Adaptive Conformal Inference (2026). arXiv:2605.24463.
- Caldara & Iacoviello (2022). *Measuring geopolitical risk*. AER 112(4).
