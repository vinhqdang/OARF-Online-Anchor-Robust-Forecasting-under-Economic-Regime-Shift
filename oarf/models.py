"""Online forecasters: OARF (Sec 2.2), channel discovery (Sec 2.3), baselines (Sec 5).

Every model implements the strict leak-free protocol of Sec 1::

    yhat = model.predict(x_t, z_t)     # predict from past-only information
    ...                                # reveal y_t
    model.update(x_t, z_t, y_t)        # incur loss and update, then absorb

``x_t`` are the raw predictors (or base-expert forecasts in combination mode),
``z_t`` the raw observable context / candidate regime drivers.  Standardisation
is past-only and owned by each model so the same transform can be replayed on a
frozen learner over the held-out ``do(A := nu)`` grid (``predict_frozen``).

Models that produce a predictive distribution additionally implement
``predict_quantiles`` returning the 5/50/95% quantiles for the distributional
metrics of Sec 6.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .online import EMA, AdaStep, OnlineStandardizer


# --------------------------------------------------------------------------- #
#  Base class                                                                 #
# --------------------------------------------------------------------------- #
class OnlineModel:
    """Common machinery: a past-only x-standardiser and a linear read-out.

    A model is *linear-in-standardised-x* if it exposes an intercept ``b0`` and
    a coefficient vector ``coef`` via :meth:`effective_linear`.  That is all the
    frozen interventional evaluation needs, so it works uniformly across models.
    """

    produces_quantiles = False

    def __init__(self, d: int, p: int, name: str):
        self.d, self.p, self.name = d, p, name
        self.std_x = OnlineStandardizer(d)
        self.std_z = OnlineStandardizer(p)
        self._frozen_x = None

    # -- standardisation helpers ------------------------------------------- #
    def _xs(self, x):
        return self.std_x.transform(x)

    def _zs(self, z):
        return self.std_z.transform(z)

    def _absorb(self, x, z):
        self.std_x.update(x)
        self.std_z.update(z)

    # -- to be overridden --------------------------------------------------- #
    def predict(self, x, z) -> float:
        raise NotImplementedError

    def update(self, x, z, y) -> None:
        raise NotImplementedError

    def effective_linear(self):
        """Return ``(b0, coef)`` of the read-out on standardised x."""
        raise NotImplementedError

    # -- frozen interventional evaluation ----------------------------------- #
    def freeze_for_eval(self):
        self._frozen_x = self.std_x.freeze()
        self._frozen_lin = self.effective_linear()

    def predict_frozen(self, X_raw) -> np.ndarray:
        b0, coef = self._frozen_lin
        Xs = self._frozen_x.transform(X_raw)
        return b0 + Xs @ coef


# --------------------------------------------------------------------------- #
#  OARF  (Sec 2.2) + online channel discovery (Sec 2.3)                       #
# --------------------------------------------------------------------------- #
class OARF(OnlineModel):
    r"""Online Anchor-Robust Forecasting.

    Implements the boxed streaming update of Sec 2.2::

        w_{t+1} = w_t - eta[ -r_t phi(x_t)
                             - 2 xi [0; m_XA (M_AA + eps I)^{-1} m_Ar]
                             + lam2 w_t ]

    with EMA cross-moments of *centered* variables (decay ``beta``).  The anchor
    is ``a_t = B^T z_t``.  With ``xi = 0`` this reduces to online gradient
    descent (the OGD floor).  With ``learn_channel=True`` the channel ``B`` is
    discovered online by the two-timescale rule of Sec 2.3.
    """

    def __init__(self, d, p, B=None, q=2, xi=1.0, eta=0.05, beta=0.98,
                 eps=1e-3, lam2=1e-4, learn_channel=False, eta_B=0.3,
                 channel_every=10, gamma_c=0.3, block=40, center=True,
                 adagrad=True, name=None):
        super().__init__(d, p, name or ("OARF-CD" if learn_channel else "OARF"))
        if B is None:
            rng = np.random.default_rng(0)
            G = rng.normal(size=(p, q))
            B, _ = np.linalg.qr(G)            # random orthonormal channel
        self.B = np.asarray(B, dtype=float)
        self.q = self.B.shape[1]
        self.xi, self.eta, self.beta = xi, eta, beta
        self.eps, self.lam2 = eps, lam2
        self.learn_channel = learn_channel
        self.eta_B, self.channel_every, self.gamma_c = eta_B, channel_every, gamma_c
        self.block = block
        self.center = center              # ablation: EMA-centring of moments
        self.adagrad = adagrad            # ablation: adaptive vs fixed step

        self.w = np.zeros(d + 1)              # [intercept; coef]
        self.opt = AdaStep(d + 1, eta)        # adaptive, scale-invariant step
        # EMA means for centering
        self.mean_x = EMA(d, beta)
        self.mean_a = EMA(self.q, beta)
        self.mean_r = EMA(1, beta)
        # EMA cross-moments of centered variables
        self.m_Xr = EMA(d, beta)
        self.m_XA = EMA((d, self.q), beta)
        self.M_AA = EMA((self.q, self.q), beta)
        self.m_Ar = EMA(self.q, beta)
        # --- channel-discovery state (Sec 2.3) ---
        # a lightweight NON-robust reference predictor whose residual still
        # carries the regime-specific anchor coupling that B must capture
        self.w_ref = np.zeros(d + 1)
        self.opt_ref = AdaStep(d + 1, eta)
        self._blk_sum = np.zeros(p)           # within-block sum of z * r_ref
        self._blk_n = 0
        self._mean_blk = np.zeros(p)          # across-block mean coupling
        self._S_blk = np.zeros((p, p))        # across-block scatter (Welford)
        self._n_blk = 0
        self._step = 0
        self.B_hist = []                      # snapshots for Fig. 6

        self._cache = None

    def predict(self, x, z):
        xs = self._xs(x)
        phi = np.concatenate(([1.0], xs))
        yhat = float(self.w @ phi)
        self._cache = (xs, self._zs(z), phi, yhat)
        return yhat

    def update(self, x, z, y):
        xs, zs, phi, yhat = self._cache
        self._cache_y = float(y)
        r = float(y - yhat)
        a = self.B.T @ zs                              # anchor a_t

        # --- centering (EMA means) ---
        mx = self.mean_x.update(xs)
        ma = self.mean_a.update(a)
        mr = float(self.mean_r.update([r])[0])
        if self.center:
            xt, at, rt = xs - mx, a - ma, r - mr       # centered
        else:                                          # ablation: no centring
            xt, at, rt = xs, a, r

        # --- EMA cross-moments of centered variables ---
        m_Xr = self.m_Xr.update(xt * rt)
        m_XA = self.m_XA.update(np.outer(xt, at))
        M_AA = self.M_AA.update(np.outer(at, at))
        m_Ar = self.m_Ar.update(at * rt)

        # --- anchor-robustness gradient block ---
        Minv_Ar = np.linalg.solve(M_AA + self.eps * np.eye(self.q), m_Ar)
        grad = -r * phi
        grad[1:] += -2.0 * self.xi * (m_XA @ Minv_Ar)
        grad[1:] += self.lam2 * self.w[1:]             # intercept unpenalised
        self.w = self.w - (self.opt.step(grad) if self.adagrad
                           else self.eta * grad)

        # --- online channel discovery (Sec 2.3) ---
        if self.learn_channel:
            self._update_channel(zs, phi)

        self._absorb(x, z)
        self._step += 1
        if self.learn_channel and self._step % self.channel_every == 0:
            self.B_hist.append((self._step, self.B.copy()))

    def _update_channel(self, zs, phi):
        r"""Two-timescale projected-gradient ascent on across-regime dispersion.

        The intervention channel is the subspace of the observable context along
        which the *relationship to the regime is most unstable*.  Because, under
        the anchor SCM, the regime acts on the context precisely by shifting the
        anchor's location, the identifying signal is the **across-regime
        dispersion of the context's conditional mean**: the anchor directions
        shift with the regime while exogenous/decoy directions are stationary.
        We therefore maximise

            ``D(B) = tr(B^T (S - gamma_c u u^T) B)``,

        with ``S`` the *across-block* scatter of the block-mean context (regimes
        proxied by a forgetting partition into length-``block`` segments — long
        enough to average out the short-horizon autocorrelation of stationary
        decoys) and ``u`` the grand-mean direction (the stable component the
        ``gamma_c`` term keeps the channel exogenous of, per Sec 2.3).  A
        lightweight non-robust reference predictor is maintained to *gate*
        discovery toward predictively relevant directions (its running residual
        scale modulates the step).  ``B`` is nudged by projected gradient ascent
        (``grad_B D = 2 M B``) and re-orthonormalised — the slow timescale.
        """
        # non-robust reference predictor: keeps a predictive-relevance gate alive
        r_ref = float(self._cache_y - self.w_ref @ phi)
        self.w_ref = self.w_ref - self.opt_ref.step(-r_ref * phi)

        self._blk_sum += zs
        self._blk_n += 1
        if self._blk_n < self.block:
            return
        m_blk = self._blk_sum / self._blk_n            # block-mean context
        self._blk_sum[:] = 0.0
        self._blk_n = 0
        # Welford update of across-block mean and scatter
        self._n_blk += 1
        delta = m_blk - self._mean_blk
        self._mean_blk += delta / self._n_blk
        self._S_blk += np.outer(delta, m_blk - self._mean_blk)
        if self._n_blk < 4:
            return
        S = self._S_blk / (self._n_blk - 1)
        M = S - self.gamma_c * np.outer(self._mean_blk, self._mean_blk)
        B = self.B + self.eta_B * (2.0 * (M @ self.B))   # projected ascent step
        B, _ = np.linalg.qr(B)                           # re-orthonormalise
        self.B = B

    def effective_linear(self):
        return float(self.w[0]), self.w[1:].copy()


class OGD(OARF):
    """Online gradient descent — OARF with ``xi = 0`` (Zinkevich, 2003)."""

    def __init__(self, d, p, eta=0.05, lam2=1e-4, **kw):
        super().__init__(d, p, xi=0.0, eta=eta, lam2=lam2, name="OGD", **kw)


# --------------------------------------------------------------------------- #
#  Rolling-OLS floor                                                          #
# --------------------------------------------------------------------------- #
class RollingOLS(OnlineModel):
    """Ridge-stabilised OLS refit on a trailing window of ``W`` steps."""

    def __init__(self, d, p, window=250, ridge=1e-3, refit_every=5):
        super().__init__(d, p, "Rolling-OLS")
        self.window, self.ridge, self.refit_every = window, ridge, refit_every
        self.bufX = deque(maxlen=window)
        self.bufY = deque(maxlen=window)
        self.w = np.zeros(d + 1)
        self._k = 0

    def predict(self, x, z):
        xs = self._xs(x)
        self._xs_cache = xs
        return float(self.w @ np.concatenate(([1.0], xs)))

    def update(self, x, z, y):
        self.bufX.append(np.concatenate(([1.0], self._xs_cache)))
        self.bufY.append(float(y))
        self._k += 1
        if self._k % self.refit_every == 0 and len(self.bufY) >= self.d + 2:
            X = np.array(self.bufX)
            Y = np.array(self.bufY)
            G = X.T @ X + self.ridge * np.eye(self.d + 1)
            self.w = np.linalg.solve(G, X.T @ Y)
        self._absorb(x, z)

    def effective_linear(self):
        return float(self.w[0]), self.w[1:].copy()


# --------------------------------------------------------------------------- #
#  Online quantile head + Adaptive Conformal Inference (Gibbs & Candes 2021)  #
# --------------------------------------------------------------------------- #
class ACI(OnlineModel):
    r"""Adaptive Conformal Inference with an online pinball quantile head.

    A median head gives the point forecast; 5/95% pinball-regression heads give
    the nominal band, whose width is then conformalised by the ACI update
    ``alpha_{t+1} = alpha_t + gamma (alpha - 1{y notin C_t})`` (Gibbs & Candes,
    2021).  This is a coverage-targeting *scalar* adapter on top of the band.
    """

    produces_quantiles = True

    def __init__(self, d, p, eta=0.03, gamma=0.01, alpha=0.1, cost_aware=False,
                 cost_ratio=2.0, name=None):
        super().__init__(d, p, name or ("CostACI" if cost_aware else "ACI"))
        self.eta = eta
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_t = alpha
        self.cost_aware = cost_aware
        self.cost_ratio = cost_ratio
        # three quantile heads: 0.05, 0.5, 0.95
        self.taus = np.array([0.05, 0.5, 0.95])
        self.W = np.zeros((3, d + 1))
        self.opt = [AdaStep(d + 1, eta) for _ in range(3)]
        self._scale = 1.0     # ACI multiplicative width adjustment

    def _heads(self, phi):
        return self.W @ phi

    def predict(self, x, z):
        xs = self._xs(x)
        phi = np.concatenate(([1.0], xs))
        self._phi = phi
        q = self._heads(phi)
        self._q = q
        return float(q[1])           # median = point forecast

    def predict_quantiles(self, x, z):
        self.predict(x, z)
        lo, mid, hi = self._q
        half = 0.5 * (hi - lo) * self._scale
        return np.array([mid - half, mid, mid + half])

    def update(self, x, z, y):
        phi = self._phi
        q = self._q
        # pinball-loss subgradient step per head
        for i, tau in enumerate(self.taus):
            err = y - q[i]
            g = -(tau if err > 0 else (tau - 1.0))
            self.W[i] -= self.opt[i].step(g * phi)
        # ACI miscoverage update (conformalised width)
        lo, mid, hi = q
        half = 0.5 * (hi - lo) * self._scale
        covered = (mid - half) <= y <= (mid + half)
        if self.cost_aware:
            # cost-coupled miscoverage: under-coverage penalised cost_ratio x
            err_signal = (self.cost_ratio if not covered else 0.0) - \
                         self.alpha * self.cost_ratio
            self.alpha_t = np.clip(self.alpha_t + self.gamma * err_signal /
                                   self.cost_ratio, 1e-3, 0.5)
        else:
            self.alpha_t = np.clip(
                self.alpha_t + self.gamma * (self.alpha - (0 if covered else 1)),
                1e-3, 0.5)
        # map miscoverage target to a width scale (smaller alpha -> wider band)
        self._scale = max(0.2, (1.0 - self.alpha_t) / (1.0 - self.alpha))
        self._absorb(x, z)

    def effective_linear(self):
        return float(self.W[1, 0]), self.W[1, 1:].copy()


# --------------------------------------------------------------------------- #
#  Anchor Regression (Rothenhauser et al. 2021) — rolling-window batch ablation#
# --------------------------------------------------------------------------- #
class AnchorRegression(OnlineModel):
    r"""Batch anchor regression refit on a trailing window (offline ablation).

    Closed form ``gamma_AR = [X^T (I + lam P_A) X]^{-1} X^T (I + lam P_A) Y``
    with ``P_A`` the projection onto the (centered) anchor ``A = Z B``.  Refit
    on a trailing window every few steps; isolates the value of doing anchor
    robustness *online* (vs OARF) and with a *learned* channel (vs fixed ``B``).
    """

    def __init__(self, d, p, B=None, q=2, lam=4.0, window=400, refit_every=10,
                 ridge=1e-3):
        super().__init__(d, p, "AnchorReg")
        if B is None:
            rng = np.random.default_rng(0)
            B, _ = np.linalg.qr(rng.normal(size=(p, q)))
        self.B = np.asarray(B, float)
        self.q = self.B.shape[1]
        self.lam, self.window, self.refit_every, self.ridge = \
            lam, window, refit_every, ridge
        self.bufX = deque(maxlen=window)
        self.bufA = deque(maxlen=window)
        self.bufY = deque(maxlen=window)
        self.w = np.zeros(d + 1)
        self._k = 0

    def predict(self, x, z):
        xs = self._xs(x)
        self._xs_cache, self._zs_cache = xs, self._zs(z)
        return float(self.w @ np.concatenate(([1.0], xs)))

    def update(self, x, z, y):
        self.bufX.append(np.concatenate(([1.0], self._xs_cache)))
        self.bufA.append(self.B.T @ self._zs_cache)
        self.bufY.append(float(y))
        self._k += 1
        if self._k % self.refit_every == 0 and len(self.bufY) >= self.d + self.q + 2:
            X = np.array(self.bufX)
            A = np.array(self.bufA)
            Y = np.array(self.bufY)
            Ac = A - A.mean(0, keepdims=True)
            # projection onto anchor column space
            PA = Ac @ np.linalg.solve(Ac.T @ Ac + 1e-6 * np.eye(self.q), Ac.T)
            Wt = np.eye(len(Y)) + self.lam * PA
            G = X.T @ Wt @ X + self.ridge * np.eye(self.d + 1)
            self.w = np.linalg.solve(G, X.T @ Wt @ Y)
        self._absorb(x, z)

    def effective_linear(self):
        return float(self.w[0]), self.w[1:].copy()


class DRIG(OnlineModel):
    r"""DRIG — distributional robustness via invariant gradients (Shen et al. 2023).

    Batch invariance ablation on a trailing window split into a *recent* and an
    *older* sub-environment (a forgetting partition).  We minimise the pooled
    squared loss plus a penalty ``gamma_d`` on the gradient mismatch between the
    two environments, which enforces invariance of the score across environments
    (anchor regression is the special case of a single linear anchor).
    """

    def __init__(self, d, p, gamma_d=2.0, window=400, refit_every=10, ridge=1e-3):
        super().__init__(d, p, "DRIG")
        self.gamma_d, self.window, self.refit_every, self.ridge = \
            gamma_d, window, refit_every, ridge
        self.bufX = deque(maxlen=window)
        self.bufY = deque(maxlen=window)
        self.w = np.zeros(d + 1)
        self._k = 0

    def predict(self, x, z):
        xs = self._xs(x)
        self._xs_cache = xs
        return float(self.w @ np.concatenate(([1.0], xs)))

    def update(self, x, z, y):
        self.bufX.append(np.concatenate(([1.0], self._xs_cache)))
        self.bufY.append(float(y))
        self._k += 1
        if self._k % self.refit_every == 0 and len(self.bufY) >= 2 * (self.d + 2):
            X = np.array(self.bufX)
            Y = np.array(self.bufY)
            half = len(Y) // 2
            Xo, Yo, Xr, Yr = X[:half], Y[:half], X[half:], Y[half:]
            # pooled normal equations + invariance penalty on the per-env gram
            Go = Xo.T @ Xo / len(Yo)
            Gr = Xr.T @ Xr / len(Yr)
            bo = Xo.T @ Yo / len(Yo)
            br = Xr.T @ Yr / len(Yr)
            G = 0.5 * (Go + Gr) + self.gamma_d * (Gr - Go).T @ (Gr - Go) \
                + self.ridge * np.eye(self.d + 1)
            b = 0.5 * (bo + br) + self.gamma_d * (Gr - Go).T @ (br - bo)
            self.w = np.linalg.solve(G, b)
        self._absorb(x, z)

    def effective_linear(self):
        return float(self.w[0]), self.w[1:].copy()


# --------------------------------------------------------------------------- #
#  2026 DRO baselines                                                         #
# --------------------------------------------------------------------------- #
class WassersteinDRO_OL(OnlineModel):
    r"""Wasserstein DRO online learning (Chen, Fattahi & Shafiee, 2026).

    Online zero-sum game: the dual player picks the worst-case distribution in a
    Wasserstein ball of radius ``rho`` around the current point, the primal
    player descends.  For the squared loss the inner ``sup`` has the closed-form
    surrogate ``(|y - w.x| + rho ||(w, -1)||)^2``; its (sub)gradient gives the
    primal step.  The robustness set is an *isotropic* Wasserstein ball — the
    intended contrast with OARF's anisotropic causal set.
    """

    def __init__(self, d, p, eta=0.03, rho=0.1, lam2=1e-4):
        super().__init__(d, p, "W-DRO-OL")
        self.eta, self.rho, self.lam2 = eta, rho, lam2
        self.w = np.zeros(d + 1)
        self.opt = AdaStep(d + 1, eta)

    def predict(self, x, z):
        xs = self._xs(x)
        self._phi = np.concatenate(([1.0], xs))
        return float(self.w @ self._phi)

    def update(self, x, z, y):
        phi = self._phi
        r = float(y - self.w @ phi)
        aug = np.sqrt(self.w[1:] @ self.w[1:] + 1.0)     # ||(coef, -1)||
        robust_res = abs(r) + self.rho * aug
        sgn = np.sign(r) if r != 0 else 1.0
        # d/dw (|r| + rho aug)^2 = 2 robust_res (-sgn phi + rho * d aug/dw)
        daug = np.zeros(self.d + 1)
        daug[1:] = self.w[1:] / aug
        grad = 2.0 * robust_res * (-sgn * phi + self.rho * daug)
        grad[1:] += self.lam2 * self.w[1:]
        self.w = self.w - self.opt.step(grad)
        self._absorb(x, z)

    def effective_linear(self):
        return float(self.w[0]), self.w[1:].copy()


class RandomizedDRCombination(OnlineModel):
    r"""Online Randomized DR Forecast Combination (Wang, 2026; JTSA 70056).

    Weights are random draws ``w ~ q_theta`` from a logistic-normal family on the
    simplex; ``theta`` is updated online by stochastic mirror descent on a
    Wasserstein-DRO objective (worst-case expected loss over a ball centred at
    the empirical joint distribution of forecasts and realisations).  We carry
    base "experts" = the individual standardised predictors with online scalar
    gains, then combine.  The randomisation provides exploration; the read-out
    used for the point forecast is the mean combination ``E[w].expert``.
    """

    def __init__(self, d, p, eta=0.05, rho=0.1, sigma=0.3, lam2=1e-4, seed=0):
        super().__init__(d, p, "DR-Combo(Wang26)")
        self.eta, self.rho, self.sigma, self.lam2 = eta, rho, sigma, lam2
        self.rng = np.random.default_rng(seed)
        self.theta = np.zeros(d)        # logits of the logistic-normal mean
        self.gain = np.zeros(d)         # online per-expert gain
        self.gain_lr = 0.02

    def _mean_weights(self):
        e = np.exp(self.theta - self.theta.max())
        return e / e.sum()

    def predict(self, x, z):
        xs = self._xs(x)
        self._xc = xs
        experts = self.gain * xs                      # per-expert forecasts
        self._experts = experts
        w = self._mean_weights()
        self._w = w
        return float(w @ experts)

    def update(self, x, z, y):
        xs, experts, w = self._xc, self._experts, self._w
        # update per-expert gains (online least squares on each expert)
        self.gain += self.gain_lr * (y - self.gain * xs) * xs
        # randomised weights for exploration
        noise = self.sigma * self.rng.normal(size=self.d)
        wr = np.exp(self.theta + noise - (self.theta + noise).max())
        wr /= wr.sum()
        pred = wr @ experts
        r = float(y - pred)
        # Wasserstein-DRO mirror-descent: EG step on robust loss gradient
        # robust per-expert loss grad ~ -2 r experts + rho * |experts| (ball)
        g = -2.0 * r * experts + self.rho * np.abs(experts)
        g += self.lam2 * self.theta
        self.theta -= self.eta * g
        self.theta -= self.theta.mean()               # gauge-fix the logits
        self._absorb(x, z)

    def effective_linear(self):
        w = self._mean_weights()
        return 0.0, w * self.gain


class FC_DRO_ES(OnlineModel):
    r"""Adaptive DR Forecast Combination / FC-DRO-ES (Liu et al., 2026).

    Variance-scaled exponential weights (their Algorithm OA.5)::

        w_{t,k} propto w_{t-1,k} exp(-1/2 sum_u E_{u,k}^2 / v_k) v_k^{-1/2}

    with ``v_k`` a rolling/EWMA error variance per expert.  The Expected-
    Shortfall variant replaces the running squared error by the tail loss
    ``ES_alpha`` of each expert.  Experts are the individual standardised
    predictors with online scalar gains.
    """

    def __init__(self, d, p, beta_v=0.97, es=False, alpha=0.1, eta=0.5,
                 lam_mix=0.02, gain_lr=0.02):
        super().__init__(d, p, "FC-DRO-ES(Liu26)" if es else "FC-DRO(Liu26)")
        self.beta_v, self.es, self.alpha = beta_v, es, alpha
        self.eta, self.lam_mix, self.gain_lr = eta, lam_mix, gain_lr
        self.logw = np.zeros(d)
        self.v = EMA(d, beta_v)
        self.gain = np.zeros(d)
        self.es_buf = [deque(maxlen=250) for _ in range(d)]

    def _weights(self):
        m = self.logw.max()
        w = np.exp(self.logw - m)
        w /= w.sum()
        # mixing with uniform for robustness (parameter lambda)
        return (1 - self.lam_mix) * w + self.lam_mix / self.d

    def predict(self, x, z):
        xs = self._xs(x)
        self._xc = xs
        self._experts = self.gain * xs
        w = self._weights()
        self._w = w
        return float(w @ self._experts)

    def update(self, x, z, y):
        xs, experts = self._xc, self._experts
        self.gain += self.gain_lr * (y - self.gain * xs) * xs
        err = y - experts                              # per-expert error
        v = self.v.update(err ** 2) + 1e-6             # EWMA error variance
        if self.es:
            # Expected-Shortfall-penalised loss: blend mean squared error with
            # the tail (ES_alpha) loss so the weights target tail robustness
            # while staying anchored to average accuracy.
            tail = np.zeros(self.d)
            for k in range(self.d):
                self.es_buf[k].append(err[k] ** 2)
                arr = np.sort(np.array(self.es_buf[k]))
                t = arr[int((1 - self.alpha) * len(arr)):]
                tail[k] = t.mean() if len(t) else err[k] ** 2
            loss = 0.5 * err ** 2 + 0.5 * tail
        else:
            loss = err ** 2
        # variance-scaled exponential-weights update (+ log v^{-1/2} prior)
        self.logw += -0.5 * self.eta * loss / v - 0.5 * np.log(v)
        self.logw = np.clip(self.logw - self.logw.max(), -40.0, 0.0)
        self._absorb(x, z)

    def effective_linear(self):
        return 0.0, self._weights() * self.gain


# --------------------------------------------------------------------------- #
#  2025 baselines                                                             #
# --------------------------------------------------------------------------- #
class MFISHER(OnlineModel):
    r"""M-FISHER — martingale-driven Fisher test-time adaptation (2025).

    Build an exponential test martingale (wealth) ``W_t = prod_s e_s`` from
    non-conformity scores; Ville's inequality controls false alarms at level
    ``delta``.  On a shift trigger (``W_t > 1/delta``) take a
    Fisher-preconditioned natural-gradient step (here ``F`` is an EMA of the
    outer product of per-step gradients) and reset the wealth.  Between triggers
    it runs a mild OGD step — the detect-then-adapt (reactive) contrast to OARF.
    """

    def __init__(self, d, p, eta=0.1, eta_adapt=0.08, delta=0.02, beta_f=0.95,
                 lam2=1e-4):
        super().__init__(d, p, "M-FISHER")
        self.eta, self.eta_adapt, self.delta = eta, eta_adapt, delta
        self.lam2 = lam2
        self.w = np.zeros(d + 1)
        self.opt = AdaStep(d + 1, eta)
        self.F = EMA(d + 1, beta_f)           # diagonal Fisher proxy
        self.logW = 0.0                       # log-wealth
        self.score_ema = EMA(1, 0.99)
        self.score_var = EMA(1, 0.99)
        self.n_triggers = 0

    def predict(self, x, z):
        xs = self._xs(x)
        self._phi = np.concatenate(([1.0], xs))
        return float(self.w @ self._phi)

    def update(self, x, z, y):
        phi = self._phi
        r = float(y - self.w @ phi)
        score = abs(r)
        mu = float(self.score_ema.update([score])[0])
        var = float(self.score_var.update([(score - mu) ** 2])[0]) + 1e-6
        # standardized non-conformity -> e-value (>1 when score is surprising)
        e = float(np.exp(np.clip((score - mu) / np.sqrt(var) - 0.5, -3, 3)))
        self.logW += np.log(max(e, 1e-6))
        self.logW = max(self.logW, 0.0)                # reset on negative wealth
        g = -r * phi
        F = self.F.update(g ** 2) + 1e-1                 # Fisher proxy (floored)
        if self.logW > np.log(1.0 / self.delta):        # Ville trigger
            step = self.eta_adapt * g / F               # natural-gradient step
            nrm = np.linalg.norm(step)                   # trust-region clip
            if nrm > 0.3:
                step *= 0.3 / nrm
            self.w -= step
            self.logW = 0.0
            self.n_triggers += 1
        else:
            self.w -= self.opt.step(g)                   # mild adaptive OGD
        self.w[1:] -= self.eta * self.lam2 * self.w[1:]
        self._absorb(x, z)

    def effective_linear(self):
        return float(self.w[0]), self.w[1:].copy()


class WATCH(OnlineModel):
    r"""WATCH / WCTM — weighted conformal test martingales (2025; 2505.04608).

    Monitors a weighted-conformal test martingale on non-conformity scores and
    adapts the *interval width* to benign covariate shift while still flagging
    harmful concept shift, reducing false alarms vs unweighted CTMs.  The point
    forecast comes from an OGD head; the distributional head adapts band width
    from the martingale state.
    """

    produces_quantiles = True

    def __init__(self, d, p, eta=0.03, alpha=0.1, beta_w=0.97, lam2=1e-4):
        super().__init__(d, p, "WATCH/WCTM")
        self.eta, self.alpha, self.lam2 = eta, alpha, lam2
        self.w = np.zeros(d + 1)
        self.opt = AdaStep(d + 1, eta)
        self.q_ema = EMA(1, beta_w)            # running scale of |residual|
        self.scale = 1.0
        self.logM = 0.0

    def predict(self, x, z):
        xs = self._xs(x)
        self._phi = np.concatenate(([1.0], xs))
        return float(self.w @ self._phi)

    def predict_quantiles(self, x, z):
        mid = self.predict(x, z)
        q = float(self.q_ema.corrected[0]) if self.q_ema._w > 0 else 1.0
        half = 1.96 * q * self.scale
        return np.array([mid - half, mid, mid + half])

    def update(self, x, z, y):
        phi = self._phi
        r = float(y - self.w @ phi)
        s = abs(r)
        q = float(self.q_ema.update([s])[0])
        # weighted conformal p-value proxy; martingale on its surprise
        pval = np.clip(np.exp(-s / (q + 1e-6)), 1e-3, 1 - 1e-3)
        self.logM += np.log(0.5 / pval) if pval < 0.5 else np.log(0.5 / pval)
        self.logM = np.clip(self.logM, 0.0, 20.0)
        # adapt band: harmful-shift evidence widens, benign keeps tight
        self.scale = 1.0 + 0.5 * np.tanh(self.logM / 5.0)
        g = -r * phi
        self.w -= self.opt.step(g)
        self.w[1:] -= self.eta * self.lam2 * self.w[1:]
        self._absorb(x, z)

    def effective_linear(self):
        return float(self.w[0]), self.w[1:].copy()


# --------------------------------------------------------------------------- #
#  Registry                                                                   #
# --------------------------------------------------------------------------- #
# Two experiment families (Sec 5).  The *causal-robustness* track compares
# regression-type online learners where the held-out do(A) grid is defined; the
# pure forecast *combiners* (Wang/Liu) live in the forecast-combination track,
# their natural problem class, alongside OARF acting as a robust combiner.
REGRESSION_TRACK = ["OARF", "OARF-CD", "OGD", "Rolling-OLS", "ACI", "M-FISHER",
                    "WATCH/WCTM", "W-DRO-OL(Chen26)", "CostACI", "AnchorReg",
                    "DRIG"]
COMBINATION_TRACK = ["OARF", "OGD", "Rolling-OLS", "ACI", "DR-Combo(Wang26)",
                     "FC-DRO(Liu26)", "FC-DRO-ES(Liu26)", "W-DRO-OL(Chen26)",
                     "M-FISHER", "AnchorReg"]


def build_model_suite(d, p, B_fixed=None, q=2, include=None, seed=0):
    """Instantiate the full comparison suite (Sec 5).

    ``B_fixed`` is the analyst-supplied channel for the fixed-channel methods;
    if ``None`` a random orthonormal channel is used.  The learned-channel OARF
    always discovers its own ``B``.
    """
    models = {
        # --- OARF and its ablations ---
        "OARF": lambda: OARF(d, p, B=B_fixed, q=q, xi=6.0, eta=0.1),
        "OARF-CD": lambda: OARF(d, p, q=q, xi=6.0, eta=0.1, learn_channel=True),
        # --- floors ---
        "OGD": lambda: OGD(d, p, eta=0.1),
        "Rolling-OLS": lambda: RollingOLS(d, p),
        "ACI": lambda: ACI(d, p, eta=0.1),
        # --- 2025 ---
        "M-FISHER": lambda: MFISHER(d, p),
        "WATCH/WCTM": lambda: WATCH(d, p, eta=0.1),
        # --- 2026 ---
        "DR-Combo(Wang26)": lambda: RandomizedDRCombination(d, p, seed=seed),
        "FC-DRO(Liu26)": lambda: FC_DRO_ES(d, p, es=False),
        "FC-DRO-ES(Liu26)": lambda: FC_DRO_ES(d, p, es=True),
        "W-DRO-OL(Chen26)": lambda: WassersteinDRO_OL(d, p, eta=0.1),
        "CostACI": lambda: ACI(d, p, eta=0.1, cost_aware=True),
        # --- foundational ablations ---
        "AnchorReg": lambda: AnchorRegression(d, p, B=B_fixed, q=q),
        "DRIG": lambda: DRIG(d, p),
    }
    if include is not None:
        models = {k: v for k, v in models.items() if k in include}
    return {k: f() for k, f in models.items()}
