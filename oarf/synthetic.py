"""Synthetic anchor-SCM fixture with economic regime shifts.

We generate a stream from a *linear structural causal model* in the canonical
anchor-regression form (Rothenhauser et al., 2021), instantiated so that the
distinction between *prediction* and *interventional robustness* is sharp and
measurable against ground truth.

Variables (Sec 1 notation): exogenous anchor ``A in R^q`` and hidden confounder
``H in R^h`` drive the predictors and target.  Crucially the predictor block is
split into

* **parents** ``X_par`` — genuine causes of ``Y`` (the *invariant* signal); and
* **children** ``X_ch`` — *descendants* of ``Y`` that are *also* driven by the
  anchor ``A``.

Structural equations::

    A_t = mu_{r(t)} + eps_A                         (anchor; regime-shifting mean)
    H_t ~ N(0, I_h)                                 (hidden confounder)
    X_par = A W_Ap + H W_Hp + eps_p                 (causal parents of Y)
    Y     = X_par . b_par + H . d_H + eps_Y         (target)
    X_ch  = Y c_ch + A W_Ac + H W_Hc + eps_ch       (anti-causal children of Y)

In-sample the children are *highly predictive* of ``Y`` (they contain ``Y``), so
ordinary least squares / OGD load on them.  But a child's relationship to ``Y``
is contaminated by the anchor term ``A W_Ac``; under an intervention
``do(A := nu)`` that term is set externally and the child becomes *misleading* —
its OLS weight now injects an error that grows with ``||nu||``.  The
*invariant* predictor that uses only the parents ``b_par`` is unaffected.  Anchor
regression drives the anchor-correlated part of the residual to zero, which
removes reliance on the children, recovering the invariant predictor and a flat
robustness curve (Fig. 3).

The observable context ``z in R^p`` is a (rotated) embedding of the anchor plus
``p - q`` stable AR(1) *decoy* drivers, so the novel channel-discovery layer
(Sec 2.3) has a ground-truth channel matrix ``B_true`` (``z @ B_true = A``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SCMConfig:
    """Configuration for the synthetic anchor-SCM stream."""

    T: int = 8000             # stream length
    d_par: int = 3            # number of causal parents of Y
    d_ch: int = 3             # number of anti-causal children of Y
    p: int = 8                # observable-context dimension
    q: int = 2                # true anchor (channel) dimension
    h: int = 2                # hidden-confounder dimension
    n_regimes: int = 8        # number of regimes (=> n_regimes-1 changepoints)
    regime_shift: float = 1.5  # across-regime anchor-mean shift scale
    anchor_noise: float = 0.6  # within-regime anchor sd
    child_coupling: float = 1.3  # strength of the Y -> child edge
    child_anchor: float = 1.1  # strength of the anchor -> child contamination
    confound: float = 0.9     # strength of the H -> (X, Y) confounding path
    noise_x: float = 0.4
    noise_y: float = 0.5
    rotate_channel: bool = True
    seed: int = 0

    @property
    def d(self) -> int:
        return self.d_par + self.d_ch


@dataclass
class SCM:
    """A drawn realisation of the synthetic stream (Sec 1 notation)."""

    cfg: SCMConfig
    X: np.ndarray            # (T, d)  predictors = [parents | children]
    Z: np.ndarray            # (T, p)  observable context / candidate drivers
    Y: np.ndarray            # (T,)    target
    A: np.ndarray            # (T, q)  true anchor (= Z @ B_true)
    regimes: np.ndarray      # (T,)    integer regime id
    changepoints: np.ndarray  # (K,)   indices where the regime changes
    B_true: np.ndarray       # (p, q)  true channel matrix (Z -> A)
    w_invariant: np.ndarray  # (d+1,)  invariant predictor [intercept; parents,0]
    parent_idx: np.ndarray   # indices of the causal parents within X
    params: dict = field(default_factory=dict)


def _make_channel(rng, p, q, rotate):
    """Return ``(B_true, Qrot)`` such that ``(z_raw @ Qrot.T) @ B_true = z_raw[:, :q]``.

    The observable context is ``Z = z_raw @ Qrot.T`` (a rotation of the
    anchor-plus-decoys frame); the true channel ``B_true = Qrot @ B_sel`` then
    satisfies ``Z @ B_true = z_raw @ B_sel = A`` exactly, giving the
    channel-discovery layer a genuine ground-truth subspace.
    """
    B_sel = np.zeros((p, q))
    B_sel[np.arange(q), np.arange(q)] = 1.0
    if rotate:
        Qrot, _ = np.linalg.qr(rng.normal(size=(p, p)))
        return Qrot @ B_sel, Qrot
    return B_sel, np.eye(p)


def make_scm(cfg: SCMConfig | None = None, **overrides) -> SCM:
    """Draw one realisation of the canonical anchor-SCM stream."""
    if cfg is None:
        cfg = SCMConfig(**overrides)
    elif overrides:
        cfg = SCMConfig(**{**cfg.__dict__, **overrides})
    rng = np.random.default_rng(cfg.seed)
    dpar, dch, q, h = cfg.d_par, cfg.d_ch, cfg.q, cfg.h

    # --- regime schedule ------------------------------------------------------
    cuts = np.sort(rng.choice(np.arange(int(0.05 * cfg.T), int(0.95 * cfg.T)),
                              size=cfg.n_regimes - 1, replace=False))
    regimes = np.zeros(cfg.T, dtype=int)
    for c in cuts:
        regimes[c:] += 1
    changepoints = cuts.astype(int)
    regime_means = rng.normal(0.0, cfg.regime_shift, size=(cfg.n_regimes, q))

    # --- structural coefficients ---------------------------------------------
    W_Ap = rng.normal(0, 1, size=(q, dpar))                  # A -> parents
    W_Hp = rng.normal(0, 1, size=(h, dpar)) * cfg.confound   # H -> parents
    b_par = rng.normal(0, 1, size=dpar)                      # parents -> Y
    d_H = rng.normal(0, 1, size=h) * cfg.confound            # H -> Y (confound)
    c_ch = rng.normal(0, 1, size=dch) * cfg.child_coupling   # Y -> children
    W_Ac = rng.normal(0, 1, size=(q, dch)) * cfg.child_anchor  # A -> children
    W_Hc = rng.normal(0, 1, size=(h, dch)) * cfg.confound    # H -> children

    B_true, Qrot = _make_channel(rng, cfg.p, q, cfg.rotate_channel)

    # --- draw the stream ------------------------------------------------------
    H = rng.normal(0, 1, size=(cfg.T, h))
    A = regime_means[regimes] + rng.normal(0, cfg.anchor_noise, size=(cfg.T, q))

    X_par = A @ W_Ap + H @ W_Hp + rng.normal(0, cfg.noise_x, size=(cfg.T, dpar))
    Y = X_par @ b_par + H @ d_H + rng.normal(0, cfg.noise_y, size=cfg.T)
    X_ch = (np.outer(Y, c_ch) + A @ W_Ac + H @ W_Hc
            + rng.normal(0, cfg.noise_x, size=(cfg.T, dch)))
    X = np.hstack([X_par, X_ch])                             # (T, d)

    # --- observable context z: anchor + stable AR(1) decoys, rotated ----------
    z_raw = np.zeros((cfg.T, cfg.p))
    z_raw[:, :q] = A
    decoy = np.zeros((cfg.T, cfg.p - q))
    for t in range(1, cfg.T):
        decoy[t] = 0.6 * decoy[t - 1] + rng.normal(0, 1, size=cfg.p - q)
    z_raw[:, q:] = decoy
    Z = z_raw @ Qrot.T if cfg.rotate_channel else z_raw

    # invariant (causal) predictor: parents with b_par, zero on children
    w_inv = np.concatenate(([0.0], b_par, np.zeros(dch)))
    parent_idx = np.arange(dpar)

    return SCM(
        cfg=cfg, X=X, Z=Z, Y=Y, A=Z @ B_true, regimes=regimes,
        changepoints=changepoints, B_true=B_true, w_invariant=w_inv,
        parent_idx=parent_idx,
        params=dict(W_Ap=W_Ap, W_Hp=W_Hp, b_par=b_par, d_H=d_H, c_ch=c_ch,
                    W_Ac=W_Ac, W_Hc=W_Hc, regime_means=regime_means),
    )


@dataclass
class InterventionGrid:
    """Held-out ``do(A := nu)`` environments for interventional metrics (Sec 6)."""

    radii: np.ndarray
    nus: list
    X: list
    Y: list


def make_intervention_grid(scm: SCM, radii=None, n_dir: int = 6,
                           n_batch: int = 400, seed: int = 12345
                           ) -> InterventionGrid:
    """Build the frozen held-out intervention grid for an SCM realisation.

    Under ``do(A := nu)`` the anchor is set externally; ``H, eps`` stay random.
    Parents, ``Y`` and children are generated from the structural equations with
    the *set* anchor value, so the children's anchor contamination is what makes
    non-invariant predictors degrade as ``||nu||`` grows.
    """
    cfg = scm.cfg
    rng = np.random.default_rng(seed)
    if radii is None:
        train_scale = float(np.sqrt(np.mean(np.sum(scm.A ** 2, axis=1))))
        radii = np.linspace(0.0, 3.0, 10) * max(train_scale, 1.0)
    radii = np.asarray(radii, dtype=float)

    P = scm.params
    nus, Xs, Ys = [], [], []
    for r in radii:
        dirs = rng.normal(size=(n_dir, cfg.q))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        nu = dirs * r
        X_r = np.zeros((n_dir, n_batch, cfg.d))
        Y_r = np.zeros((n_dir, n_batch))
        for j in range(n_dir):
            H = rng.normal(size=(n_batch, cfg.h))
            Xp = (nu[j][None, :] @ P["W_Ap"] + H @ P["W_Hp"]
                  + rng.normal(0, cfg.noise_x, size=(n_batch, cfg.d_par)))
            Yb = Xp @ P["b_par"] + H @ P["d_H"] + rng.normal(0, cfg.noise_y, n_batch)
            Xc = (np.outer(Yb, P["c_ch"]) + nu[j][None, :] @ P["W_Ac"]
                  + H @ P["W_Hc"] + rng.normal(0, cfg.noise_x, (n_batch, cfg.d_ch)))
            X_r[j] = np.hstack([Xp, Xc])
            Y_r[j] = Yb
        nus.append(nu)
        Xs.append(X_r)
        Ys.append(Y_r)
    return InterventionGrid(radii=radii, nus=nus, X=Xs, Y=Ys)


if __name__ == "__main__":
    scm = make_scm(SCMConfig(seed=0))
    print(f"SCM: T={scm.cfg.T}, d={scm.cfg.d} (par={scm.cfg.d_par},"
          f" ch={scm.cfg.d_ch}), p={scm.cfg.p}, q={scm.cfg.q}")
    print(f"changepoints: {scm.changepoints}")
    print(f"anchor reconstruction err: {np.abs(scm.Z @ scm.B_true - scm.A).max():.2e}")
    grid = make_intervention_grid(scm)
    print(f"intervention radii: {np.round(grid.radii, 2)}")
