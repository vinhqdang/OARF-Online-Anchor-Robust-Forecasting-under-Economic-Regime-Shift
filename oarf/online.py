"""Shared streaming utilities: past-only standardisation (no look-ahead).

Every model owns its own :class:`OnlineStandardizer` so that the *exact*
past-only transform used during training can be replayed at freeze time on the
held-out ``do(A := nu)`` grid (Sec 6).  This guarantees the leak-free protocol
of Sec 1: at step ``t`` we standardise ``x_t`` using statistics computed from
``x_1, ..., x_{t-1}`` only, predict, and *then* absorb ``x_t``.
"""

from __future__ import annotations

import numpy as np


class OnlineStandardizer:
    """Expanding past-only z-score with a small warm-up shrinkage.

    ``transform`` uses the running mean/variance accumulated *so far*; the new
    observation is folded in only by a subsequent :meth:`update` call.  This
    ordering is what makes the pipeline leak-free.
    """

    def __init__(self, dim: int, eps: float = 1e-8, warmup: int = 20,
                 clip: float = 8.0, std_floor: float = 1e-2):
        self.dim = dim
        self.eps = eps
        self.warmup = warmup
        self.clip = clip
        self.std_floor = std_floor
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros(dim)        # sum of squared deviations (Welford)

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.dim)
        return np.maximum(np.sqrt(self.M2 / (self.n - 1) + self.eps), self.std_floor)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        # always scale (with a variance floor) and winsorise; this keeps the
        # learner's inputs bounded even on the first, noisy observations.
        out = (x - self.mean) / self.std
        return np.clip(out, -self.clip, self.clip)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def freeze(self) -> "FrozenStandardizer":
        return FrozenStandardizer(self.mean.copy(), self.std.copy(), self.clip)


class FrozenStandardizer:
    """Immutable snapshot of a standardiser for held-out evaluation."""

    def __init__(self, mean, std, clip=8.0):
        self.mean, self._std, self.clip = mean, std, clip

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.clip((X - self.mean) / self._std, -self.clip, self.clip)


class AdaStep:
    """AdaGrad-style adaptive, per-coordinate step size (Duchi et al., 2011).

    Provides scale invariance and stability for the streaming updates while
    leaving the *gradient* exactly as specified in Sec 2.2; only the step is
    preconditioned, which preserves the online-convex-optimisation regret
    guarantees the theorem relies on.
    """

    def __init__(self, dim, eta, eps=1e-6):
        self.eta, self.eps = eta, eps
        self.G = np.zeros(dim)

    def step(self, grad):
        self.G += grad ** 2
        return self.eta * grad / (np.sqrt(self.G) + self.eps)


class EMA:
    """Exponential moving average with bias correction (decay ``beta``)."""

    def __init__(self, shape, beta: float):
        self.beta = beta
        self.value = np.zeros(shape)
        self._w = 0.0           # accumulated weight for bias correction

    def update(self, x: np.ndarray) -> np.ndarray:
        self.value = self.beta * self.value + (1.0 - self.beta) * np.asarray(x)
        self._w = self.beta * self._w + (1.0 - self.beta)
        return self.corrected

    @property
    def corrected(self) -> np.ndarray:
        if self._w < 1e-12:
            return self.value
        return self.value / self._w
