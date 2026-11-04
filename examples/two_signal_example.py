#!/usr/bin/env python3
"""Two-signal example used in the introduction.

y = x_p + e_y is the target, x_p a stable predictor, and x_c = y + a + e_c a
second predictor that tracks the target but is also moved by an observed
driver a. In the training sample a has two regimes (mean -1 and +1); in the
test regime its mean moves to +3. Prints the weights and MSEs of least squares
and of anchor regression with penalty xi, fitted on the training sample.
"""
import numpy as np

rng = np.random.default_rng(0)


def draw(n, a_mean):
    xp = rng.normal(size=n)
    y = xp + 0.5 * rng.normal(size=n)
    a = a_mean + 0.3 * rng.normal(size=n)
    xc = y + a + 0.3 * rng.normal(size=n)
    return np.column_stack([xp, xc]), y, a


def anchor_fit(X, y, a, xi):
    """Batch anchor regression on centred data: minimise
    ||(I-P)r||^2 + (1+xi)||P r||^2, P the projection on the anchor."""
    Xc, yc, ac = X - X.mean(0), y - y.mean(), (a - a.mean())[:, None]
    P = ac @ np.linalg.pinv(ac)
    W = np.eye(len(y)) + xi * P
    w = np.linalg.solve(Xc.T @ W @ Xc, Xc.T @ W @ yc)
    b0 = y.mean() - X.mean(0) @ w
    return b0, w


n = 2000
X1, y1, a1 = draw(n, -1.0)
X2, y2, a2 = draw(n, +1.0)
Xtr, ytr, atr = np.vstack([X1, X2]), np.r_[y1, y2], np.r_[a1, a2]
Xte, yte, _ = draw(4000, +3.0)

print(f"{'method':22s} {'w_stable':>9s} {'w_driver-moved':>15s} {'MSE train':>10s} {'MSE new regime':>15s}")
for name, xi in [("least squares", 0.0), ("anchor, xi=10", 10.0), ("anchor, xi=100", 100.0)]:
    b0, w = anchor_fit(Xtr, ytr, atr, xi)
    mtr = np.mean((ytr - b0 - Xtr @ w) ** 2)
    mte = np.mean((yte - b0 - Xte @ w) ** 2)
    print(f"{name:22s} {w[0]:9.3f} {w[1]:15.3f} {mtr:10.3f} {mte:15.3f}")
w_true = np.array([1.0, 0.0])
print(f"{'stable predictor only':22s} {1.0:9.3f} {0.0:15.3f} "
      f"{np.mean((ytr - Xtr @ w_true) ** 2):10.3f} {np.mean((yte - Xte @ w_true) ** 2):15.3f}")
