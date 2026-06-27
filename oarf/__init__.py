"""OARF — Online Anchor-Robust Forecasting under Economic Regime Shift.

A streaming anchor-regression learner that immunizes a forecaster against
economic regime shifts by driving the part of its residual correlated with
observable regime drivers to zero, and discovers that causal channel online.

Modules
-------
synthetic   : linear anchor-SCM fixture with regimes and a held-out do(A) grid
models      : OARF (Sec 2.2), online channel discovery (Sec 2.3), baselines (Sec 5)
evaluate    : leak-free walk-forward harness, metrics (Sec 6), DM / MCS significance
load_panel  : FRED / GPR / EPU / EUA real-data loader and feature construction
figures     : the paper's figures (Sec 7)
"""

__version__ = "1.0.0"
