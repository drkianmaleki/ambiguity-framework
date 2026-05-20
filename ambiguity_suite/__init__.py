"""
ambiguity_suite — Ambiguity Area metrics for post-hoc binary classification audit.

Public API
----------
from ambiguity_suite import compute_AAPR, compute_AAF1, compute_AAMass

All metrics share the same signature:
    compute_*(y_true, y_prob, delta=0.1, **kwargs) -> float

Parameters
----------
y_true : array-like of int   — ground truth labels {0, 1}
y_prob : array-like of float — predicted probabilities in [0, 1]
delta  : float               — half-width of the indecision interval ℐ_δ = [0.5-delta, 0.5+delta]

Returns
-------
float in [0.0, 1.0]
"""

from ambiguity_suite.AAPR import compute as compute_AAPR
from ambiguity_suite.AAF1 import compute as compute_AAF1
from ambiguity_suite.AAMass import compute as compute_AAMass
from ambiguity_suite.AARange import get_interval, interval_width, validate_delta

__version__ = "0.1.0"

__all__ = [
    "compute_AAPR",
    "compute_AAF1",
    "compute_AAMass",
    "get_interval",
    "interval_width",
    "validate_delta",
    "__version__",
]