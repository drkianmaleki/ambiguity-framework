"""
AAMass — Ambiguity Mass.

Quantifies what fraction of predictions land inside the indecision
interval ℐ_δ = [0.5 - delta, 0.5 + delta].

A score of 0.0 means no predictions are ambiguous (perfect separation).
A score of 1.0 means all predictions fall inside the indecision interval.

Note: The indecision interval itself (ℐ_δ) is Parameter 1 in the paper
(Ambiguity Range). AAMass is Parameter 2 — the sample density within it.
"""

import numpy as np
from ambiguity_suite.utils import indecision_mask, validate_inputs


def compute(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    delta: float = 0.1,
) -> float:
    """
    Compute the Ambiguity Mass score.

    Measures the fraction of predictions that fall inside the indecision
    interval ℐ_δ = [0.5 - delta, 0.5 + delta]. Higher values indicate
    more ambiguity.

    Parameters
    ----------
    y_true : array-like of int, shape (n,)
        Ground truth labels (0 or 1). Kept for API consistency; not used
        in the mass computation itself.
    y_prob : array-like of float, shape (n,)
        Predicted probabilities in [0, 1].
    delta : float, default 0.1
        Half-width of the indecision interval. Must be in (0, 0.5].

    Returns
    -------
    score : float in [0.0, 1.0]
        Fraction of predictions inside the indecision interval.
        0.0 → no ambiguity, 1.0 → maximum ambiguity.

    Examples
    --------
    >>> compute([0,0,1,1], [0.1, 0.2, 0.8, 0.9], delta=0.1)
    0.0
    >>> compute([0,0,1,1], [0.5, 0.5, 0.5, 0.5], delta=0.1)
    1.0
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    validate_inputs(y_true, y_prob, delta)

    mask = indecision_mask(y_prob, delta)
    return float(mask.mean())
