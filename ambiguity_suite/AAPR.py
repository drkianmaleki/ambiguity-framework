"""
AAPR — Ambiguity Area over Precision and Recall.

Computes a normalized definite integral of Precision(t) · Recall(t)
over all unique classifier thresholds t in y_prob.

Formula (from paper, Equations 4–5 and 8):

    AA_un   = ∫ P(t) · R(t) dt        (trapezoidal approximation)
    AA_min  = prevalence / 2           (Equation 8: random classifier baseline)
    AA_max  = 1.0                      (perfect classifier)

    AAPR    = (AA_max - AA_un) / (AA_max - AA_min)

Interpretation:
  Low AAPR  → classifier maintains high P·R across thresholds (low ambiguity)
  High AAPR → classifier's P·R degrades across thresholds (high ambiguity)

Note on delta:
  AAPR is intentionally a global measure — it integrates over all thresholds
  to assess full-spectrum ambiguity. The `delta` parameter is accepted for
  API consistency with AAF1 and AAMass but does not restrict the integration
  domain. Use AAMass(delta) to measure sample density within the interval.
"""

import warnings

import numpy as np
from sklearn.metrics import precision_score, recall_score

from ambiguity_suite.utils import validate_inputs

# NumPy 2.0 renamed trapz → trapezoid; support both environments.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def compute(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    delta: float = 0.1,
) -> float:
    """
    Compute the Ambiguity Area over Precision and Recall (AAPR) score.

    Parameters
    ----------
    y_true : array-like of int, shape (n,)
        Ground truth labels (0 or 1).
    y_prob : array-like of float, shape (n,)
        Predicted probabilities in [0, 1].
    delta : float, default 0.1
        Accepted for API consistency with AAF1 and AAMass; not used in
        AAPR computation. Must still satisfy the range constraint (0, 0.5].
        A UserWarning is raised if a non-default value is passed.

    Returns
    -------
    score : float in [0.0, 1.0]
        Higher → more ambiguous (P·R degrades across thresholds).
        Lower  → less ambiguous (P·R remains high across thresholds).
        Clipped to [0, 1] for classifiers worse than the random baseline.

    Examples
    --------
    >>> round(compute([0,0,1,1], [0.5, 0.5, 0.5, 0.5]), 4)
    0.8333
    >>> round(compute([0,0,1,1], [0.1, 0.2, 0.8, 0.9]), 4)
    0.3889
    """
    if delta != 0.1:
        warnings.warn(
            "delta is accepted for API consistency but does not affect AAPR "
            "computation. Use AAMass to measure sample density within the "
            "indecision interval.",
            UserWarning,
            stacklevel=2,
        )

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    validate_inputs(y_true, y_prob, delta)

    prevalence = float(y_true.mean())

    # Single-class edge case: metric is undefined
    if prevalence == 0.0 or prevalence == 1.0:
        return 0.0

    # Boundary thresholds anchor the integral at both ends of [0, 1].
    # Without them, a single-valued y_prob collapses to a point (area = 0),
    # artificially under-estimating AA_un and inflating the score.
    thresholds = np.sort(np.unique(np.concatenate([[0.0], y_prob, [1.0]])))

    pr_products = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        pr_products.append(p * r)

    aa_un = float(_trapezoid(pr_products, thresholds))

    aa_min = prevalence / 2.0   # Equation 8: random classifier baseline
    aa_max = 1.0

    score = (aa_max - aa_un) / (aa_max - aa_min)
    return float(np.clip(score, 0.0, 1.0))
