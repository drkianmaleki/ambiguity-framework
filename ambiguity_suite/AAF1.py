"""
AAF1 — Ambiguity Area over F1.

Computes a normalized definite integral of F_beta(t) over all unique
classifier thresholds t in y_prob.

Formula (from paper, Equations 6–7 and 9):

    AA_un   = ∫ F1(t) dt              (trapezoidal approximation)
    AA_min  = ∫ F1_random(t) dt       (Equation 9: prevalence-dependent baseline)
    AA_max  = 1.0                      (perfect classifier)

    AAF1    = (AA_max - AA_un) / (AA_max - AA_min)

The beta parameter generalizes the standard F1 integral to F_beta,
allowing recall (TPR) to be weighted more or less heavily than precision.
At beta=1.0 (default), this exactly reproduces the paper's AAF1 definition.

Interpretation:
  Low AAF1  → F_beta(t) remains high across thresholds (low fragility)
  High AAF1 → F_beta(t) degrades across thresholds (high fragility)

Note on delta:
  AAF1 is intentionally a global measure — it integrates over all thresholds.
  The `delta` parameter is accepted for API consistency with AAMass but does
  not restrict the integration domain.
"""

import warnings

import numpy as np
from sklearn.metrics import fbeta_score

from ambiguity_suite.utils import validate_inputs

# NumPy 2.0 renamed trapz → trapezoid; support both environments.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _integrate_fbeta_random(prevalence: float, beta: float,
                             n_points: int = 1000) -> float:
    """
    Numerically integrate F_beta(t) for a random classifier.

    For a random classifier:
        Precision(t) = prevalence  (constant, independent of threshold)
        Recall(t)    = 1 - t       (decreases linearly with threshold)

    This gives the prevalence-dependent baseline from Equation 9.

    Parameters
    ----------
    prevalence : float — positive class fraction p = P(Y=1)
    beta       : float — F-beta weighting parameter
    n_points   : int  — quadrature resolution

    Returns
    -------
    float — unnormalized baseline integral
    """
    t = np.linspace(0.0, 1.0, n_points)
    p_rand = prevalence
    r_rand = 1.0 - t
    denom = beta ** 2 * p_rand + r_rand
    fbeta_rand = np.where(
        denom > 0,
        (1.0 + beta ** 2) * p_rand * r_rand / denom,
        0.0,
    )
    return float(_trapezoid(fbeta_rand, t))


def compute(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    delta: float = 0.1,
    beta: float = 1.0,
) -> float:
    """
    Compute the Ambiguity Area over F-beta (AAF1) score.

    Parameters
    ----------
    y_true : array-like of int, shape (n,)
        Ground truth labels (0 or 1).
    y_prob : array-like of float, shape (n,)
        Predicted probabilities in [0, 1].
    delta : float, default 0.1
        Accepted for API consistency with AAMass; not used in AAF1
        computation. Must still satisfy the range constraint (0, 0.5].
        A UserWarning is raised if a non-default value is passed.
    beta : float, default 1.0
        F-beta weighting. beta=1.0 reproduces the paper's AAF1 exactly.
        beta>1 weights recall more heavily; beta<1 weights precision more.
        Must be > 0.

    Returns
    -------
    score : float in [0.0, 1.0]
        Higher → more fragile (F_beta degrades across thresholds).
        Lower  → more resilient (F_beta remains high across thresholds).
        Clipped to [0, 1] for classifiers worse than the random baseline.

    Examples
    --------
    >>> round(compute([0,0,1,1], [0.1, 0.2, 0.8, 0.9]), 4)
    0.3702
    >>> round(compute([0,0,1,1], [0.5, 0.5, 0.5, 0.5]), 4)
    0.9102
    """
    if delta != 0.1:
        warnings.warn(
            "delta is accepted for API consistency but does not affect AAF1 "
            "computation. Use AAMass to measure sample density within the "
            "indecision interval.",
            UserWarning,
            stacklevel=2,
        )

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    validate_inputs(y_true, y_prob, delta, beta=beta)

    prevalence = float(y_true.mean())

    # Single-class edge case: metric is undefined
    if prevalence == 0.0 or prevalence == 1.0:
        return 0.0

    # Boundary thresholds anchor the integral at both ends of [0, 1]
    thresholds = np.sort(np.unique(np.concatenate([[0.0], y_prob, [1.0]])))

    fbeta_scores = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fb = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        fbeta_scores.append(fb)

    aa_un = float(_trapezoid(fbeta_scores, thresholds))

    aa_min = _integrate_fbeta_random(prevalence, beta)  # Equation 9
    aa_max = 1.0

    score = (aa_max - aa_un) / (aa_max - aa_min)
    return float(np.clip(score, 0.0, 1.0))
