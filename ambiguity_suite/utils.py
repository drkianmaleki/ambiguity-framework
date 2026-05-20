"""
Math helpers shared across AAPR, AAF1, and AAMass.

Core concept: the indecision interval ℐ_δ = [0.5 - delta, 0.5 + delta]
identifies the score region where the classifier is ambiguous.

  - AAPR and AAF1 are global measures: they integrate over all thresholds
    t ∈ [0, 1] and do not restrict computation to the indecision interval.

  - AAMass is a local measure: it counts the fraction of predicted
    probabilities that fall inside ℐ_δ. The indecision_mask() function
    below supports this computation.
"""

from typing import Optional

import numpy as np


def indecision_mask(y_prob: np.ndarray, delta: float) -> np.ndarray:
    """
    Boolean mask: True where y_prob falls inside the indecision interval.

    Interval ℐ_δ = [0.5 - delta, 0.5 + delta], inclusive on both ends.

    Used by AAMass to identify in-band samples.

    Parameters
    ----------
    y_prob : (n,) float array of predicted probabilities in [0, 1]
    delta  : float — half-width of the indecision interval, in (0, 0.5]

    Returns
    -------
    mask : (n,) bool array
    """
    return (y_prob >= 0.5 - delta) & (y_prob <= 0.5 + delta)


def validate_inputs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    delta: float,
    beta: Optional[float] = None,
) -> None:
    """
    Validate inputs shared across all compute() functions.

    Parameters
    ----------
    y_true : array-like — ground truth labels
    y_prob : array-like — predicted probabilities
    delta  : float      — half-width of indecision interval, must be in (0, 0.5]
    beta   : float|None — F-beta weight, must be > 0 if provided

    Raises
    ------
    ValueError
        If y_true and y_prob have different lengths, delta is out of
        (0, 0.5], or beta is provided and is not positive.
    """
    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true and y_prob must have the same length, "
            f"got {len(y_true)} and {len(y_prob)}."
        )
    if not (0.0 < delta <= 0.5):
        raise ValueError(
            f"delta must be in (0, 0.5], got {delta}."
        )
    if beta is not None and beta <= 0.0:
        raise ValueError(
            f"beta must be positive, got {beta}."
        )


# ---------------------------------------------------------------------------
# Legacy helpers — retained for external use but not used by core metrics.
# weighted_balanced_accuracy and band_precision_recall were used by the
# original in-band implementation of AAPR and AAF1. Both metrics have since
# been rewritten as global threshold integrals matching the paper's
# definitions (Equations 4–9). These functions are preserved here for
# any downstream code that may reference them.
# ---------------------------------------------------------------------------

def weighted_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 1.0,
) -> float:
    """
    Beta-weighted balanced accuracy from hard predictions.

    score = (beta² * TPR + TNR) / (1 + beta²)

    Not used by the core AAPR/AAF1/AAMass metric path.

    Parameters
    ----------
    y_true : (n,) int array, values in {0, 1}
    y_pred : (n,) int array, values in {0, 1}
    beta   : float > 0

    Returns
    -------
    score : float in [0.0, 1.0]
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return (beta ** 2 * tpr + tnr) / (1.0 + beta ** 2)


def band_precision_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> tuple:
    """
    Precision and recall restricted to the in-band subset.

    Not used by the core AAPR/AAF1/AAMass metric path.

    Returns (0.0, 0.0) when the band is empty or has no positive predictions.

    Parameters
    ----------
    y_true    : (n,) int array
    y_prob    : (n,) float array
    mask      : (n,) bool array — True for in-band samples
    threshold : float — decision threshold, default 0.5

    Returns
    -------
    (precision, recall) : tuple of float
    """
    y_t = y_true[mask]
    y_p = (y_prob[mask] > threshold).astype(int)

    tp = int(((y_p == 1) & (y_t == 1)).sum())
    fp = int(((y_p == 1) & (y_t == 0)).sum())
    fn = int(((y_p == 0) & (y_t == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall
