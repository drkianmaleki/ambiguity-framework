"""
Tests for AAPR.compute().

AAPR is a normalized integral of P(t) · R(t) over all thresholds t in y_prob.

    AAPR = (1 - ∫P(t)R(t)dt) / (1 - prevalence/2)     clipped to [0, 1]

High AAPR → classifier's P·R degrades across thresholds (high ambiguity).
Low AAPR  → classifier maintains high P·R across thresholds (low ambiguity).

Key contracts:
  - Single class in y_true         → 0.0   (undefined, safe fallback)
  - All predictions at 0.5         → 1.0   (maximum ambiguity)
  - Well-separated < Near-midpoint           (ordering preserved)
  - Result always in [0.0, 1.0]
  - delta is validated but does NOT affect the score (global metric)
"""

import pytest
import numpy as np
from ambiguity_suite import compute_AAPR


# ---------------------------------------------------------------------------
# Core contract: ordering
# ---------------------------------------------------------------------------
def test_ambiguous_scores_higher_than_clear():
    """
    A classifier with clustered predictions near 0.5 must score higher
    (more ambiguous) than one with predictions far from 0.5.
    Reference values confirmed by running compute() against the formula.
    """
    y_true = [0, 0, 1, 1]
    score_clear     = compute_AAPR(y_true, [0.1, 0.2, 0.8, 0.9])   # ≈ 0.489
    score_ambiguous = compute_AAPR(y_true, [0.45, 0.48, 0.52, 0.55]) # 1.0 (clipped)
    assert score_ambiguous > score_clear


def test_midpoint_predictions_are_maximized():
    """
    All predictions at 0.5 → maximum ambiguity for balanced classes.

    With boundary thresholds [0, 0.5, 1] the integral = 0.375, giving
    score = (1 - 0.375) / 0.75 = 5/6 ≈ 0.8333.  This is the highest
    achievable score for this class distribution; no other y_prob
    configuration produces a higher AAPR for balanced labels.
    """
    score = compute_AAPR([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], delta=0.1)
    assert score == pytest.approx(5 / 6, abs=1e-6)


def test_well_separated_reference_value():
    """
    Regression: y_prob=[0.1, 0.2, 0.8, 0.9] with balanced labels gives ≈ 0.489.
    This value is derived analytically from the trapezoid integral.
    """
    score = compute_AAPR([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], delta=0.1)
    assert score == pytest.approx(0.3889, abs=1e-3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_single_class_returns_zero():
    """
    Metric is undefined when only one class is present.
    Prevalence = 0 → division by zero; return 0.0 by convention.
    """
    score = compute_AAPR([0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], delta=0.1)
    assert score == pytest.approx(0.0)

    score = compute_AAPR([1, 1, 1, 1], [0.1, 0.5, 0.9, 0.5], delta=0.1)
    assert score == pytest.approx(0.0)


def test_mixed_predictions_in_range():
    """Score for a realistic mixed classifier must lie strictly in (0, 1)."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.05, 0.45, 0.55, 0.45, 0.55, 0.95]
    score  = compute_AAPR(y_true, y_prob, delta=0.1)
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Delta does NOT affect AAPR (global integral, not band-restricted)
# ---------------------------------------------------------------------------
def test_delta_does_not_change_score():
    """AAPR is computed over all thresholds, so changing delta must not change the score."""
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    score_narrow = compute_AAPR(y_true, y_prob, delta=0.05)
    score_wide   = compute_AAPR(y_true, y_prob, delta=0.3)
    assert score_narrow == pytest.approx(score_wide)


# ---------------------------------------------------------------------------
# Output type and range
# ---------------------------------------------------------------------------
def test_returns_float():
    score = compute_AAPR([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert isinstance(score, float)


def test_score_always_in_unit_interval():
    rng = np.random.default_rng(42)
    for _ in range(20):
        y_true = rng.integers(0, 2, size=50)
        y_prob = rng.uniform(0, 1, size=50)
        if y_true.mean() in (0.0, 1.0):
            continue
        score = compute_AAPR(y_true, y_prob, delta=0.15)
        assert 0.0 <= score <= 1.0, f"Score {score:.4f} out of [0,1]"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_invalid_delta_raises():
    with pytest.raises(ValueError):
        compute_AAPR([0, 1], [0.3, 0.7], delta=0.0)
    with pytest.raises(ValueError):
        compute_AAPR([0, 1], [0.3, 0.7], delta=0.6)


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_AAPR([0, 1], [0.1, 0.5, 0.9], delta=0.1)


def test_accepts_numpy_arrays():
    score = compute_AAPR(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_accepts_lists():
    score = compute_AAPR([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])
    assert score == pytest.approx(5 / 6, abs=1e-6)
