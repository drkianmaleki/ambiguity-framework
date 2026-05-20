"""
Tests for AAF1.compute().

AAF1 is a normalized integral of F_beta(t) over all thresholds t in y_prob.

    AAF1 = (1 - ∫F_beta(t)dt) / (1 - AA_min)     clipped to [0, 1]

where AA_min is the prevalence-dependent random-classifier baseline (Eq. 9).

High AAF1 → F_beta degrades across thresholds (high ambiguity / fragility).
Low AAF1  → F_beta remains high across thresholds (low ambiguity).

Key contracts:
  - Single class in y_true         → 0.0   (undefined, safe fallback)
  - Well-separated < Near-midpoint           (ordering preserved)
  - Result always in [0.0, 1.0]
  - delta is validated but does NOT affect the score (global metric)
  - beta changes the score whenever P(t) ≠ R(t) at dominant thresholds
"""

import pytest
import numpy as np
from ambiguity_suite import compute_AAF1


# ---------------------------------------------------------------------------
# Canonical inputs
# ---------------------------------------------------------------------------
PERFECT_SEP_TRUE = [0, 0, 1, 1]
PERFECT_SEP_PROB = [0.1, 0.2, 0.8, 0.9]

MIDPOINT_TRUE = [0, 0, 1, 1]
MIDPOINT_PROB = [0.5, 0.5, 0.5, 0.5]


# ---------------------------------------------------------------------------
# Core contract: ordering
# ---------------------------------------------------------------------------
def test_ambiguous_scores_higher_than_clear():
    """
    Predictions clustered near 0.5 must score higher than well-separated ones.
    Reference values confirmed by running compute() against the formula.
    """
    score_clear     = compute_AAF1(PERFECT_SEP_TRUE, PERFECT_SEP_PROB, beta=1.0)  # ≈ 0.3702
    score_ambiguous = compute_AAF1(MIDPOINT_TRUE, MIDPOINT_PROB, beta=1.0)         # ≈ 0.9102
    assert score_ambiguous > score_clear


def test_midpoint_reference_value():
    """
    All predictions at 0.5 → high ambiguity for balanced classes.

    F1 at threshold 0.5 is 2/3; the curve is nearly flat at a low value across
    most thresholds. Normalised against the random-classifier baseline (Eq. 9)
    this gives ≈ 0.9102 — the highest achievable score for this distribution.
    """
    score = compute_AAF1(MIDPOINT_TRUE, MIDPOINT_PROB, delta=0.1, beta=1.0)
    assert score == pytest.approx(0.9102, abs=1e-3)


def test_well_separated_reference_value():
    """
    Regression: y_prob=[0.1, 0.2, 0.8, 0.9] with balanced labels gives ≈ 0.3702.
    The classifier maintains high F1 across most thresholds → low ambiguity score.
    """
    score = compute_AAF1(PERFECT_SEP_TRUE, PERFECT_SEP_PROB, delta=0.1, beta=1.0)
    assert score == pytest.approx(0.3702, abs=1e-3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_single_class_returns_zero():
    """
    Metric is undefined when only one class is present; return 0.0 by convention.
    """
    assert compute_AAF1([0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)
    assert compute_AAF1([1, 1, 1, 1], [0.1, 0.5, 0.9, 0.5]) == pytest.approx(0.0)


def test_mixed_predictions_in_range():
    """Score for a realistic mixed classifier must lie strictly in (0, 1)."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.05, 0.45, 0.55, 0.45, 0.55, 0.95]
    score  = compute_AAF1(y_true, y_prob, delta=0.1)
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Delta does NOT affect AAF1 (global integral, not band-restricted)
# ---------------------------------------------------------------------------
def test_delta_does_not_change_score():
    """AAF1 is computed over all thresholds, so changing delta must not change the score."""
    score_narrow = compute_AAF1(PERFECT_SEP_TRUE, PERFECT_SEP_PROB, delta=0.05)
    score_wide   = compute_AAF1(PERFECT_SEP_TRUE, PERFECT_SEP_PROB, delta=0.3)
    assert score_narrow == pytest.approx(score_wide)


# ---------------------------------------------------------------------------
# Beta weighting
# ---------------------------------------------------------------------------
def test_beta_changes_score_when_precision_recall_differ():
    """
    When Precision ≠ Recall at the dominant threshold, changing beta must
    produce a different score.  Here R=1.0 > P=0.75 (3 TP, 1 FP in the
    single active threshold region), so F2 sees better performance than F1,
    resulting in a lower normalized degradation score.
    """
    y_true = [0, 1, 1, 1]
    y_prob = [0.55, 0.55, 0.55, 0.55]
    score_f1 = compute_AAF1(y_true, y_prob, delta=0.1, beta=1.0)  # ≈ 0.7407
    score_f2 = compute_AAF1(y_true, y_prob, delta=0.1, beta=2.0)  # ≈ 0.5621
    # F2 weights recall (R=1.0) more → higher F_beta value → integral AA_un is larger
    # → (1 - AA_un) is smaller → lower normalized score
    assert score_f2 != pytest.approx(score_f1)
    assert score_f2 < score_f1


# ---------------------------------------------------------------------------
# Output type and range
# ---------------------------------------------------------------------------
def test_returns_float():
    score = compute_AAF1(PERFECT_SEP_TRUE, PERFECT_SEP_PROB)
    assert isinstance(score, float)


def test_score_in_unit_interval():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    for beta in [0.5, 1.0, 2.0]:
        score = compute_AAF1(y_true, y_prob, delta=0.1, beta=beta)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for beta={beta}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_invalid_delta_raises():
    with pytest.raises(ValueError):
        compute_AAF1([0, 1], [0.4, 0.6], delta=0.0)

    with pytest.raises(ValueError):
        compute_AAF1([0, 1], [0.4, 0.6], delta=0.6)


def test_invalid_beta_raises():
    with pytest.raises(ValueError):
        compute_AAF1([0, 1], [0.4, 0.6], beta=0.0)

    with pytest.raises(ValueError):
        compute_AAF1([0, 1], [0.4, 0.6], beta=-1.0)


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_AAF1([0, 1], [0.1, 0.5, 0.9], delta=0.1)


def test_accepts_numpy_arrays():
    score = compute_AAF1(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_accepts_lists():
    score = compute_AAF1([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
