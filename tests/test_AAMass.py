"""
Tests for AAMass.compute().

AAMass measures what fraction of predictions fall inside the indecision
interval ℐ_δ = [0.5 - delta, 0.5 + delta]. These tests define the ground
truth behaviour before any implementation is written.
"""

import pytest
import numpy as np
from ambiguity_suite import compute_AAMass


# ---------------------------------------------------------------------------
# Canonical inputs
# ---------------------------------------------------------------------------
PERFECT_SEP_TRUE = [0, 0, 1, 1]
PERFECT_SEP_PROB = [0.1, 0.2, 0.8, 0.9]   # all outside [0.4, 0.6] at delta=0.1

PERFECT_OVR_TRUE = [0, 0, 1, 1]
PERFECT_OVR_PROB = [0.5, 0.5, 0.5, 0.5]   # all inside [0.4, 0.6] at delta=0.1

PARTIAL_TRUE     = [0, 0, 1, 1]
PARTIAL_PROB     = [0.1, 0.5, 0.5, 0.9]   # 2 of 4 in band → 0.5


# ---------------------------------------------------------------------------
# Core contract tests
# ---------------------------------------------------------------------------
def test_perfect_separation_returns_zero():
    """No predictions in the interval → score must be exactly 0.0."""
    score = compute_AAMass(PERFECT_SEP_TRUE, PERFECT_SEP_PROB, delta=0.1)
    assert score == 0.0


def test_perfect_overlap_returns_one():
    """All predictions at 0.5 → entire sample set is ambiguous → score must be 1.0."""
    score = compute_AAMass(PERFECT_OVR_TRUE, PERFECT_OVR_PROB, delta=0.1)
    assert score == 1.0


def test_partial_overlap_returns_fraction():
    """Half the predictions in the interval → score must be 0.5."""
    score = compute_AAMass(PARTIAL_TRUE, PARTIAL_PROB, delta=0.1)
    assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Output type and range
# ---------------------------------------------------------------------------
def test_returns_float():
    score = compute_AAMass(PERFECT_SEP_TRUE, PERFECT_SEP_PROB)
    assert isinstance(score, float)


def test_score_in_unit_interval():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    score = compute_AAMass(y_true, y_prob, delta=0.1)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Delta sensitivity
# ---------------------------------------------------------------------------
def test_wider_delta_increases_score():
    """A wider indecision interval must capture at least as many samples."""
    y_true = [0, 0, 1, 1]
    y_prob = [0.35, 0.45, 0.55, 0.65]
    score_narrow = compute_AAMass(y_true, y_prob, delta=0.05)
    score_wide   = compute_AAMass(y_true, y_prob, delta=0.2)
    assert score_wide >= score_narrow


def test_delta_zero_point_five_returns_one():
    """delta=0.5 makes the interval [0.0, 1.0] — all predictions are inside."""
    y_true = [0, 1]
    y_prob = [0.1, 0.9]
    score = compute_AAMass(y_true, y_prob, delta=0.5)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_invalid_delta_raises():
    with pytest.raises(ValueError):
        compute_AAMass([0, 1], [0.3, 0.7], delta=0.0)

    with pytest.raises(ValueError):
        compute_AAMass([0, 1], [0.3, 0.7], delta=0.6)


def test_accepts_numpy_arrays():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    score = compute_AAMass(y_true, y_prob, delta=0.1)
    assert score == 0.0


def test_accepts_lists():
    score = compute_AAMass([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], delta=0.1)
    assert score == 0.0


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_AAMass([0, 1], [0.1, 0.5, 0.9], delta=0.1)
