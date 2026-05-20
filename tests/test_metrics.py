"""
tests/test_metrics.py — pytest suite for ambiguity_suite.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=ambiguity_suite
"""

import pytest
import numpy as np
import warnings

from ambiguity_suite import compute_AAPR, compute_AAF1, compute_AAMass, __version__
from ambiguity_suite import get_interval, interval_width, validate_delta

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PERFECT = ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
RANDOM  = ([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

def test_version_exists():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


# ---------------------------------------------------------------------------
# AAMass
# ---------------------------------------------------------------------------

def test_aamass_no_ambiguity():
    assert compute_AAMass(*PERFECT, delta=0.1) == 0.0

def test_aamass_full_ambiguity():
    assert compute_AAMass(*RANDOM, delta=0.1) == 1.0

def test_aamass_output_range():
    score = compute_AAMass(*PERFECT, delta=0.1)
    assert 0.0 <= score <= 1.0

def test_aamass_wider_delta_catches_more():
    score_narrow = compute_AAMass([0, 1], [0.45, 0.55], delta=0.1)
    score_wide   = compute_AAMass([0, 1], [0.45, 0.55], delta=0.2)
    assert score_wide >= score_narrow

def test_aamass_invalid_delta_raises():
    with pytest.raises(ValueError):
        compute_AAMass(*PERFECT, delta=0.0)

    with pytest.raises(ValueError):
        compute_AAMass(*PERFECT, delta=0.6)


# ---------------------------------------------------------------------------
# AAPR
# ---------------------------------------------------------------------------

def test_aapr_good_classifier_lower_than_random():
    assert compute_AAPR(*PERFECT) < compute_AAPR(*RANDOM)

def test_aapr_output_range():
    for case in [PERFECT, RANDOM]:
        score = compute_AAPR(*case)
        assert 0.0 <= score <= 1.0

def test_aapr_known_values():
    assert round(compute_AAPR(*RANDOM), 4) == 0.8333
    assert round(compute_AAPR(*PERFECT), 4) == 0.3889

def test_aapr_single_class_returns_zero():
    assert compute_AAPR([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9]) == 0.0
    assert compute_AAPR([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]) == 0.0

def test_aapr_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_AAPR([0, 1], [0.1, 0.9, 0.5])

def test_aapr_nondefault_delta_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compute_AAPR(*PERFECT, delta=0.2)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "delta" in str(w[0].message).lower()

def test_aapr_default_delta_no_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compute_AAPR(*PERFECT, delta=0.1)
        assert len(w) == 0


# ---------------------------------------------------------------------------
# AAF1
# ---------------------------------------------------------------------------

def test_aaf1_good_classifier_lower_than_random():
    assert compute_AAF1(*PERFECT) < compute_AAF1(*RANDOM)

def test_aaf1_output_range():
    for case in [PERFECT, RANDOM]:
        score = compute_AAF1(*case)
        assert 0.0 <= score <= 1.0

def test_aaf1_known_values():
    assert round(compute_AAF1(*PERFECT), 4) == 0.3702
    assert round(compute_AAF1(*RANDOM), 4) == 0.9102

def test_aaf1_single_class_returns_zero():
    assert compute_AAF1([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9]) == 0.0

def test_aaf1_beta_positive_enforced():
    with pytest.raises(ValueError):
        compute_AAF1(*PERFECT, beta=-1.0)

    with pytest.raises(ValueError):
        compute_AAF1(*PERFECT, beta=0.0)

def test_aaf1_beta_gt1_weights_recall():
    score_b1 = compute_AAF1(*PERFECT, beta=1.0)
    score_b2 = compute_AAF1(*PERFECT, beta=2.0)
    # Different beta values should produce different scores
    assert score_b1 != score_b2

def test_aaf1_nondefault_delta_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compute_AAF1(*PERFECT, delta=0.2)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)

def test_aaf1_default_delta_no_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        compute_AAF1(*PERFECT, delta=0.1)
        assert len(w) == 0


# ---------------------------------------------------------------------------
# AARange utilities
# ---------------------------------------------------------------------------

def test_get_interval():
    assert get_interval(0.1) == (0.4, 0.6)
    assert get_interval(0.5) == (0.0, 1.0)

def test_interval_width():
    assert interval_width(0.1) == pytest.approx(0.2)
    assert interval_width(0.05) == pytest.approx(0.1)

def test_validate_delta_valid():
    validate_delta(0.1)   # should not raise
    validate_delta(0.5)   # boundary — should not raise

def test_validate_delta_invalid():
    with pytest.raises(ValueError):
        validate_delta(0.0)

    with pytest.raises(ValueError):
        validate_delta(0.51)
