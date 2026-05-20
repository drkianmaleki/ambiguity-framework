"""
AARange — Ambiguity Interval utilities.

The Ambiguity Range ℐ_δ = [0.5 - delta, 0.5 + delta] is Parameter 1
of the Ambiguity Range Framework (paper, Section 3.1, Equation 2).

Unlike AAPR, AAF1, and AAMass — which are metrics computed from data —
the Ambiguity Range is an input parameter set by the practitioner. It
defines the coordinate window within which ambiguity is evaluated.

This module provides:
  - get_interval(delta)     → returns (lower, upper) bounds of ℐ_δ
  - interval_width(delta)   → returns the full width 2δ
  - validate_delta(delta)   → raises ValueError if delta is out of range

These utilities are used internally by AAMass, AAPR, and AAF1, and are
exposed here for practitioners who need to inspect or report the interval.
"""

import numpy as np


def get_interval(delta: float) -> tuple:
    """
    Return the lower and upper bounds of the ambiguity interval ℐ_δ.

    Parameters
    ----------
    delta : float
        Scalar half-width of the indecision band. Must be in (0, 0.5].

    Returns
    -------
    (lower, upper) : tuple of float
        The interval [0.5 - delta, 0.5 + delta].

    Examples
    --------
    >>> get_interval(0.1)
    (0.4, 0.6)
    >>> get_interval(0.5)
    (0.0, 1.0)
    """
    validate_delta(delta)
    return (round(0.5 - delta, 10), round(0.5 + delta, 10))


def interval_width(delta: float) -> float:
    """
    Return the full width of the ambiguity interval (2 * delta).

    This is the value reported in the paper's Tables 2–5 under
    the column 'δ Range'.

    Parameters
    ----------
    delta : float
        Scalar half-width. Must be in (0, 0.5].

    Returns
    -------
    float — full width 2δ

    Examples
    --------
    >>> interval_width(0.1)
    0.2
    >>> interval_width(0.05)
    0.1
    """
    validate_delta(delta)
    return round(2.0 * delta, 10)


def validate_delta(delta: float) -> None:
    """
    Raise ValueError if delta is outside the valid range (0, 0.5].

    Parameters
    ----------
    delta : float

    Raises
    ------
    ValueError
        If delta <= 0 or delta > 0.5.

    Examples
    --------
    >>> validate_delta(0.1)   # no error
    >>> validate_delta(0.0)
    Traceback (most recent call last):
        ...
    ValueError: delta must be in (0, 0.5], got 0.0
    """
    if not (0.0 < delta <= 0.5):
        raise ValueError(f"delta must be in (0, 0.5], got {delta}")
