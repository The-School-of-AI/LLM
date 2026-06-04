"""Unit tests for calculate_proportions script logic."""

import math

import pytest  # noqa: F401

from scripts.calculate_proportions import (
    alignment_weight,
    apply_floors_and_caps,
    model_capacity,
    renormalize,
)


def test_model_capacity():
    """Test normalized capacity calculation."""
    # Min params -> 0.0
    assert math.isclose(model_capacity(1e9, 1e9, 70e9), 0.0, abs_tol=1e-5)

    # Max params -> 1.0
    assert math.isclose(model_capacity(70e9, 1e9, 70e9), 1.0, abs_tol=1e-5)

    # Mid params check (geometric mean should be ~0.5 but it's log scale)
    # log(sqrt(max*min)) is exactly halfway
    mid_params = math.sqrt(1e9 * 70e9)
    assert math.isclose(model_capacity(mid_params, 1e9, 70e9), 0.5, abs_tol=1e-5)

    # Out of bounds clamping
    assert math.isclose(model_capacity(0.5e9, 1e9, 70e9), 0.0, abs_tol=1e-5)
    assert math.isclose(model_capacity(100e9, 1e9, 70e9), 1.0, abs_tol=1e-5)


def test_alignment_weight():
    """Test alignment weight calculation."""
    # Perfect match
    assert alignment_weight(0.5, 0.5) == 1.0

    # Distance decay
    w1 = alignment_weight(0.5, 0.6)
    w2 = alignment_weight(0.5, 0.7)
    assert w1 > w2


def test_apply_floors_and_caps():
    """Test floor and cap enforcement."""
    weights = {"B0": 0.05, "B1": 0.50}
    floors = {"B0": 0.10, "B1": 0.10}

    constrained = apply_floors_and_caps(weights, floors)

    # B0 should be raised to floor
    assert constrained["B0"] == 0.10
    # B1 should stay same (above floor)
    assert constrained["B1"] == 0.50


def test_renormalize():
    """Test renormalization to sum to 1.0."""
    weights = {"A": 0.1, "B": 0.3}
    norm = renormalize(weights)

    assert math.isclose(sum(norm.values()), 1.0)
    assert math.isclose(norm["A"], 0.25)  # 0.1 / 0.4
    assert math.isclose(norm["B"], 0.75)  # 0.3 / 0.4
