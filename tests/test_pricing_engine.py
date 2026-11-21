"""Tests for src.pricing_engine: demand curve, guardrails, markdown ladder."""

from __future__ import annotations

import pytest

from src.pricing_engine import demand_curve, markdown_ladder, optimize_price


def test_demand_curve_identity_at_reference_price():
    assert demand_curve(predicted_demand=50, reference_price=100, price=100, elasticity=-1.2) == pytest.approx(50)


def test_demand_curve_rises_as_price_falls():
    base = demand_curve(50, 100, 100, -1.2)
    lower = demand_curve(50, 100, 80, -1.2)
    assert lower > base


def test_demand_curve_uses_magnitude_of_elasticity_either_sign():
    assert demand_curve(50, 100, 80, -1.2) == pytest.approx(demand_curve(50, 100, 80, 1.2))


def test_demand_curve_rejects_non_positive_price():
    with pytest.raises(ValueError):
        demand_curve(50, 100, 0, -1.0)
    with pytest.raises(ValueError):
        demand_curve(50, 0, 10, -1.0)


def test_optimize_price_respects_cost_floor_and_ceiling():
    res = optimize_price(
        predicted_demand=20, unit_cost=70, inventory=40, historical_max_price=90, reference_price=90, elasticity=-1.4
    )
    assert res["floor_price"] == pytest.approx(1.15 * 70)
    assert res["ceiling_price"] == pytest.approx(min(1.2 * 90, 2 * 90))
    assert res["floor_price"] - 1e-6 <= res["optimal_price"] <= res["ceiling_price"] + 1e-6


def test_optimize_price_caps_units_sold_at_inventory():
    res = optimize_price(
        predicted_demand=1000, unit_cost=10, inventory=5, historical_max_price=20, reference_price=20, elasticity=-0.5
    )
    assert res["expected_units_sold"] <= 5 + 1e-9


def test_optimize_price_handles_degenerate_floor_above_ceiling():
    # unit_cost far above what the SKU has ever sold for: floor collapses onto ceiling.
    res = optimize_price(
        predicted_demand=10, unit_cost=100, inventory=20, historical_max_price=50, reference_price=50, elasticity=-1.0
    )
    assert res["floor_price"] == res["ceiling_price"] == res["optimal_price"]


def test_optimize_price_returns_only_plain_numeric_values():
    res = optimize_price(
        predicted_demand=20, unit_cost=70, inventory=40, historical_max_price=90, reference_price=90, elasticity=-1.4
    )
    # generate_sales.py does `{k: round(v, 2) for k, v in res.items()}` — every
    # value must support round().
    for value in res.values():
        assert isinstance(value, float)
        round(value, 2)


def test_markdown_ladder_skips_rungs_below_cost_floor():
    # unit_cost so high that even the shallowest (10%) markdown would sell below floor.
    rungs = markdown_ladder(
        predicted_demand=20, unit_cost=90, inventory=40, reference_price=100, elasticity=-1.2, rungs=(0.10, 0.25, 0.40)
    )
    floor = 1.15 * 90
    for rung in rungs:
        assert rung["price"] >= floor - 1e-9


def test_markdown_ladder_deeper_discount_sells_through_more():
    rungs = markdown_ladder(
        predicted_demand=10, unit_cost=50, inventory=40, reference_price=100, elasticity=-1.5, rungs=(0.10, 0.25, 0.40)
    )
    sell_throughs = [r["sell_through_pct"] for r in rungs]
    assert sell_throughs == sorted(sell_throughs)
