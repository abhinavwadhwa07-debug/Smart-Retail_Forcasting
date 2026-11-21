"""ML Core — Price elasticity & dynamic pricing.

Two pieces, per the Neural Map spec:

- :func:`demand_curve` — the log-log elasticity model
  ``Q = Q0 * (P0 / P) ** |e|``: how many units sell at a candidate price
  ``P``, given a reference price ``P0`` at which ``Q0`` units were expected.
- :func:`optimize_price` — the bounded, profit-maximising price search: a
  1-D ``scipy.optimize.minimize_scalar`` over the demand curve, constrained
  to ``[1.15 x unit_cost, min(1.2 x historical_max_price, 2 x reference_price)]``
  so a clearance markdown can never dip below a viable margin and a
  "dynamic" price can never spike into surge-pricing territory.

:func:`markdown_ladder` is a small extra for the Decision & Action Engine's
markdown action (discount rungs 10% -> 25% -> 40%), built on the same demand
curve, clipped to the same cost floor.
"""

from __future__ import annotations

from typing import Final

from scipy.optimize import minimize_scalar

__all__ = ["demand_curve", "optimize_price", "markdown_ladder"]

#: Minimum acceptable margin over unit cost: a price can never be set below
#: ``COST_FLOOR_MULT * unit_cost``, even to clear expiring stock.
COST_FLOOR_MULT: Final[float] = 1.15
#: A "dynamic" ceiling price can never exceed this multiple of the SKU's own
#: historical maximum price...
HISTORICAL_CEILING_MULT: Final[float] = 1.2
#: ...nor this multiple of the current reference/shelf price — whichever
#: ceiling is tighter wins. Together these are the anti-surge-pricing
#: guardrail (see the Problem & Vision > Customers "no surge pricing on
#: essentials" constraint in the Neural Map).
REFERENCE_CEILING_MULT: Final[float] = 2.0


def demand_curve(predicted_demand: float, reference_price: float, price: float, elasticity: float) -> float:
    """Log-log constant-elasticity demand curve: ``Q = Q0 * (P0 / P) ** |e|``.

    Parameters
    ----------
    predicted_demand:
        ``Q0`` — the forecast demand (units) at the reference price, i.e.
        the forecaster's output for this SKU-day *before* any price change
        is considered.
    reference_price:
        ``P0`` — the price the forecast was made at (normally today's shelf
        price).
    price:
        ``P`` — the candidate price to evaluate demand at.
    elasticity:
        Price elasticity of demand for this SKU. Only its magnitude is
        used (retail elasticities are conventionally negative — demand
        falls as price rises — but the curve is defined in terms of
        ``|e|`` so either sign convention works): ``|e| < 1`` is inelastic
        (a price cut doesn't make up for itself in extra volume), ``|e| > 1``
        is elastic (it does).

    Returns
    -------
    float
        Expected demand at ``price``. Equals ``predicted_demand`` exactly
        when ``price == reference_price``.

    Raises
    ------
    ValueError
        If ``price`` or ``reference_price`` is not strictly positive — the
        curve is undefined (division by zero or a non-physical negative
        price).
    """
    if price <= 0 or reference_price <= 0:
        raise ValueError(f"price and reference_price must be > 0, got price={price}, reference_price={reference_price}")
    return predicted_demand * (reference_price / price) ** abs(elasticity)


def optimize_price(
    predicted_demand: float,
    unit_cost: float,
    inventory: float,
    historical_max_price: float,
    reference_price: float,
    elasticity: float,
) -> dict[str, float]:
    """Bounded, profit-maximising price search within guardrails.

    Maximises expected profit ``units_sold * (price - unit_cost)``, where
    ``units_sold = min(demand_curve(price), inventory)`` (you cannot sell
    more than you have on the shelf), over a 1-D bounded search
    (``scipy.optimize.minimize_scalar(..., method="bounded")``) between:

    - **floor** = ``COST_FLOOR_MULT * unit_cost`` (never sell below a
      viable margin, even to clear near-expiry stock), and
    - **ceiling** = ``min(HISTORICAL_CEILING_MULT * historical_max_price,
      REFERENCE_CEILING_MULT * reference_price)`` (never price above what
      the SKU has realistically fetched before, or more than double
      today's price — the anti-surge-pricing guardrail).

    Parameters
    ----------
    predicted_demand:
        Forecast demand (units) at ``reference_price`` — ``Q0`` in
        :func:`demand_curve`.
    unit_cost:
        Cost per unit; defines the profit floor.
    inventory:
        Units physically available to sell (a batch quantity, for a
        near-expiry clearance call, or total on-hand for routine pricing).
        Caps ``units_sold`` regardless of how much the demand curve says
        would sell.
    historical_max_price:
        The highest price this SKU has credibly sold at — bounds how far
        "dynamic" pricing can reach upward.
    reference_price:
        Today's shelf price — both the ``P0`` the demand curve is anchored
        to and the baseline every result in the return value is compared
        against.
    elasticity:
        This SKU's price elasticity of demand (see :func:`demand_curve`).

    Returns
    -------
    dict[str, float]
        All-numeric (every value is a plain ``float``, so callers can
        ``round()`` the whole dict uniformly):

        - ``optimal_price`` — the recommended price.
        - ``floor_price`` / ``ceiling_price`` — the guardrail bounds used.
        - ``expected_demand`` — raw demand-curve output at ``optimal_price``
          (may exceed ``inventory``; see ``expected_units_sold``).
        - ``expected_units_sold`` — demand capped at ``inventory``.
        - ``expected_revenue`` / ``expected_profit`` — at ``optimal_price``.
        - ``baseline_units_sold`` / ``baseline_revenue`` / ``baseline_profit``
          — the same quantities at today's ``reference_price``, for
          comparison.
        - ``discount_pct`` — how far ``optimal_price`` sits below
          ``reference_price`` (negative if the optimum is actually an
          increase).
        - ``profit_uplift_rs`` — ``expected_profit - baseline_profit``, the
          rupee case for repricing at all.
    """
    floor = COST_FLOOR_MULT * unit_cost
    ceiling = min(HISTORICAL_CEILING_MULT * historical_max_price, REFERENCE_CEILING_MULT * reference_price)
    if floor > ceiling:
        # Degenerate inputs (e.g. unit_cost close to or above historical_max_price):
        # collapse to the single feasible point rather than raising, so a bad
        # upstream estimate for one SKU can't take down a whole batch job.
        floor = ceiling

    def neg_profit(price: float) -> float:
        units_sold = min(demand_curve(predicted_demand, reference_price, price, elasticity), inventory)
        return -(units_sold * (price - unit_cost))

    if floor == ceiling:
        best_price = floor
    else:
        result = minimize_scalar(neg_profit, bounds=(floor, ceiling), method="bounded")
        best_price = float(result.x)

    expected_demand = demand_curve(predicted_demand, reference_price, best_price, elasticity)
    expected_units_sold = min(expected_demand, inventory)
    expected_revenue = expected_units_sold * best_price
    expected_profit = expected_units_sold * (best_price - unit_cost)

    baseline_units_sold = min(predicted_demand, inventory)
    baseline_revenue = baseline_units_sold * reference_price
    baseline_profit = baseline_units_sold * (reference_price - unit_cost)

    return {
        "optimal_price": best_price,
        "floor_price": floor,
        "ceiling_price": ceiling,
        "expected_demand": float(expected_demand),
        "expected_units_sold": float(expected_units_sold),
        "expected_revenue": float(expected_revenue),
        "expected_profit": float(expected_profit),
        "baseline_units_sold": float(baseline_units_sold),
        "baseline_revenue": float(baseline_revenue),
        "baseline_profit": float(baseline_profit),
        "discount_pct": float(100.0 * (1 - best_price / reference_price)),
        "profit_uplift_rs": float(expected_profit - baseline_profit),
    }


def markdown_ladder(
    predicted_demand: float,
    unit_cost: float,
    inventory: float,
    reference_price: float,
    elasticity: float,
    rungs: tuple[float, ...] = (0.10, 0.25, 0.40),
) -> list[dict[str, float]]:
    """Evaluate the Action Catalogue's markdown ladder (10% -> 25% -> 40% off).

    For a near-expiry batch the question usually isn't "what's the
    profit-optimal price" (that can land arbitrarily close to cost) but "how
    deep a discount clears the batch" — so this walks fixed, store-friendly
    discount rungs instead of a free-form optimum, still respecting the same
    cost floor as :func:`optimize_price`.

    Parameters
    ----------
    predicted_demand, unit_cost, inventory, reference_price, elasticity:
        Same meaning as in :func:`optimize_price`.
    rungs:
        Discount fractions to evaluate, e.g. ``0.10`` for 10% off.

    Returns
    -------
    list[dict[str, float]]
        One entry per rung actually evaluated (a rung whose price would
        fall below the ``COST_FLOOR_MULT * unit_cost`` floor is skipped
        rather than silently sold under cost), each with: ``discount_pct``,
        ``price``, ``expected_units_sold``, ``expected_revenue``,
        ``expected_profit``, ``sell_through_pct`` (``expected_units_sold /
        inventory``).
    """
    floor = COST_FLOOR_MULT * unit_cost
    out = []
    for rung in rungs:
        price = reference_price * (1 - rung)
        if price < floor:
            continue
        demand = demand_curve(predicted_demand, reference_price, price, elasticity)
        units_sold = min(demand, inventory)
        out.append(
            {
                "discount_pct": float(rung * 100),
                "price": float(price),
                "expected_units_sold": float(units_sold),
                "expected_revenue": float(units_sold * price),
                "expected_profit": float(units_sold * (price - unit_cost)),
                "sell_through_pct": float(100 * units_sold / inventory) if inventory else 0.0,
            }
        )
    return out
