"""Smart Retail Precision Demand Forecasting & Dynamic Pricing Engine.

Core library package. See ``docs/Smart_Retail_Neural_Map.md`` for the full
product and architecture spec this package implements against.

Modules
-------
data_loader
    Raw POS ingestion, cleaning rules and the zero-filled daily sales grid.
features
    :class:`~src.features.RetailFeatureEngineer` — calendar, lag, rolling and
    price-perception features used by the forecaster.
forecasting
    WAPE scoring, expanding-window cross-validation and the LightGBM Tweedie
    + quantile demand forecaster (with a dependency-free fallback).
pricing_engine
    Log-log elasticity demand curve and bounded, profit-maximising price
    optimisation with cost-floor / historical-ceiling guardrails.
"""

from __future__ import annotations

__all__ = ["data_loader", "features", "forecasting", "pricing_engine"]
__version__ = "1.0.0"
