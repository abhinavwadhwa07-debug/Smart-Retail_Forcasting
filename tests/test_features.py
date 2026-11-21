"""Tests for src.features.RetailFeatureEngineer: contract + leakage-safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import RetailFeatureEngineer, WARMUP_DAYS


def _write_series(tmp_path, item_values: dict[str, list[float]], start="2026-01-01", price: float = 10.0):
    rows = []
    for item, values in item_values.items():
        dates = pd.date_range(start, periods=len(values), freq="D")
        for d, v in zip(dates, values):
            rows.append((item, d, v, price))
    df = pd.DataFrame(rows, columns=["item_id", "invoice_date", "quantity", "unit_price"])
    path = tmp_path / "series.csv"
    df.to_csv(path, index=False)
    return path


def test_create_temporal_features_columns_and_values(tmp_path):
    path = _write_series(tmp_path, {"A": list(range(10))})
    fe = RetailFeatureEngineer(path)
    fe.load_data()
    t = fe.create_temporal_features()
    assert {"day_of_week", "day_of_month", "month", "quarter", "is_weekend", "payday_flag"} <= set(t.columns)
    # 2026-01-01 is a Thursday -> day_of_week 3, not a weekend, is a payday (day 1).
    first = t.iloc[0]
    assert first["day_of_week"] == 3
    assert first["is_weekend"] == 0
    assert first["payday_flag"] == 1


def test_lag_features_are_shifted_and_filled(tmp_path):
    path = _write_series(tmp_path, {"A": list(range(10))})  # 0,1,2,...,9
    fe = RetailFeatureEngineer(path)
    fe.load_data()
    lags = fe.create_lag_features()
    assert list(lags["lag_1_days"]) == [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    # lag_7_days at row index 7 (value 7) must be row 0's value (0), never its own.
    assert lags.loc[7, "lag_7_days"] == 0
    assert lags.loc[8, "lag_7_days"] == 1


def test_rolling_features_never_see_current_day(tmp_path):
    # Strictly increasing series makes leakage obvious: if today's value ever
    # leaked in, rolling_7_max would equal the current row's own quantity.
    path = _write_series(tmp_path, {"A": list(range(10))})
    fe = RetailFeatureEngineer(path)
    fe.load_data()
    r = fe.create_rolling_features()
    for i in range(1, 10):
        assert r.loc[i, "rolling_7_max"] < i, "rolling window leaked the current day's own value"


def test_relative_price_ratio_reflects_discount_and_markup(tmp_path):
    prices = [10.0] * 10 + [5.0]  # a markdown on the 11th day
    rows = []
    dates = pd.date_range("2026-01-01", periods=len(prices), freq="D")
    for d, p in zip(dates, prices):
        rows.append(("A", d, 4, p))
    df = pd.DataFrame(rows, columns=["item_id", "invoice_date", "quantity", "unit_price"])
    path = tmp_path / "price.csv"
    df.to_csv(path, index=False)

    fe = RetailFeatureEngineer(path)
    fe.load_data()
    r = fe.create_rolling_features()
    last = r.iloc[-1]
    # Price dropped to 5 while the trailing 30-day average is still ~10 -> ratio well below 1.
    assert last["relative_price_ratio"] < 0.6


def test_transform_matches_generate_sales_merge_contract(tmp_path):
    """Reproduces the exact merge generate_sales.py performs, to pin the
    column-name contract (lag_*_days, rolling_* prefix, relative_price_ratio)
    that the demo generator (and any other caller) depends on."""
    path = _write_series(tmp_path, {"A": list(range(40)), "B": list(range(40, 0, -1))})
    fe = RetailFeatureEngineer(path)
    fe.load_data()
    k = ["item_id", "invoice_date"]
    t, l, r = fe.create_temporal_features(), fe.create_lag_features(), fe.create_rolling_features()
    merged = t.merge(l[k + [c for c in l.columns if c.startswith("lag_")]], on=k).merge(
        r[k + [c for c in r.columns if c.startswith("rolling_") or c == "relative_price_ratio"]], on=k
    )
    warm = merged[merged.groupby("item_id").cumcount() >= 30].reset_index(drop=True)

    full = RetailFeatureEngineer(path).transform()
    # transform() adds item_code on top of the manual merge; compare the shared columns.
    shared = [c for c in warm.columns]
    pd.testing.assert_frame_equal(
        full[shared].reset_index(drop=True), warm.reset_index(drop=True), check_like=False
    )
    assert "item_code" in full.columns
    # 40-row series, 30-day warmup per item -> 10 rows survive per item.
    assert (full.groupby("item_id").size() == 40 - WARMUP_DAYS).all()


def test_compute_intermittency_profile_classifies_dead_sku(tmp_path):
    path = _write_series(tmp_path, {"STEADY": [5] * 60, "DEAD": [0] * 60})
    fe = RetailFeatureEngineer(path)
    fe.load_data()
    profile = fe.compute_intermittency_profile().set_index("item_id")
    assert profile.loc["DEAD", "demand_class"] == "dead"
    assert profile.loc["STEADY", "demand_class"] == "smooth"


def test_methods_raise_before_load_data(tmp_path):
    path = _write_series(tmp_path, {"A": [1, 2, 3]})
    fe = RetailFeatureEngineer(path)
    with pytest.raises(RuntimeError):
        fe.create_temporal_features()
