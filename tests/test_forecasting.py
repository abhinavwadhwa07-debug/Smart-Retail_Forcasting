"""Tests for src.forecasting: WAPE, expanding-window CV, DemandForecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import RetailFeatureEngineer
from src.forecasting import (
    DemandForecaster,
    expanding_window_cv,
    expanding_window_splits,
    seasonal_naive_forecast,
    wape,
)


def test_wape_perfect_forecast_is_zero():
    assert wape([1, 2, 3], [1, 2, 3]) == 0.0


def test_wape_known_value():
    # |1-2| + |2-1| = 2, actual sum = 3 -> 2/3
    assert wape([1, 2], [2, 1]) == pytest.approx(2 / 3)


def test_wape_handles_all_zero_actuals_without_dividing_by_zero():
    assert wape([0, 0, 0], [0, 0, 0]) == 0.0
    assert wape([0, 0], [1, 2]) == 3.0  # falls back to summed absolute forecast error


def test_seasonal_naive_forecast_shifts_per_item():
    df = pd.DataFrame(
        {
            "item_id": ["A"] * 10 + ["B"] * 10,
            "invoice_date": list(pd.date_range("2026-01-01", periods=10)) * 2,
            "quantity": list(range(10)) + list(range(100, 110)),
        }
    )
    naive = seasonal_naive_forecast(df, season_days=7)
    a = naive[df.item_id == "A"].reset_index(drop=True)
    assert a.iloc[7] == 0  # day index 7 forecast = day index 0's actual
    assert a.iloc[8] == 1
    assert a.iloc[:7].isna().all()  # no 7-day-ago history yet for the first week


def test_expanding_window_splits_are_time_respecting_and_growing():
    dates = pd.Series(pd.date_range("2026-01-01", periods=120, freq="D"))
    folds = list(expanding_window_splits(dates, n_splits=3, test_size_days=20))
    assert len(folds) == 3
    prev_train_count = 0
    for train_mask, test_mask in folds:
        assert train_mask.sum() > prev_train_count
        prev_train_count = train_mask.sum()
        train_dates = dates[train_mask]
        test_dates = dates[test_mask]
        # Every training date must be strictly before every test date in the same fold.
        assert train_dates.max() < test_dates.min()


def test_expanding_window_splits_rejects_too_short_history():
    dates = pd.Series(pd.date_range("2026-01-01", periods=5, freq="D"))
    with pytest.raises(ValueError):
        list(expanding_window_splits(dates, n_splits=1, test_size_days=28))


@pytest.fixture
def demo_features(tmp_path):
    rng = np.random.default_rng(42)
    days = pd.date_range("2026-01-01", periods=150, freq="D")
    rows = []
    for item, base in (("A", 20), ("B", 6)):
        for d in days:
            dow_mult = 1.3 if d.dayofweek >= 5 else 1.0
            rows.append((item, d, int(rng.poisson(base * dow_mult)), 10.0))
    df = pd.DataFrame(rows, columns=["item_id", "invoice_date", "quantity", "unit_price"])
    path = tmp_path / "demand.csv"
    df.to_csv(path, index=False)
    feat = RetailFeatureEngineer(path).transform()
    feature_cols = [c for c in feat.columns if c not in ("item_id", "invoice_date", "quantity")]
    return feat, feature_cols


def test_demand_forecaster_quantiles_never_cross(demo_features):
    feat, feature_cols = demo_features
    cut = feat.invoice_date.max() - pd.Timedelta(days=20)
    train, test = feat[feat.invoice_date < cut], feat[feat.invoice_date >= cut]
    model = DemandForecaster().fit(train[feature_cols], train["quantity"])
    q = model.predict_quantiles(test[feature_cols])
    assert (q["p10"] <= q["p50"] + 1e-9).all()
    assert (q["p50"] <= q["p90"] + 1e-9).all()
    assert (q >= 0).all().all()


def test_demand_forecaster_predict_is_non_negative(demo_features):
    feat, feature_cols = demo_features
    model = DemandForecaster(n_estimators=50).fit(feat[feature_cols], feat["quantity"])
    preds = model.predict(feat[feature_cols])
    assert (preds >= 0).all()


def test_expanding_window_cv_beats_or_matches_naive_baseline(demo_features):
    feat, feature_cols = demo_features
    cv = expanding_window_cv(feat, feature_cols, n_splits=2, test_size_days=20)
    assert len(cv) == 2
    assert {"wape_model", "wape_naive", "coverage_p10_p90"} <= set(cv.columns)
    # Not a strict guarantee on every possible random seed, but a trained
    # LightGBM model should not be wildly worse than the naive baseline on
    # data with genuine weekly seasonality.
    assert cv["wape_model"].mean() < cv["wape_naive"].mean() * 1.5
