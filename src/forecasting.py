"""ML Core — Demand forecaster.

Predicts units per SKU per day with uncertainty (P10/P50/P90), scored with
WAPE against expanding-window, time-respecting cross-validation. Primary
model is LightGBM with a Tweedie objective (gradient boosting handles the
zero-heavy, non-negative, right-skewed shape of retail demand far better
than a Gaussian-loss model); if LightGBM isn't installed the forecaster
degrades gracefully to a :class:`~sklearn.dummy.DummyRegressor` so the rest
of the pipeline (expiry risk, pricing, the action engine) still runs rather
than hard-failing on a missing optional dependency.

Contents
--------
:func:`wape`
    The repo's primary accuracy metric.
:func:`seasonal_naive_forecast`
    The benchmark every model must beat ("same weekday last week").
:func:`expanding_window_splits`
    Time-respecting cross-validation folds: training windows only grow,
    test windows are always strictly after their training window.
:class:`DemandForecaster`
    Fit once, get a point forecast (:meth:`~DemandForecaster.predict`) or
    quantile forecasts (:meth:`~DemandForecaster.predict_quantiles`).
:func:`expanding_window_cv`
    Runs :class:`DemandForecaster` through every fold from
    :func:`expanding_window_splits` and reports WAPE (model vs. naive
    baseline) and P10-P90 interval coverage per fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - exercised only without the optional dep
    _HAS_LIGHTGBM = False

__all__ = [
    "wape",
    "seasonal_naive_forecast",
    "expanding_window_splits",
    "DemandForecaster",
    "expanding_window_cv",
]


def wape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Weighted Absolute Percentage Error: ``sum(|actual - forecast|) / sum(|actual|)``.

    The repo's primary accuracy metric. Unlike MAPE it stays well-defined
    and stable for zero-heavy intermittent retail demand (no division by a
    per-row actual that can be zero); unlike plain MAE it's scale-free, so a
    high-volume SKU and a low-volume SKU can be compared or averaged
    meaningfully.

    Parameters
    ----------
    y_true, y_pred:
        Actual and forecast quantities, same length.

    Returns
    -------
    float
        ``0.0`` is a perfect forecast; ``1.0`` means the forecast errs by as
        much, in total, as all the actual demand itself. ``0.0`` is also
        returned (rather than dividing by zero) in the degenerate case where
        actual demand was zero throughout *and* the forecast was too.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom <= 0:
        return 0.0 if np.allclose(y_pred, 0) else float(np.abs(y_pred).sum())
    return float(np.abs(y_true - y_pred).sum() / denom)


def seasonal_naive_forecast(
    df: pd.DataFrame,
    date_col: str = "invoice_date",
    item_col: str = "item_id",
    target_col: str = "quantity",
    season_days: int = 7,
) -> pd.Series:
    """Benchmark baseline: predict the same weekday ``season_days`` ago, per item.

    Every model in the repo must beat this (and the plainer moving-average
    baseline) on held-out WAPE before it's considered worth its complexity.

    Parameters
    ----------
    df:
        Must contain ``item_col``, ``date_col`` and ``target_col``. Rows do
        not need to be pre-sorted.
    season_days:
        Lag, in days, defining "last season" — 7 reproduces last week's same
        weekday; use 1 for a pure persistence baseline.

    Returns
    -------
    pandas.Series
        Forecast aligned to ``df``'s original index (``NaN`` for a row
        whose item has fewer than ``season_days`` days of prior history —
        the caller decides how to treat that, e.g. ``.fillna(0)``).
    """
    order = df[[item_col, date_col]].copy()
    order["_pos"] = np.arange(len(df))
    sorted_df = df.loc[order.sort_values([item_col, date_col])["_pos"]].copy()
    sorted_df["forecast"] = sorted_df.groupby(item_col)[target_col].shift(season_days)
    return sorted_df["forecast"].reindex(df.index)


def expanding_window_splits(
    dates: pd.Series,
    n_splits: int = 3,
    test_size_days: int = 28,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_mask, test_mask)`` boolean arrays for expanding-window CV.

    Walks backward from the end of the calendar, carving out ``n_splits``
    consecutive ``test_size_days``-day test windows. Each fold's training
    window is *every* day strictly before its test window starts — so the
    training window only ever grows between folds and a model is always
    evaluated purely on dates it could not have seen, which is what makes
    the resulting WAPE a fair estimate of live performance (ordinary k-fold
    CV would leak the future into training and overstate accuracy).

    Parameters
    ----------
    dates:
        The date column of the feature table being split (one entry per
        row; duplicates across SKUs on the same day are fine).
    n_splits:
        Number of folds.
    test_size_days:
        Width of each fold's test window, in calendar days.

    Yields
    ------
    tuple[numpy.ndarray, numpy.ndarray]
        Boolean masks the same length as ``dates``, aligned to its index:
        ``train_mask`` for dates before the fold's cutoff, ``test_mask`` for
        the fold's test window. Folds with an empty train or test window
        (not enough history yet) are skipped.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    unique_days = np.sort(dates.unique())
    if len(unique_days) <= test_size_days:
        raise ValueError(
            f"Only {len(unique_days)} distinct days available; need more than "
            f"test_size_days={test_size_days} for even one fold."
        )
    dates_arr = dates.to_numpy()
    for split in range(n_splits, 0, -1):
        cutoff_idx = len(unique_days) - split * test_size_days
        if cutoff_idx <= 0:
            continue
        cutoff = unique_days[cutoff_idx]
        test_end_idx = min(cutoff_idx + test_size_days, len(unique_days))
        test_end = unique_days[test_end_idx - 1]
        train_mask = dates_arr < cutoff
        test_mask = (dates_arr >= cutoff) & (dates_arr <= test_end)
        if train_mask.any() and test_mask.any():
            yield train_mask, test_mask


@dataclass
class DemandForecaster:
    """LightGBM Tweedie point forecaster + per-quantile LightGBM models.

    Falls back to a mean-predicting :class:`~sklearn.dummy.DummyRegressor`
    (point forecast) and unconditional empirical quantiles of the training
    target (quantile forecasts) if LightGBM is not installed, so callers
    never need to special-case a missing optional dependency.

    Parameters
    ----------
    quantiles:
        Quantile levels to forecast in addition to the point (P50) forecast.
        ``0.5`` is always implicitly included via :meth:`predict`.
    tweedie_variance_power:
        LightGBM's ``tweedie_variance_power``, 1-2. Closer to 1 behaves more
        like Poisson (good for very sparse/intermittent demand); closer to 2
        more like Gamma (good for smoother, higher-volume demand). 1.2 is a
        reasonable default for mixed retail catalogues.
    n_estimators, learning_rate, num_leaves, min_child_samples, random_state:
        Forwarded to the underlying ``LGBMRegressor``.

    Attributes
    ----------
    feature_names_:
        Column names the model was fit on, set by :meth:`fit`.
    uses_lightgbm_:
        Whether LightGBM is actually backing this instance (``False`` means
        the DummyRegressor fallback is active).
    """

    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    tweedie_variance_power: float = 1.2
    n_estimators: int = 400
    learning_rate: float = 0.04
    num_leaves: int = 31
    min_child_samples: int = 10
    random_state: int = 42

    feature_names_: list[str] = field(default_factory=list, init=False, repr=False)
    uses_lightgbm_: bool = field(default=False, init=False, repr=False)
    _point_model: object = field(default=None, init=False, repr=False)
    _quantile_models: dict[float, object] = field(default_factory=dict, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "DemandForecaster":
        """Fit the point model and every non-0.5 quantile model.

        Parameters
        ----------
        X:
            Feature matrix, e.g. the ``X_cols`` subset of
            :meth:`src.features.RetailFeatureEngineer.transform`'s output.
        y:
            Target quantities, same length as ``X``.

        Returns
        -------
        DemandForecaster
            ``self``, so ``.fit(...).predict(...)`` chains.
        """
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else []
        self.uses_lightgbm_ = _HAS_LIGHTGBM
        side_quantiles = [q for q in self.quantiles if q != 0.5]

        if _HAS_LIGHTGBM:
            self._point_model = lgb.LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=self.tweedie_variance_power,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbose=-1,
            ).fit(X, y)
            self._quantile_models = {
                q: lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves,
                    min_child_samples=self.min_child_samples,
                    random_state=self.random_state,
                    verbose=-1,
                ).fit(X, y)
                for q in side_quantiles
            }
        else:  # pragma: no cover - exercised only without the optional dep
            self._point_model = DummyRegressor(strategy="mean").fit(X, y)
            self._quantile_models = {q: float(np.quantile(y, q)) for q in side_quantiles}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point (P50-equivalent) forecast, clipped at zero (demand can't be negative)."""
        if self._point_model is None:
            raise RuntimeError("Call fit() before predict().")
        return np.clip(np.asarray(self._point_model.predict(X), dtype=float), 0, None)

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """Quantile forecasts, columns ordered ``p10, p50, p90, ...``.

        The non-0.5 quantiles are clipped so the band never crosses the
        point forecast (``p10 <= p50 <= p90``), which LightGBM's
        independently-trained quantile models don't otherwise guarantee.

        Parameters
        ----------
        X:
            Same feature columns :meth:`fit` was called with.

        Returns
        -------
        pandas.DataFrame
            One ``p{quantile*100}`` column per entry in ``self.quantiles``,
            indexed like ``X``.
        """
        if self._point_model is None:
            raise RuntimeError("Call fit() before predict_quantiles().")
        index = X.index if hasattr(X, "index") else pd.RangeIndex(len(X))
        p50 = self.predict(X)
        cols: dict[str, np.ndarray] = {"p50": p50}
        for q in self.quantiles:
            if q == 0.5:
                continue
            label = f"p{round(q * 100):d}"
            if self.uses_lightgbm_:
                raw = np.clip(np.asarray(self._quantile_models[q].predict(X), dtype=float), 0, None)
            else:  # pragma: no cover - exercised only without the optional dep
                raw = np.full(len(X), self._quantile_models[q])
            cols[label] = np.minimum(raw, p50) if q < 0.5 else np.maximum(raw, p50)
        out = pd.DataFrame(cols, index=index)
        ordered = sorted(out.columns, key=lambda c: int(c[1:]))
        return out[ordered]


def expanding_window_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "quantity",
    date_col: str = "invoice_date",
    n_splits: int = 3,
    test_size_days: int = 28,
    forecaster_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Expanding-window CV: fit :class:`DemandForecaster` per fold, score WAPE.

    This is the repo's model-acceptance test — ``wape_model`` must beat
    ``wape_naive`` (the seasonal-naive baseline) before a model change ships.

    Parameters
    ----------
    df:
        Feature table, e.g. the output of
        :meth:`src.features.RetailFeatureEngineer.transform`.
    feature_cols:
        Columns of ``df`` to use as model inputs.
    n_splits, test_size_days:
        Forwarded to :func:`expanding_window_splits`.
    forecaster_kwargs:
        Extra keyword arguments for :class:`DemandForecaster` (e.g. a
        non-default ``tweedie_variance_power``).

    Returns
    -------
    pandas.DataFrame
        One row per fold: ``fold``, ``train_rows``, ``test_rows``,
        ``wape_model``, ``wape_naive``, ``coverage_p10_p90``.
    """
    forecaster_kwargs = forecaster_kwargs or {}
    naive = seasonal_naive_forecast(df, date_col=date_col, target_col=target_col).fillna(0)
    rows = []
    for fold, (train_mask, test_mask) in enumerate(
        expanding_window_splits(df[date_col], n_splits=n_splits, test_size_days=test_size_days), start=1
    ):
        train, test = df.loc[train_mask], df.loc[test_mask]
        model = DemandForecaster(**forecaster_kwargs).fit(train[feature_cols], train[target_col])
        q = model.predict_quantiles(test[feature_cols])
        lo_col = min((c for c in q.columns if c != "p50" and int(c[1:]) < 50), key=lambda c: int(c[1:]), default=None)
        hi_col = max((c for c in q.columns if c != "p50" and int(c[1:]) > 50), key=lambda c: int(c[1:]), default=None)
        coverage = (
            float(
                (
                    (test[target_col].to_numpy() >= np.floor(q[lo_col].to_numpy()))
                    & (test[target_col].to_numpy() <= np.ceil(q[hi_col].to_numpy()))
                ).mean()
            )
            if lo_col and hi_col
            else float("nan")
        )
        rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "wape_model": wape(test[target_col], q["p50"]),
                "wape_naive": wape(test[target_col], naive.loc[test_mask]),
                "coverage_p10_p90": coverage,
            }
        )
    return pd.DataFrame(rows)
