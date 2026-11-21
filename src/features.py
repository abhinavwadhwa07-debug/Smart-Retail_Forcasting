"""Feature Engineering — :class:`RetailFeatureEngineer`.

Turns a per-SKU-per-day demand table into the leakage-safe signals the
forecaster in :mod:`src.forecasting` trains on: calendar rhythm, demand
momentum (lags, rolling stats) and price perception. Every rolling/lag
computation is shifted by at least one day before the window is taken, so no
row ever sees its own or a future day's demand — the "closed='left'" rule
from the Neural Map's Feature Engineering spec.

Typical use (mirrors how ``data/generator/generate_sales.py`` drives this
class)::

    fe = RetailFeatureEngineer("daily_demand_clean.csv")
    fe.load_data()
    temporal = fe.create_temporal_features()
    lags = fe.create_lag_features()
    rolling = fe.create_rolling_features()
    k = ["item_id", "invoice_date"]
    feat = (
        temporal
        .merge(lags[k + [c for c in lags.columns if c.startswith("lag_")]], on=k)
        .merge(rolling[k + [c for c in rolling.columns if c.startswith("rolling_") or c == "relative_price_ratio"]], on=k)
    )

or, for the same result in one call::

    feat = RetailFeatureEngineer("daily_demand_clean.csv").transform()
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

__all__ = ["RetailFeatureEngineer"]

#: Lag horizons (in days) computed by :meth:`RetailFeatureEngineer.create_lag_features`.
LAG_DAYS: Final[tuple[int, ...]] = (1, 7, 14, 30)
#: Rolling-window widths (in days) computed by :meth:`RetailFeatureEngineer.create_rolling_features`.
ROLLING_WINDOWS: Final[tuple[int, ...]] = (7, 30)
#: Rows per item dropped by :meth:`RetailFeatureEngineer.transform` so every
#: remaining row has a fully-populated 30-day lag/rolling history.
WARMUP_DAYS: Final[int] = 30


class RetailFeatureEngineer:
    """Builds calendar, lag, rolling and price-perception features per SKU-day.

    Parameters
    ----------
    path:
        CSV with at least ``item_id``, ``invoice_date``, ``quantity`` and
        ``unit_price`` columns — the shape produced by
        :func:`src.data_loader.expand_daily_sales_grid`. The grid should
        already be zero-filled (no missing SKU-days): lags and rolling
        windows are computed by row position within each item's time series,
        so a gap in the calendar would silently shrink the effective window.

    Attributes
    ----------
    df:
        The loaded, sorted source table. Populated by :meth:`load_data`;
        ``None`` until then.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load_data(self) -> pd.DataFrame:
        """Read the CSV, parse dates and sort by (``item_id``, ``invoice_date``).

        Returns
        -------
        pandas.DataFrame
            The loaded table (also stored on ``self.df`` for the
            ``create_*`` methods to use).
        """
        df = pd.read_csv(self.path)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="mixed")
        df = df.sort_values(["item_id", "invoice_date"]).reset_index(drop=True)
        self.df = df
        return df

    def _require_df(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Call load_data() before computing features.")
        return self.df

    # ------------------------------------------------------------------ #
    # Calendar features
    # ------------------------------------------------------------------ #
    def create_temporal_features(self) -> pd.DataFrame:
        """When the item-day happened: calendar rhythm, no leakage risk.

        Returns
        -------
        pandas.DataFrame
            ``item_id``, ``invoice_date``, ``quantity``, ``unit_price`` plus:

            - ``day_of_week`` (0=Monday..6=Sunday)
            - ``day_of_month``
            - ``month``
            - ``quarter``
            - ``is_weekend`` (Saturday/Sunday)
            - ``payday_flag`` (day of month 1-5, when salaries typically land)
        """
        df = self._require_df()
        out = df[["item_id", "invoice_date", "quantity", "unit_price"]].copy()
        dt = out["invoice_date"].dt
        out["day_of_week"] = dt.dayofweek
        out["day_of_month"] = dt.day
        out["month"] = dt.month
        out["quarter"] = dt.quarter
        out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
        out["payday_flag"] = (out["day_of_month"] <= 5).astype(int)
        return out

    # ------------------------------------------------------------------ #
    # Lag features
    # ------------------------------------------------------------------ #
    def create_lag_features(self) -> pd.DataFrame:
        """Demand on earlier days, per item.

        Each ``lag_{n}_days`` column is ``quantity`` shifted ``n`` rows
        within its own ``item_id`` group — strictly past observations only.
        Rows with no history that far back (the first ``n`` rows of each
        item) are filled with ``0``, matching the repo's intermittent-demand
        convention that "no data yet" and "sold nothing" are both ``0`` at
        this stage (the :data:`WARMUP_DAYS` cut in :meth:`transform` removes
        the ambiguous rows for training).

        Returns
        -------
        pandas.DataFrame
            ``item_id``, ``invoice_date`` plus ``lag_1_days``, ``lag_7_days``,
            ``lag_14_days``, ``lag_30_days``.
        """
        df = self._require_df()
        out = df[["item_id", "invoice_date"]].copy()
        grouped = df.groupby("item_id")["quantity"]
        for n in LAG_DAYS:
            out[f"lag_{n}_days"] = grouped.shift(n).fillna(0)
        return out

    # ------------------------------------------------------------------ #
    # Rolling statistics + price perception
    # ------------------------------------------------------------------ #
    def create_rolling_features(self) -> pd.DataFrame:
        """Recent level, volatility and price perception, leakage-safe.

        Every statistic is computed on ``quantity`` (or ``unit_price``)
        shifted one day first, i.e. an equivalent of pandas'
        ``rolling(window, closed='left')``: the window covers the ``window``
        days strictly *before* the current row, so no row's own demand or
        price ever contributes to its own features.

        Returns
        -------
        pandas.DataFrame
            ``item_id``, ``invoice_date`` plus, for each window in
            :data:`ROLLING_WINDOWS` (7 and 30 days): ``rolling_{w}_mean``,
            ``rolling_{w}_std``, ``rolling_{w}_min``, ``rolling_{w}_max``.
            Also:

            - ``rolling_30d_avg_unit_price`` — mean shelf price over the
              prior 30 days.
            - ``relative_price_ratio`` — ``unit_price / rolling_30d_avg_unit_price``:
              above 1 means today's price is high versus the SKU's recent
              normal; below 1 means it's discounted. ``1.0`` where there is
              no prior price history yet.
        """
        df = self._require_df()
        out = df[["item_id", "invoice_date"]].copy()
        grp_id = df["item_id"]

        shifted_qty = df.groupby("item_id")["quantity"].shift(1)
        for window in ROLLING_WINDOWS:
            roll = shifted_qty.groupby(grp_id).rolling(window, min_periods=1)
            out[f"rolling_{window}_mean"] = roll.mean().reset_index(level=0, drop=True)
            out[f"rolling_{window}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
            out[f"rolling_{window}_min"] = roll.min().reset_index(level=0, drop=True)
            out[f"rolling_{window}_max"] = roll.max().reset_index(level=0, drop=True)

        shifted_price = df.groupby("item_id")["unit_price"].shift(1)
        avg_price = shifted_price.groupby(grp_id).rolling(30, min_periods=1).mean().reset_index(level=0, drop=True)
        out["rolling_30d_avg_unit_price"] = avg_price
        ratio = df["unit_price"] / avg_price.replace(0, np.nan)
        out["relative_price_ratio"] = ratio.fillna(1.0).astype(float)

        # First row per item has no shifted history at all: rolling_* NaN -> 0
        # (zero prior demand/level is the right neutral value; relative_price_ratio
        # is handled separately above because 0 would read as "free").
        rolling_cols = [c for c in out.columns if c.startswith("rolling_")]
        out[rolling_cols] = out[rolling_cols].fillna(0.0)
        return out

    # ------------------------------------------------------------------ #
    # One-call pipeline
    # ------------------------------------------------------------------ #
    def transform(self) -> pd.DataFrame:
        """Full feature matrix: calendar + lag + rolling, warm-up rows dropped.

        Calls :meth:`load_data` if it hasn't been called yet, computes all
        three feature groups, merges them on (``item_id``, ``invoice_date``),
        drops each item's first :data:`WARMUP_DAYS` rows (so every remaining
        row has a true 30-day lag/rolling history instead of zero-filled
        placeholders), and adds ``item_code`` — a stable integer encoding of
        ``item_id`` for models that want a numeric SKU identity feature.

        Returns
        -------
        pandas.DataFrame
            One row per (SKU, day) with ``quantity`` as the prediction
            target and every other non-identifier column as a model feature.
        """
        if self.df is None:
            self.load_data()
        key = ["item_id", "invoice_date"]
        temporal = self.create_temporal_features()
        lags = self.create_lag_features()
        rolling = self.create_rolling_features()

        feat = temporal.merge(
            lags[key + [c for c in lags.columns if c.startswith("lag_")]], on=key
        ).merge(
            rolling[key + [c for c in rolling.columns if c.startswith("rolling_") or c == "relative_price_ratio"]],
            on=key,
        )
        feat = feat[feat.groupby("item_id").cumcount() >= WARMUP_DAYS].reset_index(drop=True)
        feat["item_code"] = feat["item_id"].astype("category").cat.codes
        return feat

    # ------------------------------------------------------------------ #
    # Extension point (Neural Map: "Extend features.py ... intermittency")
    # ------------------------------------------------------------------ #
    def compute_intermittency_profile(self) -> pd.DataFrame:
        """Per-SKU demand-pattern profile (Syntetos-Boylan classification).

        Not merged into :meth:`transform`'s per-row matrix — this is a
        per-item diagnostic, used by the SKU health classifier and ordering
        policy to decide *how* to forecast a SKU (e.g. Croston/TSB for
        lumpy demand) rather than a row-level model feature.

        Returns
        -------
        pandas.DataFrame
            One row per ``item_id`` with:

            - ``adi`` — Average Demand Interval: mean number of days between
              non-zero-demand days. Low ADI means demand is frequent.
            - ``cv2`` — squared coefficient of variation of the *non-zero*
              demand sizes. Low CV² means sale sizes are stable.
            - ``demand_class`` — ``smooth`` (ADI<1.32, CV2<0.49),
              ``intermittent`` (ADI>=1.32, CV2<0.49), ``erratic``
              (ADI<1.32, CV2>=0.49) or ``lumpy`` (ADI>=1.32, CV2>=0.49),
              the standard Syntetos-Boylan thresholds.
        """
        df = self._require_df()
        rows = []
        for item_id, g in df.sort_values("invoice_date").groupby("item_id"):
            nonzero = g.loc[g["quantity"] > 0, "quantity"]
            n_periods = len(g)
            n_sales = len(nonzero)
            adi = n_periods / n_sales if n_sales > 0 else np.inf
            cv2 = float((nonzero.std(ddof=0) / nonzero.mean()) ** 2) if n_sales > 1 and nonzero.mean() > 0 else 0.0
            if not np.isfinite(adi):
                demand_class = "dead"
            elif adi < 1.32 and cv2 < 0.49:
                demand_class = "smooth"
            elif adi >= 1.32 and cv2 < 0.49:
                demand_class = "intermittent"
            elif adi < 1.32 and cv2 >= 0.49:
                demand_class = "erratic"
            else:
                demand_class = "lumpy"
            rows.append({"item_id": item_id, "adi": adi, "cv2": cv2, "demand_class": demand_class})
        return pd.DataFrame(rows)
