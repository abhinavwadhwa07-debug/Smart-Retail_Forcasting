"""Data Layer — ingestion, cleaning and the zero-filled daily sales grid.

Implements the three ``[Built]`` pieces of the Data Layer described in
``docs/Smart_Retail_Neural_Map.md``:

1. **POS transactions** — :func:`load_raw_transactions` reads line-level POS
   data (the UCI Online Retail II dataset by default) with the encoding it
   actually ships in, and aliases vendor column names onto the repo's
   canonical schema (``item_id``, ``quantity``, ``unit_price``,
   ``invoice_date``).
2. **Cleaning rules** — :func:`clean_transactions` drops rows that cannot be
   real demand (nulls, blank IDs, non-positive quantity or price), floors
   timestamps to the day, and adds ``gross_revenue``.
3. **Zero-filled daily grid** — :func:`expand_daily_sales_grid` takes the
   Cartesian product of active SKUs x calendar days so a day with no sale is
   an explicit zero, not a missing row. This is what makes lag/rolling
   features and intermittency statistics meaningful downstream in
   :mod:`src.features`.

Every function is a pure transform (input DataFrame -> output DataFrame) so
each step can be unit tested and composed on its own; :func:`load_and_prepare`
chains all three for the common case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "DataQualityError",
    "CANONICAL_COLUMNS",
    "load_raw_transactions",
    "clean_transactions",
    "expand_daily_sales_grid",
    "load_and_prepare",
]

# Vendor column name -> canonical repo column name. Matched case-insensitively
# after stripping whitespace, so "Invoice Date", "invoicedate" etc. all hit.
_COLUMN_ALIASES: Final[dict[str, str]] = {
    "stockcode": "item_id",
    "itemid": "item_id",
    "item_id": "item_id",
    "productid": "item_id",
    "sku": "item_id",
    "description": "item_name",
    "quantity": "quantity",
    "qty": "quantity",
    "unitprice": "unit_price",
    "price": "unit_price",
    "unit_price": "unit_price",
    "invoicedate": "invoice_date",
    "date": "invoice_date",
    "invoice_date": "invoice_date",
    "invoice": "invoice_no",
    "invoiceno": "invoice_no",
    "customerid": "customer_id",
    "country": "country",
}

CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "item_id",
    "invoice_date",
    "quantity",
    "unit_price",
)


class DataQualityError(ValueError):
    """Raised when a transactions table is missing a required column.

    Fails loudly rather than silently imputing, per the Data Layer's quality
    policy ("fail loudly, never silently impute").
    """


def _alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "")
        if key in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[key]
    return df.rename(columns=rename)


def load_raw_transactions(path: str | Path, encoding: str = "ISO-8859-1", **read_csv_kwargs: object) -> pd.DataFrame:
    """Read a line-level POS transactions CSV and alias it to the canonical schema.

    Defaults are tuned for the UCI Online Retail II dataset (the repo's
    reference source — see ``docs/SOURCES.md``), which ships as
    Latin-1 / ``ISO-8859-1`` encoded CSV with columns such as ``StockCode``,
    ``Quantity``, ``UnitPrice`` and ``InvoiceDate``. Any file whose columns
    alias onto ``item_id`` / ``quantity`` / ``unit_price`` / ``invoice_date``
    (see :data:`_COLUMN_ALIASES`) will load the same way.

    Parameters
    ----------
    path:
        Path to the raw transactions CSV.
    encoding:
        Text encoding to read the file with. UCI Online Retail II requires
        ``ISO-8859-1``; override for a UTF-8 export.
    **read_csv_kwargs:
        Forwarded to :func:`pandas.read_csv` (e.g. ``nrows`` for a quick
        sample read).

    Returns
    -------
    pandas.DataFrame
        Raw rows with canonical column names. No cleaning is applied yet —
        call :func:`clean_transactions` next.

    Raises
    ------
    DataQualityError
        If, after aliasing, any of ``item_id``, ``quantity``, ``unit_price``,
        ``invoice_date`` is still missing.
    """
    df = pd.read_csv(path, encoding=encoding, **read_csv_kwargs)
    df = _alias_columns(df)
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(
            f"Transactions file {path!r} is missing required column(s) {missing} "
            f"after alias resolution. Columns present: {list(df.columns)}"
        )
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows so downstream models see real demand only.

    Rules (per the Data Layer's cleaning spec):

    - Drop rows with a null ``item_id``, ``quantity``, ``unit_price`` or
      ``invoice_date``.
    - Drop rows with a blank/whitespace-only ``item_id``.
    - Drop rows with ``quantity <= 0`` or ``unit_price <= 0`` (returns,
      cancellations and free-of-charge lines are excluded here; the planned
      improvement is to route them to a separate returns table instead of
      discarding them outright).
    - Floor ``invoice_date`` to the day (line-level timestamps collapse to a
      single demand observation per SKU-day).
    - Add ``gross_revenue = quantity * unit_price``.

    Parameters
    ----------
    df:
        Output of :func:`load_raw_transactions` (or anything already on the
        canonical schema).

    Returns
    -------
    pandas.DataFrame
        Cleaned transactions with an added ``gross_revenue`` column, index
        reset.
    """
    out = df.copy()
    out["item_id"] = out["item_id"].astype("string")
    # format="mixed" parses each value independently instead of inferring one
    # format from the first row — real POS exports routinely mix
    # "YYYY-MM-DD" and "YYYY-MM-DD HH:MM" timestamps in the same column, and
    # a single inferred format would silently turn the other rows into NaT.
    out["invoice_date"] = pd.to_datetime(out["invoice_date"], errors="coerce", format="mixed")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce")

    required = ["item_id", "quantity", "unit_price", "invoice_date"]
    out = out.dropna(subset=required)
    out = out[out["item_id"].str.strip().str.len() > 0]
    out = out[(out["quantity"] > 0) & (out["unit_price"] > 0)]

    out["invoice_date"] = out["invoice_date"].dt.floor("D")
    out["gross_revenue"] = out["quantity"] * out["unit_price"]

    return out.reset_index(drop=True)


def expand_daily_sales_grid(df: pd.DataFrame, min_sales: int = 50) -> pd.DataFrame:
    """Build the zero-filled SKU x calendar-day grid.

    Cleaned transactions are line-level and sparse: a SKU that didn't sell on
    a given day simply has no row for that day. That is ambiguous with "we
    have no data for that day" and silently breaks lags, rolling windows and
    intermittency statistics. This expands every *active* SKU (total
    lifetime quantity >= ``min_sales``, to drop one-off/noise codes) across
    the full observed calendar range, so every SKU-day combination exists
    explicitly with ``quantity = 0`` where nothing sold.

    Parameters
    ----------
    df:
        Cleaned, day-floored transactions (see :func:`clean_transactions`),
        at minimum with ``item_id``, ``invoice_date``, ``quantity`` and
        ``unit_price``.
    min_sales:
        Minimum lifetime quantity for a SKU to be considered "active" and
        included in the expanded grid. SKUs below this threshold are too
        sparse to forecast meaningfully and are dropped.

    Returns
    -------
    pandas.DataFrame
        One row per (``item_id``, ``invoice_date``) for every active SKU
        across the full date range, sorted by ``item_id`` then
        ``invoice_date``, with:

        - ``quantity`` — actual quantity sold, ``0`` on non-sale days.
        - ``unit_price`` — the SKU's observed price, forward-filled then
          back-filled across its own zero-sale days (a SKU's price is
          sticky; it doesn't become unknown just because nothing sold).
        - ``gross_revenue`` — recomputed as ``quantity * unit_price`` so it
          is consistent with the filled price on zero-sale days too.
    """
    daily = (
        df.groupby(["item_id", "invoice_date"], as_index=False)
        .agg(quantity=("quantity", "sum"), unit_price=("unit_price", "mean"))
    )

    totals = daily.groupby("item_id")["quantity"].sum()
    active_items = totals[totals >= min_sales].index
    daily = daily[daily["item_id"].isin(active_items)]

    if daily.empty:
        return daily.assign(gross_revenue=pd.Series(dtype=float))

    full_dates = pd.date_range(daily["invoice_date"].min(), daily["invoice_date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([active_items, full_dates], names=["item_id", "invoice_date"]).to_frame(
        index=False
    )

    merged = grid.merge(daily, on=["item_id", "invoice_date"], how="left")
    merged["quantity"] = merged["quantity"].fillna(0)
    merged["unit_price"] = merged.groupby("item_id")["unit_price"].transform(lambda s: s.ffill().bfill())
    merged["gross_revenue"] = merged["quantity"] * merged["unit_price"]

    merged = merged.sort_values(["item_id", "invoice_date"]).reset_index(drop=True)
    return merged


def load_and_prepare(path: str | Path, encoding: str = "ISO-8859-1", min_sales: int = 50) -> pd.DataFrame:
    """Convenience pipeline: load -> clean -> zero-filled daily grid.

    Equivalent to::

        expand_daily_sales_grid(clean_transactions(load_raw_transactions(path, encoding)), min_sales)
    """
    raw = load_raw_transactions(path, encoding=encoding)
    cleaned = clean_transactions(raw)
    return expand_daily_sales_grid(cleaned, min_sales=min_sales)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    grid = load_and_prepare(sys.argv[1])
    print(grid.head(10).to_string())
    print(f"\n{grid['item_id'].nunique()} active SKUs, {len(grid)} SKU-day rows, "
          f"{grid['invoice_date'].min().date()} to {grid['invoice_date'].max().date()}")
