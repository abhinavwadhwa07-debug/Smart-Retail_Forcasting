"""Tests for src.data_loader: aliasing, cleaning rules, zero-filled grid."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (
    DataQualityError,
    clean_transactions,
    expand_daily_sales_grid,
    load_raw_transactions,
)


def _write(tmp_path, df: pd.DataFrame, name: str = "raw.csv", encoding: str = "ISO-8859-1"):
    path = tmp_path / name
    df.to_csv(path, index=False, encoding=encoding)
    return path


def test_load_raw_transactions_aliases_uci_columns(tmp_path):
    df = pd.DataFrame(
        {
            "StockCode": ["A1"],
            "Description": ["Widget"],
            "Quantity": [3],
            "InvoiceDate": ["2026-01-01"],
            "UnitPrice": [9.5],
            "CustomerID": [101],
            "Country": ["India"],
        }
    )
    path = _write(tmp_path, df)
    out = load_raw_transactions(path)
    assert {"item_id", "quantity", "unit_price", "invoice_date"} <= set(out.columns)
    assert out.loc[0, "item_id"] == "A1"
    assert out.loc[0, "unit_price"] == 9.5


def test_load_raw_transactions_requires_core_columns(tmp_path):
    path = _write(tmp_path, pd.DataFrame({"foo": [1], "bar": [2]}))
    with pytest.raises(DataQualityError):
        load_raw_transactions(path)


def test_clean_transactions_drops_invalid_rows():
    raw = pd.DataFrame(
        {
            "item_id": ["A", "A", "B", None, "  ", "C"],
            "quantity": [5, -1, 0, 3, 3, 4],
            "unit_price": [10.0, 10.0, 10.0, 10.0, 10.0, -5.0],
            "invoice_date": ["2026-01-01"] * 6,
        }
    )
    cleaned = clean_transactions(raw)
    # Only the first row is valid: A/qty5/price10 survives; negative qty,
    # zero qty, null id, blank id and negative price are all dropped.
    assert len(cleaned) == 1
    assert cleaned.loc[0, "item_id"] == "A"


def test_clean_transactions_adds_gross_revenue_and_floors_dates():
    raw = pd.DataFrame(
        {
            "item_id": ["A", "A"],
            "quantity": [2, 3],
            "unit_price": [10.0, 5.0],
            "invoice_date": ["2026-01-01 08:15:00", "2026-01-01 23:59:00"],
        }
    )
    cleaned = clean_transactions(raw)
    assert (cleaned["invoice_date"] == pd.Timestamp("2026-01-01")).all()
    assert list(cleaned["gross_revenue"]) == [20.0, 15.0]


def test_clean_transactions_handles_mixed_date_formats():
    """A single column mixing 'YYYY-MM-DD' and 'YYYY-MM-DD HH:MM' must not
    silently turn either variant into NaT (a real pandas footgun when the
    format is inferred from only the first row)."""
    raw = pd.DataFrame(
        {
            "item_id": ["A", "A", "A"],
            "quantity": [1, 1, 1],
            "unit_price": [10.0, 10.0, 10.0],
            "invoice_date": ["2026-01-01 10:00", "2026-01-02", "2026-01-03 23:00"],
        }
    )
    cleaned = clean_transactions(raw)
    assert len(cleaned) == 3
    assert cleaned["invoice_date"].notna().all()


def test_expand_daily_sales_grid_fills_zero_sale_days_and_keeps_active_skus():
    cleaned = pd.DataFrame(
        {
            "item_id": ["A", "A", "B"],
            "invoice_date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-02"]),
            "quantity": [5.0, 3.0, 2.0],
            "unit_price": [10.0, 10.0, 7.0],
            "gross_revenue": [50.0, 30.0, 14.0],
        }
    )
    grid = expand_daily_sales_grid(cleaned, min_sales=1)
    # Full calendar span (Jan 1-3) x both active SKUs = 6 rows.
    assert len(grid) == 6
    a_gap_day = grid[(grid.item_id == "A") & (grid.invoice_date == "2026-01-02")].iloc[0]
    assert a_gap_day["quantity"] == 0
    # Price is forward/back-filled, not zeroed, on a zero-sale day.
    assert a_gap_day["unit_price"] == 10.0
    assert a_gap_day["gross_revenue"] == 0.0


def test_expand_daily_sales_grid_drops_skus_below_min_sales():
    cleaned = pd.DataFrame(
        {
            "item_id": ["A", "RARE"],
            "invoice_date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "quantity": [100.0, 1.0],
            "unit_price": [10.0, 5.0],
            "gross_revenue": [1000.0, 5.0],
        }
    )
    grid = expand_daily_sales_grid(cleaned, min_sales=50)
    assert set(grid["item_id"]) == {"A"}
