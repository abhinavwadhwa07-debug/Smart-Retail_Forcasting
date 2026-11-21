# Smart Retail Intelligence Network

Demand forecasting, expiry/dead-stock risk detection and dynamic pricing for
retail stores — plus a ready-to-run frontend dashboard and a FastAPI backend
stub that serves it.

This is the `Smart-Retail_Forcasting` repo referenced throughout
`docs/Smart_Retail_Neural_Map.md`: the four core modules the demo generator
and the architecture doc both depend on (`src/data_loader.py`,
`src/features.py`, `src/forecasting.py`, `src/pricing_engine.py`) have been
built out, unit-tested and verified end-to-end against the sample data pack.

## Layout

```
src/                  the ML core (see "Modules" below)
tests/                pytest suite — 33 tests, run with `pytest`
frontend/             SmartRetail_Frontend.html (single-file dashboard) + backend_stub.py (FastAPI) + mock_data.json
data/
  csv/                flat CSV exports of mock_data.json, for Excel/BI tools
  generator/          scripts that build the demo data by running it through src/
  mock_data.json       the data the frontend/backend ship with
docs/
  Smart_Retail_Neural_Map.md   full product + architecture spec
  DATA_DICTIONARY.md           every field in mock_data.json explained
  SOURCES.md                   real datasets to replace the sample with
  charts/                      PNGs from the last verified generator run
requirements.txt
```

## Modules

| Module | What it does |
|---|---|
| `src/data_loader.py` | Reads raw POS transactions (UCI Online Retail II by default, `ISO-8859-1`, column-aliased), applies the cleaning rules (drop nulls/blank IDs/non-positive qty or price, floor to day, add `gross_revenue`), then expands active SKUs into a zero-filled SKU x calendar-day grid so a day with no sale is an explicit `0`, not a missing row. |
| `src/features.py` | `RetailFeatureEngineer`: calendar features (day-of-week, payday flag, ...), lag features (`lag_1/7/14/30_days`), rolling features (7d/30d mean/std/min/max + `relative_price_ratio`) — every lag/rolling value is computed one day *shifted* first, so no row ever sees its own or a future day's demand. `transform()` chains all three and drops each SKU's first 30 days so every surviving row has a real history. |
| `src/forecasting.py` | `wape()` (the repo's accuracy metric), `seasonal_naive_forecast()` (the baseline every model must beat), `expanding_window_splits()` (time-respecting CV — training window only grows, test window is always strictly after it), and `DemandForecaster` — LightGBM Tweedie point forecast + per-quantile LightGBM models (P10/P50/P90), degrading gracefully to a `DummyRegressor` if LightGBM isn't installed. |
| `src/pricing_engine.py` | `demand_curve()` — log-log elasticity model `Q = Q0 x (P0/P)^|e|` — and `optimize_price()` — bounded, profit-maximising price search (`scipy.optimize.minimize_scalar`) within a `1.15x unit_cost` floor and a `min(1.2x historical max, 2x reference price)` ceiling, so a clearance markdown can't dip below margin and a price increase can't become surge pricing. `markdown_ladder()` evaluates the fixed 10/25/40% discount rungs from the Action Catalogue against the same floor. |

Every function/class above has full docstrings (parameters, return shape,
the "why", and the specific edge cases it guards against) — read the
modules directly for the details.

## Running the tests

```bash
pip install -r requirements.txt
pytest
```

33 tests cover: cleaning-rule correctness, the zero-fill grid, leakage-safety
of every lag/rolling feature (proven directly — a strictly increasing demand
series would expose any leak), WAPE edge cases, expanding-window CV never
mixing future into training, quantile ordering, and every pricing guardrail
(cost floor, ceiling, inventory cap, the degenerate-inputs case).

## Regenerating the demo data

The sample data pack (`data/mock_data.json`, `data/csv/*.csv`, the charts in
`docs/charts/`) is not hand-written — it's the output of running 180 days of
synthetic SKU-day demand through this repo's own `src/` code:

```bash
cd data/generator
python generate_sales.py ../../out/demo ../..                 # trains through src/features.py + src/pricing_engine.py
python build_mock.py ../../data/mock_data.json ../../out/demo neural_map.json
python export_csv.py ../../data/mock_data.json ../../out/demo ../csv   # flat CSV tables
```

`generate_sales.py` is the single piece of evidence that `src/` actually
works end-to-end: it imports `RetailFeatureEngineer` and
`optimize_price`/`demand_curve` directly ("repo code" in its own comments),
trains a real LightGBM Tweedie + quantile forecaster on the generated
series, and plots the forecast band, SKU health map, expiry-risk value,
spike detection, markdown-clearance curve and peer comparison — all six
charts are in `docs/charts/` from the last run, and the resulting WAPE
(~0.28, beating the ~0.33 seasonal-naive baseline) and P10-P90 coverage
(~0.74) are baked into `data/mock_data.json`'s `summary`.

## Running the dashboard

```bash
cd frontend
pip install fastapi uvicorn
uvicorn backend_stub:app --reload --port 8000
# open http://localhost:8000
```

Or just open `SmartRetail_Frontend.html` directly — it ships with the same
`mock_data.json` embedded inline, so it works with no backend at all. Swap
in a live API later from the app's Settings page; the JSON shapes in
`mock_data.json` double as the API contract (see `backend_stub.py`'s
`TODO`s for where real model output replaces the sample data).

## Where to go next

`docs/Smart_Retail_Neural_Map.md` is the full map of what's built, partly
built, planned or still a gap (SKU health classifier, expiry risk estimator,
spike detector, supplier transparency, peer comparison, the decision/action
engine) — each section carries a ready-to-use prompt seed for building the
next piece against this same `src/` package.
