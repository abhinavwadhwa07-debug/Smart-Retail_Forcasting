# Smart Retail — Sample Data Pack

Synthetic data for one demo store: 14 SKUs x 180 days (15 May to 10 Nov 2026).
Used by SmartRetail_Frontend.html and backend_stub.py.

- `mock_data.json` — everything in the exact shape the frontend API expects
- `csv/` — the same data as flat tables (open in Excel)
- `generator/` — scripts that rebuild it:
    python generator/generate_sales.py out/demo <path-to-Smart-Retail_Forcasting-repo>
    python generator/build_mock.py mock_data.json out/demo generator/neural_map.json
- `DATA_DICTIONARY.md` — every field explained
- `SOURCES.md` — real datasets to replace this sample, and references

This is NOT real store data. Use it to build and test; replace it before drawing any business conclusion.
