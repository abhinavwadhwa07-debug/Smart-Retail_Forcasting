"""Usage: python export_csv.py <mock_data.json> <generate_sales out_dir> <csv_out_dir>

Flattens mock_data.json (plus the generator's full 180-day sales.csv and
28-day forecast.csv) into the same 8 "open in Excel" CSV tables shipped in
the Sample Data Pack, so the flat exports always match whatever the JSON
behind the live frontend demo currently says.
"""
import json
import sys
from pathlib import Path

import pandas as pd

mock_path, gen_dir, csv_out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
csv_out.mkdir(parents=True, exist_ok=True)
M = json.loads(mock_path.read_text())
name_of = {s["id"]: s["name"] for s in M["skus"]}

# skus.csv — product master flattened with its health snapshot
skus = pd.DataFrame(
    [
        dict(
            id=s["id"], name=s["name"], category=s["category"], shelf_life_days=s["shelf_life_days"],
            unit_cost=s["unit_cost"], price=s["price"], elasticity=s["elasticity"], supplier_id=s["supplier_id"],
            **{f"health_{k}": v for k, v in s["health"].items()},
        )
        for s in M["skus"]
    ]
)
skus.to_csv(csv_out / "skus.csv", index=False)

# batches.csv
pd.DataFrame(M["batches"]).to_csv(csv_out / "batches.csv", index=False)

# spikes.csv
pd.DataFrame(M["spikes"]).to_csv(csv_out / "spikes.csv", index=False)

# suppliers.csv (top-level fields only; open_pos goes to its own table below)
suppliers = pd.DataFrame(
    [
        {k: (", ".join(v) if isinstance(v, list) else v) for k, v in s.items() if k != "open_pos"}
        for s in M["suppliers"]
    ]
)
suppliers.to_csv(csv_out / "suppliers.csv", index=False)

# open_purchase_orders.csv — supplier_id re-attached per PO
pos = []
for s in M["suppliers"]:
    for po in s["open_pos"]:
        pos.append({**po, "supplier_id": s["id"]})
pd.DataFrame(pos, columns=["id", "sku_id", "qty", "eta", "status", "supplier_id"]).to_csv(
    csv_out / "open_purchase_orders.csv", index=False
)

# tasks.csv — params as a JSON string, matching the shipped format
tasks = pd.DataFrame(M["tasks"]).copy()
tasks["params"] = tasks["params"].apply(json.dumps)
tasks = tasks[["id", "type", "sku_id", "params", "impact_rs", "minutes", "urgency", "role", "status", "priority"]]
tasks.to_csv(csv_out / "tasks.csv", index=False)

# sales_daily.csv — the full 180-day history straight from the generator
sales = pd.read_csv(gen_dir / "sales.csv")
sales = sales.rename(columns={"item_id": "sku_name"})
sales.insert(0, "sku_id", sales["sku_name"].map({v: k for k, v in name_of.items()}))
sales[["sku_id", "sku_name", "invoice_date", "quantity", "unit_price"]].rename(
    columns={"invoice_date": "date"}
).to_csv(csv_out / "sales_daily.csv", index=False)

# forecast_holdout.csv — 28-day holdout: actual + P10/P50/P90
fc = pd.read_csv(gen_dir / "forecast.csv")
fc = fc.rename(columns={"item_id": "sku_name", "invoice_date": "date"})
fc.insert(0, "sku_id", fc["sku_name"].map({v: k for k, v in name_of.items()}))
fc[["sku_id", "sku_name", "date", "quantity", "p10", "p50", "p90"]].rename(columns={"quantity": "actual"}).to_csv(
    csv_out / "forecast_holdout.csv", index=False
)

print("Exported 8 CSVs to", csv_out)
