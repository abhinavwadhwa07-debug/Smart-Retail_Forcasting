"""Usage: python build_mock.py <mock_data.json out> <generate_sales out_dir> <neural_map.json>
Builds mock_data.json for the frontend (same shapes the backend API must return)."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import poisson

D = Path(sys.argv[2])  # output folder of generate_sales.py
NUM = json.load(open(D / "demo_numbers.json")); MAP = json.load(open(sys.argv[3]))  # neural_map.json
sales = pd.read_csv(D / "sales.csv", parse_dates=["invoice_date"])
fc = pd.read_csv(D / "forecast.csv", parse_dates=["invoice_date"])
master = pd.read_csv(D / "master.csv"); onhand = json.load(open(D / "onhand.json"))
FEST = pd.Timestamp("2026-11-05")
slug = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
H = {h["item"]: h for h in NUM["health"]}
SUP = {"Dairy": "sup-gokul", "Bakery": "sup-sunrise", "Snacks": "sup-crunch", "Sweets": "sup-mithas",
       "Staples": "sup-annapurna", "Flour": "sup-annapurna", "Beverages": "sup-leafline", "Frozen": "sup-coldchain"}
today = sales.invoice_date.max()

skus, series, batches, spikes = [], {}, [], []
for _, m in master.iterrows():
    sid = slug(m.item_id); h = H[m.item_id]
    score = {"Healthy": 85, "Watch": 58, "Expiry risk": 42, "Shelf gap": 35, "Dead": 12}[h["cls"]]
    score = int(np.clip(score + (h["trend"] - 1) * 10 - min(h["cover_weeks"], 20) * .3, 3, 98))
    skus.append(dict(id=sid, name=m.item_id, category=m.category, shelf_life_days=int(m.shelf_life), unit_cost=float(m.unit_cost),
                     price=float(m.price), elasticity=float(m.elasticity), supplier_id=SUP[m.category],
                     health=dict(cls=h["cls"], score=score, cover_weeks=h["cover_weeks"], trend=h["trend"], days_since_sale=h["days_since_sale"],
                                 on_hand=h["on_hand"], daily_rate=h["daily_rate"], at_risk_units=h["at_risk_units"],
                                 at_risk_value=h["at_risk_value"], p_expire=h["p_expire"], reason=h["reason"])))
    s = sales[sales.item_id == m.item_id].set_index("invoice_date").quantity
    f = fc[fc.item_id == m.item_id].set_index("invoice_date")
    start = today - pd.Timedelta(days=89)
    dates = pd.date_range(start, today)
    series[sid] = dict(dates=[d.strftime("%Y-%m-%d") for d in dates], actual=[int(s.get(d, 0)) for d in dates],
                       p10=[round(float(f.p10[d]), 2) if d in f.index else None for d in dates],
                       p50=[round(float(f.p50[d]), 2) if d in f.index else None for d in dates],
                       p90=[round(float(f.p90[d]), 2) if d in f.index else None for d in dates],
                       forecast_start=f.index.min().strftime("%Y-%m-%d"))
    # batches FEFO
    mu = max(h["daily_rate"], .05); before = 0
    for bi, (q, life) in enumerate(sorted(onhand[m.item_id], key=lambda b: b[1])):
        lam = mu * life; ks = np.arange(0, int(lam * 3 + q + before + 30))
        left = float((poisson.pmf(ks, lam) * np.clip(before + q - ks, 0, q)).sum())
        batches.append(dict(id=f"{sid}-b{bi+1}", sku_id=sid, qty=q, days_left=life,
                            expiry_date=(today + pd.Timedelta(days=life)).strftime("%Y-%m-%d"),
                            expected_unsold=round(left, 1), value_at_risk=round(left * m.unit_cost), p_expire=round(float(poisson.cdf(before + q - 1, lam)), 2)))
        before += q
    # spikes, last 60 days
    base = s.shift(1).rolling(28, min_periods=7).median()
    mad = (s.shift(1) - base).abs().rolling(28, min_periods=7).median().replace(0, 1)
    z = (s - base) / (1.4826 * mad)
    for d, zz in z[(z.index > today - pd.Timedelta(days=60)) & (z > 3.5)].items():
        kind = "festival" if abs((d - FEST).days) <= 10 else "one-off"
        spikes.append(dict(sku_id=sid, date=d.strftime("%Y-%m-%d"), units=int(s[d]), baseline=round(float(base[d]), 1), z=round(float(zz), 1), type=kind))

suppliers = [
 dict(id="sup-gokul", name="Gokul Dairy Co-op", categories=["Dairy"], otif=.91, fill_rate=.96, lead_time_mean=1.1, lead_time_sd=.3, open_pos=[dict(id="PO-2291", sku_id="toned-milk-500ml", qty=120, eta="2026-11-11", status="on time")]),
 dict(id="sup-sunrise", name="Sunrise Bakers", categories=["Bakery"], otif=.84, fill_rate=.93, lead_time_mean=1.4, lead_time_sd=.6, open_pos=[dict(id="PO-2288", sku_id="brown-bread-400g", qty=24, eta="2026-11-11", status="on time")]),
 dict(id="sup-crunch", name="Crunch Foods Distributors", categories=["Snacks"], otif=.88, fill_rate=.95, lead_time_mean=3.2, lead_time_sd=.9, open_pos=[]),
 dict(id="sup-mithas", name="Mithas Sweets", categories=["Sweets"], otif=.72, fill_rate=.86, lead_time_mean=2.8, lead_time_sd=1.6, open_pos=[dict(id="PO-2296", sku_id="festive-sweets-box-500g", qty=60, eta="2026-11-13", status="late 2 days")]),
 dict(id="sup-annapurna", name="Annapurna Grains", categories=["Staples", "Flour"], otif=.94, fill_rate=.98, lead_time_mean=4.0, lead_time_sd=.8, open_pos=[dict(id="PO-2283", sku_id="wheat-atta-5kg", qty=40, eta="2026-11-14", status="on time")]),
 dict(id="sup-leafline", name="Leafline Teas", categories=["Beverages"], otif=.97, fill_rate=.99, lead_time_mean=6.5, lead_time_sd=1.1, open_pos=[]),
 dict(id="sup-coldchain", name="ColdChain Frozen", categories=["Frozen", "Dairy"], otif=.66, fill_rate=.81, lead_time_mean=5.2, lead_time_sd=2.4, open_pos=[dict(id="PO-2279", sku_id="imported-cheese-spread", qty=24, eta="2026-11-12", status="cancel suggested")]),
]
for s_ in suppliers:
    s_["score"] = round(100 * (.5 * s_["otif"] + .3 * s_["fill_rate"] + .2 * max(0, 1 - s_["lead_time_sd"] / 3)))

MD = NUM["markdown"]
tasks = [
 dict(id="t1", type="markdown", sku_id="paneer-200g", params=dict(price=MD["best_price"], units=MD["batch_units"], days=MD["days_left"]), impact_rs=MD["rev_best"] - MD["rev_full"], minutes=10, urgency=3, role="floor", status="open"),
 dict(id="t2", type="shelf_check", sku_id="ragi-flour-1kg", params=dict(units=H["Ragi Flour 1kg"]["on_hand"], days=H["Ragi Flour 1kg"]["days_since_sale"]), impact_rs=round(H["Ragi Flour 1kg"]["daily_rate"] * 0 + 3 * 95 * 2), minutes=8, urgency=2, role="floor", status="open"),
 dict(id="t3", type="move_front", sku_id="festive-sweets-box-500g", params=dict(mult=6, qty=60), impact_rs=4200, minutes=15, urgency=3, role="floor", status="open"),
 dict(id="t4", type="return_vendor", sku_id="imported-cheese-spread", params=dict(weeks=int(H["Imported Cheese Spread"]["cover_weeks"]), units=36), impact_rs=H["Imported Cheese Spread"]["at_risk_value"], minutes=20, urgency=2, role="floor", status="open"),
 dict(id="t5", type="fefo", sku_id="toned-milk-500ml", params=dict(units=70), impact_rs=round(H["Toned Milk 500ml"]["at_risk_value"]), minutes=5, urgency=3, role="floor", status="open"),
 dict(id="t6", type="markdown", sku_id="brown-bread-400g", params=dict(price=34, units=30, days=2), impact_rs=410, minutes=6, urgency=2, role="floor", status="open"),
 dict(id="t7", type="reorder", sku_id="instant-noodles-70g", params=dict(qty=180), impact_rs=900, minutes=3, urgency=1, role="manager", status="open"),
 dict(id="t8", type="donate", sku_id="brown-bread-400g", params=dict(units=8), impact_rs=0, minutes=10, urgency=1, role="manager", status="open"),
]
for t in tasks:
    t["priority"] = round(t["impact_rs"] * t["urgency"] / max(t["minutes"], 1))
tasks.sort(key=lambda t: -t["priority"])

M = NUM["metrics"]
summary = dict(store="Demo Store", as_of=today.strftime("%Y-%m-%d"), sku_count=len(skus),
               at_risk_value=int(sum(b["value_at_risk"] for b in batches)), counts=NUM["totals"]["counts"],
               wape=round(M["wape_lgbm"], 4), wape_baseline=round(M["wape_naive"], 4), interval_coverage=round(M["coverage_p10_p90"], 4),
               open_tasks=len(tasks), active_spikes=len({s["sku_id"] for s in spikes if s["date"] >= (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d")}),
               late_pos=sum(1 for s_ in suppliers for p in s_["open_pos"] if p["status"] != "on time"),
               sample_data=True)
mock = dict(summary=summary, skus=skus, series=series, batches=batches, spikes=spikes, suppliers=suppliers, tasks=tasks, map=MAP)
out = Path(sys.argv[1]); out.write_text(json.dumps(mock, ensure_ascii=False, separators=(",", ":")))
print(out.stat().st_size, len(spikes), summary)
