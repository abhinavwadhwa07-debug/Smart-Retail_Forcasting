"""Usage: python generate_sales.py <out_dir> <path_to_Smart-Retail_Forcasting_repo>
Needs: pandas numpy scipy matplotlib lightgbm (and the Carlito font, or change font.family).

Illustrative demo on SYNTHETIC store data, run through the repo's own feature + pricing code.
Outputs charts (PNG) and a numbers JSON used by the deck and the report."""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import poisson
warnings.filterwarnings("ignore")

REPO = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(".").resolve()  # path to the Smart-Retail_Forcasting repo
sys.path.insert(0, str(REPO))
from src.features import RetailFeatureEngineer      # repo code
from src.pricing_engine import optimize_price, demand_curve  # repo code

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
INK, TEAL, SAFF, CORAL, SAGE, SLATE, MUTED, GRID = "#12302F", "#1F7A78", "#D98A1C", "#C9502F", "#4F8468", "#3D6A88", "#5A6E6C", "#DCE5E3"
plt.rcParams.update({"font.family": "Carlito", "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "axes.titleweight": "bold",
                     "axes.titlesize": 16, "axes.titlecolor": INK, "axes.titlelocation": "left", "figure.dpi": 160,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})
rng = np.random.default_rng(7)
days = pd.date_range("2026-05-15", periods=180, freq="D")
FEST = pd.Timestamp("2026-11-05")  # synthetic festival week centre

# name, category, base/day, shelf_life_d, unit_cost, price, elasticity, pattern
SKUS = [
 ("Toned Milk 500ml", "Dairy", 38, 3, 24, 28, -0.6, "steady"),
 ("Paneer 200g", "Dairy", 9, 7, 70, 90, -1.4, "steady"),
 ("Brown Bread 400g", "Bakery", 7, 5, 32, 45, -1.2, "declining"),
 ("Instant Noodles 70g", "Snacks", 30, 240, 10, 14, -1.1, "steady"),
 ("Masala Chips 50g", "Snacks", 22, 120, 14, 20, -1.3, "steady"),
 ("Festive Sweets Box 500g", "Sweets", 2, 10, 180, 260, -1.6, "festival"),
 ("Basmati Rice 5kg", "Staples", 4, 365, 520, 640, -0.8, "steady"),
 ("Sunflower Oil 1L", "Staples", 6, 270, 120, 150, -0.9, "steady"),
 ("Wheat Atta 5kg", "Flour", 7, 120, 210, 260, -0.9, "steady"),
 ("Besan 1kg", "Flour", 4, 180, 85, 110, -1.0, "steady"),
 ("Ragi Flour 1kg", "Flour", 3, 150, 70, 95, -1.0, "shelf_gap"),
 ("Green Tea 25 bags", "Beverages", 1.2, 540, 110, 160, -1.2, "slow"),
 ("Imported Cheese Spread", "Dairy", 0.6, 60, 210, 320, -1.8, "dead"),
 ("Frozen Peas 500g", "Frozen", 5, 180, 55, 75, -1.1, "steady"),
]
rows = []
for name, cat, base, life, cost, price, el, pat in SKUS:
    t = np.arange(len(days))
    dow = np.array([1.0, .92, .9, .95, 1.05, 1.3, 1.25])[days.dayofweek]
    pay = np.where(days.day <= 5, 1.12, 1.0)
    level = np.ones(len(days))
    if pat == "declining": level = np.linspace(1.4, .45, len(days))
    if pat == "dead": level = np.linspace(1.0, .05, len(days))
    if pat == "slow": level = np.full(len(days), .9)
    fest = np.ones(len(days))
    if pat == "festival":
        dd = (days - FEST).days.values
        fest = 1 + 11 * np.exp(-(dd / 4.0) ** 2) * (dd <= 3) + 2 * np.exp(-(dd / 6.0) ** 2)
    elif cat in ("Sweets", "Snacks", "Staples", "Flour"):
        dd = (days - FEST).days.values
        fest = 1 + .5 * np.exp(-(dd / 6.0) ** 2)
    p = price * np.where(rng.random(len(days)) < .08, .9, 1.0)  # occasional promo
    lam = base * dow * pay * level * fest * (price / p) ** abs(el)
    q = rng.poisson(lam)
    if pat == "shelf_gap":
        q[-14:] = 0  # stock exists in system but item is not on shelf
    for d, qq, pp in zip(days, q, p):
        rows.append((name, d, int(qq), round(float(pp), 2)))
sales = pd.DataFrame(rows, columns=["item_id", "invoice_date", "quantity", "unit_price"])
master = pd.DataFrame(SKUS, columns=["item_id", "category", "base", "shelf_life", "unit_cost", "price", "elasticity", "pattern"])

# ---------------- forecasting with the repo feature engineer -----------------
tmp = OUT / "_tmp"; tmp.mkdir(exist_ok=True)
sales.to_csv(tmp / "daily_demand_clean.csv", index=False)
fe = RetailFeatureEngineer(tmp / "daily_demand_clean.csv")
fe.load_data()
_k = ["item_id", "invoice_date"]
_t = fe.create_temporal_features(); _l = fe.create_lag_features(); _r = fe.create_rolling_features()
feat = _t.merge(_l[_k + [c for c in _l.columns if c.startswith("lag_")]], on=_k).merge(
    _r[_k + [c for c in _r.columns if c.startswith("rolling_") or c == "relative_price_ratio"]], on=_k)
feat = feat[feat.groupby("item_id").cumcount() >= 30].reset_index(drop=True)  # same 30-day warm-up as transform()
feat["invoice_date"] = pd.to_datetime(feat["invoice_date"])
feat["relative_price_ratio"] = feat["relative_price_ratio"].astype(float)
feat["item_code"] = feat["item_id"].astype("category").cat.codes
dd = (feat["invoice_date"] - FEST).dt.days
feat["days_to_festival"] = dd.clip(-30, 30)
X_cols = [c for c in feat.columns if c not in ("item_id", "invoice_date", "quantity")]
cut = days[-28]
tr, te = feat[feat.invoice_date < cut], feat[feat.invoice_date >= cut]
import lightgbm as lgb
p50m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, n_estimators=400, learning_rate=.04, num_leaves=31, min_child_samples=10, random_state=42, verbose=-1).fit(tr[X_cols], tr.quantity)
qm = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=300, learning_rate=.05, num_leaves=31, min_child_samples=10, random_state=42, verbose=-1).fit(tr[X_cols], tr.quantity) for a in (.1, .9)}
te = te.copy()
te["p50"] = p50m.predict(te[X_cols]); te["p10"] = qm[.1].predict(te[X_cols]).clip(0); te["p90"] = qm[.9].predict(te[X_cols])
te["p90"] = np.maximum(te["p90"], te["p50"]); te["p10"] = np.minimum(te["p10"], te["p50"])
def wape(y, f): return float(np.abs(y - f).sum() / max(np.abs(y).sum(), 1e-9))
# seasonal naive baseline (same weekday last week)
lag7 = te["lag_7_days"]
metrics = {"wape_lgbm": wape(te.quantity, te.p50), "wape_naive": wape(te.quantity, lag7),
           "coverage_p10_p90": float(((te.quantity >= np.floor(te.p10)) & (te.quantity <= np.ceil(te.p90))).mean()),
           "train_rows": int(len(tr)), "test_rows": int(len(te)), "features": len(X_cols)}

# Chart 1: forecast band for milk
s = "Toned Milk 500ml"
hist = sales[(sales.item_id == s) & (sales.invoice_date >= days[-70])]
f = te[te.item_id == s]
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(hist.invoice_date, hist.quantity, color=MUTED, lw=1.4, label="Actual units sold")
ax.fill_between(f.invoice_date, f.p10, f.p90, color=TEAL, alpha=.18, lw=0, label="P10 to P90 range")
ax.plot(f.invoice_date, f.p50, color=TEAL, lw=2.6, label="Forecast (P50)")
ax.axvline(cut, color=INK, lw=1, ls="--"); ax.text(cut, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*.03, "  forecast starts", color=INK, va="bottom", fontsize=11)
ax.set_title("Toned Milk 500ml: 28-day forecast with uncertainty"); ax.set_ylabel("Units per day")
ax.legend(frameon=False, loc="upper left", fontsize=11, ncol=3); fig.savefig(OUT / "c1_forecast.png"); plt.close(fig)

# ---------------- inventory snapshot + SKU health -----------------
today = days[-1]
recent = sales[sales.invoice_date > today - pd.Timedelta(days=28)].groupby("item_id").quantity.sum()
prev = sales[(sales.invoice_date <= today - pd.Timedelta(days=28)) & (sales.invoice_date > today - pd.Timedelta(days=56))].groupby("item_id").quantity.sum()
last_sale = sales[sales.quantity > 0].groupby("item_id").invoice_date.max()
next_f = te.groupby("item_id").p50.mean()  # avg daily forecast proxy
onhand = {"Toned Milk 500ml": [(70, 1), (60, 3)], "Paneer 200g": [(40, 2), (25, 6)], "Brown Bread 400g": [(30, 2)],
          "Instant Noodles 70g": [(300, 200)], "Masala Chips 50g": [(260, 90)], "Festive Sweets Box 500g": [(45, 4)],
          "Basmati Rice 5kg": [(40, 300)], "Sunflower Oil 1L": [(80, 200)], "Wheat Atta 5kg": [(90, 80)], "Besan 1kg": [(45, 120)],
          "Ragi Flour 1kg": [(38, 110)], "Green Tea 25 bags": [(60, 400)], "Imported Cheese Spread": [(36, 12)], "Frozen Peas 500g": [(70, 150)]}
H = []
for _, m in master.iterrows():
    n = m.item_id; batches = sorted(onhand[n], key=lambda b: b[1])
    qty = sum(b[0] for b in batches)
    mu = max(recent.get(n, 0) / 28, .05)
    cover_w = qty / (mu * 7)
    trend = (recent.get(n, 0) + 1) / (prev.get(n, 0) + 1)
    dsls = (today - last_sale.get(n, days[0])).days
    # expiry risk, FEFO: demand consumes oldest batch first
    at_risk_units = 0.0; risk_list = []
    consumed_before = 0
    for bq, life in batches:
        # P(cumulative demand over `life` days < consumed_before + bq) -> expected leftover of this batch
        lam = mu * life
        ks = np.arange(0, int(lam * 3 + bq + consumed_before + 30))
        pmf = poisson.pmf(ks, lam)
        leftover = np.clip(consumed_before + bq - ks, 0, bq)
        exp_left = float((pmf * leftover).sum())
        p_any = float(poisson.cdf(consumed_before + bq - 1, lam))
        risk_list.append((bq, life, exp_left, p_any))
        at_risk_units += exp_left; consumed_before += bq
    at_risk_val = at_risk_units * m.unit_cost
    p_exp = max(r[3] for r in risk_list)
    # health class
    if dsls >= 10 and m.pattern == "shelf_gap":
        cls, reason = "Shelf gap", f"In stock ({qty}) but no sale for {dsls} days while flour peers sell daily"
    elif mu < .4 and cover_w > 8:
        cls, reason = "Dead", f"{cover_w:.0f} weeks of cover, sales down {100*(1-trend):.0f}% vs prior 4 weeks"
    elif at_risk_units >= .15 * qty and at_risk_units >= 3:
        cls, reason = "Expiry risk", f"{at_risk_units:.0f} of {qty} units likely to expire unsold"
    elif trend < .7 or cover_w > 6:
        cls, reason = "Watch", f"Demand trend {trend:.2f}x, {cover_w:.1f} weeks of cover"
    else:
        cls, reason = "Healthy", f"{cover_w:.1f} weeks of cover, trend {trend:.2f}x"
    H.append(dict(item=n, category=m.category, on_hand=qty, daily_rate=round(mu, 2), cover_weeks=round(cover_w, 1),
                  trend=round(trend, 2), days_since_sale=int(dsls), at_risk_units=round(at_risk_units, 1),
                  at_risk_value=round(at_risk_val), p_expire=round(p_exp, 2), cls=cls, reason=reason,
                  unit_cost=m.unit_cost, price=m.price, elasticity=m.elasticity, shelf_life=m.shelf_life))
health = pd.DataFrame(H)
CLSCOL = {"Healthy": SAGE, "Watch": SAFF, "Expiry risk": CORAL, "Dead": INK, "Shelf gap": SLATE}

# Chart 2: health map
fig, ax = plt.subplots(figsize=(10, 5.2))
for c, g in health.groupby("cls"):
    ax.scatter(g.cover_weeks, g.trend, s=90 + g.at_risk_value / 25, color=CLSCOL[c], alpha=.85, label=c, edgecolor="white", lw=1.2, zorder=3)
for _, r in health[health.cls != "Healthy"].iterrows():
    ax.annotate(r["item"], (r.cover_weeks, r.trend), xytext=(-8, -20) if r["item"].startswith("Paneer") else (8, 6), textcoords="offset points", fontsize=11, color=INK,
                ha="right" if (r.cover_weeks > 50 or r["item"].startswith("Paneer")) else "left")
hh = health[health.cls == "Healthy"]
ax.annotate(f"{len(hh)} healthy SKUs\n(staples, snacks, frozen)", (hh.cover_weeks.median(), hh.trend.max()), xytext=(10, 45),
            textcoords="offset points", fontsize=11, color=SAGE, arrowprops=dict(arrowstyle="-", color=SAGE))
ax.set_xscale("log"); ax.set_xlabel("Weeks of stock cover (log scale)"); ax.set_ylabel("Demand trend (last 4 wks / prior 4 wks)")
ax.axhline(1, color=MUTED, lw=.8); ax.axvline(8, color=MUTED, lw=.8, ls=":")
ax.set_title("SKU health map: where cash and shelf space are stuck")
ax.legend(frameon=False, fontsize=11, loc="upper right"); fig.savefig(OUT / "c2_health.png"); plt.close(fig)

# Chart 3: expiry risk value
er = health[health.at_risk_value > 0].sort_values("at_risk_value").tail(7)
fig, ax = plt.subplots(figsize=(10, 4.4))
bars = ax.barh(er["item"], er.at_risk_value, color=[CORAL if v > 500 else SAFF for v in er.at_risk_value], height=.6, zorder=3)
for b, v, u in zip(bars, er.at_risk_value, er.at_risk_units):
    ax.text(b.get_width() + 30, b.get_y() + b.get_height() / 2, f"Rs {v:,.0f}  ({u:.0f} units)", va="center", fontsize=11, color=INK)
ax.set_xlim(0, er.at_risk_value.max() * 1.35); ax.grid(axis="y", visible=False)
ax.set_title("Stock value likely to expire unsold (at cost)"); ax.set_xlabel("Rupees at cost")
fig.savefig(OUT / "c3_expiry.png"); plt.close(fig)

# Chart 4: spike detection on festival item
sw = sales[sales.item_id == "Festive Sweets Box 500g"].set_index("invoice_date").quantity
base = sw.shift(1).rolling(28, min_periods=7).median()
mad = (sw.shift(1) - base).abs().rolling(28, min_periods=7).median().replace(0, 1)
z = (sw - base) / (1.4826 * mad)
win = sw[sw.index >= days[-60]]
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.bar(win.index, win.values, color=GRID, width=.85, zorder=2, label="Units sold")
flag = z[(z.index >= days[-60]) & (z > 3.5)]
ax.bar(flag.index, sw[flag.index], color=CORAL, width=.85, zorder=3, label="Flagged as spike (robust z > 3.5)")
ax.plot(base[base.index >= days[-60]], color=INK, lw=2, label="Normal level (28-day median)")
first = flag.index.min()
if pd.notna(first):
    ax.annotate(f"First alert {first:%d %b}", (first, sw[first]), xytext=(-150, 30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=INK), fontsize=11, color=INK)
ax.set_title("Festive Sweets Box: demand hike flagged before the peak")
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); ax.set_ylabel("Units per day")
ax.legend(frameon=False, fontsize=11, loc="upper left"); fig.savefig(OUT / "c4_spike.png"); plt.close(fig)
spike = {"first_alert": str(first.date()) if pd.notna(first) else None, "peak_units": int(sw.max()), "peak_date": str(sw.idxmax().date()), "normal_level": float(base.loc[first]) if pd.notna(first) else None, "flag_days": int(len(flag))}

# Chart 5: markdown optimisation for Paneer using repo optimize_price
pn = health[health["item"] == "Paneer 200g"].iloc[0]
near = [b for b in onhand["Paneer 200g"] if b[1] <= 2][0]
pred_2d = pn.daily_rate * near[1]
res_norm = optimize_price(predicted_demand=pred_2d, unit_cost=pn.unit_cost, inventory=near[0], historical_max_price=pn.price,
                          reference_price=pn.price, elasticity=pn.elasticity)
# salvage view: units unsold are lost, so maximise revenue - cost of all stock => equivalently revenue (cost sunk)
prices = np.linspace(pn.unit_cost * 0.75, pn.price * 1.1, 120)
rev = [p * min(demand_curve(pred_2d, pn.price, p, pn.elasticity), near[0]) for p in prices]
sold = [min(demand_curve(pred_2d, pn.price, p, pn.elasticity), near[0]) for p in prices]
best_i = int(np.argmax(rev)); best_p = prices[best_i]
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(prices, rev, color=TEAL, lw=2.6)
ax.axvline(pn.price, color=MUTED, ls=":", lw=1.2); ax.text(pn.price, min(rev) + (max(rev) - min(rev)) * .55, " today's shelf price", color=MUTED, fontsize=11)
ax.scatter([pn.price], [rev_full0 := pn.price * min(pred_2d, near[0])], color=MUTED, s=50, zorder=4)
ax.annotate(f"Full price: sells {min(pred_2d, near[0]):.0f}, wastes {near[0]-min(pred_2d, near[0]):.0f}", (pn.price, rev_full0), xytext=(-210, -30), textcoords="offset points", fontsize=11, color=MUTED)
ax.scatter([best_p], [rev[best_i]], color=CORAL, s=80, zorder=4)
ax.annotate(f"Best clearance price Rs {best_p:.0f}\nsells {sold[best_i]:.0f} of {near[0]} units", (best_p, rev[best_i]), xytext=(16, 10), textcoords="offset points", fontsize=11, color=INK, va="bottom")
ax.set_ylim(top=max(rev) + (max(rev) - min(rev)) * .3)
ax.set_title(f"Paneer batch expiring in {near[1]} days: recovery by price"); ax.set_xlabel("Price (Rs)"); ax.set_ylabel("Expected revenue in 2 days (Rs)")
fig.savefig(OUT / "c5_markdown.png"); plt.close(fig)
rev_full = pn.price * min(pred_2d, near[0])
markdown = {"batch_units": near[0], "days_left": near[1], "shelf_price": pn.price, "best_price": round(float(best_p)), "units_sold_best": round(float(sold[best_i])),
            "units_sold_full": round(float(min(pred_2d, near[0]))), "rev_best": round(float(rev[best_i])), "rev_full": round(float(rev_full)),
            "discount_pct": round(100 * (1 - best_p / pn.price)), "repo_optimize_price": {k: round(v, 2) for k, v in res_norm.items()}}

# Chart 6: peer comparison for Ragi
fl = sales[(sales.item_id.isin(["Wheat Atta 5kg", "Besan 1kg", "Ragi Flour 1kg"])) & (sales.invoice_date >= days[-42])]
wk = fl.assign(week=((fl.invoice_date - days[-42]).dt.days // 7) + 1).groupby(["item_id", "week"]).quantity.sum().unstack(0)
fig, ax = plt.subplots(figsize=(10, 4.4))
for col, c in zip(["Wheat Atta 5kg", "Besan 1kg", "Ragi Flour 1kg"], [SAGE, SLATE, CORAL]):
    ax.plot(wk.index, wk[col], marker="o", lw=2.6 if "Ragi" in col else 1.8, color=c, label=col)
ax.set_xticks(wk.index, [f"Wk {i}" for i in wk.index]); ax.set_ylabel("Units per week")
ax.set_title("Ragi Flour vs its flour peers: sales stop while stock says 38")
ax.legend(frameon=False, fontsize=11); fig.savefig(OUT / "c6_peers.png"); plt.close(fig)

summary = {"metrics": metrics, "spike": spike, "markdown": markdown,
           "health": health.to_dict(orient="records"),
           "totals": {"at_risk_value": int(health.at_risk_value.sum()), "skus": len(health),
                      "counts": health.cls.value_counts().to_dict(),
                      "dead_value": int((health[health.cls == "Dead"].on_hand * health[health.cls == "Dead"].unit_cost).sum())}}
json.dump(summary, open(OUT / "demo_numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: summary[k] for k in ("metrics", "spike", "markdown", "totals")}, indent=1, default=float))
print(health[["item", "cls", "cover_weeks", "trend", "at_risk_units", "at_risk_value", "reason"]].to_string())
# ---- exports for the web frontend mock data
sales.to_csv(OUT / "sales.csv", index=False)
te[["item_id", "invoice_date", "quantity", "p10", "p50", "p90"]].to_csv(OUT / "forecast.csv", index=False)
json.dump({k: v for k, v in onhand.items()}, open(OUT / "onhand.json", "w"))
master.to_csv(OUT / "master.csv", index=False)
