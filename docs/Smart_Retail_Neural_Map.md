# Smart Retail Intelligence Network — Neural Map

_Paste this whole file (or one branch) into a chat as context, then ask for code, a notebook, a design or a report section. Status tags: Built / Partly built / Planned / Gap._

# Smart Retail Intelligence Network  `[Partly built]`

An AI/ML system that forecasts demand per product per day, flags stock that is expiring, dead or losing shelf presence, catches demand hikes early, makes the supply chain transparent, and turns all of it into clear daily actions for store workers and managers.

- Base repo: Smart Retail Precision Demand Forecasting & Dynamic Pricing Engine (Python, pandas, LightGBM, SciPy)
- Unit of analysis: one SKU x one store x one day
- North-star outcome: less waste, fewer stockouts, healthier margin, less guesswork for workers

> **Prompt seed:** You are helping build the Smart Retail Intelligence Network: a Python AI/ML system that forecasts SKU-day demand, detects expiry risk, dead stock, weak shelf presence and demand spikes, scores supplier transparency, compares weak SKUs against well-fulfilled peers, and outputs plain-language daily actions for retail workers. The existing repo has data_loader.py (cleaning + zero-filled daily grid), features.py (calendar, lag 1/7/14/30, rolling 7/30, price ratio), forecasting.py (WAPE + expanding-window CV with LightGBM Tweedie) and pricing_engine.py (log-log elasticity + bounded profit optimisation with a 1.15x cost floor). Keep code typed, tested with pytest, leakage-free, and explainable.

## Problem & Vision  `[Built]`

The retail pain points the network exists to remove, and what success looks like.

> **Prompt seed:** Write a crisp problem statement and vision for a retail AI that tackles expired stock, dead stock, weak shelf presence, missed demand hikes and opaque supply chains. Quantify the cost of each pain point for a mid-size Indian grocery/FMCG store and define measurable success targets.

### Expired & near-expiry stock  `[Planned]`

Perishables and short-shelf-life goods that cross their expiry date unsold become write-offs and waste.

- Signal: on-hand batch qty > forecast demand over remaining shelf life
- Cost: write-off value, disposal cost, lost shelf space
- Lever: FEFO picking, early markdown, transfer, donation

> **Prompt seed:** Context: Smart Retail Intelligence Network > Problem & Vision > Expired & near-expiry stock. Perishables and short-shelf-life goods that cross their expiry date unsold become write-offs and waste. Known points: Signal: on-hand batch qty > forecast demand over remaining shelf life; Cost: write-off value, disposal cost, lost shelf space; Lever: FEFO picking, early markdown, transfer, donation. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Dead & slow-moving stock  `[Planned]`

SKUs with little or no demand that tie up cash and shelf space.

- Signal: long zero-sale streaks, falling sell-through, high weeks-of-cover
- Lever: clearance, bundling, return-to-vendor, delist

> **Prompt seed:** Context: Smart Retail Intelligence Network > Problem & Vision > Dead & slow-moving stock. SKUs with little or no demand that tie up cash and shelf space. Known points: Signal: long zero-sale streaks, falling sell-through, high weeks-of-cover; Lever: clearance, bundling, return-to-vendor, delist. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Weak shelf presence  `[Planned]`

Products that exist in the store but are not visible or available enough to sell (phantom stock, poor facing, out-of-shelf).

- Signal: system stock > 0 but sales = 0 for an unusually long run
- Signal: sales far below peers with similar price and category
- Lever: re-facing, planogram fix, cycle count

> **Prompt seed:** Context: Smart Retail Intelligence Network > Problem & Vision > Weak shelf presence. Products that exist in the store but are not visible or available enough to sell (phantom stock, poor facing, out-of-shelf). Known points: Signal: system stock > 0 but sales = 0 for an unusually long run; Signal: sales far below peers with similar price and category; Lever: re-facing, planogram fix, cycle count. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Missed demand hikes  `[Planned]`

Sudden surges (festivals, weather, viral trends, competitor stockout) that cause stockouts and lost sales.

- Signal: forecast residuals breaking the P90 band, changepoints
- Lever: early reorder, safety stock bump, staff reallocation

> **Prompt seed:** Context: Smart Retail Intelligence Network > Problem & Vision > Missed demand hikes. Sudden surges (festivals, weather, viral trends, competitor stockout) that cause stockouts and lost sales. Known points: Signal: forecast residuals breaking the P90 band, changepoints; Lever: early reorder, safety stock bump, staff reallocation. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Opaque supply chain  `[Planned]`

Store teams cannot see where an order is, how reliable a supplier is, or which batch came from where.

- Signal: lead-time variance, OTIF misses, short shipments
- Lever: supplier scorecards, ETA prediction, batch traceability

> **Prompt seed:** Context: Smart Retail Intelligence Network > Problem & Vision > Opaque supply chain. Store teams cannot see where an order is, how reliable a supplier is, or which batch came from where. Known points: Signal: lead-time variance, OTIF misses, short shipments; Lever: supplier scorecards, ETA prediction, batch traceability. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Worker burden  `[Planned]`

Staff spend time on manual counts, guesswork ordering and firefighting instead of serving customers.

- Goal: one short, ranked task list per shift
- Goal: every alert comes with a reason in plain language (English / Hindi)

> **Prompt seed:** Context: Smart Retail Intelligence Network > Problem & Vision > Worker burden. Staff spend time on manual counts, guesswork ordering and firefighting instead of serving customers. Known points: Goal: one short, ranked task list per shift; Goal: every alert comes with a reason in plain language (English / Hindi). Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Success targets  `[Planned]`

What the pilot must prove.

- Waste / expiry write-off down 20-30%
- Stockout rate down 15-25%
- Forecast WAPE below 35% on A-class SKUs
- Inventory turns up, GMROI up
- Worker time on ordering down 30%

> **Prompt seed:** Context: Smart Retail Intelligence Network > Problem & Vision > Success targets. What the pilot must prove. Known points: Waste / expiry write-off down 20-30%; Stockout rate down 15-25%; Forecast WAPE below 35% on A-class SKUs; Inventory turns up, GMROI up; Worker time on ordering down 30%. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## People & Users  `[Planned]`

Who touches the system and what each one needs from it.

> **Prompt seed:** Create user personas and jobs-to-be-done for a retail demand-intelligence system: store manager, floor/shelf worker, buyer/procurement, warehouse staff, supplier, owner/finance. For each: goals, pains, decisions they make, the screen they need, and the alerts they should and should not receive.

### Store manager  `[Planned]`

Owns store P&L; approves orders, markdowns and transfers.

- Needs: morning brief, exceptions only, approve/override buttons

> **Prompt seed:** Context: Smart Retail Intelligence Network > People & Users > Store manager. Owns store P&L; approves orders, markdowns and transfers. Known points: Needs: morning brief, exceptions only, approve/override buttons. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Floor & shelf workers  `[Planned]`

Restock, face, count, mark down and pull expiring items.

- Needs: ranked task list, barcode scan, bilingual UI, voice, big touch targets
- Must never be blamed by the system; tasks are suggestions with reasons

> **Prompt seed:** Context: Smart Retail Intelligence Network > People & Users > Floor & shelf workers. Restock, face, count, mark down and pull expiring items. Known points: Needs: ranked task list, barcode scan, bilingual UI, voice, big touch targets; Must never be blamed by the system; tasks are suggestions with reasons. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Buyer / procurement  `[Planned]`

Decides what and how much to order from whom.

- Needs: forecast P10-P90, recommended order qty, supplier scorecard

> **Prompt seed:** Context: Smart Retail Intelligence Network > People & Users > Buyer / procurement. Decides what and how much to order from whom. Known points: Needs: forecast P10-P90, recommended order qty, supplier scorecard. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Warehouse / DC  `[Planned]`

Receives, stores, picks and ships.

- Needs: FEFO pick lists, inbound ETA, transfer requests

> **Prompt seed:** Context: Smart Retail Intelligence Network > People & Users > Warehouse / DC. Receives, stores, picks and ships. Known points: Needs: FEFO pick lists, inbound ETA, transfer requests. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Suppliers  `[Planned]`

Deliver goods; benefit from shared forecasts.

- Needs: portal with demand outlook and their own OTIF score

> **Prompt seed:** Context: Smart Retail Intelligence Network > People & Users > Suppliers. Deliver goods; benefit from shared forecasts. Known points: Needs: portal with demand outlook and their own OTIF score. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Owner / finance  `[Planned]`

Cares about margin, cash in stock, waste.

- Needs: KPI dashboard, waste and margin trend, ROI of the system

> **Prompt seed:** Context: Smart Retail Intelligence Network > People & Users > Owner / finance. Cares about margin, cash in stock, waste. Known points: Needs: KPI dashboard, waste and margin trend, ROI of the system. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Customers (indirect)  `[Planned]`

Feel the result: fresher shelves, fewer stockouts, fair prices.

- Guardrail: no surge pricing on essentials

> **Prompt seed:** Context: Smart Retail Intelligence Network > People & Users > Customers (indirect). Feel the result: fresher shelves, fewer stockouts, fair prices. Known points: Guardrail: no surge pricing on essentials. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## Data Layer  `[Partly built]`

Every input the network learns from, and the rules that make it trustworthy.

> **Prompt seed:** Design the data layer for a retail demand-intelligence system. Define schemas (column, type, grain, key) for POS sales, inventory batches with expiry, product master, purchase orders and receipts, prices/promotions, calendar/festivals, weather and store ops. Include data-quality rules, the zero-filled SKU-day grid, and how each table joins.

### POS transactions  `[Built]`

Line-level sales: item, quantity, unit price, timestamp, store. Currently the UCI Online Retail II dataset.

- Repo: load_raw_transactions() reads CSV with ISO-8859-1
- Column aliasing: StockCode/item_id, UnitPrice/Price, InvoiceDate/date

> **Prompt seed:** Context: Smart Retail Intelligence Network > Data Layer > POS transactions. Line-level sales: item, quantity, unit price, timestamp, store. Currently the UCI Online Retail II dataset. Known points: Repo: load_raw_transactions() reads CSV with ISO-8859-1; Column aliasing: StockCode/item_id, UnitPrice/Price, InvoiceDate/date. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Cleaning rules  `[Built]`

Drop invalid rows so the model sees real demand only.

- Repo: drop nulls, empty IDs, quantity <= 0, price <= 0
- Floor timestamps to day; gross_revenue = qty x price
- Planned: keep returns/cancellations in a separate table instead of discarding them

> **Prompt seed:** Context: Smart Retail Intelligence Network > Data Layer > Cleaning rules. Drop invalid rows so the model sees real demand only. Known points: Repo: drop nulls, empty IDs, quantity <= 0, price <= 0; Floor timestamps to day; gross_revenue = qty x price; Planned: keep returns/cancellations in a separate table instead of discarding them. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Zero-filled daily grid  `[Built]`

Full Cartesian product of active SKUs x calendar days so zero-sale days are explicit, not missing.

- Repo: expand_daily_sales_grid(min_sales=50)
- Price forward/back-filled per item
- Why it matters: missing != no demand; enables lags and intermittency stats

> **Prompt seed:** Context: Smart Retail Intelligence Network > Data Layer > Zero-filled daily grid. Full Cartesian product of active SKUs x calendar days so zero-sale days are explicit, not missing. Known points: Repo: expand_daily_sales_grid(min_sales=50); Price forward/back-filled per item; Why it matters: missing != no demand; enables lags and intermittency stats. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Inventory & batches  `[Gap / fix]`

On-hand quantity per SKU per batch with manufacture and expiry dates.

- Needed for expiry risk and weeks-of-cover
- Grain: store x SKU x batch x day snapshot

> **Prompt seed:** Context: Smart Retail Intelligence Network > Data Layer > Inventory & batches. On-hand quantity per SKU per batch with manufacture and expiry dates. Known points: Needed for expiry risk and weeks-of-cover; Grain: store x SKU x batch x day snapshot. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Product master  `[Gap / fix]`

Category, brand, pack size, shelf life, unit cost, supplier, perishability class.

- Needed for peer comparison and margin floor

> **Prompt seed:** Context: Smart Retail Intelligence Network > Data Layer > Product master. Category, brand, pack size, shelf life, unit cost, supplier, perishability class. Known points: Needed for peer comparison and margin floor. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Supply chain events  `[Gap / fix]`

Purchase orders, advance shipping notices, goods receipts, short-shipments, returns to vendor.

- Derives: lead time, fill rate, OTIF per supplier

> **Prompt seed:** Context: Smart Retail Intelligence Network > Data Layer > Supply chain events. Purchase orders, advance shipping notices, goods receipts, short-shipments, returns to vendor. Known points: Derives: lead time, fill rate, OTIF per supplier. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Prices & promotions  `[Partly built]`

Shelf price history, discounts, promo flags, display/endcap placement.

- Repo: unit_price + rolling_30d_avg_unit_price + relative_price_ratio
- Planned: promo calendar, discount depth

> **Prompt seed:** Context: Smart Retail Intelligence Network > Data Layer > Prices & promotions. Shelf price history, discounts, promo flags, display/endcap placement. Known points: Repo: unit_price + rolling_30d_avg_unit_price + relative_price_ratio; Planned: promo calendar, discount depth. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### External signals  `[Planned]`

Things outside the store that move demand.

- Festival & holiday calendar (Diwali, Holi, Eid, Christmas, wedding season)
- Weather (temperature, rain)
- Local events, school terms, salary days
- Competitor prices (optional)

> **Prompt seed:** Context: Smart Retail Intelligence Network > Data Layer > External signals. Things outside the store that move demand. Known points: Festival & holiday calendar (Diwali, Holi, Eid, Christmas, wedding season); Weather (temperature, rain); Local events, school terms, salary days; Competitor prices (optional). Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Store operations  `[Gap / fix]`

Planogram, shelf facings, footfall, staff rota.

- Needed for shelf-presence analysis and task assignment

> **Prompt seed:** Context: Smart Retail Intelligence Network > Data Layer > Store operations. Planogram, shelf facings, footfall, staff rota. Known points: Needed for shelf-presence analysis and task assignment. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Data quality monitor  `[Planned]`

Automated checks before anything trains or predicts.

- Schema, nulls, negative stock, duplicate invoices, sudden volume drops
- Great Expectations / pandera style rules
- Fail loudly, never silently impute

> **Prompt seed:** Context: Smart Retail Intelligence Network > Data Layer > Data quality monitor. Automated checks before anything trains or predicts. Known points: Schema, nulls, negative stock, duplicate invoices, sudden volume drops; Great Expectations / pandera style rules; Fail loudly, never silently impute. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## Feature Engineering  `[Partly built]`

Signals computed from the data that let models see rhythm, momentum, price perception and risk.

> **Prompt seed:** Extend features.py (RetailFeatureEngineer) with leakage-safe features for expiry risk, intermittency, stock cover, promotions, festivals and peer behaviour. Keep closed='left' on all rolling windows, compute per item_id, add pytest tests proving no future data leaks into any row.

### Calendar features  `[Built]`

When the item-day happened.

- Repo: day_of_week, day_of_month, month, quarter, is_weekend, payday_flag (day 1-5)

> **Prompt seed:** Context: Smart Retail Intelligence Network > Feature Engineering > Calendar features. When the item-day happened. Known points: Repo: day_of_week, day_of_month, month, quarter, is_weekend, payday_flag (day 1-5). Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Lag features  `[Built]`

Demand on earlier days.

- Repo: lag_1, lag_7, lag_14, lag_30 days, per item, fillna(0)

> **Prompt seed:** Context: Smart Retail Intelligence Network > Feature Engineering > Lag features. Demand on earlier days. Known points: Repo: lag_1, lag_7, lag_14, lag_30 days, per item, fillna(0). Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Rolling statistics  `[Built]`

Recent level and volatility.

- Repo: 7d and 30d mean, std, min, max with closed='left'
- Warm-up rows (30 days) dropped in transform()

> **Prompt seed:** Context: Smart Retail Intelligence Network > Feature Engineering > Rolling statistics. Recent level and volatility. Known points: Repo: 7d and 30d mean, std, min, max with closed='left'; Warm-up rows (30 days) dropped in transform(). Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Price perception  `[Built]`

Is today's price high or low versus normal?

- Repo: rolling_30d_avg_unit_price, relative_price_ratio

> **Prompt seed:** Context: Smart Retail Intelligence Network > Feature Engineering > Price perception. Is today's price high or low versus normal? Known points: Repo: rolling_30d_avg_unit_price, relative_price_ratio. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Expiry & ageing features  `[Planned]`

How close stock is to being unsellable.

- days_to_expiry (min across batches)
- stock_age_days
- share_of_stock_expiring_7d
- shelf_life_used_pct

> **Prompt seed:** Context: Smart Retail Intelligence Network > Feature Engineering > Expiry & ageing features. How close stock is to being unsellable. Known points: days_to_expiry (min across batches); stock_age_days; share_of_stock_expiring_7d; shelf_life_used_pct. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Inventory cover  `[Planned]`

How long current stock will last.

- weeks_of_cover = on_hand / forecast_weekly
- sell_through_rate = sold / (sold + on_hand)
- days_since_last_sale

> **Prompt seed:** Context: Smart Retail Intelligence Network > Feature Engineering > Inventory cover. How long current stock will last. Known points: weeks_of_cover = on_hand / forecast_weekly; sell_through_rate = sold / (sold + on_hand); days_since_last_sale. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Intermittency profile  `[Planned]`

How lumpy demand is.

- ADI (avg demand interval)
- CV-squared of non-zero demand
- Syntetos-Boylan class: smooth / erratic / intermittent / lumpy

> **Prompt seed:** Context: Smart Retail Intelligence Network > Feature Engineering > Intermittency profile. How lumpy demand is. Known points: ADI (avg demand interval); CV-squared of non-zero demand; Syntetos-Boylan class: smooth / erratic / intermittent / lumpy. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Promotion & event features  `[Planned]`

Special-day effects.

- promo_flag, discount_depth
- days_to_next_festival, days_since_festival
- weather_temp, rain_flag

> **Prompt seed:** Context: Smart Retail Intelligence Network > Feature Engineering > Promotion & event features. Special-day effects. Known points: promo_flag, discount_depth; days_to_next_festival, days_since_festival; weather_temp, rain_flag. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Peer & cross-SKU features  `[Planned]`

How the product behaves relative to similar products.

- category share of sales
- peer median demand at same price band
- substitute stockout flag (cannibalisation in reverse)

> **Prompt seed:** Context: Smart Retail Intelligence Network > Feature Engineering > Peer & cross-SKU features. How the product behaves relative to similar products. Known points: category share of sales; peer median demand at same price band; substitute stockout flag (cannibalisation in reverse). Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Supplier features  `[Planned]`

Supply-side reliability.

- lead_time_mean / std per supplier-SKU
- last_fill_rate
- open_po_qty, days_to_next_delivery

> **Prompt seed:** Context: Smart Retail Intelligence Network > Feature Engineering > Supplier features. Supply-side reliability. Known points: lead_time_mean / std per supplier-SKU; last_fill_rate; open_po_qty, days_to_next_delivery. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## ML Core (the Network)  `[Partly built]`

The models that think: forecasting, health scoring, risk, spikes, pricing, inventory and comparison.

> **Prompt seed:** Design the ML core of a retail demand-intelligence system as cooperating models: a probabilistic demand forecaster, SKU health classifier, expiry-risk estimator, demand-spike detector, elasticity/pricing optimiser, inventory policy engine, supplier reliability model and peer-comparison engine. Show inputs, outputs, algorithms, metrics, and how outputs feed each other.

### Demand forecaster  `[Partly built]`

Predicts units per SKU per day for the next 1-28 days, with uncertainty.

> **Prompt seed:** Context: Smart Retail Intelligence Network > ML Core (the Network) > Demand forecaster. Predicts units per SKU per day for the next 1-28 days, with uncertainty. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

- **Baselines** `[Planned]` — Benchmarks every model must beat.
  - Seasonal naive (same weekday last week)
  - Moving average 7/28
  - Croston / TSB for intermittent SKUs
- **LightGBM Tweedie (primary)** `[Partly built]` — Gradient boosting with Tweedie loss for zero-heavy, non-negative demand.
  - Repo: LGBMRegressor(objective='regression_tweedie', n_estimators=200, lr=0.05)
  - Falls back to DummyRegressor if LightGBM missing
  - Tune tweedie_variance_power 1.1-1.5
- **XGBoost challenger** `[Planned]` — Second GBM for ensembling and robustness.
  - objective='reg:tweedie'
- **Quantile forecasts** `[Planned]` — P10 / P50 / P90 so ordering can respect risk.
  - LightGBM objective='quantile' per alpha, or conformal intervals
- **Hierarchical reconciliation** `[Planned]` — SKU forecasts add up to category and store totals.
  - MinT / bottom-up reconciliation
- **Deep models (later)** `[Planned]` — For large catalogues with rich covariates.
  - Temporal Fusion Transformer, N-BEATS, DeepAR
  - Only if they beat GBM on WAPE in back-tests
### SKU health classifier  `[Planned]`

Labels each SKU's life stage and whether it is healthy, at risk or dead.

- ABC (value) x XYZ (variability) matrix
- Life stage: launch / growth / mature / decline / dead
- Dead-stock rule + model: zero-sale streak, cover > N weeks, falling trend
- Output: health score 0-100 with reason codes

> **Prompt seed:** Context: Smart Retail Intelligence Network > ML Core (the Network) > SKU health classifier. Labels each SKU's life stage and whether it is healthy, at risk or dead. Known points: ABC (value) x XYZ (variability) matrix; Life stage: launch / growth / mature / decline / dead; Dead-stock rule + model: zero-sale streak, cover > N weeks, falling trend; Output: health score 0-100 with reason codes. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Expiry risk estimator  `[Planned]`

Probability that each batch expires before it sells.

- P(expire) = P(cumulative demand over remaining shelf life < batch qty) using the quantile forecast
- FEFO-aware: oldest batch consumes demand first
- Output: at-risk units, at-risk value, days left, suggested action

> **Prompt seed:** Context: Smart Retail Intelligence Network > ML Core (the Network) > Expiry risk estimator. Probability that each batch expires before it sells. Known points: P(expire) = P(cumulative demand over remaining shelf life < batch qty) using the quantile forecast; FEFO-aware: oldest batch consumes demand first; Output: at-risk units, at-risk value, days left, suggested action. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Demand spike / hike detector  `[Planned]`

Spots surges early and tells you whether they will last.

- Residual z-score vs forecast, P90 band breaks
- STL decomposition + changepoint (ruptures / PELT)
- Isolation Forest for multivariate anomalies
- Classify: one-off, festival, trend shift, data error

> **Prompt seed:** Context: Smart Retail Intelligence Network > ML Core (the Network) > Demand spike / hike detector. Spots surges early and tells you whether they will last. Known points: Residual z-score vs forecast, P90 band breaks; STL decomposition + changepoint (ruptures / PELT); Isolation Forest for multivariate anomalies; Classify: one-off, festival, trend shift, data error. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Price elasticity & dynamic pricing  `[Partly built]`

How demand responds to price and the best price within guardrails.

- Repo: demand_curve() log-log, Q = Q0 x (P0/P)^|e|
- Repo: optimize_price() via scipy minimize_scalar, bounded
- Guardrails: floor 1.15 x unit cost, ceiling min(1.2 x hist max, 2 x ref price)
- Planned: per-SKU elasticity by regression on log-price, markdown ladders for near-expiry

> **Prompt seed:** Context: Smart Retail Intelligence Network > ML Core (the Network) > Price elasticity & dynamic pricing. How demand responds to price and the best price within guardrails. Known points: Repo: demand_curve() log-log, Q = Q0 x (P0/P)^|e|; Repo: optimize_price() via scipy minimize_scalar, bounded; Guardrails: floor 1.15 x unit cost, ceiling min(1.2 x hist max, 2 x ref price); Planned: per-SKU elasticity by regression on log-price, markdown ladders for near-expiry. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Inventory policy engine  `[Planned]`

Turns forecasts into order quantities.

- Safety stock = z x sigma_demand x sqrt(lead time)
- Reorder point, order-up-to level
- Newsvendor critical ratio for perishables
- Respects MOQ, pack size, shelf capacity

> **Prompt seed:** Context: Smart Retail Intelligence Network > ML Core (the Network) > Inventory policy engine. Turns forecasts into order quantities. Known points: Safety stock = z x sigma_demand x sqrt(lead time); Reorder point, order-up-to level; Newsvendor critical ratio for perishables; Respects MOQ, pack size, shelf capacity. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Supplier transparency model  `[Planned]`

Scores and predicts supplier behaviour.

- OTIF, fill rate, lead-time mean and variance
- ETA prediction for open POs
- Batch-to-supplier traceability for recalls and quality

> **Prompt seed:** Context: Smart Retail Intelligence Network > ML Core (the Network) > Supplier transparency model. Scores and predicts supplier behaviour. Known points: OTIF, fill rate, lead-time mean and variance; ETA prediction for open POs; Batch-to-supplier traceability for recalls and quality. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Peer comparison engine  `[Planned]`

Analyses weak products side-by-side with well-fulfilled ones.

- Find peers: same category, price band, pack size (k-NN on product embeddings)
- Compare demand, price ratio, facings, availability, promo exposure
- Explains the gap: price too high? not on shelf? supply short?
- Detects cannibalisation and substitution

> **Prompt seed:** Context: Smart Retail Intelligence Network > ML Core (the Network) > Peer comparison engine. Analyses weak products side-by-side with well-fulfilled ones. Known points: Find peers: same category, price band, pack size (k-NN on product embeddings); Compare demand, price ratio, facings, availability, promo exposure; Explains the gap: price too high? not on shelf? supply short?; Detects cannibalisation and substitution. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Explainability  `[Planned]`

Every prediction comes with reasons people trust.

- SHAP values per forecast
- Reason codes mapped to plain sentences (EN/HI)
- Elasticity and peer-gap charts

> **Prompt seed:** Context: Smart Retail Intelligence Network > ML Core (the Network) > Explainability. Every prediction comes with reasons people trust. Known points: SHAP values per forecast; Reason codes mapped to plain sentences (EN/HI); Elasticity and peer-gap charts. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## Decision & Action Engine  `[Planned]`

Converts model outputs into ranked, explainable actions with a human in the loop.

> **Prompt seed:** Build a rules-plus-optimisation action engine that turns forecasts, expiry risk, SKU health, spike alerts, supplier scores and peer gaps into a ranked daily action list (reorder, markdown, re-face, transfer, return, donate, delist). Each action has expected rupee impact, urgency, owner role, reason text, and approve/override capture for feedback.

### Action catalogue  `[Planned]`

What the system can recommend.

- Reorder qty / skip order
- Markdown % (ladder: 10 -> 25 -> 40)
- Move to eye level / endcap, add facings
- Bundle with a fast mover
- Transfer to another store
- Return to vendor
- Donate before expiry
- Delist SKU

> **Prompt seed:** Context: Smart Retail Intelligence Network > Decision & Action Engine > Action catalogue. What the system can recommend. Known points: Reorder qty / skip order; Markdown % (ladder: 10 -> 25 -> 40); Move to eye level / endcap, add facings; Bundle with a fast mover; Transfer to another store; Return to vendor; Donate before expiry; Delist SKU. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Prioritisation  `[Planned]`

Rank by money at stake x urgency x effort.

- Score = expected rupee impact x urgency weight / effort minutes
- Caps tasks per worker per shift

> **Prompt seed:** Context: Smart Retail Intelligence Network > Decision & Action Engine > Prioritisation. Rank by money at stake x urgency x effort. Known points: Score = expected rupee impact x urgency weight / effort minutes; Caps tasks per worker per shift. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Alerts  `[Planned]`

Push only what matters.

- Expiry in <= 3 days with risk > 60%
- Spike detected on A-class SKU
- Supplier ETA slip > 2 days
- Quiet hours and digest mode

> **Prompt seed:** Context: Smart Retail Intelligence Network > Decision & Action Engine > Alerts. Push only what matters. Known points: Expiry in <= 3 days with risk > 60%; Spike detected on A-class SKU; Supplier ETA slip > 2 days; Quiet hours and digest mode. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Human in the loop  `[Planned]`

People approve, edit or reject; the system learns.

- Override reasons captured as labels
- Track acceptance rate per action type

> **Prompt seed:** Context: Smart Retail Intelligence Network > Decision & Action Engine > Human in the loop. People approve, edit or reject; the system learns. Known points: Override reasons captured as labels; Track acceptance rate per action type. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Feedback loop  `[Planned]`

Outcomes flow back into training.

- Did the markdown sell through?
- Was the spike real?
- Retrain weekly with new labels

> **Prompt seed:** Context: Smart Retail Intelligence Network > Decision & Action Engine > Feedback loop. Outcomes flow back into training. Known points: Did the markdown sell through?; Was the spike real?; Retrain weekly with new labels. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## Interfaces  `[Planned]`

Where people meet the network.

> **Prompt seed:** Design the interfaces for a retail demand-intelligence system: manager web dashboard, worker mobile app (bilingual English/Hindi, barcode scan, voice, offline-first), supplier portal and a REST API. Give screen lists, key components, and the data each screen reads.

### Manager dashboard  `[Planned]`

Web app with KPIs, exceptions and approvals.

- Waste, stockouts, margin, forecast accuracy
- SKU health map, expiry calendar, spike feed

> **Prompt seed:** Context: Smart Retail Intelligence Network > Interfaces > Manager dashboard. Web app with KPIs, exceptions and approvals. Known points: Waste, stockouts, margin, forecast accuracy; SKU health map, expiry calendar, spike feed. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Worker mobile app  `[Planned]`

The humane front line.

- Shift task list ranked by impact
- Scan barcode -> see stock, expiry, action
- English / Hindi, voice prompts, large buttons, works offline

> **Prompt seed:** Context: Smart Retail Intelligence Network > Interfaces > Worker mobile app. The humane front line. Known points: Shift task list ranked by impact; Scan barcode -> see stock, expiry, action; English / Hindi, voice prompts, large buttons, works offline. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Supplier portal  `[Planned]`

Shared demand outlook and scorecards.

- Forecast for next 4 weeks, open POs, OTIF score

> **Prompt seed:** Context: Smart Retail Intelligence Network > Interfaces > Supplier portal. Shared demand outlook and scorecards. Known points: Forecast for next 4 weeks, open POs, OTIF score. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### REST API  `[Planned]`

For POS, ERP and other apps.

- FastAPI: /forecast, /sku-health, /expiry-risk, /price, /actions
- Auth: API keys + role-based access

> **Prompt seed:** Context: Smart Retail Intelligence Network > Interfaces > REST API. For POS, ERP and other apps. Known points: FastAPI: /forecast, /sku-health, /expiry-risk, /price, /actions; Auth: API keys + role-based access. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## Evaluation & KPIs  `[Partly built]`

How we know the models and the business are getting better.

> **Prompt seed:** Write an evaluation plan for a retail forecasting and action system: expanding-window back-tests, metrics (WAPE, MAE, RMSE, bias, pinball loss, interval coverage), per-segment reporting (ABC/XYZ, intermittency class), and business KPIs (waste %, stockout rate, sell-through, turns, GMROI, worker hours). Include an A/B or pilot-vs-control store design.

### Validation scheme  `[Built]`

No peeking into the future.

- Repo: TimeSeriesSplit expanding window, n_splits=3
- Planned: rolling-origin back-test per week, horizon 1-28 days

> **Prompt seed:** Context: Smart Retail Intelligence Network > Evaluation & KPIs > Validation scheme. No peeking into the future. Known points: Repo: TimeSeriesSplit expanding window, n_splits=3; Planned: rolling-origin back-test per week, horizon 1-28 days. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Model metrics  `[Partly built]`

Accuracy and calibration.

- Repo: WAPE, RMSE, MAE
- Planned: bias, pinball loss, P10-P90 coverage, MASE

> **Prompt seed:** Context: Smart Retail Intelligence Network > Evaluation & KPIs > Model metrics. Accuracy and calibration. Known points: Repo: WAPE, RMSE, MAE; Planned: bias, pinball loss, P10-P90 coverage, MASE. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Business KPIs  `[Planned]`

What owners care about.

- Waste / expiry write-off %
- Stockout rate, on-shelf availability
- Sell-through, inventory turns, GMROI
- Gross margin
- Worker hours on ordering

> **Prompt seed:** Context: Smart Retail Intelligence Network > Evaluation & KPIs > Business KPIs. What owners care about. Known points: Waste / expiry write-off %; Stockout rate, on-shelf availability; Sell-through, inventory turns, GMROI; Gross margin; Worker hours on ordering. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Pilot design  `[Planned]`

Prove impact causally.

- Matched pilot vs control stores
- 8-12 week run, difference-in-differences

> **Prompt seed:** Context: Smart Retail Intelligence Network > Evaluation & KPIs > Pilot design. Prove impact causally. Known points: Matched pilot vs control stores; 8-12 week run, difference-in-differences. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## Architecture & MLOps  `[Partly built]`

How the system is built, run and kept healthy.

> **Prompt seed:** Propose a lean production architecture for a retail ML system built in Python: ingestion (batch from POS/ERP), PostgreSQL + Parquet storage, feature pipeline, training with MLflow tracking, model registry, FastAPI serving, Prefect/Airflow scheduling, drift monitoring, pytest + GitHub Actions CI, Docker deployment. Include a folder layout that extends the existing src/ modules.

### Repo modules (today)  `[Built]`

What exists in the codebase.

- src/data_loader.py - ingestion, cleaning, zero-fill grid
- src/features.py - RetailFeatureEngineer
- src/forecasting.py - WAPE, time_series_cv_evaluate
- src/pricing_engine.py - demand_curve, optimize_price
- notebooks/02_eda_analysis.ipynb
- tests/test_pipeline.py

> **Prompt seed:** Context: Smart Retail Intelligence Network > Architecture & MLOps > Repo modules (today). What exists in the codebase. Known points: src/data_loader.py - ingestion, cleaning, zero-fill grid; src/features.py - RetailFeatureEngineer; src/forecasting.py - WAPE, time_series_cv_evaluate; src/pricing_engine.py - demand_curve, optimize_price; notebooks/02_eda_analysis.ipynb; tests/test_pipeline.py. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Pipelines  `[Planned]`

Daily batch flow.

- ingest -> validate -> grid -> features -> predict -> actions -> publish
- Weekly retrain, monthly elasticity refresh

> **Prompt seed:** Context: Smart Retail Intelligence Network > Architecture & MLOps > Pipelines. Daily batch flow. Known points: ingest -> validate -> grid -> features -> predict -> actions -> publish; Weekly retrain, monthly elasticity refresh. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Storage  `[Planned]`

Where data lives.

- PostgreSQL for ops tables
- Parquet / DuckDB for analytics
- Feature store (Feast) when multi-store

> **Prompt seed:** Context: Smart Retail Intelligence Network > Architecture & MLOps > Storage. Where data lives. Known points: PostgreSQL for ops tables; Parquet / DuckDB for analytics; Feature store (Feast) when multi-store. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Training & registry  `[Planned]`

Reproducible models.

- MLflow experiments + registry
- Seeds fixed (random_state=42 in repo)

> **Prompt seed:** Context: Smart Retail Intelligence Network > Architecture & MLOps > Training & registry. Reproducible models. Known points: MLflow experiments + registry; Seeds fixed (random_state=42 in repo). Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Serving  `[Planned]`

Predictions to apps.

- FastAPI + Docker
- Batch scoring job writes action table

> **Prompt seed:** Context: Smart Retail Intelligence Network > Architecture & MLOps > Serving. Predictions to apps. Known points: FastAPI + Docker; Batch scoring job writes action table. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Monitoring  `[Planned]`

Catch decay early.

- Data drift (PSI), forecast error drift
- Alert if WAPE worsens > 20% week on week

> **Prompt seed:** Context: Smart Retail Intelligence Network > Architecture & MLOps > Monitoring. Catch decay early. Known points: Data drift (PSI), forecast error drift; Alert if WAPE worsens > 20% week on week. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Testing & CI  `[Partly built]`

Trust every change.

- pytest unit + leakage tests
- GitHub Actions on push
- Notebook smoke test (tmp_validate_notebook.py)

> **Prompt seed:** Context: Smart Retail Intelligence Network > Architecture & MLOps > Testing & CI. Trust every change. Known points: pytest unit + leakage tests; GitHub Actions on push; Notebook smoke test (tmp_validate_notebook.py). Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## Current State & Gaps  `[Gap / fix]`

An honest read of the reference repo as uploaded.

> **Prompt seed:** Fix the Smart Retail Forecasting repo so tests pass: (1) add add_feature_columns(df) to src/features.py or update forecasting.py and tests to use RetailFeatureEngineer; (2) align test column names with the loader's item_id / quantity / unit_price / invoice_date schema; (3) make transform() merge temporal, lag and rolling features instead of overwriting them; (4) cast relative_price_ratio to float; (5) add an expiry-aware salvage term to optimize_price(); (6) add src/__init__.py and notebooks 01, 03, 04; (7) remove .venv from the archive and add a .gitignore. Show the diff and the passing pytest output.

### Works  `[Built]`

Solid foundations already in place.

- Robust cleaning and zero-filled grid
- Leakage-aware lags and rolling windows
- Tweedie GBM with time-series CV
- Bounded price optimiser with margin floor
- EDA notebook with 4 diagnostics

> **Prompt seed:** Context: Smart Retail Intelligence Network > Current State & Gaps > Works. Solid foundations already in place. Known points: Robust cleaning and zero-filled grid; Leakage-aware lags and rolling windows; Tweedie GBM with time-series CV; Bounded price optimiser with margin floor; EDA notebook with 4 diagnostics. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Broken import  `[Gap / fix]`

forecasting.py and tests import add_feature_columns, which features.py does not define, so pytest fails at collection.

> **Prompt seed:** Context: Smart Retail Intelligence Network > Current State & Gaps > Broken import. forecasting.py and tests import add_feature_columns, which features.py does not define, so pytest fails at collection. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Schema mismatch  `[Gap / fix]`

Tests expect StockCode / Quantity columns, but clean_transactions() renames them to item_id / quantity.

> **Prompt seed:** Context: Smart Retail Intelligence Network > Current State & Gaps > Schema mismatch. Tests expect StockCode / Quantity columns, but clean_transactions() renames them to item_id / quantity. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Missing pieces  `[Gap / fix]`

Things the README promises but the repo lacks.

- notebooks 01, 03, 04
- src/__init__.py
- raw data (UCI Online Retail II)
- no inventory, expiry or supplier data yet

> **Prompt seed:** Context: Smart Retail Intelligence Network > Current State & Gaps > Missing pieces. Things the README promises but the repo lacks. Known points: notebooks 01, 03, 04; src/__init__.py; raw data (UCI Online Retail II); no inventory, expiry or supplier data yet. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### transform() drops features  `[Gap / fix]`

RetailFeatureEngineer.transform() reassigns 'combined' from self.data at each step, so calendar and lag features are silently lost; only rolling features reach the output CSV.

- Fix: build temporal, lag and rolling frames from one base and merge on item_id + invoice_date
- Add a test asserting every LAG_WINDOWS column is present

> **Prompt seed:** Context: Smart Retail Intelligence Network > Current State & Gaps > transform() drops features. RetailFeatureEngineer.transform() reassigns 'combined' from self.data at each step, so calendar and lag features are silently lost; only rolling features reach the output CSV. Known points: Fix: build temporal, lag and rolling frames from one base and merge on item_id + invoice_date; Add a test asserting every LAG_WINDOWS column is present. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Object dtype bug  `[Gap / fix]`

relative_price_ratio becomes dtype object after replace(0, pd.NA), and LightGBM refuses to train on it.

- Fix: .astype(float) after fillna, or replace(0, np.nan)

> **Prompt seed:** Context: Smart Retail Intelligence Network > Current State & Gaps > Object dtype bug. relative_price_ratio becomes dtype object after replace(0, pd.NA), and LightGBM refuses to train on it. Known points: Fix: .astype(float) after fillna, or replace(0, np.nan). Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Pricing ignores expiry  `[Gap / fix]`

optimize_price() assumes unsold stock keeps its value, so for a batch expiring in 2 days it recommends raising the price.

- Fix: add a salvage value / waste cost term for units left at expiry
- Demo: Paneer 40 units, 2 days left -> repo says Rs 108; expiry-aware says Rs 58

> **Prompt seed:** Context: Smart Retail Intelligence Network > Current State & Gaps > Pricing ignores expiry. optimize_price() assumes unsold stock keeps its value, so for a batch expiring in 2 days it recommends raising the price. Known points: Fix: add a salvage value / waste cost term for units left at expiry; Demo: Paneer 40 units, 2 days left -> repo says Rs 108; expiry-aware says Rs 58. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Toy sample data  `[Gap / fix]`

processed CSV has 3 items x 30 days with 80% zeros, so EDA charts are flat; real data is needed before any conclusions.

> **Prompt seed:** Context: Smart Retail Intelligence Network > Current State & Gaps > Toy sample data. processed CSV has 3 items x 30 days with 80% zeros, so EDA charts are flat; real data is needed before any conclusions. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Housekeeping  `[Gap / fix]`

The archive ships a Windows .venv (thousands of files) and a tmp script.

> **Prompt seed:** Context: Smart Retail Intelligence Network > Current State & Gaps > Housekeeping. The archive ships a Windows .venv (thousands of files) and a tmp script. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## Roadmap  `[Planned]`

Phased build from today's repo to a store pilot.

> **Prompt seed:** Turn this roadmap into a sprint plan with tasks, owners (roles), estimates and acceptance criteria for a 4-person student team building a retail AI system over 16 weeks.

### Phase 0 - Fix foundation (wk 1-2)  `[Planned]`

Green tests, clean repo, real dataset loaded.

- Fix imports and schema
- Load UCI Online Retail II
- CI on GitHub

> **Prompt seed:** Context: Smart Retail Intelligence Network > Roadmap > Phase 0 - Fix foundation (wk 1-2). Green tests, clean repo, real dataset loaded. Known points: Fix imports and schema; Load UCI Online Retail II; CI on GitHub. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Phase 1 - Forecast MVP (wk 3-5)  `[Planned]`

Beat baselines with LightGBM Tweedie.

- Back-tests, WAPE by segment
- Quantile P10/P50/P90

> **Prompt seed:** Context: Smart Retail Intelligence Network > Roadmap > Phase 1 - Forecast MVP (wk 3-5). Beat baselines with LightGBM Tweedie. Known points: Back-tests, WAPE by segment; Quantile P10/P50/P90. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Phase 2 - Risk radar (wk 6-8)  `[Planned]`

SKU health, expiry risk, spike detector.

- Synthetic batch/expiry generator until real data
- Reason codes

> **Prompt seed:** Context: Smart Retail Intelligence Network > Roadmap > Phase 2 - Risk radar (wk 6-8). SKU health, expiry risk, spike detector. Known points: Synthetic batch/expiry generator until real data; Reason codes. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Phase 3 - Act (wk 9-11)  `[Planned]`

Pricing, markdown ladders, reorder engine, action list.

- Guardrails reviewed with a store owner

> **Prompt seed:** Context: Smart Retail Intelligence Network > Roadmap > Phase 3 - Act (wk 9-11). Pricing, markdown ladders, reorder engine, action list. Known points: Guardrails reviewed with a store owner. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Phase 4 - See the chain (wk 12-13)  `[Planned]`

Supplier scorecards and peer comparison.

- OTIF, lead-time model, k-NN peers

> **Prompt seed:** Context: Smart Retail Intelligence Network > Roadmap > Phase 4 - See the chain (wk 12-13). Supplier scorecards and peer comparison. Known points: OTIF, lead-time model, k-NN peers. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Phase 5 - Pilot (wk 14-16)  `[Planned]`

Dashboard + worker app in 1-2 real stores.

- Pilot vs control KPIs
- Collect worker feedback

> **Prompt seed:** Context: Smart Retail Intelligence Network > Roadmap > Phase 5 - Pilot (wk 14-16). Dashboard + worker app in 1-2 real stores. Known points: Pilot vs control KPIs; Collect worker feedback. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## Ethics, Risk & Humane Design  `[Planned]`

Guardrails so the system helps people and treats customers fairly.

> **Prompt seed:** Write the responsible-AI section for a retail demand-intelligence system: worker augmentation not surveillance, explainability, fair pricing guardrails for essentials, customer data privacy (DPDP Act 2023 in India), food donation over destruction, bias across store locations, cold-start and failure modes, and a human override policy.

### Augment, don't replace  `[Planned]`

Workers decide; the system suggests and explains.

- No individual productivity scoring from the app

> **Prompt seed:** Context: Smart Retail Intelligence Network > Ethics, Risk & Humane Design > Augment, don't replace. Workers decide; the system suggests and explains. Known points: No individual productivity scoring from the app. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Fair pricing  `[Planned]`

No surge pricing on essentials during spikes or emergencies.

- Hard ceiling on staples, audit log of price changes

> **Prompt seed:** Context: Smart Retail Intelligence Network > Ethics, Risk & Humane Design > Fair pricing. No surge pricing on essentials during spikes or emergencies. Known points: Hard ceiling on staples, audit log of price changes. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Privacy  `[Planned]`

Aggregate demand only; no customer PII needed.

- Align with India's DPDP Act 2023

> **Prompt seed:** Context: Smart Retail Intelligence Network > Ethics, Risk & Humane Design > Privacy. Aggregate demand only; no customer PII needed. Known points: Align with India's DPDP Act 2023. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Waste to worth  `[Planned]`

Prefer donation and transfer over destruction.

- Food-bank partner workflow for near-expiry items

> **Prompt seed:** Context: Smart Retail Intelligence Network > Ethics, Risk & Humane Design > Waste to worth. Prefer donation and transfer over destruction. Known points: Food-bank partner workflow for near-expiry items. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Failure modes  `[Planned]`

Know when not to trust the model.

- Cold-start SKUs use category priors
- Low-confidence flag hides auto-actions
- Fallback to last good model

> **Prompt seed:** Context: Smart Retail Intelligence Network > Ethics, Risk & Humane Design > Failure modes. Know when not to trust the model. Known points: Cold-start SKUs use category priors; Low-confidence flag hides auto-actions; Fallback to last good model. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

## Prompt Library  `[Planned]`

Ready-to-use prompts that reuse this map as context.

> **Prompt seed:** Using the Smart Retail Intelligence Network map as context, generate the next most valuable prompt for me based on the branch I name.

### Code prompt template  `[Planned]`

For implementation work.

- 'Using <node> context, implement <function> in src/<module>.py with type hints, docstring, pytest tests, no future leakage.'

> **Prompt seed:** Context: Smart Retail Intelligence Network > Prompt Library > Code prompt template. For implementation work. Known points: 'Using <node> context, implement <function> in src/<module>.py with type hints, docstring, pytest tests, no future leakage.'. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Analysis prompt template  `[Planned]`

For notebooks and EDA.

- 'Write notebook 0X that <goal>, using daily_demand_features.csv, with charts and a 3-line takeaway per chart.'

> **Prompt seed:** Context: Smart Retail Intelligence Network > Prompt Library > Analysis prompt template. For notebooks and EDA. Known points: 'Write notebook 0X that <goal>, using daily_demand_features.csv, with charts and a 3-line takeaway per chart.'. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Design prompt template  `[Planned]`

For UI / architecture.

- 'Design <screen/system> for <persona> showing <data>, bilingual, mobile-first.'

> **Prompt seed:** Context: Smart Retail Intelligence Network > Prompt Library > Design prompt template. For UI / architecture. Known points: 'Design <screen/system> for <persona> showing <data>, bilingual, mobile-first.'. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.

### Report prompt template  `[Planned]`

For writing.

- 'Write the <section> of the project report in formal academic tone, 400 words, cite the repo module names.'

> **Prompt seed:** Context: Smart Retail Intelligence Network > Prompt Library > Report prompt template. For writing. Known points: 'Write the <section> of the project report in formal academic tone, 400 words, cite the repo module names.'. Task: design and implement this part in Python for the existing repo (src/ modules, pandas, LightGBM, SciPy). Give the approach, data needed, code with type hints, pytest tests, and how its output feeds the action engine.
