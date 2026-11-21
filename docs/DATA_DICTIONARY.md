# Sample Data Pack — Data dictionary

All values are **synthetic** (simulated store, seed 7). Supplier names are invented. Prices are illustrative.


## skus
One row per product (14)

| Field | Meaning |
|---|---|
| `id` | slug key used by every other table |
| `name, category` | product name and category |
| `shelf_life_days` | days from receipt to expiry |
| `unit_cost, price` | rupees; illustrative, not market quotes |
| `elasticity` | price elasticity used in the log-log demand curve |
| `supplier_id` | link to suppliers |
| `health.cls` | Healthy | Watch | Expiry risk | Shelf gap | Dead |
| `health.score` | 0-100, higher is healthier |
| `health.cover_weeks` | on hand / weekly sales rate |
| `health.trend` | last 4 weeks sales / prior 4 weeks |
| `health.days_since_sale` | days since last non-zero sale |
| `health.on_hand` | units in stock |
| `health.daily_rate` | average units per day, last 28 days |
| `health.at_risk_units / at_risk_value` | expected unsold units at expiry and their cost |
| `health.reason` | plain-language reason code |

## series
Per SKU: last 90 days

| Field | Meaning |
|---|---|
| `dates[]` | ISO dates |
| `actual[]` | units sold per day |
| `p10[], p50[], p90[]` | forecast quantiles; null before forecast_start |
| `forecast_start` | first forecast date (28-day hold-out) |

## batches
Stock batches (16)

| Field | Meaning |
|---|---|
| `id, sku_id` | batch key and product |
| `qty` | units in batch |
| `days_left, expiry_date` | time to expiry |
| `expected_unsold` | FEFO Poisson estimate of units left at expiry |
| `value_at_risk` | expected_unsold x unit_cost |
| `p_expire` | probability some units of this batch expire |

## spikes
Flagged days (16)

| Field | Meaning |
|---|---|
| `sku_id, date` | product and day |
| `units` | units sold that day |
| `baseline` | rolling 28-day median before that day |
| `z` | robust z-score (MAD based) |
| `type` | festival | one-off |

## suppliers
Suppliers (7, fictional)

| Field | Meaning |
|---|---|
| `id, name, categories` | supplier key, invented name, what they supply |
| `otif` | on-time-in-full share |
| `fill_rate` | share of ordered units delivered |
| `lead_time_mean / sd` | days |
| `score` | 0.5 OTIF + 0.3 fill + 0.2 lead-time reliability, x100 |
| `open_pos[]` | id, sku_id, qty, eta, status |

## tasks
Ranked actions (8)

| Field | Meaning |
|---|---|
| `id, type` | markdown | shelf_check | move_front | return_vendor | fefo | reorder | donate |
| `sku_id, params` | product and template values |
| `impact_rs, urgency, minutes` | inputs to priority |
| `priority` | impact_rs x urgency / minutes |
| `role, status` | floor | manager; open | done | not_possible | snoozed |

## summary
KPIs for the overview

| Field | Meaning |
|---|---|
| `at_risk_value` | sum of batch value_at_risk |
| `counts` | SKUs per health class |
| `wape, wape_baseline` | hold-out error, model vs seasonal naive |
| `interval_coverage` | share of actuals inside P10-P90 |
| `open_tasks, active_spikes, late_pos` | counters |
| `sample_data` | true for this pack |
