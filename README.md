# Lunch Counter Demand Forecast

One-day-ahead demand prediction for the Siemens lunch counters — a frozen
LightGBM (Poisson) model with conformally calibrated P10–P90 intervals, a
SmartQ-branded Streamlit frontend, a monthly retraining pipeline, and a
shadow-mode evaluation harness for challenger models.

**Locked June 2026 internal test:** 6.06% counter-level WAPE · 3.54%
day-level WAPE (vs 26.4% for the incumbent 7-day moving average).
Methodology: [`HANDOFF.md`](HANDOFF.md) · ongoing research:
[`model_research/FINDINGS.md`](model_research/FINDINGS.md).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Build tomorrow's menu plan in the app (pre-filled with each counter's latest
menu) or upload a `plan.csv`/`.xlsx` — just Date, Counter, Item and Category;
**Day Type is derived automatically from the workbook's Holiday List sheet and
Panchangam is computed astronomically from the date** (validated against all
203 recorded days: 203/203 day types, 83/84 observances). Hit **Run
forecast** and get per-counter demand, the calibrated P10–P90 range, a
suggested order quantity and a risk level. The app currently runs in MVP mode (numbers only); set
`MVP_MODE = False` in `app.py` for the full tool (driver explanations, result
cards, charts, history explorer, model performance pages).

## The three workflows

**Daily forecasting** — keep the history workbook current through the last
served day, plan the next day, run the forecast, order the suggested
quantity. If yesterday's actuals aren't entered yet the app automatically
switches to a dedicated lag-2 fallback model (silently using stale lags
would cost ~2.1pt WAPE; the fallback costs ~0.5pt).

**Monthly retraining** — the learning curve says every extra month of data
helps, so retrain monthly:

```bash
python3 retrain.py               # trains everything, writes a report — no deploy
# review model_research/retrain_reports/retrain_<date>.md, then:
python3 retrain.py --promote     # deploys; previous models are backed up first
```

**Challenger evaluation (shadow mode)** — every forecast silently logs the
frozen model, the challenger and the blend to `shadow_log.csv`. At month end:

```bash
python3 model_research/shadow_eval.py
```

joins actuals, scores all three, bootstraps the difference and prints an
adopt / keep / extend verdict. Adopt a challenger only when the 95% CI
excludes zero — a 30-row month cannot adjudicate a half-point difference.

## Repository layout

```
app.py                        Streamlit entry point (page config + routing)
retrain.py                    monthly retraining pipeline (train → report → promote)
requirements.txt
Lunch_Master_Data_FINAL(cleaned).xlsx   history workbook (Aug 2025 – Jun 2026)
HANDOFF.md                    full modelling handoff / methodology

ui/                           Streamlit frontend package
  branding.py                 SmartQ brand tokens, CSS, chart chrome
  data.py                     paths, domain constants, cached loaders
  forecasting.py              plan scoring, lag-regime routing, shadow logging
  plan_input.py               plan upload + interactive menu builder
  results.py                  MVP results panel
  full_tool.py                full-mode pages (loaded only when MVP_MODE=False)

siemens_model_bundle/         the model itself
  features.py                 leakage-safe feature builder (frozen, audited)
  predict.py                  CLI scorer (frozen)
  evaluate.py                 metric harness (frozen)
  shadow.py                   k-shift feature builder + shadow/fallback loading
  auto_calendar.py            Day Type (holiday list) + Panchangam (ephem) derivation
  plan_template.csv           plan input format example
  artifacts/                  DEPLOYED boosters + conformal config
  artifacts_shadow/           challenger, blend member, lag-2 fallback set
  figs/                       EDA / diagnostic figures

model_research/               evidence behind every modelling decision
  FINDINGS.md                 research log: CV results, noise floor, sweeps
  harness.py                  expanding-window CV harness
  build_shadow_bundle.py      one-off builder for the current shadow set
  validate_auto_calendar.py   checks the calendar derivation against recorded labels
  shadow_eval.py              month-end shadow comparison
  retrain_reports/            one report per retraining run
```

## Ground rules encoded in this repo

- **Leakage safety**: every history feature is shift-protected — it uses only
  information available the evening before service. Verified by truncation-
  invariance tests; all shift-chain features are computed before any
  dataframe merge (merges reset the index and silently misalign groupby
  results — this bit us once, see FINDINGS.md).
- **The June test stays scored-once**: model selection happens on
  expanding-window CV over pre-June data; June was touched exactly once per
  candidate generation.
- **No silent deploys**: `retrain.py` never replaces the deployed models
  without `--promote`, and promote always backs up the previous deployment.
- **Shortage is an order-quantity decision, not a point-accuracy one**: the
  service-level table in FINDINGS.md prices shortage risk from the current
  P75 policy (30% shortage days) to zero-shortage (P90 + CQR + 5%).
```
