# Lunch Counter Demand Forecast

One-day-ahead demand prediction for the Siemens lunch counters — a frozen
LightGBM (Poisson) model with conformally calibrated P10–P90 intervals, plus a
Streamlit frontend for planning, forecasting and exploring history.

**Locked June 2026 internal test:** 6.06% counter-level WAPE · 3.54% day-level
WAPE (vs 26.4% for the incumbent 7-day moving average). Full methodology in
[`HANDOFF.md`](HANDOFF.md).

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens on `http://localhost:8501` with four tabs:

| Tab | What it does |
|---|---|
| 🔮 **Forecast** | Build tomorrow's menu plan interactively (pre-filled with each counter's latest menu) or upload a `plan.csv`, then get per-counter predictions, calibrated ranges, suggested order quantities, risk levels and plain-language explanations. Results download as CSV. |
| 📊 **History explorer** | Daily consumption trends, weekday seasonality and counter shares with date/counter filters. |
| 📈 **Model performance** | Locked test metrics, baseline comparison and diagnostic figures. |
| ℹ️ **About the model** | How forecasts are made, what drives them, honest limitations, operating checklist. |

The bundled history workbook (`Lunch_Master_Data_FINAL(cleaned).xlsx`) loads by
default; upload a newer version from the sidebar to keep forecasts current —
every lag feature is rebuilt from it on each run.

## Command-line equivalent

```bash
cd siemens_model_bundle
python predict.py --plan plan.csv --history "../Lunch_Master_Data_FINAL(cleaned).xlsx"
```

## Repository layout

```
app.py                      Streamlit frontend
requirements.txt            Python dependencies
Lunch_Master_Data_FINAL(cleaned).xlsx   history workbook (Aug 2025 – Jun 2026)
HANDOFF.md                  full modelling handoff / methodology
siemens_model_bundle/
  features.py               leakage-safe feature builder (63 features)
  predict.py                CLI scorer
  evaluate.py               metric harness
  plan_template.csv         plan input format example
  artifacts/                frozen LightGBM boosters + conformal config
  figs/                     EDA / diagnostic figures
```
