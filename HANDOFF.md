# Siemens Lunch Counter Demand Prediction — Final Handoff

**Model:** LightGBM, Poisson objective, direct counter-day prediction · **Frozen:** trained on 1 Aug 2025 – 31 May 2026 · **Internal test:** June 2026, scored once · **Next:** July 2026 live forward test

---

## 1. Executive summary

The final model predicts Counter Consumed at the Date + Counter grain one day ahead of service, using only information available at vendor-ordering time. On the locked June 2026 internal test it achieved **6.06% counter-level WAPE** and **3.54% day-level WAPE** (sum of counter predictions vs actual total), against the current moving-average practice whose counter-level proxy scored 26.4% and whose business-reported error is ±10–12% at facility level. The first business target (±6–8%) is met on the internal test; the July live run is the real confirmation.

| Split | MAE | RMSE | MAPE% | WAPE% | Bias | Over% | Under% |
|---|---|---|---|---|---|---|---|
| Train (Aug–Mar) | 33.3 | 42.0 | 7.4 | 6.28 | −0.1 | 47 | 53 |
| Validation (Apr–May) | 52.1 | 65.2 | 10.2 | 9.10 | +6.0 | 53 | 48 |
| **June test (once)** | **37.3** | **46.3** | **6.1** | **6.06** | **−4.8** | **50** | **50** |

Reference points on the same June window: same-weekday rolling-4 baseline 7.70% WAPE; 7-day moving average 26.4%. The June set is only 30 counter-days, so treat the 6.06% as an encouraging point estimate, not a guarantee — validation's 9.10% over 120 rows is the more conservative expectation for July.

Calibrated uncertainty: conformalized P10–P90 intervals achieved 87% June coverage (80% target); the suggested-order quantile (calibrated P75) covered demand on 70% of June counter-days with mean over-provision of 49 plates.

## 2. Data facts and handling rules (Phase 1 log)

One raw row = Date + Counter + Item (5,595 rows, zero duplicates), collapsed to 624 counter-days across 203 working days. Invariants verified: Counter Consumed and Counter Ordered constant within every counter-day; Total Lunch Consumed equals the sum of counter consumption on every date. Handling rules applied and logged: (1) North Non Veg ran only 18 days (Aug 2025) — retained in training for redistribution signal, but treated as inactive at scoring time unless planned; (2) Receiving Qty / Bainmarie Wastage missing 11% including the entire 12 May–12 Jun window — reference-only columns, never features, never imputed as zero; (3) one row with wastage > received and four days with consumption > headcount — logged as source-data quirks in reference columns, no action since they never enter the feature set; (4) two working days absent without a holiday-list entry (8 Oct 2025, 12 Mar 2026) — treated as unplanned closures, no rows fabricated; (5) three imputed headcount dates are irrelevant because headcount is excluded as a feature by the leakage rule.

## 3. What the EDA established (Phase 2)

Weekly seasonality is the dominant structure: Wednesday mean 677 vs Friday 363 per counter, and same-weekday autocorrelation (0.565 at lag 5) exceeds next-day autocorrelation (0.384). This mirrors the ridership pattern in Géron ch. 15, where the seasonal-naive difference is the correct baseline transform. Holiday-adjacent days drop roughly 130–150 pax per counter. Headcount correlates 0.966 with total consumption but is unavailable at ordering time, so calendar features must carry that signal by proxy. South Non Veg is the volatile counter (share σ = 0.109 vs ≈0.06 for veg counters) and menu composition explains it: in train data, non-veg biryani days lift SNV from 502 to 742 mean pax while draining South Veg from 600 to 437 — cannibalization is real and became the competitor-feature family. The Ekadashi fasting hypothesis was tested and rejected (554 vs 553 on non-veg counters).

## 4. Splits (Phase 3)

Chronological throughout: Train 1 Aug 2025 – 31 Mar 2026 (474 rows / 153 days), Validation 1 Apr – 31 May 2026 (120 / 40), Internal test 1–12 Jun 2026 (30 / 10, locked until the end). Note the master prompt stated coverage to 30 Jun; the file ends 12 Jun, so the test window is smaller than designed. Every lag/rolling feature passed a truncation-invariance audit: rebuilding features on data cut at 1 Mar 2026 reproduces the full-build values exactly, proving no future information enters any historical feature. The one standing assumption: the previous working day's consumption is known by ordering time (order placed the evening before service, after lunch closes) — confirmed by your brief listing "previous counter consumed" as valid.

## 5. Escalation ladder — evidence at each gate (Phases 4, 6, 7)

Tier 0 established the bar on validation: per-counter mean 23.8% WAPE, 3-day moving average 33.3%, 7-day 27.4%, same-weekday-last-week 12.3%, same-weekday rolling-4 10.9%. The naive moving average — the incumbent business method — is the *worst* reasonable baseline at counter level because it smears the weekday cycle.

Tier 1 (Ridge/Lasso/Poisson GLM on 63 features + one-hots): best was Lasso (α=3) at 10.13%, marginal over the smart baseline. Residuals showed unambiguous structure: +89 mean residual on non-veg-biryani days, counter-specific biases, residual–prediction correlation 0.48, and |residual| growing with prediction (funnel, correlation 0.46). Per the protocol, Tier 1 was given a fair chance via explicit interaction features (counter × biryani, counter × weekday, counter × star, star × level): WAPE improved to 9.49% but the biryani residual (+88) and the funnel persisted — the effects are multiplicative and conditional, beyond what a fixed linear basis captures. **Decision gate: a statistical model cannot fully solve this problem; escalate to Tier 2.** Tier 2 (tuned LightGBM-Poisson) reached 9.10% and cut the biryani residual to +41 and residual–prediction correlation to 0.24. Tier 3 was not entered: two categoricals of cardinality 4 and 5 give embeddings nothing to learn that one-hot/native categorical splits don't already capture, and 474 training rows cannot feed a neural network past a regularized GBM — the learning curve (below) shows the model is data-limited, which deep learning worsens, not fixes.

Learning curve (Phase 6): validation WAPE falls from 11.05% at 142 training rows to 9.10% at 474 with train WAPE in the 6–9% band — a modest, healthy gap. The model sits in the "more data will help" regime; every month of accumulated history should improve July-onward accuracy. Retrain monthly.

Phase 7 iteration was residual-driven and two candidate feature families were **rejected on evidence**: expanding target-encodings per counter×biryani / counter×star-bucket / counter×weekday degraded validation to 9.74% (they duplicate what the trees learn from the raw flags while adding noisy early-window estimates), and permutation-based pruning of 28 zero-importance features degraded to 9.32% (LightGBM's own feature_fraction sampling already performs soft selection; hard removal reduced useful ensemble diversity). The full 63-feature set with native categoricals stands. Top validated drivers by permutation importance: wd_roll4 (same-weekday rolling-4 mean, dominant), wd_lag1, wd_share_roll4, dt_prev_holiday, menu_strength, lag1, weekday, star_minus_oth, tlc_lag1.

## 6. Loss and metric choices with mechanisms (Phase 5)

**Training objective — Poisson deviance.** Like-for-like comparison on the identical split: L2 10.32%, L1 11.15%, Huber 18.69% (failed to converge sensibly at α=2), Poisson 10.78%, Tweedie(1.2) 10.77% at base settings; after tuning, Poisson won at 9.10% and was retained. Mechanism: LightGBM fits log-intensity F(x), predicting exp(F); the gradient of the Poisson negative log-likelihood is exp(F) − y and the hessian exp(F). This makes the per-leaf Newton step (Σgradient/Σhessian) scale errors *relative to the predicted level* — a 50-plate miss at level 200 outweighs the same miss at level 1,100 — exactly matching the funnel heteroscedasticity found in Tier 1 residuals and the WAPE business metric's relative-error character. It also guarantees positive predictions, correct for count data. This choice aligns with practitioner literature on retail GBTs, which evaluates LightGBM with Poisson/Tweedie losses as the standard for count-like demand. Géron anchor: gradient boosting as sequential residual-correction, ch. 7; loss-gradient reasoning, ch. 4.

**Uncertainty — quantile (pinball) loss + conformal calibration.** Separate LightGBM models at α = 0.10, 0.75, 0.90. Mechanism: pinball loss penalizes under-prediction by α and over-prediction by (1−α) per unit, so its minimizer is the conditional α-quantile — the direct answer to "how many plates cover demand α% of the time," which encodes the business asymmetry (shortage penalties + reputation vs wastage) without distorting the point forecast. Raw quantile GBMs under-covered on small data (64% for a nominal 80% band), so conformalized quantile regression was applied: the validation conformity score max(q̂₁₀−y, y−q̂₉₀) at its 80th percentile widens the band by ±28 plates, restoring exact 80% validation coverage and delivering 87% on June. The order quantile received a one-sided +7-plate correction to hit 75% validation service rate. Alternatives considered: asymmetric MSE (requires an arbitrary cost ratio and distorts the point model — rejected), fixed percentage buffer (ignores per-day uncertainty; the quantile spread is wider on biryani days, which is exactly when shortage risk peaks — rejected).

**Evaluation metric — WAPE primary**, with MAE/RMSE/MAPE/bias/over-under% reported alongside every model in every phase. WAPE = Σ|error|/Σactual weights errors by volume, is stable when small counters would explode MAPE, and translates directly to "plates wrong per plates served" — the business's cost driver.

## 7. Phase 8 — challenger round (all judged on identical split and metric)

XGBoost-Poisson 9.65% and CatBoost-Poisson 9.95% versus incumbent 9.10% — mechanism-level note: XGBoost's level-wise growth and CatBoost's symmetric/loss-guided trees are less parameter-efficient here than LightGBM's leaf-wise growth constrained to 7 leaves, which concentrates depth where the loss reduction is (essentially learning the few dominant interactions) on a small sample. The **two-model total × share architecture (your Phase-2 preference) was tested honestly and lost at counter grain: 11.14%** despite the total-day model being excellent alone (7.02% day-level WAPE). The failure mode is instructive: normalizing predicted shares within a day couples every counter's error to every other's, and the share model smooths exactly the menu-driven share swings the direct model captures. Verdict: direct counter model for ordering; day-level totals from summing counter predictions are *better* (3.96% validation, 3.54% June) than the dedicated total model, so hierarchical consistency comes free. A 30/70 Lasso–LGB blend tied at 9.10% and was rejected on parsimony. These results echo the published finding that direct/bottom-up GBTs are competitive with top-down schemes; adoption rule was "beat the incumbent," and nothing did.

## 8. Decision log

| Choice | Alternatives considered | Evidence / reasoning | Mechanism (1–3 lines) |
|---|---|---|---|
| Grain: counter-day aggregation | Train on item rows | Target repeats per item → pseudo-replication inflates n and biases toward big menus | One row per Date+Counter; menu becomes features |
| Lag-1 features allowed | Lag-2-only (conservative) | Order placed evening before service, after lunch closes; brief lists prev-day consumed as valid | shift(1) within counter, truncation-audited |
| Median imputation of early-window lag NaNs (linear only) | Drop rows; zero-fill | Preserves 474-row sample; LightGBM handles NaN natively so final model needs none | Train-median fill, fitted on train only |
| Tier 1 → Tier 2 escalation | Stay linear + interactions | Interactions: 9.49% but biryani residual +88 and funnel persisted | Conditional multiplicative effects need adaptive partitioning |
| No Tier 3 | Embedding MLP | Cardinality 4–5 categoricals; 474 rows; learning curve shows data-limited regime | Embeddings need high cardinality to beat one-hot trees |
| LightGBM (implementation) | XGBoost 9.65, CatBoost 9.95 | Won on val 9.10; fastest to iterate | Leaf-wise growth, native categoricals, NaN-aware splits |
| Poisson objective | L2 10.32, L1 11.15, Huber 18.69, Tweedie 10.77 (tuned Poisson 9.10) | Best val WAPE; matches count target + funnel residuals | grad=exp(F)−y scales error relative to level |
| Hyperparams: 7 leaves, min_child 10, ff 0.6, λ=1, lr 0.03, early stop | 90-config grid | Best of grid on val; small leaves = strong regularizer at n=474 | Capacity control via structure, not depth |
| Reject expanding target-encodings | Keep (9.74%) | Degraded val by 0.64pt vs 9.10 | Duplicates tree-learnable signal; noisy early estimates |
| Reject permutation pruning | Keep 41 feats (9.32%) | Degraded val by 0.22pt | feature_fraction already soft-selects; hard removal cuts diversity |
| Direct counter model | Total×share (11.14%), renorm hybrid (10.84%), blend (tie) | Direct wins at counter grain; summed counters beat total model at day grain too | Share normalization couples errors across counters |
| Quantile+CQR intervals | Raw quantiles (64% cov.), asymmetric MSE, fixed % buffer | CQR restores exact 80% val coverage; 87% on June | Pinball minimizer = conditional quantile; conformal widening ±28 |
| P75 (+7) as suggested order | P80/P90; fixed buffer | 75% val service rate ≈ business's current posture with less over-provision | One-sided conformal correction on order quantile |
| WAPE primary metric | MAPE (unstable at small counters), RMSE (scale-dominated) | Volume-weighted; matches plates-wrong cost | Σ\|e\|/Σy |

## 9. Feature list (63 numeric + 2 categorical)

Calendar: month, day-of-month, week number, is_monday, is_friday, month start/end flags, previous/next-day-of-holiday flags, eight Panchangam observance flags plus pan_any. Counter identity: Counter Name (categorical), is_nonveg_counter, is_south. Menu (from planned items, known in advance): n_items, n_categories, category flags (dessert/rice/bread/gravy/dry), star keyword flags (biryani, nv_biryani, mutton, fish, chicken, egg, paneer, mushroom, chole, indo-chinese), rule-based star_score (max pull) and menu_strength (summed pull). Active-counter plan: n_active, n_active_veg, n_active_nonveg. Competitor (other active counters same day): oth_has_{biryani, nv_biryani, paneer, mutton, chicken, fish}, oth_max_star, star_minus_oth, star_rank. History (all shift(1)-protected): lag1, roll3/7/14, roll7_std, same-weekday lag and rolling-4, share_roll7, weekday share rolling-4, and day-level tlc_lag1, tlc_roll5, tlc_roll10, tlc_wd_roll4. Excluded by leakage rule: Headcount, Counter Ordered, Total Lunch Consumed (same-day), Receiving Qty, wastage columns.

## 10. Limitations and July live-test protocol

Honest caveats: (a) June's 6.06% comes from 30 rows — expect July closer to the 8–10% validation band; (b) intervals are calibrated on Apr–May exchangeability, so a regime change (new counter, client policy shift) breaks coverage until recalibration; (c) star scoring is keyword-rule-based — genuinely novel dishes get conservative scores; (d) North Non Veg predictions rest on 18 historical days — treat as low confidence if reactivated; (e) two-day-ahead ordering would invalidate lag1/roll3 and needs a retrained variant.

To run July: keep the history file current through the last served day, prepare `plan.csv` (one row per planned item; only active counters; columns Date, Counter Name, Item Name, Category, optional Day Type and Panchangam), then `python predict.py --plan plan.csv --history <file>`. Output prints per-counter prediction, calibrated range, suggested order, risk, and explanation, and writes predictions_out.csv. Score the month with evaluate.metrics against actuals and return the table — the pre-registered diagnosis plan for any train-live gap is: check bias sign first (level shift → recalibrate intervals + retrain), then weekday residual pattern (attendance regime change), then biryani-day residuals (menu drift → extend star rules).

## 11. File manifest

`features.py` (feature builder, leakage-audited) · `evaluate.py` (metric harness) · `predict.py` (July scoring CLI) · `plan_template.csv` (input format example) · `artifacts/model_point.txt`, `model_q10/75/90.txt` (frozen LightGBM boosters) · `artifacts/final_config.pkl` (hyperparams, iterations, conformal corrections) · `figs/` (EDA, learning curve, prediction-vs-actual plots).

---

## Appendix: prediction-time information contract (added July 2026)

Known the evening before service (T−1), usable as features:
calendar (weekday, month, gaps to adjacent working days), planned menu and
active-counter plan (incl. Sub Category), Day Type derived from the workbook's
Holiday List sheet, Panchangam computed astronomically (ephem), and all
actuals through the previous working day (consumption, day totals) as
shift-protected lags.

NOT known at T−1, never same-day features: Headcount, Total Lunch Consumed,
Counter Ordered, Receiving Qty, Bainmarie Wastage. These enter only as lags.
When even yesterday's actuals are missing at ordering time, the dedicated
lag-2 fallback models take over (tests/test_leakage.py guards both regimes).
