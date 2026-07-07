# Model improvement research — July 2026 round

Goal: reduce predicted-vs-actual error further without overfitting. Every
candidate was selected on **4-fold expanding-window CV** (val windows
Jan–Feb / Feb–Mar / Mar–Apr / Apr–May; June 1–12 stayed locked until one
final scoring). Metric: counter-level WAPE, same harness as the bundle.

## Verification first

- Incumbent reproduced: frozen booster scores **6.06** on June (exact match);
  a fresh retrain of the same config lands 6.22–6.26 (seed noise ≈ ±0.3 on
  120-row windows — differences smaller than that are not evidence).
- Leakage audit passed before this round (truncation invariance at two cuts,
  lag alignment exact, no target-side features in the boosters).

## What was tried (CV mean WAPE across the 4 folds)

| Candidate | CV WAPE | Verdict |
|---|---|---|
| Incumbent (frozen config, 63 feats) | 11.81 | baseline |
| Objectives: L2 / Tweedie 1.1–1.5 / log-L2 | 11.69–12.09 | no gain — Poisson stands |
| Recency weighting (hl 60/120/240d) | 11.67–11.86 | within noise |
| Random search, 150 configs, top-8 re-seeded | **11.56** | real: smaller lr + bagging + tighter ff |
| + feature families (tested one at a time): trend/vol/share-lag/menu-comp/tlc-wd-lag | — | rejected (≥ baseline) |
| + lag2, wd_lag2, EWMA(3,7), **calendar gaps** | **11.32** | all 4 folds improve; gaps strongest (ablation +0.16 when dropped) |
| Re-tuned on new features | **11.13** | `nl=10, mcs=8, ff=0.35, rl=0.5, l1=0.5, lr=0.015, bf=0.8` |
| 3-config × 3-seed ensemble | 11.14 | no gain over best single — rejected on parsimony |
| DART / linear_tree / lagged headcount | 29.2 / 11.37 / 11.10 | rejected (last one −0.03 = noise; tlc lags already carry it) |

**Challenger** = Poisson LightGBM, 63 base features + `lag2, wd_lag2, ewm3,
ewm7, gap_prev, gap_next`, tuned config above, 3-seed averaged.
Beats the incumbent on **all four** CV folds (margins 0.36–1.03pt).

## The one-shot June test (n = 30, touched once)

| Model | June counter WAPE | June day WAPE |
|---|---|---|
| Frozen incumbent | 6.06 | 3.54 |
| Incumbent retrained (3-seed) | 6.26 | — |
| Challenger (3-seed) | 6.50 | 3.92 |

Paired bootstrap of the difference (challenger − frozen): **+0.44,
95% CI [−0.50, +1.36]**. June is too small to confirm or refute a ~0.5pt
improvement; the CV gain and the June point estimate disagree, and neither
is decisive at this sample size. Part of the 0.68pt CV gain is selection
pressure (~250 evaluations against the same folds), so the honest expected
gain is ~0.3–0.5pt.

## Decision

**Keep the frozen incumbent as the committed July model** (pre-registered,
its June anchor stands). **Run the challenger in shadow** through July:
score both daily, compare at month end on ~90 fresh counter-days — three
times June's evidence, zero selection pressure. Adopt the challenger for
August retraining only if it wins the shadow month.

## Structural ceiling and the real next steps

The learning curve says the model is data-limited; tuning is nearly
exhausted (150-config search moved it 0.25pt). The gains that remain live
outside the current feature set:

1. **More months of history** — retrain monthly; each month should be worth
   more than any tuning found here.
2. **A real holiday calendar** — `gap_prev/gap_next` (working-day gaps) was
   the strongest new feature; an explicit company holiday/event list would
   sharpen it further.
3. **Planned headcount signals** — leave approvals, floor occupancy or
   meeting-load forecasts from HR/facilities systems, if available a day
   ahead, would attack the single biggest error source (attendance swings).
4. **Menu-swing modelling** — biryani-day bias (−53 on validation) is still
   the largest structured residual; item-level popularity learned from more
   history (not target encoding — that was re-confirmed harmful) needs more
   data to work.

Reproduce: `python3 -c "from harness import *; ..."` — `harness.py` holds the
splits, CV folds, fitting and metric code used for every number above.

---

# Round 2 — accuracy target, algorithm sweep, lag fallback (Jul 2026)

## Is "3–5% WAPE or ≤30 pax" achievable? (critical evaluation)

- **Pure counting noise floor: 3.36% WAPE.** If demand were Poisson at the
  observed levels (~543 pax mean), a PERFECT model of the true rate would
  still score 3.36% — ±23 pax per counter-day is one sigma of pure chance.
  A 3% target at counter-day grain is below the physical floor. **4–5% is
  the theoretical best-case zone**; the frozen model's June median error is
  already exactly 30 pax (50% of rows within 30 pax).
- Empirical repeatability (same counter + weekday + menu tier, consecutive
  occurrences): actuals differ from each other by ~84 pax (≈11% single-obs
  bound including real signal). The process itself does not repeat to 3%.
- Conclusion: chase the remaining 6% → ~5% with data + shadow winner, but
  **shortage elimination must come from the ORDER QUANTITY, not the point
  forecast** (newsvendor logic).

## "No shortage at all" is purchasable today (June 2026, 30 counter-days)

| Order policy | Shortage days | Worst shortfall | Avg over-provision |
|---|---|---|---|
| Point forecast itself | 15/30 (50%) | 82 pax | 16 pax/day |
| P75 + corr (current suggested order) | 9/30 (30%) | 120 pax | 49 pax/day |
| P90 | 7/30 (23%) | 33 pax | 61 pax/day |
| P90 + CQR widening | 4/30 (13%) | 5 pax | 84 pax/day |
| **P90 + CQR + 5% buffer** | **0/30 (0%)** | **0 pax** | 119 pax/day |

Pick the row whose wastage cost is acceptable; no new model required.

## Full classical sweep (same 4-fold CV; deep learning not yet justified)

Ranked: LGB blend **10.96** < LGB challenger 11.13 < ExtraTrees 11.35 <
RandomForest 11.63 < LGB incumbent 11.81 < Lasso 11.96 < XGB-Poisson 12.02 ≈
seasonal-naive wd_roll4 12.02 < kNN 12.07 < HistGB 12.56 < CatBoost 12.99 <
seasonal-naive wd_lag1 14.30 < Ridge 14.92 < PoissonGLM 15.95 < SARIMAX
(1,0,1)(1,0,1,5) 33.2 ≈ ETS 34.5 < SARIMAX+exog (unstable) < SVR 41.1.

Pure time-series models fail because the signal is in the menu + lag
covariates, which they cannot exploit. With 474–654 training rows, deep
learning has no case yet (Tier-3 logic re-confirmed; revisit at ~2+ years of
data). New best: **0.7·LGB-challenger + 0.3·ExtraTrees = 10.96** (beats LGB
alone on all 4 folds) — added to the shadow run.

## Lag-1 availability (order-time reality check)

Simulated "yesterday's actuals missing" on all 4 CV folds:

| Scenario | CV WAPE | vs normal |
|---|---|---|
| Lag-1 model, yesterday available | 11.13 | — |
| Lag-1 model silently fed 2-day-old lags | 13.27 | **+2.13** |
| Dedicated model trained on 2-day lags | 11.60 | **+0.47** |

→ Built a dedicated **lag-2 fallback set** (point + q10/q75/q90 + its own
conformal corrections: CQR ±14.2, order +1.9). The app auto-detects
staleness (business-day gap from last history date to plan date > 1) and
routes official numbers to the fallback with a visible notice.

## Correctness incident (logged deliberately)

The first lag-fallback experiment produced wrong numbers: extra lag columns
were computed AFTER a dataframe merge that reset the index, misaligning
groupby shifts. Caught by diffing the two pipelines column-by-column;
verified the main pipeline (all Round-1/Round-2 headline numbers) was clean;
re-ran the experiment fixed. Rule now encoded in shadow.py: all shift-chain
features are computed on one frame before any merge.

## Shadow mode (built)

- `siemens_model_bundle/artifacts_shadow/` — frozen challenger (3 seeds),
  ExtraTrees blend member, lag-2 fallbacks, meta.json.
- Every app forecast silently logs official / challenger / blend to
  `shadow_log.csv` (gitignored; deduped on Date+Counter).
- Month-end: `python3 model_research/shadow_eval.py` joins actuals, scores
  all three, bootstraps the difference and prints an adopt/keep/extend
  verdict. Decision rule: adopt only if the 95% CI excludes zero.
