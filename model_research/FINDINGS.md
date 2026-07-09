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

---

# Round 3 — subcat verdict, shadow roster, calendar-gap fix (Jul 2026)

**Rule zero adopted: June 2026 is retired as an evaluation set.** The July
shadow month is the only referee from here on.

## Dataset v2 swapped in

Same target (cc verified identical), Day Type column now derived live from
the Holiday List (reproduces old labels 203/203), plus curated **Sub
Category** (75 groups) and Review Flag columns. Incumbent feature values
verified bit-identical after the swap.

## gap_next leak found and fixed

The data-derived gap_next "knew" about the two unplanned closures a day
early. Now computed from the calendar (weekends + Holiday List) — genuinely
known at T−1, defined at scoring time (the old version was NaN on the plan
date), and the truncation gate passes. Baseline moved 11.13 → 11.08.

## Round-3 candidates, re-validated on this harness

| Candidate | CV WAPE | Verdict |
|---|---|---|
| challenger + gap interactions (2) | **10.80** | confirmed, all-fold gain → in challenger4 |
| + subcat features (4 reconstructions tried) | 10.72–11.30 | NOT reproduced (claimed 10.43); best variant −0.08 = noise. Parked until the exact round-3 definitions are shareable; enters shadow only as their own pre-registered entrant |
| tuned ExtraTrees (msl=5, mf=0.6) | **10.70** | shadow entrant |
| tuned KNN (k=10, L1, distance, scaled) | 10.92 | hybrid member |
| hybrid 0.65·LGB + 0.35·KNN | **10.40** | shadow entrant (0.50 weight scored 10.35; kept 0.65 as pre-specified) |

## July shadow roster (frozen before actuals accumulate)

official (frozen incumbent) · challenger4 (LGB, +8 features, 3 seeds) ·
tuned ExtraTrees · hybrid LGB+KNN. Logged by the app and `shadow_run.py`;
`shadow_eval.py` scores counter+day WAPE with paired bootstrap and prints
adopt/keep verdicts. Lag-2 fallbacks rebuilt on the same pipeline.

## Kitchen benchmark (verified, now in evaluate.py)

WAPE(Ordered vs Consumed): **7.79% lifetime · 6.91% June**, June bias −40
plates, short on **93% of June days** — systematic under-ordering. This is
the business KPI line; the ordering fix is the quantile heads (order-policy
selector now in the app: P75 / P90+CQR / P90+CQR+5%).

## Guard rails added

`tests/test_leakage.py` (truncation invariance, 78 features × 2 cuts × 2 lag
regimes) runs standalone and gates every retrain. Official scoring path
regression-anchored: replaying a pre-round-3 input reproduces 1,341 total
exactly.

---

# Round 4 — subcategory play: reproduced, challenged, corrected (Jul 2026)

The round-3/4 playbook + exact subcat code (`subcat_features_exact.py`) were
re-run under the official regression-anchored harness. Verdict protocol from
the document followed exactly.

## The −0.65 vs −0.08 discrepancy is resolved: it was the subcat COLUMN

Running the *exact* code, 4-fold CV, both columns × both param sets:

| subcat source | my params (poisson) | champion params (L2) |
|---|---|---|
| curated 75-group "Sub Category" | base 10.79 → **+0.37 (hurts)** | 10.51 → +0.14 (hurts) |
| head-word derived (≥3 rule) | base 10.79 → **−0.49 (helps)** | 10.51 → −0.52 (helps) |

My round-3 reconstruction used the curated column → it genuinely hurt, so my
"−0.08 = noise" was correct *for that column*. The document's gain is real —
it lives in the **head-word derivation**, not the curated one. Both were right
about different objects. (Why crude head-word beats hand-curated: the last-
token grouping happens to separate demand signatures better than the curated
taxonomy; noted, not fully explained — the shadow month is the arbiter.)

## Challenge: is the gain leakage-safe? (the real-world gate)

- **As written the exact code LEAKS**: truncation invariance fails on all 6
  popularity/recency features. Cause (Challenge 3): the head-word "≥3 distinct
  items" grouping is recomputed on all visible data, so **12 items change
  subcat label retroactively** as history grows — a past row's feature value
  depends on the future.
- **Fix — freeze the mapping at train time.** With a train-only frozen
  head-word map: truncation invariance is **clean** at both cuts, and the
  honest per-fold CV gain is **−0.41** (fold deltas −0.30/−0.50/−0.64/−0.18).
  Only **−0.04** of the leaky −0.45 was the leak. The signal is genuine.

## Defect found in my own round-3 freeze

`challenger4` was frozen with the CURATED-column subcat features, which hurt
~0.1pt (CV 10.91 vs 10.80 for interactions-only). Since no July actuals exist
yet, correcting the pre-registered roster now is still pre-registration, not a
mid-month edit. Corrected: subcat features rebuilt from the **frozen head-word
mapping**; `build_cd_k` threads `subcat_map`; the map is stored in
`meta.json` and applied identically at score time. Leakage gate re-run green.

## Does this serve the goal (no shortage, no waste)?

Honest framing: this is a **point-forecast** gain (~0.4pt CV WAPE, deep in the
diminishing-returns zone above the 3.36% Poisson floor). Shortage/waste is a
**service-level** decision driven by which quantile you order at — the
order-policy selector, not the point model. The subcat play slightly tightens
the quantiles; it does not by itself move the shortage/waste frontier. It is
worth adopting *if* July shadow confirms it, but the money still lives in the
order quantile (P90+CQR+5% = 0 shortage days in June) and in new information
(day-ahead attendance), per Phase 9 of the playbook.

## Disposition

Head-word frozen-mapping subcat enters July as part of the corrected
`challenger4` (leakage-safe, CV −0.41 vs interaction-only base). Adopt for
August only on the shadow verdict with a bootstrap CI that excludes zero —
same rule as everything else. Curated-column subcat: parked (do-not-add
ledger), retest at higher n.
