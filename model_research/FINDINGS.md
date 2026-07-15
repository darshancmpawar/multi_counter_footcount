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

---

# Model improvement research — August 2026 round (post-July-forward-test)

Trigger: history extended to **9 Jul 2026** (19 new working days, 15 Jun –
9 Jul; the workbook now also carries a `Festival` column and a Holiday List
`Facility Status`/`Holiday Type` schema). Two questions: how is the frozen
model doing on genuinely unseen data, and can the new columns help?

## The July forward test — the model degraded, and the cause is in the data

Frozen incumbent, scored one-day-ahead exactly as the June report:

| Window | n | Counter WAPE | Day WAPE | Bias | Over% |
|---|---|---|---|---|---|
| June report (recap) | 30 | 6.07 | 3.54 | −4.7 | 50 |
| **15 Jun – 9 Jul (new)** | 57 | **11.45** | **9.95** | **+53.6** | **86** |
| — Jun 15–30 | 36 | 10.67 | 9.10 | +49.2 | |
| — Jul 1–9 | 21 | 12.80 | 11.43 | +61.2 | |

The error is one-sided (86% over-predictions, +54 plates/counter-day). Running
the pre-registered diagnosis ladder (HANDOFF §10: bias sign first) → a genuine
**demand level shift**: consumption per head dropped ~0.84 → ~0.73–0.75 (−11–13%)
starting the week of 15 Jun, while headcount *rose* +4.6%. People are in the
building but eating at these counters less — an external regime change, not a
model defect. Every weekday is down 5–12%.

Retraining does **not** fix it (the shift is inside the window, unseen at
train time): a policy-simulated retrain @12 Jun then @30 Jun scores 11.54%
counter / 8.95% day — statistically tied with frozen (bootstrap +0.09,
CI [−0.75, +0.92]). Guardrails held: P10–P90 coverage stayed 89% and the
P90+CQR order policy had **0 shortfall days** — the bias surfaced as
over-provision (waste), not shortage. Shadow roster on the window (all
ambiguous vs frozen, CIs include zero): challenger4 10.75, ExtraTrees 10.49,
hybrid 11.19.

## Change 1 — trailing level-shift corrector (adopted, gated, CV-neutral)

A multiplicative debias on the official point + quantiles: compare the model's
own predictions on the last N served days with actuals, scale by the trailing
actual/pred ratio. The design question was **how to help in a shift without
hurting in stable noise**. Swept post-hoc on stored predictions across the 4
stable CV folds (must stay neutral) and the shifted July/June windows (must
help):

| Config | CV mean | CV worst Δ | Jul(base) | Jul(fest) | Frozen 15Jun–9Jul |
|---|---|---|---|---|---|
| raw (no debias) | 12.15 | — | 12.18 | 10.92 | 11.45 |
| N=7 pooled, always-on | 12.33 | +0.65 | 9.16 | 8.61 | 8.35 |
| N=5 per-counter, always-on | 12.97 | +1.5 | 7.47 | 6.79 | 8.20 |
| **N=10 pooled, gate 80%** | **12.12** | **+0.18** | 9.16 | 8.61 | **8.15** |

Two design findings from iterating:
1. **Always-on hurts stable regimes** (+0.65pt CV worst fold); a **sign-gate**
   — apply only when ≥80% of the trailing days' *day-level* errors share one
   sign — makes it dormant in noise (fires 2–5/40 stable days at factors within
   ±6%) yet fully active in the shift (fires 15/19 frozen-window days at
   0.89–0.95). Net CV effect ≈ 0.
2. **Per-counter factors chase counter-level noise** (worst CV fold +1.5pt)
   despite a better shifted-window number — rejected on the same
   robustness-over-peak-performance principle the repo already applies to
   target encodings. **Pooled day-level factor** is the honest choice.

Adopted: `DEBIAS = {window 10, min_days 5, gate 0.80, clip ±15%}` in
`shadow.py`; recovers **11.45 → 8.15** counter WAPE (bias +54 → +9) on the
shifted window, ~0 effect on stable data. Conformal margins are left untouched
(the corrector scales the model output; the calibrated ±cqr / +corr are added
after). The raw forecast is logged alongside (`pred_official_raw`) so the
month-end referee scores the corrector as official-vs-raw. The MVP UI surfaces
a banner whenever the factor departs from 1.0. **This is a bridge, not a cure**
— the fix is a retrain once the new level is represented, plus a leakage-safe
attendance signal.

## Change 2 — festival features, pre-registered as a shadow entrant

The `Festival` column is 21/23 days redundant with the existing holiday-
adjacency flag (only Onam and Kartik Purnima are festivals on open days,
effect −2% / −6%, n=2). Its genuinely new signal is **severity**: the model
has one `dt_prev/next_holiday` flag for effects ranging −74% (pre-Christmas
shutdown) to −3% (Republic Day). Five leakage-safe flags derived from the
**Holiday List sheet** (not the `Festival` column — plans carry no Festival
column, so features must come from calendar data available at score time):
`fest_any`, `fest_op_today`, `adj_hol_{important,compulsory,shutdown}`.

Evidence under the repo's protocol:

| Test | Baseline | + Festival |
|---|---|---|
| 4-fold expanding CV (Jan–May, selection) | 12.15 | 12.18 (no gain) |
| Jul 1–9 one-shot (mean of 8 seed-triples) | 12.02 ±0.19 | **11.10 ±0.13** |
| Jul 1–9 paired bootstrap | — | **−1.26, CI [−1.96, −0.43]** |

The July gain is **indirect** — no festival fell in Jul 1–9; the flags absorb
the extreme festival days in *training*, cleaning up how correlated features
(panchangam, lags) are used on ordinary days. But CV over Jan–May shows nothing,
21 rows is below the repo's evidence bar, and the trailing-2-month validation
window used at build time (May 10 – Jul 9, festival-sparse) shows base=festival
to two decimals — this window simply can't discriminate the effect. **Not
adopted; pre-registered as a shadow entrant** (`festival_s*` + lag-2
`fest_fb_s*` in `artifacts_shadow`, incumbent architecture + the 5 flags,
built on history through 9 Jul, frozen official model untouched). The
decisive test is **mid-August onward** — Independence Day, Ganesh Chaturthi,
Dasara, Diwali — when the flags actually activate and n is large enough.
Adoption rule unchanged: fold into a retrain only if the shadow month's
bootstrap CI vs official excludes zero.

Honest confound noted in the build script: the entrant has ~2 more months of
data than the frozen official, so festival-vs-official conflates "festival
features help" with "more data helps". The clean adjudicator is the
identical-snapshot base-vs-festival A/B (the table above), re-run at August
scale — `model_research/add_festival_entrant.py` records both the entrant and
its matched baseline WAPE for exactly this reason.

## Disposition

- **Deploy the debias corrector now** (it is in the official scoring path,
  gated and CV-neutral). Retrain monthly regardless; the corrector is a bridge
  between retrains and an early-warning signal for regime change.
- **Festival entrant runs in shadow**; adjudicate on the August festival month.
- **Still the biggest levers** (unchanged from the July FINDINGS): more history,
  and a leakage-safe day-ahead attendance/headcount signal — the −11% per-head
  drop is exactly the variance an attendance feature would capture.
