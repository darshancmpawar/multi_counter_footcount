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
