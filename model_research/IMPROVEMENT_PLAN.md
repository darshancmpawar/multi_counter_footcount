# Improvement plan — August 2026 (post-forward-test audit)

Produced by a full adversarial audit (code, data, features, decisions) after
the frozen model degraded 6.07% → 11.45% counter WAPE on 15 Jun – 9 Jul.

**Root cause (from the never-read workbook sheets + kiosk orderlog):** a
menu/caterer overhaul landed exactly on 15 Jun 2026 — 54 brand-new dishes
(Considerations G6), following a vendor transition (the two largest caterers
exited by May; orderlog vendor column) — and plates-per-head fell **−13.8%**
(not −9%: Apr 13/20/24 + Jul 1/2/3/6 have circularly-imputed headcount that
masks the drop, Considerations D1/G4). Festivals/Panchangam could never have
caught this: they are calendar features; the miss is a *behavioral propensity*
shift. That is why error got worse despite new calendar features and more data.

Evaluation protocol for everything below: **selection on the 6-fold expanding
CV** (4 stable folds + 2 shift folds; baseline 12.15 / 9.09 / 11.13 overall),
burned window 15 Jun–9 Jul reported but never selected on, **live data from
10 Jul onward stays untouched** as the only clean confirmation.

## Item 1 — Production correctness (bugs that change served numbers) ✅ criteria: gates green, both regimes smoke-tested
- [ ] 1a `requirements.txt`: add scikit-learn, joblib, pyarrow — a
      requirements-only install crashes every forecast; this exact gap
      already silently killed the ET/KNN shadow entrants for weeks.
- [ ] 1b `detect_lag_regime` holiday-aware — the first working day after any
      mid-week holiday is misrouted to the inferior lag-2 fallback (+wrong
      conformal margins) even though lag-1 is current. Verified repro.
- [ ] 1c `predict.py` → thin wrapper over `shadow.score_plan` — the CLI today
      is a divergent second implementation: no debias, no lag-2 fallback, no
      auto-calendar (Day Type silently 'Regular'), broken default history
      path; verified ~9% higher orders than the app on an identical plan.
- [ ] 1d `score_plan` stale-regime crash when artifacts_shadow absent →
      clear RuntimeError instead of TypeError.
- [ ] 1e Silent failures surfaced: `load_shadow` warns when an artifact
      exists but fails to load; `ui/forecasting` shadow-log failures get a
      toast + log instead of `pass`; `auto_calendar` distinguishes
      missing-sheet from read-error.

## Item 2 — Dead code / duplication removal ✅ criteria: no behavior change, gates green
- [ ] 2a One `wape()` (evaluate.py) imported everywhere (currently 5 copies);
      delete dead `features.design_matrix`, dead `evaluate.metrics` path.
- [ ] 2b Delete `model_research/build_shadow_bundle.py` — superseded by
      retrain.py, and re-running it would clobber the festival keys in
      meta.json.
- [ ] 2c Delete dead committed artifacts: model_q50.txt, featlist.pkl,
      design_meta.pkl, lgb_params.pkl (referenced by no code; promote()
      never refreshes them).
- [ ] 2d Cache model loading (boosters are re-read from disk on every
      forecast; the st.cache_resource loaders in ui/data.py are dead code);
      single CAT_LEVELS source; un-hardcode harness paths.
- [ ] 2e Single source for driver-explanation text (duplicated verbatim).
- [ ] 2f Make tests pytest-collectable; add unit tests: holiday regime case,
      debias gate on/off, plan-column filling.

## Item 3 — Data integrity for modeling
- [ ] 3a Flag the 7 imputed-headcount dates; exclude from any
      propensity/debias fitting; use −13.8% as the true drop.
- [ ] 3b Client asks: confirm the Jun-15 menu/caterer overhaul; obtain
      orderlog_2026_06/07.csv (orderlog ends 25 May — it MISSES the shift
      window) and Raw_Data.xlsx (Summary Report, 25 incident days, covers to
      2 Jul); sign off the 47 unapproved New Items Review rows.

## Item 4 — Error-% reduction experiments (the actual ask)
Ranked by expected value; each selected on the 6-fold CV, never on the
burned window. Combinations of individual winners tested as a final matrix.
- [ ] 4a **Menu-novelty features** (root-cause aligned, leakage-safe from the
      plan itself): per counter-day, share of planned items unseen at that
      counter in the prior 60/90 days; count of first-ever items; novelty ×
      is_veg. New-item share doubled at the shift (9.8%→15.2%) and correlates
      −0.22 with plates-per-head.
- [ ] 4b **Per-head propensity trend** (lagged, leakage-safe): rolling means
      and short-vs-long ratio of tlc_lag/hc_lag, imputed days masked — lets
      the model *learn* level shifts instead of relying on the debias layer.
- [ ] 4c **Recency weighting re-test** (half-life 30/60/120d sweep) — the old
      "within noise" rejection predates the regime shift.
- [ ] 4d **Seed-average the official model** (challengers are 3-seed, the
      official point model is single-seed — an inconsistency, and averaging
      is a known cheap variance cut).
- [ ] 4e **Pruning ablations**: 9 panchangam flags (EDA rejected the Ekadashi
      effect yet all flags stayed), month-start/end, star_rank, competitor
      tail — re-test "keep all 63" at n=681 with permutation importance first.
- [ ] 4f **Hyperparameter re-search** at n=681 (tuned at n=474; leaf count /
      lr / bagging may want more capacity now).
- [ ] 4g Combination matrix of winners → single final config.

## Item 5 — Deploy + monitor
- [ ] 5a Retrain with the winning config; conformal recalibrated excluding
      imputed days; `--promote` after report review.
- [ ] 5b Keep the debias layer armed post-promote (it auto-disarms when the
      gate sees balanced errors); add a weekly drift monitor (plates-per-head
      + rolling bias) so the next regime shift is caught in days, not 4 weeks.
- [ ] 5c Replace hardcoded June-2026 metrics in the UI with values generated
      at promote time; update FINDINGS.md; push.

## Standing decisions re-affirmed (challenged, kept)
- Direct counter-grain LightGBM-Poisson architecture: nothing in the new
  failure implicates it (the shape is still right; the level moved).
- Debias layer design (pooled, N=10, gate 0.8): CV-neutral, shift-effective;
  it correctly fired through the shift window.
- Festival entrant stays in shadow — the August festival season adjudicates.
