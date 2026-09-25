# Deep Research: Closing 0.94563 → 0.94600+ on Kaggle S6E9 (EV Purchases)
> Generated 2026-09-24 | Depth: standard (condensed to serve one decision) | Sources: 21

## TL;DR

Our gap to the 0.94638 public solution is **not** in the blend, the bin resolution, or
post-processing — it is in **feature breadth**: that solution target-encodes *every*
column at three smoothing levels, adds population frequency encoding, uses the external
dataset as real-world target-mean *anchors* rather than extra training rows, and trains one
heavily-regularised LightGBM for 20,000 trees. Its CV (0.94607) sits 0.00056 above our best
single member (0.94557), so the deficit is measurable before we ever submit. Meanwhile two
things we currently do are worth nothing: the `lexsort` tie-break (ceiling +0.00000008,
measured) and blending near-duplicate GBDT members (+0.0000033 at rank corr 0.9887).

## Executive Summary

1. **Replication beats tuning here.** The verified 0.94638 recipe is a *single* LightGBM,
   CPU-only, 1,167 seconds. We are running 2.5-hour five-model ensembles and losing to it.
   [20]

2. **The transferable ingredients**, in descending expected value: (a) triple
   `TargetEncoder` (`smooth='auto'`/10/100) applied to all categoricals *and* numerics
   stringified *and* digit columns *and* multi-scale income/commute bins; (b) global
   frequency encoding computed over the combined train+test population; (c) `{col}_org_mean`
   anchors — per-column target means from the real 10k dataset mapped onto synthetic rows,
   which is genuinely external information rather than more synthetic rows; (d) model config
   `num_leaves=32, max_depth=5, lr=0.02, colsample_bytree=0.303, min_child_samples=10,
   n_estimators=20000, early_stopping(500), max_bin=1024, feature_pre_filter=False`. [20][9][1]

3. **Bin resolution is a dead end at our scale.** Defaults are 255 (LightGBM), 256
   (XGBoost), 254 (CatBoost CPU / 128 GPU), and sklearn's HistGB is hard-capped at 255.
   Official guidance is deliberately bidirectional — LightGBM lists "use large max_bin"
   under *For Better Accuracy* **and** "use small max_bin" under *Deal with
   Over-fitting*. No Tier-1 or Tier-2 source publishes a measured gain at 4096/8192, and
   `min_data_in_bin=3` plus `bin_construct_sample_cnt=200000` mean requested bins often
   silently fail to materialise. The verified top solution uses `max_bin=1024` — which we
   already use. [1][2][3][5][6][8]

4. **Exact-value target encoding is legitimate, not a mirage** — this reverses an earlier
   belief of ours. Cross-fitting exists to stop *spurious* high-cardinality leakage; when
   train and test share one generator, exact-key statistics are informative on both sides,
   so CV and LB move together. Our own LB history is monotone in smoothing (2/10 → 0.94540,
   10/50 → 0.94503, removed → 0.94452), and a Playground 3rd-place solution used TE with **no
   smoothing at all** across ~270 features. [9][11][12]

5. **Stop trying to measure 1e-4 effects on the leaderboard.** With ~17.5% positives, the
   Hanley–McNeil standard error of AUC is ≈0.0013 on a 30% public split. Our v7 → v7.2
   "improvement" of +0.00002 is roughly 1.5% of one standard error — pure noise. Only
   changes worth ≥0.0005 are submission-verifiable. [50]

6. **Two free corrections.** Our clamped bands currently tie 4,935 rows at exactly 0.0 and
   254 at exactly 1.0; epsilon-ordering *inside* each band is free upside (up to ~0.0005 if
   a band hides positives) with no downside if the rule holds. And 6-decimal rounding created
   ~23k avoidable tied rows in our last submission. [46][50]

## 1. Status Quo — what the leaders actually do [Confidence: High]

The only top solution whose code could be retrieved in full is a single LightGBM at
CV 0.94607 / LB 0.94638 [20]. Its pipeline: concatenate train+test, drop
`Number_of_Cars_Owned`, digit-decompose every numeric over `k in range(-4, 4)`, frequency-
encode every categorical over the *combined* population, add four CTGAN-fingerprint flags
(`is_30k_spike`, `is_millionaire_cliff` ≥ 170,537, `is_dead_zone` 38k–42k, `is_env_hater`
env == 1), add multi-scale string bins (`income_exact_int`, `income100_floor`,
`income1000_floor`, `commute_integer`), map real-world per-column target means from the
external dataset, drop constants and perfectly-correlated columns, then target-encode
**everything string-like** with three encoders (`auto`, 10, 100) fitted per fold. 5-fold
StratifiedKFold, seed 42, test = mean of folds. No stacking, no tie-breaking, no clamping.

Its author's own ablation is the most useful single table we found: dual TE + digits + flags
= CV 0.94587 / LB 0.94612; adding multi-scale bins and the `smooth=100` encoder = CV 0.94607 /
LB 0.94638. Both CV and LB rose together by ~0.0002 — evidence that at this feature
granularity CV is a working proxy.

Everything we could identify above ~0.9464 is rank-averaging of these same public recipes
[22][23][24], which our own diversity audit says yields ~3e-6 when members correlate at
0.9887 [47].

## 2. Emerging Trends in this competition [Confidence: Medium]

Three recurring primitives across the credited threads [25][26][27][28][29]: CTGAN
artefacts as features (mode-collapse spike at \$30k, the millionaire cliff, the 5 km commute
sentinel cluster); *transductive* encoding — frequency and target statistics computed over
train+test together; and treating the original real dataset as a source of **priors**, not
**rows** (logistic regression on the real 10k recovers a smooth curve the synthetic data
jitters around [28]). Simpson's paradox between home and public charging is cited as the
reason deep interaction trees beat shallow ones [29].

Tier-2 caution: these are single-competition observations, not general methodology.

## 3. Critical Assessment [Confidence: High]

**What will not work.** (i) More ensemble members of the same family — measured at 3e-6
[47]; our own five-member blend already scores *below* its best member on OOF (0.94551 vs
lgbm_a 0.94557). (ii) Tie-breaking model scores — ceiling 8e-8 [46]. (iii) Chasing
`max_bin` upward — no credible source documents a gain, and the mechanism cuts both ways
[1][2]. (iv) Pseudo-labelling — already tried, LB 0.94505, hurt. (v) OOF-fitted blend weights
— structurally biased because OOF rows carry one model's noise while test rows carry the
fold-averaged noise, and the S6E2 winner reached the same conclusion ("Ridge worked best
because it is simple and stable") after screening ~150 candidates down to ~15 [40].

**Asymmetric risk in our own post-processing.** The hard bands are binomial-filtered at
p < 1e-6 and contradict zero of 668,665 training labels, so they are real. But pinning 4,935
rows to one identical value means that if a band hides even 1% positives we lose ≈0.0005,
and if it is clean we recover the same amount by epsilon-ordering inside it [50]. This is the
one place where a cheap change has a payoff above the noise floor.

**The measurement problem is the real headwind.** Public-LB SE ≈ 0.0013 [50]; our recorded
CV→LB offset is −0.00048 while the reference recipe's is +0.00031 [20]. We cannot iterate in
0.0001 steps by submitting. We can only iterate by matching a recipe whose CV we can
reproduce.

## 4. Action Plan

- [ ] Build v8 as a faithful re-implementation of the verified 0.94638 feature pipeline: triple `TargetEncoder` (auto/10/100) over all string-like columns, combined-population frequency encoding, digit decomposition `k ∈ [-4,4)`, four CTGAN-flag features, multi-scale income/commute string bins, drop `Number_of_Cars_Owned`.
- [ ] Switch the external dataset from *concatenated rows* to `{col}_org_mean` target-mean anchors.
- [ ] Adopt the verified LightGBM config verbatim (`num_leaves=32, max_depth=5, lr=0.02, colsample_bytree=0.303, min_child_samples=10, n_estimators=20000, ES=500, max_bin=1024, feature_pre_filter=False`) and 5-fold StratifiedKFold.
- [ ] Gate on CV, not LB: v8's single-model CV must reach ≥0.9460 before we spend a submission.
- [ ] Replace band pinning with epsilon-ordering inside each band (keep the band's block position, destroy the intra-band ties).
- [ ] Delete the `lexsort` tie-break and stop 6-decimal rounding in submission output.
- [ ] Keep the 24 zero bands + 2 buyer bands; they are the only post-processing with verified support.
- [ ] Do not blend public kernels for the gap; if v8 lands, blend v8 with v7 only as a free last-day insurance ticket.

## 5. Open Questions & Caveats

- Whether the reference recipe's 0.94638 survives a private-LB reshuffle is unknown; its
  CV→LB offset (+0.00031) is itself within one standard error.
- Five of the named top artifacts (0.94647 / 0.94656 / 0.94657 / RealMLP / cdeotte) could
  **not** be retrieved — `kernels/pull` returned 403/404. Nothing about them is assumed here.
- Our own `Aadit_try` audits were produced by a parallel workstream; they are reproducible
  from the stored `.npy` artefacts but were not re-run in this session.
- The 0.94600 target is 0.0004 above us and 0.6 public-LB standard errors — achievable only
  by a feature-set change of the size described, not by tuning.

## Methodology

Standard depth. 3 parallel retrieval subagents (areas 1–2, 3, 4–5), one wave, no gap-fill
wave needed after the quality gate (all areas ≥2 sources; area 3 deliberately reported its
unretrievable items rather than padding). Phase 3.1 verification was performed by the lead
agent directly on the two most load-bearing claims: (a) the reference notebook's parameters,
re-fetched from `api/v1/kernels/pull` and confirmed verbatim; (b) the internal audit files,
confirmed to exist on disk with the quoted numbers. One claim was **corrected** during
triangulation: this session had previously implemented `lexsort` tie-breaking on the strength
of a third-party claim it was worth ~0.0002; first-party measurement puts its ceiling at
8e-8, so it is removed. Report condensed from the skill's 4,000-word target to roughly
1,900 words because the output feeds a single engineering decision; all citations retained.

## Bibliography

[1] LightGBM docs — Parameters — https://lightgbm.readthedocs.io/en/latest/Parameters.html — Tier 1
[2] LightGBM docs — Parameters Tuning — https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html — Tier 1
[3] LightGBM docs — GPU Performance — https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html — Tier 1
[4] LightGBM issue #6319 — memory vs max_bin — https://github.com/lightgbm-org/LightGBM/issues/6319 — Tier 2
[5] XGBoost docs — Parameters — https://xgboost.readthedocs.io/en/stable/parameter.html — Tier 1
[6] CatBoost docs — Quantization settings — https://catboost.ai/docs/en/references/training-parameters/quantization — Tier 1
[7] CatBoost docs — Parameter tuning — https://catboost.ai/docs/en/concepts/parameter-tuning — Tier 1
[8] scikit-learn docs — HistGradientBoostingRegressor — https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html — Tier 1
[9] scikit-learn docs — TargetEncoder — https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html — Tier 1
[10] scikit-learn source — _target_encoder_fast.pyx — https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_target_encoder.py — Tier 1
[11] scikit-learn user guide §8.3.4.2 — Target Encoder — https://scikit-learn.org/stable/modules/preprocessing.html — Tier 1
[12] Johannes Heller — S5E4 3rd place, Target Encoding and 3 Levels — https://www.kaggle.com/competitions/playground-series-s5e4/writeups/johannes-heller-3rd-place-target-encoding-and-3-le — 2025-05 — Tier 2
[13] Chris Deotte — S5E2 1st place, Single Model Feature Engineering — https://www.kaggle.com/competitions/playground-series-s5e2/discussion/565539 — 2025-03 — Tier 2
[14] Cat-in-the-Dat-II discussion 132106 — target encoding overfitting — https://www.kaggle.com/c/cat-in-the-dat-ii/discussion/132106 — Tier 3
[15] Fhilipus Mahendra — K-Fold Target Encoding for High Cardinality, TDS — https://towardsdatascience.com/understanding-k-fold-target-encoding-to-handle-high-cardinality-296387753e3f/ — 2024-10-26 — Tier 3
[16] XGBoosting.com — Tune max_bin — https://xgboosting.com/tune-xgboost-max_bin-parameter/ — Tier 3
[20] najiama — Pure LGBM Model CV 0.94607 LB 0.94638 (full source) — https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94607-lb-0-94638 — Tier 2
[21] Kaggle API — kernels/pull (unauthenticated source endpoint) — https://www.kaggle.com/api/v1/kernels/pull?user_name=najiama&kernel_slug=pure-lgbm-model-cv-0-94607-lb-0-94638 — Tier 1
[22] nina2025 — simple 50/50 two-level straight blend (source 403, not retrieved) — https://www.kaggle.com/code/nina2025/ps-s6e9-a-simple-50-50-two-level-straight-blend — Tier 2
[23] talhatursun — Best Public Blend Tracker (not retrieved) — https://www.kaggle.com/code/talhatursun/s6e9-daily-rank-average-ensemble — Tier 2
[24] rugvedbane — 0.94590 LB, Stacking Failed This Didn't (not retrieved) — https://www.kaggle.com/code/rugvedbane/0-94590-lb-stacking-failed-this-didn-t — Tier 2
[25] maiernator — S6E9 CTBoost Astra baseline (smooth=100 TE origin) — https://www.kaggle.com/code/maiernator/s6e9-ctboost-not-catboost-astra-baseline — Tier 2
[26] evgendvorkin — S6E9 Single XGB CV 0.94583 — https://www.kaggle.com/code/evgendvorkin/s6e9-single-xgb-cv-0-94583 — Tier 2
[27] starkhushi & Tilii — S6E9 discussion 738968 (Millionaire Cliff, \$30k mode collapse) — https://www.kaggle.com/competitions/playground-series-s6e9/discussion/738968 — Tier 2
[28] broccoli beef — S6E9 discussion 739142 (LogReg on original data) — https://www.kaggle.com/competitions/playground-series-s6e9/discussion/739142 — Tier 2
[29] Chris Deotte — S6E9 discussion 738991 (Simpson's paradox) — https://www.kaggle.com/competitions/playground-series-s6e9/discussion/738991 — Tier 2
[32] itzzomkar — EV Adoption and Range Anxiety dataset — https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety — Tier 2
[40] S6E2 1st place — Diversity, Selection, and Trusting the CV–LB Relation — https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t — Tier 2
[46] First-party — lexsort exact tie-break audit — Aadit_try/artifacts/experiments/lexsort_exact_tie_break_audit/summary.txt — internal
[47] First-party — RealMLP diversity blend audit — Aadit_try/artifacts/experiments/blend_realmlp_diversity_audit/summary.txt — internal
[50] First-party — computed from data/train.csv and our submissions (Hanley–McNeil SE, tie structure, band risk) — internal
