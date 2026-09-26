# S6E9 — reverse-engineering findings

Date 2026-09-26. Trigger: the hypothesis that above ~0.9462 the only remaining gains come
from reverse-engineering / "hardcoding" the synthetic data rather than modelling. Bar to beat:
Aadit's `candidate_submission.csv` at **0.94643**. Ours: v8 0.94617, v10 0.94616, **v11 0.94620**.

Every number below is measured against real labels on data the rule was not fitted on.
Probes are in `research/` and are throwaway.

## 1. Hardcoding is exhausted. There is nothing left to extract.

`research/spike_hardcode.py` — 15 rule families, each discovered on half of train and scored
on the other half against a real CatBoost ranking. Exact-value cells (income, commute,
income × subsidy / anxiety / home-charge / env, income × commute-bin, one 3-way, and the full
13-feature tuple) plus income bands quantised at $1/$10/$50/$100/$250.

| family | pure cells on discovery half | held-out rows | impurities on held-out | AUC delta |
|---|---|---|---|---|
| income exact (sup≥60) | 4 neg / 1 pos | 554 | **11 positives** | −0.000149 |
| income × subsidy | 28 neg | 3,327 | 9 | −0.000052 |
| income × anxiety | 4 neg | 537 | 11 | −0.000146 |
| income × commute bin | 1 neg | 131 | 1 | −0.000007 |
| full 13-feature tuple | **none at support ≥ 5** | — | — | — |
| income bands $1…$250 | ≤ 14 neg each | ≤ 3,385 | ≤ 9 | best **+0.000019** |

**Zero of 15 survived.** Best delta across all of them, +0.000019, is noise — and that
candidate still carried 9 mislabeled rows inside its own "certain" set.

The control that makes this conclusive: the 24+2 bands v11 **already ships** score, on the
same held-out half, **6,118 rows with ZERO impurities and +0.000422**. The determinism in this
dataset is real, it is worth about +0.0004, and **we are already collecting all of it.**

Corollary: there are no exact train↔test duplicate rows to memorise (0 of 286,571 match on the
full 13-feature tuple), and the generator's label function is not a recoverable closed form
— plain logistic regression on raw features reaches only 0.9377 and its coefficients are not
round numbers.

## 2. Blending is worth almost nothing, and that is now calibrated.

`research/spike3_gain_curve.py` — all 376 distinct pairs among the 28 OOF vectors in this repo
with AUC ≥ 0.944. For each pair: Spearman rho, and the actual gain of a 50/50 rank blend over
the better member.

corr(rho, gain) = **−0.869** on equal-strength pairs. The theory holds perfectly. The magnitude
does not:

| rho bucket | pairs | median gain | max gain |
|---|---|---|---|
| 0.990 – 0.994 | 68 | **−0.000013** | +0.000218 |
| 0.994 – 0.996 | 110 | +0.000026 | +0.000225 |
| 0.996 – 0.998 | 106 | −0.000005 | +0.000200 |
| 0.9995 + (our v8/v10/v11) | 19 | −0.000003 | +0.000026 |

Best single pair out of 376 tried: **+0.000225** — and that is a maximum over 376 attempts,
i.e. selection, not expectation.

**Aadit's local champion sits at rho 0.9912 against v11.** That is in the bucket whose *median*
gain is negative. Blending it in is a coin flip, not a plan.

`research/blend_search.py` independently confirms this at the level of the whole model zoo:
leave-one-fold-out Caruana selection over the top 40 of 117 de-duplicated OOF vectors gives an
honest cross-fitted **0.946133**, only +0.000078 over v8 and +0.000005 over the best single
vector. In-sample was 0.946142, so selection overfitting was 1e-5 — the number is trustworthy,
and it says the ceiling of everything in this family is ~0.9461.

## 3. v11's two changes: one was dead, one is the biggest real effect found.

`research/spike2_faithful_ablation.py` — this is the answer to the question the missing v11
Kaggle log was supposed to settle. v11 landed at +0.00003 over v8 when the earlier toy-feature
proxy predicted +0.0005. The proxy was the problem: it used 13 raw columns plus digits, so a
feature that adds information to a toy model can add nothing to a 291-column design that
already target-encodes the same keys three times. Re-running the ablation on the **real**
`ev_s6e9_v11.build_features` / `fold_matrix`, 50/50 held-out:

| change | delta on the faithful pipeline | verdict |
|---|---|---|
| C1 full 69 anchors vs v8's 13 | **−0.000019** | dead. The toy +0.000225 did not transfer. |
| $50/$250 fine-income bins on/off | +0.000027 | neutral |
| C2 3-seed bagging vs single model | **+0.000347** | real, and large |

The bagging number is honest: measured against the **mean** of three single seeds (0.944605),
not the luckiest baseline. K=1 → 0.944605, K=2 → +0.000260, K=3 → +0.000347. Seed-to-seed spread
is sd 0.000048, so the K=3 gain is **7.3 seed-sigma**.

Fitting the standard ρ + (1−ρ)/K saturation gives inter-seed correlation **ρ = 0.99948** and:

| K | predicted gain |
|---|---|
| 3 | +0.000347 (measured) |
| 5 | +0.000416 |
| 8 | +0.000455 |
| 10 | +0.000468 |
| ∞ | +0.000494 |

**v11 already runs 5 repeats**, so v11 should already be carrying roughly +0.0004 of true gain
over v8. It scored +0.00003 on the public LB. With a public-LB standard error of ~0.0013, a
true +0.0004 showing as +0.00003 is a 0.28-sigma miss — entirely ordinary. The most probable
reading is that **v11 is genuinely better than v8 and we drew an unlucky public sample.**

## 4. What this means for beating 0.94643

- The gap from v11 (0.94620) to Aadit (0.94643) is **+0.00023**. That is 18% of one public
  standard error, and it is exactly the size of the luckiest single pair-blend out of 376 tried.
  **There is no evidence his candidate is a better model.** It is inside the noise.
- No untried lever on this list has an expected value above +0.00005. Modelling, feature
  engineering, hardcoding and blending are all measured exhausted.
- The one lever with headroom is bagging depth, and it is nearly saturated: 5 → 10 repeats buys
  a predicted +0.00005 for double runtime.

**Recommendation.** Stop engineering for 0.0002 that does not exist measurably. Run **v12**
(v11 minus the dead anchor change — same expectation, ~15% faster, 20 fewer columns to
overfit), keep it and v11 as the two submissions, and when Aadit's files land run
`make_v12_blend.py` to check rho against v11 before spending anything on a blend. If rho comes
in below ~0.990 — genuinely different information, not just a different seed — the blend is
worth a slot. At 0.991 it is not.
