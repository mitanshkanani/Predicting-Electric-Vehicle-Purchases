"""
v6.2 post-processing: hardcode the deterministic income bands of S6E9.

WHAT THIS DOES
  Takes an existing submission and rewrites the rows that fall in income bands
  whose behaviour is STRUCTURALLY fixed in the generator, then writes
  submission_v6_2.csv. AUC only cares about ranking, so "force to buyer" = put
  the row above everything, "force to non-buyer" = put it below everything.

WHY THESE BANDS AND NOT JUST THE $170,537 CLIFF
  Every band below was tested against its own LOCAL base rate (the positive rate
  of the +-4,000 USD window around it, excluding the band) with an exact binomial
  test. Only bands with p < 1e-6 are used. That filter matters: with 678k rows
  there are hundreds of income stretches that happen to contain 0 buyers by pure
  luck (e.g. subsidy=No slices), and hardcoding those would LOSE score.

  Verified on train.csv (668,665 rows):
    [170537,188549]  393/393 BUYERS     p=4.0e-141   156 test rows
    [92106, 92207]   223/223 BUYERS     p=5.5e-162    98 test rows   <- new find
    24 zero-buyer bands, 9,082 train rows, 0 buyers, 4,857 test rows

MEASURED EFFECT (not guessed) -- applied to our stored out-of-fold predictions:
    v5_oof.npy  AUC 0.947479 -> 0.947567   (+0.000089)
    v4_oof.npy  AUC 0.943598 -> 0.944253   (+0.000654)
  The gain is larger for a weaker base model. Read it as an UPPER bound: the
  bands were derived from these same train labels, so the true LB gain is
  smaller -- realistically +0.0001 to +0.0003. It is free and cannot hurt if the
  generator rules hold (they hold on 9,475 train rows with zero exceptions), so
  it is worth a submission slot, but this is not what closes the gap to 0.94675.

USAGE
    python hardcode_v62.py                                  # auto-pick best submission + test.csv
    python hardcode_v62.py submission_v7.csv test.csv        # -> submission_v7_2.csv
    python hardcode_v62.py sub.csv test.csv --out my.csv     # explicit output name
    python hardcode_v62.py --check-oof v5/v5_oof.npy         # report the AUC delta on train
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET = "Will_Buy_EV"
ID_COL = "id"


def default_output(base_path):
    """submission_v7.csv -> submission_v7_2.csv; falls back to _hard.csv."""
    name = os.path.basename(base_path)
    m = re.search(r"v(\d+(?:\.\d+)?)", name, flags=re.IGNORECASE)
    return f"submission_v{m.group(1)}_2.csv" if m else name.replace(".csv", "_hard.csv")

# --- (low, high) inclusive income bands, in USD, rounded to integer ---
BUYER_BANDS = [
    (170_537, 188_549),   # 393/393, p=4.0e-141
    (92_106, 92_207),     # 223/223, p=5.5e-162
]
NON_BUYER_BANDS = [
    (38_174, 41_384), (48_002, 48_589), (48_657, 48_779), (48_981, 49_491),
    (49_646, 49_809), (50_345, 50_525), (56_945, 57_208), (59_081, 59_192),
    (59_227, 59_296), (59_602, 59_711), (59_754, 59_802), (59_989, 60_417),
    (60_425, 60_499), (62_223, 62_260), (62_998, 63_361), (63_363, 63_384),
    (63_991, 64_404), (64_640, 64_706), (65_108, 65_190), (65_228, 65_409),
    (83_985, 84_164), (84_223, 84_353), (90_151, 90_189), (103_103, 103_314),
]
# Borderline (p=3.3e-04 on 186 rows) and reported independently in the
# discussion threads. Small, so it is opt-in.
COMMUTE_NO_BUYERS_FROM = 83.0
USE_COMMUTE_EDGE = True


def _first_existing(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def find_paths(argv):
    base = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else None
    rest = [a for i, a in enumerate(argv[1:], start=1)
            if not a.startswith("--") and a != base]
    out_arg = None
    if "--out" in argv:
        out_arg = argv[argv.index("--out") + 1]
        rest = [a for a in rest if a != out_arg]
    test_path = rest[0] if rest else _first_existing([
        "test.csv", "data/test.csv", "/kaggle/input/**/test.csv", "/content/test.csv",
    ])
    if base is None:
        # prefer the strongest known submission that exists locally
        for cand in ["submission_v7.csv", "submission_v7_2.csv", "submission_v6_2.csv",
                     "submission_merged_nn25.csv", "submission_v6_kaggle.csv", "submission.csv"]:
            if os.path.isfile(cand):
                base = cand
                break
    if base is None or test_path is None:
        raise SystemExit(__doc__ + "\nERROR: could not locate a submission and/or test.csv")
    return base, test_path, out_arg


def apply_rules(probs, income, commute):
    """Move structurally-fixed rows to the extreme top/bottom of the ranking.
    Everything stays inside [0,1] (Kaggle may reject out-of-range probabilities);
    the middle rows are linearly squeezed, which is monotone so their relative
    order — and therefore the AUC contribution — is unchanged."""
    out = np.array(probs, dtype=np.float64, copy=True)

    mask_buy = np.zeros(len(out), dtype=bool)
    for lo, hi in BUYER_BANDS:
        mask_buy |= (income >= lo) & (income <= hi)
    mask_no = np.zeros(len(out), dtype=bool)
    for lo, hi in NON_BUYER_BANDS:
        mask_no |= (income >= lo) & (income <= hi)
    if USE_COMMUTE_EDGE:
        mask_no |= commute >= COMMUTE_NO_BUYERS_FROM

    overlap = mask_buy & mask_no          # bands are disjoint; belt and braces
    mask_no &= ~mask_buy

    mid = ~(mask_buy | mask_no)
    lo, hi = out[mid].min(), out[mid].max()
    span = max(hi - lo, 1e-12)
    eps = 1e-4
    out[mid] = eps + (out[mid] - lo) / span * (1.0 - 2.0 * eps)
    out[mask_buy] = 1.0                   # ties among true positives cost no AUC
    out[mask_no] = 0.0                    # ties among true negatives cost no AUC
    return out, mask_buy, mask_no, overlap


def main():
    argv = sys.argv
    if "--check-oof" in argv:
        i = argv.index("--check-oof")
        oof_path = argv[i + 1]
        train_path = _first_existing(["train.csv", "data/train.csv", "/content/train.csv"])
        tr = pd.read_csv(train_path)
        y = (tr[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int).to_numpy()
        oof = np.load(oof_path)
        if len(oof) != len(y):
            raise SystemExit(f"{oof_path} has {len(oof)} rows but train has {len(y)} "
                             "(that OOF includes the merged 10k source rows)")
        new, mb, mn, _ = apply_rules(oof,
                                     np.rint(pd.to_numeric(tr["Annual_Income_USD"])).to_numpy(),
                                     pd.to_numeric(tr["Daily_Commute_km"]).to_numpy())
        print(f"train rows forced to buyer: {int(mb.sum())} (all truly buyers: "
              f"{bool(y[mb].all())})")
        print(f"train rows forced to non-buyer: {int(mn.sum())} (all truly non-buyers: "
              f"{bool((~y[mn].astype(bool)).all())})")
        print(f"AUC {roc_auc_score(y, oof):.6f} -> {roc_auc_score(y, new):.6f} "
              f"({roc_auc_score(y, new) - roc_auc_score(y, oof):+.6f})")
        return

    base, test_path, out_arg = find_paths(argv)
    sub = pd.read_csv(base)
    test = pd.read_csv(test_path)
    if ID_COL not in sub.columns or TARGET not in sub.columns:
        raise SystemExit(f"{base} needs '{ID_COL}' and '{TARGET}' columns")

    merged = test.merge(sub[[ID_COL, TARGET]], on=ID_COL, how="left")
    if merged[TARGET].isna().any():
        raise SystemExit(f"{len(merged[merged[TARGET].isna()])} test ids missing from {base}")
    if "Annual_Income_USD" not in merged.columns or "Daily_Commute_km" not in merged.columns:
        raise SystemExit(f"{test_path} is missing Annual_Income_USD / Daily_Commute_km")

    income = np.rint(pd.to_numeric(merged["Annual_Income_USD"], errors="coerce")).to_numpy(dtype=np.float64)
    commute = pd.to_numeric(merged["Daily_Commute_km"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    new_probs, mb, mn, overlap = apply_rules(merged[TARGET].to_numpy(dtype=np.float64), income, commute)

    print(f"base   : {base} ({len(merged)} rows)")
    print(f"test   : {test_path}")
    print(f"forced to BUYER      : {int(mb.sum()):5d} rows  {BUYER_BANDS}")
    print(f"forced to NON-BUYER  : {int(mn.sum()):5d} rows  ({len(NON_BUYER_BANDS)} income bands"
          + (f" + commute>={COMMUTE_NO_BUYERS_FROM:g})" if USE_COMMUTE_EDGE else ")"))
    if overlap.any():
        print(f"WARNING: {int(overlap.sum())} rows matched both a buyer and a non-buyer band (buyer wins)")

    out = sub.copy()
    lookup = dict(zip(merged[ID_COL].to_numpy(), new_probs))
    out[TARGET] = [lookup[i] for i in out[ID_COL].to_numpy()]
    out = out[[ID_COL, TARGET]]

    out_path = out_arg or default_output(base)
    if os.path.isdir("/kaggle/working") and not os.path.dirname(out_path):
        out_path = os.path.join("/kaggle/working", out_path)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} | {len(out)} rows | prob {out[TARGET].min():.4f}..{out[TARGET].max():.4f}")


if __name__ == "__main__":
    main()
