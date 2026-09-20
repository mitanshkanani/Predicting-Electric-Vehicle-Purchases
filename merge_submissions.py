"""
merge_submissions.py — rank-average two Kaggle submission CSVs.

AUC is rank-based, so blending the RANKS of two decorrelated models is a valid,
robust ensemble. Use this to test whether the v4 NN adds signal to the v3 GBDT
blend. Run it anywhere both files are present (Colab or Kaggle).

Edit the three paths + weight below, then run. It writes submission_merged.csv.
"""

import pandas as pd
import numpy as np

# ---- EDIT THESE ----
GBDT_FILE = "submission_v3.csv"   # your best GBDT submission (the 0.94540 one)
NN_FILE = "submission_v4.csv"     # the neural-net submission (0.94354)
NN_WEIGHT = 0.25                  # trust GBDT more; 0.25 = 75% GBDT / 25% NN
TARGET = "Will_Buy_EV"
ID_COL = "id"
OUT = "submission_merged.csv"
# ---------------------

g = pd.read_csv(GBDT_FILE)
n = pd.read_csv(NN_FILE)

# Align the NN onto the GBDT row order by id (safe if orders differ).
g = g.sort_values(ID_COL).reset_index(drop=True)
n = n.sort_values(ID_COL).reset_index(drop=True)
assert np.array_equal(g[ID_COL].values, n[ID_COL].values), "id columns do not match between files"

rg = g[TARGET].rank(pct=True).to_numpy()
rn = n[TARGET].rank(pct=True).to_numpy()
merged = (1 - NN_WEIGHT) * rg + NN_WEIGHT * rn

out = pd.DataFrame({ID_COL: g[ID_COL], TARGET: merged})
out.to_csv(OUT, index=False)
print(f"Wrote {OUT}: {len(out)} rows | blended {GBDT_FILE} (w={1-NN_WEIGHT:.2f}) + {NN_FILE} (w={NN_WEIGHT:.2f})")
print("Submit this and compare its LB score to the GBDT alone.")
print("Tip: also try NN_WEIGHT = 0.1, 0.2, 0.3 and keep whichever scores best on the LB.")
