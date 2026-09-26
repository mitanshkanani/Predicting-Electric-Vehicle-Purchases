# Archived experiments

This folder contains completed, rejected, superseded, or diagnostic experiment
scripts from the Kaggle S6E9 EV purchase project.

They are intentionally archived rather than deleted so experiment history stays
available through the working tree and Git history.

## Active pipeline

The validated champion remains in the Aadit_try root.

Key active/reproducibility scripts intentionally left at root include:

- `lightgbm_engineered_learned_margin_cpu.py`
- `xgboost_hierarchical_income_te_gpu.py`
- `xgboost_hierarchical_commute_te_gpu.py`
- `xgboost_fine_income_te_gpu.py`
- `catboost_hierarchical_income_buckets_gpu.py`
- `catboost_hierarchical_income_commute_buckets_gpu.py`
- `catboost_hierarchical_income_commute_multiseed_gpu.py`
- `blend_fine_income_xgb_validated_submission.py`

Validated secondary models intentionally left at root:

- `catboost_fine_income_multiseed_gpu.py`
- `lightgbm_fine_income_te_cpu.py`
- `realmlp_frozen5_vectorized_gpu.py`

Do not move active dependency scripts without updating imports first.
