'''lightgbm_fine_income_highres_singleton_cpu.py

Kaggle Playground Series S6E9

CONTROLLED HIGH-RESOLUTION + ADDITIVE LIGHTGBM SYNERGY TEST

Immediate baseline:
    fine-income LightGBM with max_bin=13214 OOF = 0.94595344

ONLY CHANGE RELATIVE TO THAT BASELINE:
    add one singleton LightGBM interaction-constraint group per feature

No feature changes. No HPO. No blend tuning.
'''

from __future__ import annotations

from pathlib import Path

BASE_SCRIPT = Path(__file__).resolve().parent / "lightgbm_fine_income_te_cpu.py"

EXPECTED_HIGHRES_AUC = 0.94595344
HIGH_RES_MAX_BIN = 13214
EXPECTED_INCOME_UNIQUE = 13214


def replace_exact(source: str, old: str, new: str, expected_count: int = 1) -> str:
    count = source.count(old)
    if count != expected_count:
        raise RuntimeError(
            "Guarded source transformation failed.\n"
            f"Expected {expected_count} occurrence(s), found {count}.\n"
            f"Pattern starts with: {old[:160]!r}\n"
            "The validated baseline script may have changed. "
            "Do not run an uncontrolled experiment."
        )
    return source.replace(old, new)


def main() -> None:
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            "Missing validated baseline script:\n"
            f"{BASE_SCRIPT}\n"
            "Place this file in the same repo root as "
            "lightgbm_fine_income_te_cpu.py."
        )

    source = BASE_SCRIPT.read_text(encoding="utf-8")

    source = replace_exact(
        source,
        "EXPECTED_BASELINE_AUC = 0.94578042",
        f"EXPECTED_BASELINE_AUC = {EXPECTED_HIGHRES_AUC:.8f}",
    )

    source = replace_exact(
        source,
        '    / "lightgbm_engineered_learned_margin_cpu"\n'
        '    / "oof_predictions.csv"',
        '    / "lightgbm_fine_income_highres_cpu"\n'
        '    / "oof_predictions.csv"',
    )

    source = replace_exact(
        source,
        '    / "lightgbm_fine_income_te_cpu"\n)',
        '    / "lightgbm_fine_income_highres_singleton_cpu"\n)',
    )

    source = replace_exact(
        source,
        '    train = pd.read_csv(TRAIN_PATH)\n'
        '    test = pd.read_csv(TEST_PATH)',
        '''    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    income_unique = int(train["Annual_Income_USD"].nunique(dropna=True))
    if income_unique != 13214:
        raise RuntimeError(
            "Annual_Income_USD distinct-value count mismatch.\\n"
            f"Expected: 13214\\n"
            f"Found:    {income_unique}\\n"
            "Do not silently change the high-resolution max_bin target."
        )''',
    )

    source = replace_exact(
        source,
        '        model = base.build_model()\n\n'
        '        fit_start = time.perf_counter()',
        '''        model = base.build_model()

        singleton_constraints = [
            [i]
            for i in range(X_train.shape[1])
        ]

        model.set_params(
            max_bin=13214,
            interaction_constraints=singleton_constraints,
        )

        if outer_fold == 0:
            print(
                "Singleton interaction groups: "
                f"{len(singleton_constraints)} "
                "(one group per feature)"
            )

        fit_start = time.perf_counter()''',
    )

    source = replace_exact(
        source,
        '"Old engineered LightGBM baseline mismatch.\\n"',
        '"High-resolution LightGBM baseline mismatch.\\n"',
    )

    source = replace_exact(
        source,
        'print("LIGHTGBM + FINE $50/$250 INCOME TE")',
        'print("LIGHTGBM HIGH-RESOLUTION + SINGLETON CONSTRAINTS")',
    )
    source = replace_exact(
        source,
        'print("CONTROLLED TRANSFER INTO OLD DIVERSE LIGHTGBM")',
        'print("CONTROLLED ADDITIVE-SYNERGY EXPERIMENT")',
    )
    source = replace_exact(
        source,
        'print(f"Old engineered LightGBM baseline: {baseline_auc:.8f}")',
        'print(f"Verified high-resolution LightGBM baseline: {baseline_auc:.8f}")',
    )

    old_experiment_text = '''    print("HYPOTHESIS:")
    print(
        "  The $50/$250 local-income signal can improve the old diverse "
        "LightGBM without adding the hierarchical income/commute features "
        "that previously made LightGBM less useful in the ensemble."
    )
    print()
    print("ONLY CHANGE:")
    print("  Add nested leakage-safe:")
    print("    HTE__income_50_bucket")
    print("    HTE__income_250_bucket")
    print()
    print("HELD FIXED:")
    print("  - old engineered LightGBM representation")
    print("  - raw 13 features")
    print("  - income digits")
    print("  - exact income/commute frequency features")
    print("  - exact income/commute nested TE")
    print("  - learned nested logistic base margin")
    print("  - smoothing m=2")
    print("  - fixed LightGBM parameters")
    print("  - seed 42")
    print("  - CPU device")
    print("  - frozen competition folds")
    print()
    print("INTENTIONALLY NOT ADDED:")
    print("  - hierarchical income 1k/10k/100k TE")
    print("  - hierarchical commute 1/5/10km TE")
    print("  - any parameter search")
    print()'''

    new_experiment_text = '''    print("HYPOTHESIS:")
    print(
        "  High histogram resolution may become useful when LightGBM is forced "
        "to model features additively rather than combining different columns "
        "along the same decision path."
    )
    print()
    print("ONLY CHANGE VS VERIFIED HIGH-RES BASELINE:")
    print("  add one singleton interaction-constraint group per feature")
    print()
    print("HELD FIXED:")
    print("  - max_bin=13214")
    print("  - complete current fine-income LightGBM representation")
    print("  - raw features")
    print("  - income digits")
    print("  - exact income/commute frequency features")
    print("  - exact income/commute nested TE")
    print("  - fine income TE at $50/$250")
    print("  - learned nested logistic base margin")
    print("  - smoothing m=2")
    print("  - all other LightGBM parameters")
    print("  - seed 42")
    print("  - CPU device")
    print("  - frozen competition folds + SHA256")
    print("  - no blend tuning")
    print(f"  - verified Annual_Income_USD distinct values: {income_unique}")
    print()'''

    source = replace_exact(source, old_experiment_text, new_experiment_text)

    source = replace_exact(
        source,
        '"POSITIVE_FINE_INCOME_LGBM_SIGNAL"',
        '"POSITIVE_HIGHRES_SINGLETON_LGBM_SYNERGY"',
    )
    source = replace_exact(
        source,
        '"NEGATIVE_FINE_INCOME_LGBM_SIGNAL"',
        '"NEGATIVE_HIGHRES_SINGLETON_LGBM_SYNERGY"',
    )
    source = replace_exact(
        source,
        '"WEAK_OR_INCONSISTENT_FINE_INCOME_LGBM_SIGNAL"',
        '"WEAK_OR_INCONSISTENT_HIGHRES_SINGLETON_LGBM_SYNERGY"',
    )

    source = replace_exact(
        source,
        'print("FINE-INCOME LIGHTGBM RESULT")',
        'print("HIGH-RESOLUTION + SINGLETON LIGHTGBM RESULT")',
    )
    source = replace_exact(source, 'f"Old engineered LGBM : "', 'f"High-res baseline    : "')
    source = replace_exact(source, 'f"Fine-income LGBM     : "', 'f"Singleton candidate  : "')
    source = replace_exact(source, 'f"Prob corr vs old LGBM: "', 'f"Prob corr vs baseline: "')
    source = replace_exact(source, 'f"Rank corr vs old LGBM: "', 'f"Rank corr vs baseline: "')

    summary_start = source.index('    summary = [')
    summary_end_marker = '        "FOLD RESULTS",\n    ]'
    summary_end = source.index(summary_end_marker, summary_start) + len(summary_end_marker)

    new_summary = '''    summary = [
        "EXPERIMENT: LIGHTGBM HIGH-RESOLUTION + SINGLETON CONSTRAINTS",
        "=" * 84,
        "",
        "HYPOTHESIS",
        (
            "Can high histogram resolution become useful when LightGBM is "
            "prevented from combining different features along one branch?"
        ),
        "",
        "ONLY CHANGE VS VERIFIED HIGH-RES BASELINE",
        "Add one singleton interaction-constraint group per feature.",
        "",
        "HELD FIXED",
        "- max_bin=13214",
        "- complete current fine-income LightGBM representation",
        "- raw features",
        "- income digits",
        "- exact income/commute frequency features",
        "- exact income/commute nested TE",
        "- fine income TE at $50/$250",
        "- learned nested logistic base margin",
        "- smoothing m=2",
        "- all other LightGBM parameters",
        "- seed 42",
        "- CPU",
        "- frozen 5 folds",
        "",
        f"Frozen fold SHA256: {fold_hash}",
        f"Annual_Income_USD unique values: {income_unique}",
        f"Verified high-res baseline: {baseline_auc:.8f}",
        f"High-res + singleton candidate: {candidate_auc:.8f}",
        f"Delta: {delta_vs_baseline:+.8f}",
        f"Folds improved: {folds_improved}/5",
        f"Folds worse: {folds_worse}/5",
        f"Probability corr vs baseline: {prob_corr_vs_baseline:.6f}",
        f"Rank corr vs baseline: {rank_corr_vs_baseline:.6f}",
        f"Primitive: {primitive}",
        f"Runtime: {total_seconds:.2f}s",
        "",
        "FOLD RESULTS",
    ]'''

    source = source[:summary_start] + new_summary + source[summary_end:]

    globals_dict = {
        "__name__": "__main__",
        "__file__": str(Path(__file__).resolve()),
        "__package__": None,
    }

    compiled = compile(
        source,
        str(BASE_SCRIPT) + " [highres-singleton transformed]",
        "exec",
    )
    exec(compiled, globals_dict)


if __name__ == "__main__":
    main()
