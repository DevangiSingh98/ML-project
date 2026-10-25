"""
Phase C: Leakage-Safe EEG Channel Reduction Experiments.

Compares:
1. All 62 Channels (Reference Baseline from Phase B)
2. Random Channel Selection (at budgets k in [8, 12, 16, 24, 32], evaluated across repeated random draws)
3. Predictive-Importance Selection (Grouped Permutation Importance on inner training splits)
4. Stability-Aware Selection (Combined predictive utility and rank consistency Q_c = alpha*I_c + beta*S_c)

Guarantees:
- Outer test subjects are completely excluded from channel ranking, scoring, scaling, and fitting.
- Physical channel grouping (all 5 frequency bands per channel) is strictly preserved.
- Full output persistence and reproducibility.
"""

import sys
from pathlib import Path
# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import json
import time
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_seed_iv_dataset, load_config, LABEL_NAMES
from src.subject_split import get_loso_splits
from src.channel_features import slice_channel_features, get_channel_names
from src.channel_selection import (
    select_random_channels,
    compute_inner_grouped_permutation_importance,
    select_predictive_importance_channels,
    select_stability_aware_channels,
    compute_pairwise_jaccard_similarity,
)
from src.preprocessing import create_baseline_pipeline
from src.metrics import compute_classification_metrics
from src.make_figures import (
    plot_performance_vs_channel_count,
    plot_channel_selection_stability,
)


def run_single_fold_channel_reduction(
    fold_idx: int,
    test_sub: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    all_channel_names: List[str],
    budgets: List[int],
    alpha: float = 0.5,
    beta: float = 0.5,
    cache_size: int = 500,
    random_seeds: List[int] = [42, 142, 242],
) -> Tuple[List[dict], List[pd.DataFrame], Dict[str, Dict[int, List[int]]]]:
    """
    Execute channel reduction evaluations for a single outer LOSO fold.
    
    Returns:
        (fold_metric_records, prediction_dfs, selected_channels_by_method_and_budget)
    """
    t_start = time.time()

    # Leakage check
    train_subs = set(metadata.iloc[train_indices]["subject_id"].unique())
    test_subs = set(metadata.iloc[test_indices]["subject_id"].unique())
    assert train_subs.isdisjoint(test_subs), f"Fold {fold_idx} leakage detected!"
    assert test_subs == {test_sub}

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    train_meta = metadata.iloc[train_indices].reset_index(drop=True)
    test_meta = metadata.iloc[test_indices].copy().reset_index(drop=True)

    # 1. Compute training-only channel rankings (inner splits on 14 training subjects)
    importance_mat, ranks_mat, avg_imp = compute_inner_grouped_permutation_importance(
        X_train=X_train,
        y_train=y_train,
        train_metadata=train_meta,
        n_splits=3,
        seed=42 + fold_idx,
    )

    # Predictive Importance ranking (descending average drop)
    pred_rank_order = np.argsort(-avg_imp).tolist()

    # Stability-Aware ranking
    min_i = np.min(avg_imp)
    max_i = np.max(avg_imp)
    denom_i = (max_i - min_i) if (max_i - min_i) > 1e-8 else 1.0
    I_norm = (avg_imp - min_i) / denom_i
    mean_ranks = np.mean(ranks_mat, axis=0)
    S_c = np.clip(1.0 - (mean_ranks - 1.0) / 61.0, 0.0, 1.0)
    Q_c = alpha * I_norm + beta * S_c
    stab_rank_order = np.argsort(-Q_c).tolist()

    fold_metric_records = []
    pred_dfs = []
    channel_selections: Dict[str, Dict[int, List[int]]] = {
        "stability_aware": {},
        "predictive_importance": {},
        "random": {},
    }

    # Evaluate each budget k
    for k in budgets:
        # A. Stability-Aware top k
        stab_k_chs = sorted(stab_rank_order[:k])
        channel_selections["stability_aware"][k] = stab_k_chs

        X_tr_stab, _ = slice_channel_features(X_train, stab_k_chs)
        X_te_stab, _ = slice_channel_features(X_test, stab_k_chs)

        pipe_stab = create_baseline_pipeline(C=1.0, gamma="scale", cache_size=cache_size)
        pipe_stab.fit(X_tr_stab, y_train)
        y_pred_stab = pipe_stab.predict(X_te_stab)

        m_stab = compute_classification_metrics(y_test, y_pred_stab)
        fold_metric_records.append({
            "fold": fold_idx,
            "test_subject": test_sub,
            "method": "stability_aware",
            "budget": k,
            "seed": 42,
            "selected_channels": ",".join([all_channel_names[c] for c in stab_k_chs]),
            **m_stab,
        })

        pred_stab_df = test_meta.copy()
        pred_stab_df["method"] = "stability_aware"
        pred_stab_df["budget"] = k
        pred_stab_df["seed"] = 42
        pred_stab_df["predicted_label"] = y_pred_stab
        pred_stab_df["predicted_label_name"] = [LABEL_NAMES[p] for p in y_pred_stab]
        pred_stab_df["is_correct"] = (y_pred_stab == y_test).astype(int)
        pred_dfs.append(pred_stab_df)

        # B. Predictive Importance top k
        pred_k_chs = sorted(pred_rank_order[:k])
        channel_selections["predictive_importance"][k] = pred_k_chs

        if pred_k_chs == stab_k_chs:
            y_pred_imp = y_pred_stab
            m_pred = m_stab
        else:
            X_tr_pred, _ = slice_channel_features(X_train, pred_k_chs)
            X_te_pred, _ = slice_channel_features(X_test, pred_k_chs)

            pipe_pred = create_baseline_pipeline(C=1.0, gamma="scale", cache_size=cache_size)
            pipe_pred.fit(X_tr_pred, y_train)
            y_pred_imp = pipe_pred.predict(X_te_pred)
            m_pred = compute_classification_metrics(y_test, y_pred_imp)
        fold_metric_records.append({
            "fold": fold_idx,
            "test_subject": test_sub,
            "method": "predictive_importance",
            "budget": k,
            "seed": 42,
            "selected_channels": ",".join([all_channel_names[c] for c in pred_k_chs]),
            **m_pred,
        })

        pred_imp_df = test_meta.copy()
        pred_imp_df["method"] = "predictive_importance"
        pred_imp_df["budget"] = k
        pred_imp_df["seed"] = 42
        pred_imp_df["predicted_label"] = y_pred_imp
        pred_imp_df["predicted_label_name"] = [LABEL_NAMES[p] for p in y_pred_imp]
        pred_imp_df["is_correct"] = (y_pred_imp == y_test).astype(int)
        pred_dfs.append(pred_imp_df)

        # C. Random selection (primary seed 42 + k)
        rnd_k_chs = select_random_channels(budget=k, seed=42 + k)
        channel_selections["random"][k] = rnd_k_chs

        X_tr_rnd, _ = slice_channel_features(X_train, rnd_k_chs)
        X_te_rnd, _ = slice_channel_features(X_test, rnd_k_chs)

        pipe_rnd = create_baseline_pipeline(C=1.0, gamma="scale", cache_size=cache_size)
        pipe_rnd.fit(X_tr_rnd, y_train)
        y_pred_rnd = pipe_rnd.predict(X_te_rnd)

        m_rnd = compute_classification_metrics(y_test, y_pred_rnd)
        fold_metric_records.append({
            "fold": fold_idx,
            "test_subject": test_sub,
            "method": "random",
            "budget": k,
            "seed": 42 + k,
            "selected_channels": ",".join([all_channel_names[c] for c in rnd_k_chs]),
            **m_rnd,
        })

        pred_rnd_df = test_meta.copy()
        pred_rnd_df["method"] = "random"
        pred_rnd_df["budget"] = k
        pred_rnd_df["seed"] = 42 + k
        pred_rnd_df["predicted_label"] = y_pred_rnd
        pred_rnd_df["predicted_label_name"] = [LABEL_NAMES[p] for p in y_pred_rnd]
        pred_rnd_df["is_correct"] = (y_pred_rnd == y_test).astype(int)
        pred_dfs.append(pred_rnd_df)

        # Additional random seeds for budgets 8 and 16 to evaluate random subset variability
        if k in [8, 16]:
            for alt_seed in [142 + k, 242 + k]:
                alt_chs = select_random_channels(budget=k, seed=alt_seed)
                X_tr_alt, _ = slice_channel_features(X_train, alt_chs)
                X_te_alt, _ = slice_channel_features(X_test, alt_chs)

                pipe_alt = create_baseline_pipeline(C=1.0, gamma="scale", cache_size=cache_size)
                pipe_alt.fit(X_tr_alt, y_train)
                y_pred_alt = pipe_alt.predict(X_te_alt)

                m_alt = compute_classification_metrics(y_test, y_pred_alt)
                fold_metric_records.append({
                    "fold": fold_idx,
                    "test_subject": test_sub,
                    "method": "random_repeat",
                    "budget": k,
                    "seed": alt_seed,
                    "selected_channels": ",".join([all_channel_names[c] for c in alt_chs]),
                    **m_alt,
                })

    return fold_metric_records, pred_dfs, channel_selections


def run_channel_reduction_experiments(
    config_path: Union[str, Path] = "config.yaml",
    n_jobs: int = 4,
    cache_size: int = 500,
) -> Dict[str, Union[pd.DataFrame, dict]]:
    """
    Execute full 15-fold LOSO channel reduction across all budgets and methods.
    """
    cfg = load_config(config_path)
    budgets = cfg.get("channel_reduction", {}).get("budgets", [8, 12, 16, 24, 32])
    reduced_budgets = [b for b in budgets if b != 62]

    print("=" * 80)
    print("PHASE C: LEAKAGE-SAFE EEG CHANNEL REDUCTION EXPERIMENT")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Budgets evaluated: {reduced_budgets} (and all 62 channels)")
    print(f"Selection methods: All-62, Random, Predictive-Importance, Stability-Aware")
    print(f"Parallel worker processes: {n_jobs}")
    print("=" * 80)

    t0 = time.time()
    data = load_seed_iv_dataset(config=cfg)
    splits = list(get_loso_splits(data.metadata))
    print(f"Loaded dataset ({data.X.shape}) and generated 15 outer LOSO folds in {time.time()-t0:.2f}s.")

    # Run outer folds in parallel
    print(f"\nLaunching 15-fold LOSO channel reduction across {n_jobs} workers...")
    t_start = time.time()

    results = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
        joblib.delayed(run_single_fold_channel_reduction)(
            fold_idx=fold_idx,
            test_sub=test_sub,
            train_indices=train_indices,
            test_indices=test_indices,
            X=data.X,
            y=data.y,
            metadata=data.metadata,
            all_channel_names=data.channels,
            budgets=reduced_budgets,
            alpha=0.5,
            beta=0.5,
            cache_size=cache_size,
        )
        for fold_idx, test_sub, train_indices, test_indices in splits
    )

    total_experiment_time = time.time() - t_start
    print(f"\nAll channel reduction evaluations completed in {total_experiment_time:.2f}s ({total_experiment_time/60:.2f} min).")

    # Aggregate records
    all_fold_records = []
    all_pred_dfs = []
    fold_selections = []

    for r in results:
        all_fold_records.extend(r[0])
        all_pred_dfs.extend(r[1])
        fold_selections.append(r[2])

    fold_metrics_df = pd.DataFrame(all_fold_records)
    predictions_df = pd.concat(all_pred_dfs, ignore_index=True)

    # Load Phase B All-62 baseline to integrate into the comparison
    b_fold_file = Path("outputs/metrics/baseline_fold_metrics.csv")
    b_preds_file = Path("outputs/predictions/baseline_predictions.csv")

    if b_fold_file.is_file() and b_preds_file.is_file():
        b_folds = pd.read_csv(b_fold_file)
        b_preds = pd.read_csv(b_preds_file)

        # Add all-62 fold records
        for _, r in b_folds.iterrows():
            rec = r.to_dict()
            rec["method"] = "all"
            rec["budget"] = 62
            rec["seed"] = 42
            rec["selected_channels"] = ",".join(data.channels)
            all_fold_records.append(rec)

        # Add all-62 predictions
        b_preds_tagged = b_preds.copy()
        b_preds_tagged["method"] = "all"
        b_preds_tagged["budget"] = 62
        b_preds_tagged["seed"] = 42
        predictions_df = pd.concat([predictions_df, b_preds_tagged], ignore_index=True)

        fold_metrics_df = pd.DataFrame(all_fold_records)

    # 1. Compute summary table across all conditions
    summary_rows = []
    methods_to_summarize = ["all", "stability_aware", "predictive_importance", "random"]

    for method in methods_to_summarize:
        m_budgets = [62] if method == "all" else reduced_budgets
        for b in m_budgets:
            sub_preds = predictions_df[(predictions_df["method"] == method) & (predictions_df["budget"] == b)]
            sub_folds = fold_metrics_df[(fold_metrics_df["method"] == method) & (fold_metrics_df["budget"] == b)]

            if sub_preds.empty or sub_folds.empty:
                continue

            # Pooled window-level metrics
            pooled_m = compute_classification_metrics(
                y_true=sub_preds["label"].to_numpy(),
                y_pred=sub_preds["predicted_label"].to_numpy(),
            )

            # Subject-level mean ± std
            acc_mean = float(sub_folds["accuracy"].mean())
            acc_std = float(sub_folds["accuracy"].std())
            bal_mean = float(sub_folds["balanced_accuracy"].mean())
            bal_std = float(sub_folds["balanced_accuracy"].std())
            f1_mean = float(sub_folds["macro_f1"].mean())
            f1_std = float(sub_folds["macro_f1"].std())

            summary_rows.append({
                "method": method,
                "budget": b,
                "features": b * 5,
                "pooled_accuracy": pooled_m["accuracy"],
                "pooled_balanced_accuracy": pooled_m["balanced_accuracy"],
                "pooled_macro_f1": pooled_m["macro_f1"],
                "accuracy": acc_mean,
                "accuracy_std": acc_std,
                "balanced_accuracy": bal_mean,
                "balanced_accuracy_std": bal_std,
                "macro_f1": f1_mean,
                "macro_f1_std": f1_std,
            })

    comparison_df = pd.DataFrame(summary_rows)

    # 2. Channel Stability Analysis (Jaccard similarity and selection frequencies)
    stability_records = []
    freq_counter: Dict[str, Dict[str, int]] = {
        "stability_aware": {ch: 0 for ch in data.channels},
        "predictive_importance": {ch: 0 for ch in data.channels},
    }

    for b in reduced_budgets:
        # Collect sets across the 15 outer folds
        stab_sets = [f_sel["stability_aware"][b] for f_sel in fold_selections if b in f_sel["stability_aware"]]
        pred_sets = [f_sel["predictive_importance"][b] for f_sel in fold_selections if b in f_sel["predictive_importance"]]
        rand_sets = [f_sel["random"][b] for f_sel in fold_selections if b in f_sel["random"]]

        jaccard_stab = compute_pairwise_jaccard_similarity(stab_sets)
        jaccard_pred = compute_pairwise_jaccard_similarity(pred_sets)
        jaccard_rand = compute_pairwise_jaccard_similarity(rand_sets)

        stability_records.append({
            "budget": b,
            "stability_aware_jaccard": jaccard_stab,
            "predictive_jaccard": jaccard_pred,
            "random_jaccard": jaccard_rand,
        })

        # Count frequencies for budget k=16 (central budget)
        if b == 16:
            for s in stab_sets:
                for c in s:
                    freq_counter["stability_aware"][data.channels[c]] += 1
            for s in pred_sets:
                for c in s:
                    freq_counter["predictive_importance"][data.channels[c]] += 1

    stability_df = pd.DataFrame(stability_records)

    # Channel frequency DataFrame
    freq_rows = []
    for ch in data.channels:
        freq_rows.append({
            "channel_name": ch,
            "stability_aware_freq": freq_counter["stability_aware"][ch] / 15.0,
            "predictive_freq": freq_counter["predictive_importance"][ch] / 15.0,
        })
    freq_df = pd.DataFrame(freq_rows)

    # 3. Save all outputs
    outputs_dir = Path("outputs")
    preds_file = outputs_dir / "predictions/channel_reduction_predictions.csv"
    fold_file = outputs_dir / "metrics/channel_reduction_fold_metrics.csv"
    comp_file = outputs_dir / "tables/channel_reduction_comparison.csv"
    stab_file = outputs_dir / "tables/channel_stability_summary.csv"
    freq_file = outputs_dir / "tables/channel_selection_frequencies.csv"
    report_file = outputs_dir / "metrics/channel_reduction_run_report.json"

    predictions_df.to_csv(preds_file, index=False)
    fold_metrics_df.to_csv(fold_file, index=False)
    comparison_df.to_csv(comp_file, index=False)
    stability_df.to_csv(stab_file, index=False)
    freq_df.to_csv(freq_file, index=False)

    print(f"Saved: {preds_file} ({len(predictions_df)} prediction rows)")
    print(f"Saved: {fold_file}")
    print(f"Saved: {comp_file}")
    print(f"Saved: {stab_file}")
    print(f"Saved: {freq_file}")

    # Generate Figures
    fig_perf = outputs_dir / "figures/channel_reduction_performance.png"
    fig_stab = outputs_dir / "figures/channel_selection_stability.png"

    plot_performance_vs_channel_count(
        summary_df=comparison_df,
        output_path=fig_perf,
        metric_name="accuracy",
    )
    plot_channel_selection_stability(
        selection_freq_df=freq_df,
        all_channel_names=data.channels,
        output_path=fig_stab,
        top_n=20,
    )
    print(f"Saved: {fig_perf}")
    print(f"Saved: {fig_stab}")

    # Run report
    report_dict = {
        "timestamp": datetime.now().isoformat(),
        "total_runtime_seconds": round(time.time() - t0, 2),
        "experiment_time_seconds": round(total_experiment_time, 2),
        "n_workers": n_jobs,
        "budgets": budgets,
        "methods": methods_to_summarize,
        "leakage_checks_passed": True,
        "comparison_summary": comparison_df.to_dict(orient="records"),
        "stability_summary": stability_df.to_dict(orient="records"),
    }
    with open(report_file, "w", encoding="utf-8") as fp:
        json.dump(report_dict, fp, indent=2)
    print(f"Saved: {report_file}")

    # Print summary table
    print("\n" + "=" * 92)
    print("PHASE C: PERFORMANCE VERSUS CHANNEL COUNT SUMMARY")
    print("=" * 92)
    print(f"{'Method':<22} {'Budget':<8} {'Features':<10} {'Pooled Acc':<12} {'Subject Acc (Mean ± Std)':<25} {'Macro-F1':<12}")
    print("-" * 92)
    for _, r in comparison_df.sort_values(by=["budget", "method"]).iterrows():
        print(f"{r['method']:<22} {int(r['budget']):<8} {int(r['features']):<10} "
              f"{r['pooled_accuracy']*100:6.2f}%      "
              f"{r['accuracy']*100:5.2f}% ± {r['accuracy_std']*100:4.2f}%          "
              f"{r['macro_f1']*100:5.2f}%")
    print("=" * 92)

    print("\n" + "=" * 80)
    print("CHANNEL SELECTION STABILITY (Mean Pairwise Jaccard Similarity across 15 Folds)")
    print("=" * 80)
    print(f"{'Budget (k)':<12} {'Stability-Aware':<22} {'Predictive-Importance':<25} {'Random':<15}")
    print("-" * 80)
    for _, r in stability_df.iterrows():
        print(f"{int(r['budget']):<12} {r['stability_aware_jaccard']:<22.4f} "
              f"{r['predictive_jaccard']:<25.4f} {r['random_jaccard']:<15.4f}")
    print("=" * 80)

    return {
        "comparison": comparison_df,
        "stability": stability_df,
        "frequencies": freq_df,
        "predictions": predictions_df,
    }


if __name__ == "__main__":
    run_channel_reduction_experiments(n_jobs=4)
