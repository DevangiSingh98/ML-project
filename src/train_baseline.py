"""
15-Fold Leave-One-Subject-Out (LOSO) Baseline for SEED-IV Emotion Recognition.

Evaluates an all-62-channel RBF-SVM baseline under strict cross-subject isolation.
Guarantees:
- Feature scaling (StandardScaler) is strictly fit on outer training subjects only.
- Strict subject disjointness assertions in every fold.
- No probability calibration or thresholding in Phase B.
- Comprehensive output persistence (predictions, fold metrics, summaries, confusion matrix).
"""

import sys
from pathlib import Path
# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_seed_iv_dataset, load_config, DatasetContainer, LABEL_NAMES
from src.subject_split import get_loso_splits
from src.preprocessing import create_baseline_pipeline
from src.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix_df,
    plot_and_save_confusion_matrix,
)


def run_single_fold(
    fold_idx: int,
    test_sub: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    C: float = 1.0,
    gamma: Union[str, float] = "scale",
    cache_size: int = 500,
) -> Tuple[Dict[str, Union[int, float]], pd.DataFrame]:
    """
    Execute a single outer LOSO fold.
    
    Returns:
        (fold_metrics_dict, fold_predictions_dataframe)
    """
    t_start = time.time()

    # Mandatory Leakage Checks
    train_subs = set(metadata.iloc[train_indices]["subject_id"].unique())
    test_subs = set(metadata.iloc[test_indices]["subject_id"].unique())

    assert train_subs.isdisjoint(test_subs), (
        f"CRITICAL LEAKAGE DETECTED in Fold {fold_idx}: Train subjects {train_subs} "
        f"overlap with test subjects {test_subs}!"
    )
    assert test_subs == {test_sub}, (
        f"Fold {fold_idx} test set mismatch: expected {{{test_sub}}}, got {test_subs}"
    )
    assert len(train_subs) == 14, f"Fold {fold_idx} expected 14 train subjects, got {len(train_subs)}"
    assert len(test_indices) == 2505, f"Fold {fold_idx} expected 2505 test windows, got {len(test_indices)}"
    assert len(train_indices) == 35070, f"Fold {fold_idx} expected 35070 train windows, got {len(train_indices)}"

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Pipeline: StandardScaler fit exclusively on X_train, then SVC
    pipeline = create_baseline_pipeline(C=C, gamma=gamma, kernel="rbf", cache_size=cache_size)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    fold_time = time.time() - t_start

    # Metrics
    metrics = compute_classification_metrics(y_test, y_pred)
    fold_record: Dict[str, Union[int, float]] = {
        "fold": fold_idx,
        "test_subject": test_sub,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "fit_time_seconds": round(fold_time, 2),
        **metrics,
    }

    # Prediction records with full metadata alignment
    test_meta = metadata.iloc[test_indices].copy()
    test_meta["predicted_label"] = y_pred
    test_meta["predicted_label_name"] = [LABEL_NAMES[p] for p in y_pred]
    test_meta["is_correct"] = (y_pred == y_test).astype(int)
    test_meta["fold"] = fold_idx

    pred_df = test_meta[[
        "fold",
        "subject_id",
        "session_id",
        "trial_id",
        "window_id",
        "label",
        "label_name",
        "predicted_label",
        "predicted_label_name",
        "is_correct",
    ]]

    return fold_record, pred_df


def run_baseline_loso(
    config_path: Union[str, Path] = "config.yaml",
    n_jobs: int = 4,
    cache_size_per_worker: int = 500,
) -> Dict[str, Union[pd.DataFrame, dict]]:
    """
    Run the full 15-fold Leave-One-Subject-Out (LOSO) baseline.
    
    Parameters:
        config_path: Path to master config.yaml.
        n_jobs: Number of parallel fold workers. (Recommended: 4 on multi-core CPU).
        cache_size_per_worker: Libsvm kernel cache size in MB per worker.
        
    Returns:
        Dictionary containing fold metrics, pooled metrics, summary metrics, and predictions.
    """
    cfg = load_config(config_path)
    print("=" * 78)
    print("PHASE B: LEAKAGE-SAFE LOSO BASELINE (All 62 Channels, RBF-SVM)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parallel fold workers (n_jobs): {n_jobs}")
    print("=" * 78)

    t0 = time.time()
    data = load_seed_iv_dataset(config=cfg)
    load_time = time.time() - t0
    print(f"Loaded SEED-IV dataset in {load_time:.2f}s: X={data.X.shape}, y={data.y.shape}")

    splits = list(get_loso_splits(data.metadata))
    print(f"Generated {len(splits)} outer LOSO folds. All subject disjointness checks PASSED.")

    # Execute 15 folds
    print(f"\nStarting 15 outer LOSO folds (Parallel execution with {n_jobs} workers)...")
    t_train_start = time.time()

    results = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
        joblib.delayed(run_single_fold)(
            fold_idx=fold_idx,
            test_sub=test_sub,
            train_indices=train_indices,
            test_indices=test_indices,
            X=data.X,
            y=data.y,
            metadata=data.metadata,
            C=1.0,
            gamma="scale",
            cache_size=cache_size_per_worker,
        )
        for fold_idx, test_sub, train_indices, test_indices in splits
    )

    total_training_time = time.time() - t_train_start
    print(f"\nAll 15 LOSO folds completed in {total_training_time:.2f}s ({total_training_time/60:.2f} min).")

    # Aggregate results
    fold_records = [r[0] for r in results]
    pred_dfs = [r[1] for r in results]

    fold_metrics_df = pd.DataFrame(fold_records)
    all_predictions_df = pd.concat(pred_dfs, ignore_index=True)

    # Sort predictions deterministically by subject_id, session_id, trial_id, window_id
    all_predictions_df.sort_values(
        by=["subject_id", "session_id", "trial_id", "window_id"],
        inplace=True
    )
    all_predictions_df.reset_index(drop=True, inplace=True)

    assert len(all_predictions_df) == 37575, (
        f"Expected exactly 37,575 predictions, got {len(all_predictions_df)}"
    )

    # 1. Pooled sample-level metrics (calculated across all 37,575 windows)
    pooled_metrics = compute_classification_metrics(
        y_true=all_predictions_df["label"].to_numpy(),
        y_pred=all_predictions_df["predicted_label"].to_numpy(),
    )

    # 2. Per-subject summary (mean and std across the 15 folds)
    subject_mean_acc = float(fold_metrics_df["accuracy"].mean())
    subject_std_acc = float(fold_metrics_df["accuracy"].std())
    subject_mean_bal_acc = float(fold_metrics_df["balanced_accuracy"].mean())
    subject_std_bal_acc = float(fold_metrics_df["balanced_accuracy"].std())
    subject_mean_macro_f1 = float(fold_metrics_df["macro_f1"].mean())
    subject_std_macro_f1 = float(fold_metrics_df["macro_f1"].std())

    summary_rows = [
        {
            "metric_type": "pooled_window_level",
            "description": "Computed directly across all 37,575 pooled out-of-subject predictions",
            "accuracy": pooled_metrics["accuracy"],
            "balanced_accuracy": pooled_metrics["balanced_accuracy"],
            "macro_f1": pooled_metrics["macro_f1"],
            "macro_precision": pooled_metrics["macro_precision"],
            "macro_recall": pooled_metrics["macro_recall"],
            "accuracy_std": 0.0,
            "balanced_accuracy_std": 0.0,
            "macro_f1_std": 0.0,
        },
        {
            "metric_type": "subject_level_average",
            "description": "Mean and standard deviation across the 15 held-out subjects",
            "accuracy": subject_mean_acc,
            "balanced_accuracy": subject_mean_bal_acc,
            "macro_f1": subject_mean_macro_f1,
            "macro_precision": float(fold_metrics_df["macro_precision"].mean()),
            "macro_recall": float(fold_metrics_df["macro_recall"].mean()),
            "accuracy_std": subject_std_acc,
            "balanced_accuracy_std": subject_std_bal_acc,
            "macro_f1_std": subject_std_macro_f1,
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    # 3. Save outputs
    paths = cfg.get("paths", {})
    preds_dir = Path(paths.get("predictions_dir", "outputs/predictions"))
    metrics_dir = Path(paths.get("metrics_dir", "outputs/metrics"))
    tables_dir = Path(paths.get("tables_dir", "outputs/tables"))
    figures_dir = Path(paths.get("figures_dir", "outputs/figures"))

    for d in [preds_dir, metrics_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save predictions
    preds_file = preds_dir / "baseline_predictions.csv"
    all_predictions_df.to_csv(preds_file, index=False)
    print(f"Saved: {preds_file} ({len(all_predictions_df)} rows)")

    # Save fold metrics
    fold_file = metrics_dir / "baseline_fold_metrics.csv"
    fold_metrics_df.to_csv(fold_file, index=False)
    print(f"Saved: {fold_file} (15 fold rows)")

    # Save summary metrics
    summary_file = metrics_dir / "baseline_summary_metrics.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved: {summary_file}")

    # Save confusion matrix CSV
    cm_df = compute_confusion_matrix_df(
        y_true=all_predictions_df["label"].to_numpy(),
        y_pred=all_predictions_df["predicted_label"].to_numpy(),
    )
    cm_file = tables_dir / "baseline_confusion_matrix.csv"
    cm_df.to_csv(cm_file)
    print(f"Saved: {cm_file}")

    # Save confusion matrix figure
    cm_fig_file = figures_dir / "baseline_confusion_matrix.png"
    plot_and_save_confusion_matrix(
        y_true=all_predictions_df["label"].to_numpy(),
        y_pred=all_predictions_df["predicted_label"].to_numpy(),
        output_path=cm_fig_file,
        title="Baseline RBF-SVM Confusion Matrix (Pooled 15-Fold LOSO)",
    )
    print(f"Saved: {cm_fig_file}")

    # Save experiment execution report
    report_dict = {
        "timestamp": datetime.now().isoformat(),
        "total_runtime_seconds": round(time.time() - t0, 2),
        "training_time_seconds": round(total_training_time, 2),
        "n_workers": n_jobs,
        "n_samples": len(all_predictions_df),
        "n_channels": 62,
        "n_features": 310,
        "model": "RBF-SVM",
        "C": 1.0,
        "gamma": "scale",
        "pooled_accuracy": pooled_metrics["accuracy"],
        "pooled_balanced_accuracy": pooled_metrics["balanced_accuracy"],
        "pooled_macro_f1": pooled_metrics["macro_f1"],
        "subject_mean_accuracy": subject_mean_acc,
        "subject_std_accuracy": subject_std_acc,
        "subject_mean_balanced_accuracy": subject_mean_bal_acc,
        "subject_std_balanced_accuracy": subject_std_bal_acc,
        "subject_mean_macro_f1": subject_mean_macro_f1,
        "subject_std_macro_f1": subject_std_macro_f1,
        "per_subject_accuracies": {
            f"sub_{int(r['test_subject'])}": round(float(r['accuracy']), 4)
            for _, r in fold_metrics_df.iterrows()
        },
        "leakage_checks_passed": True,
    }
    report_file = metrics_dir / "baseline_run_report.json"
    with open(report_file, "w", encoding="utf-8") as fp:
        json.dump(report_dict, fp, indent=2)
    print(f"Saved: {report_file}")

    # Print summary table to stdout
    print("\n" + "=" * 78)
    print("PHASE B LOSO BASELINE RESULTS SUMMARY:")
    print("=" * 78)
    print(f"Pooled Window Accuracy:    {pooled_metrics['accuracy']*100:.2f}%")
    print(f"Pooled Balanced Accuracy:  {pooled_metrics['balanced_accuracy']*100:.2f}%")
    print(f"Pooled Macro-F1:           {pooled_metrics['macro_f1']*100:.2f}%")
    print("-" * 78)
    print(f"Subject Mean Accuracy:     {subject_mean_acc*100:.2f}% ± {subject_std_acc*100:.2f}%")
    print(f"Subject Mean Balanced Acc: {subject_mean_bal_acc*100:.2f}% ± {subject_std_bal_acc*100:.2f}%")
    print(f"Subject Mean Macro-F1:     {subject_mean_macro_f1*100:.2f}% ± {subject_std_macro_f1*100:.2f}%")
    print("-" * 78)
    print("Per-Subject Accuracy Breakdown:")
    for _, r in fold_metrics_df.iterrows():
        print(f"  Subject {int(r['test_subject']):02d}: Acc = {r['accuracy']*100:5.2f}% | "
              f"BalAcc = {r['balanced_accuracy']*100:5.2f}% | "
              f"Macro-F1 = {r['macro_f1']*100:5.2f}% (Fit: {r['fit_time_seconds']:5.1f}s)")
    print("=" * 78)

    return {
        "fold_metrics": fold_metrics_df,
        "summary_metrics": summary_df,
        "predictions": all_predictions_df,
        "confusion_matrix": cm_df,
        "report": report_dict,
    }


if __name__ == "__main__":
    run_baseline_loso(n_jobs=4)
