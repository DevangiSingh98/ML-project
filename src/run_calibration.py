"""
Phase D Experiment Runner: Leakage-Safe Probability Calibration & Uncertainty Estimation.

Evaluates 4 Channel Conditions:
1. all_62: All 62 channels (reference baseline)
2. pred_32: 32 channels selected by Predictive Importance on training subjects
3. stab_32: 32 channels selected by Stability-Aware criterion on training subjects
4. pred_12: 12 channels selected by Predictive Importance on training subjects

Evaluates Probability Calibration Regimes:
A. uncalibrated_softmax: Raw decision scores passed through standard temperature-1 softmax
B. group_safe_temperature: Temperature scaling (optimal T fitted on subject-disjoint out-of-fold scores)
C. group_safe_platt: Multiclass vector scaling (multinomial logistic regression on out-of-fold scores)

Guarantees:
- Strict subject disjointness: Outer test subjects never contribute to channel selection,
  scaling, model fitting, out-of-fold calibration generation, or calibrator training.
- Complete output persistence and reproducibility across all 15 outer LOSO folds.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import json
import time
from typing import Dict, List, Optional, Tuple, Union

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from src.data_loader import load_seed_iv_dataset, load_config, LABEL_NAMES
from src.subject_split import get_loso_splits
from src.channel_features import slice_channel_features
from src.channel_selection import (
    compute_inner_grouped_permutation_importance,
)
from src.preprocessing import create_baseline_pipeline
from src.metrics import compute_classification_metrics
from src.calibration import (
    generate_group_safe_oof_decision_scores,
    GroupSafeTemperatureScalingCalibrator,
    MulticlassPlattCalibrator,
    UncalibratedSoftmaxCalibrator,
)
from src.uncertainty_metrics import (
    compute_uncertainty_scores,
    compute_probability_metrics,
    compute_error_detection_metrics,
    compute_expected_calibration_error,
)


def run_single_fold_calibration(
    fold_idx: int,
    test_sub: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    all_channel_names: List[str],
    cache_size: int = 500,
) -> Tuple[List[dict], List[pd.DataFrame]]:
    """
    Execute Phase D calibration for a single outer LOSO fold across all conditions.
    """
    train_subs = set(metadata.iloc[train_indices]["subject_id"].unique())
    test_subs = set(metadata.iloc[test_indices]["subject_id"].unique())
    assert train_subs.isdisjoint(test_subs), f"Fold {fold_idx} leakage detected!"
    assert test_subs == {test_sub}

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    train_meta = metadata.iloc[train_indices].reset_index(drop=True)
    test_meta = metadata.iloc[test_indices].copy().reset_index(drop=True)

    # 1. Training-only channel rankings (inner splits on 14 training subjects)
    importance_mat, ranks_mat, avg_imp = compute_inner_grouped_permutation_importance(
        X_train=X_train,
        y_train=y_train,
        train_metadata=train_meta,
        n_splits=3,
        seed=42 + fold_idx,
    )
    pred_rank_order = np.argsort(-avg_imp).tolist()

    min_i = np.min(avg_imp)
    max_i = np.max(avg_imp)
    denom_i = (max_i - min_i) if (max_i - min_i) > 1e-8 else 1.0
    I_norm = (avg_imp - min_i) / denom_i
    mean_ranks = np.mean(ranks_mat, axis=0)
    S_c = np.clip(1.0 - (mean_ranks - 1.0) / 61.0, 0.0, 1.0)
    Q_c = 0.5 * I_norm + 0.5 * S_c
    stab_rank_order = np.argsort(-Q_c).tolist()

    # Define conditions to evaluate
    conditions = {
        "all_62": list(range(62)),
        "pred_32": sorted(pred_rank_order[:32]),
        "stab_32": sorted(stab_rank_order[:32]),
        "pred_12": sorted(pred_rank_order[:12]),
    }

    fold_metric_records = []
    pred_dfs = []

    for cond_name, ch_indices in conditions.items():
        X_tr_sub, _ = slice_channel_features(X_train, ch_indices)
        X_te_sub, _ = slice_channel_features(X_test, ch_indices)
        k_channels = len(ch_indices)
        ch_str = ",".join([all_channel_names[c] for c in ch_indices])

        # Step A: Generate group-safe out-of-fold decision scores on outer training subjects
        Z_oof = generate_group_safe_oof_decision_scores(
            X_train=X_tr_sub,
            y_train=y_train,
            train_metadata=train_meta,
            group_key="subject_id",
            n_splits=3,
            C=1.0,
            gamma="scale",
            cache_size=cache_size,
        )

        # Step B: Fit calibrators on out-of-fold decision scores
        uncal_cal = UncalibratedSoftmaxCalibrator()
        uncal_cal.fit(Z_oof, y_train)

        temp_cal = GroupSafeTemperatureScalingCalibrator()
        temp_cal.fit(Z_oof, y_train)

        platt_cal = MulticlassPlattCalibrator()
        platt_cal.fit(Z_oof, y_train)

        # Step C: Fit final SVM on all outer training subjects
        final_pipe = create_baseline_pipeline(C=1.0, gamma="scale", cache_size=cache_size)
        t_fit_start = time.time()
        final_pipe.fit(X_tr_sub, y_train)
        fit_time = time.time() - t_fit_start

        # Step D: Predict raw decision scores for held-out subject
        Z_test = final_pipe.decision_function(X_te_sub)

        # Evaluate Uncalibrated, Temperature-Scaled, and Platt regimes
        for method_name, calibrator in [
            ("uncalibrated_softmax", uncal_cal),
            ("group_safe_temperature", temp_cal),
            ("group_safe_platt", platt_cal),
        ]:
            probs = calibrator.predict_proba(Z_test)
            preds = calibrator.predict(Z_test)

            # Basic classification metrics
            clf_metrics = compute_classification_metrics(y_test, preds)

            # Probability calibration metrics
            prob_metrics = compute_probability_metrics(y_test, probs, n_bins=10)

            # Uncertainty scores
            u_dict = compute_uncertainty_scores(probs)

            # Error detection metrics (for u_max and entropy)
            err_max = compute_error_detection_metrics(y_test, preds, u_dict["u_max"])
            err_ent = compute_error_detection_metrics(y_test, preds, u_dict["entropy"])

            # Fitted temperature if applicable
            fitted_temp = getattr(calibrator, "temperature_", np.nan)

            rec = {
                "fold": fold_idx,
                "test_subject": test_sub,
                "condition": cond_name,
                "channel_count": k_channels,
                "method": method_name,
                "fit_time_seconds": round(fit_time, 2),
                "temperature": round(fitted_temp, 3) if not np.isnan(fitted_temp) else np.nan,
                "selected_channels": ch_str,
                "accuracy": clf_metrics["accuracy"],
                "balanced_accuracy": clf_metrics["balanced_accuracy"],
                "macro_f1": clf_metrics["macro_f1"],
                "brier_score": prob_metrics["brier_score"],
                "log_loss": prob_metrics["log_loss"],
                "top_label_ece": prob_metrics["top_label_ece"],
                "classwise_ece": prob_metrics["classwise_ece"],
                "error_rate": err_max["error_rate"],
                "auroc_error_umax": err_max["auroc_error"],
                "auprc_error_umax": err_max["auprc_error"],
                "auroc_error_entropy": err_ent["auroc_error"],
                "auprc_error_entropy": err_ent["auprc_error"],
            }
            fold_metric_records.append(rec)

            # Predictions dataframe for this fold and method
            p_df = test_meta.copy()
            p_df["condition"] = cond_name
            p_df["channel_count"] = k_channels
            p_df["method"] = method_name
            p_df["temperature"] = fitted_temp
            p_df["true_label"] = y_test
            p_df["true_label_name"] = [LABEL_NAMES[y] for y in y_test]
            p_df["predicted_label"] = preds
            p_df["predicted_label_name"] = [LABEL_NAMES[p] for p in preds]
            p_df["prob_neutral"] = probs[:, 0]
            p_df["prob_sad"] = probs[:, 1]
            p_df["prob_fear"] = probs[:, 2]
            p_df["prob_happy"] = probs[:, 3]
            p_df["confidence"] = np.max(probs, axis=1)
            p_df["u_max"] = u_dict["u_max"]
            p_df["entropy"] = u_dict["entropy"]
            p_df["norm_entropy"] = u_dict["norm_entropy"]
            p_df["is_correct"] = (preds == y_test).astype(int)
            p_df["is_error"] = (preds != y_test).astype(int)
            pred_dfs.append(p_df)

    return fold_metric_records, pred_dfs


def run_phase_d_calibration(
    config_path: Union[str, Path] = "config.yaml",
    n_jobs: int = 4,
    cache_size: int = 500,
) -> Dict[str, Union[pd.DataFrame, dict]]:
    """
    Execute full Phase D calibration across all 15 outer LOSO folds.
    """
    t0 = time.time()
    cfg = load_config(config_path)

    print("=" * 80)
    print("PHASE D: LEAKAGE-SAFE PROBABILITY CALIBRATION & UNCERTAINTY ESTIMATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Conditions: all_62, pred_32, stab_32, pred_12")
    print("Methods: uncalibrated_softmax, group_safe_temperature, group_safe_platt")
    print(f"Parallel worker processes: {n_jobs}")
    print("=" * 80)

    dataset = load_seed_iv_dataset(config=cfg)
    X, y, metadata = dataset.X, dataset.y, dataset.metadata
    all_channel_names = dataset.channels
    splits = get_loso_splits(metadata)

    # Output paths
    pred_dir = Path("outputs/predictions")
    metrics_dir = Path("outputs/metrics")
    tables_dir = Path("outputs/tables")
    figures_dir = Path("outputs/figures")
    for d in [pred_dir, metrics_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\nLaunching 15 outer LOSO folds across workers...")
    fold_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_fold_calibration)(
            fold_idx=f_idx,
            test_sub=t_sub,
            train_indices=tr_idx,
            test_indices=te_idx,
            X=X,
            y=y,
            metadata=metadata,
            all_channel_names=all_channel_names,
            cache_size=cache_size,
        )
        for f_idx, t_sub, tr_idx, te_idx in splits
    )

    # Aggregate fold outputs
    all_metrics = []
    all_pred_dfs = []
    for f_metrics, f_preds in fold_results:
        all_metrics.extend(f_metrics)
        all_pred_dfs.extend(f_preds)

    fold_metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_pred_dfs, ignore_index=True)

    # Save primary predictions and fold metrics
    pred_path = pred_dir / "phase_d_calibrated_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"\nSaved: {pred_path} ({len(predictions_df)} prediction rows)")

    fold_metrics_path = metrics_dir / "phase_d_fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    print(f"Saved: {fold_metrics_path}")

    # Summary table across conditions and methods
    summary_records = []
    for (cond, method), grp in fold_metrics_df.groupby(["condition", "method"]):
        ch_count = grp["channel_count"].iloc[0]
        # Subject-level mean and std
        acc_mean = grp["accuracy"].mean()
        acc_std = grp["accuracy"].std()
        f1_mean = grp["macro_f1"].mean()
        f1_std = grp["macro_f1"].std()
        brier_mean = grp["brier_score"].mean()
        brier_std = grp["brier_score"].std()
        logloss_mean = grp["log_loss"].mean()
        logloss_std = grp["log_loss"].std()
        ece_mean = grp["top_label_ece"].mean()
        ece_std = grp["top_label_ece"].std()
        class_ece_mean = grp["classwise_ece"].mean()
        class_ece_std = grp["classwise_ece"].std()
        temp_mean = grp["temperature"].mean() if "temperature" in grp else np.nan

        # Pooled metrics across all 37,575 windows
        sub_preds = predictions_df[
            (predictions_df["condition"] == cond) & (predictions_df["method"] == method)
        ]
        p_true = sub_preds["true_label"].to_numpy()
        p_pred = sub_preds["predicted_label"].to_numpy()
        p_probs = sub_preds[["prob_neutral", "prob_sad", "prob_fear", "prob_happy"]].to_numpy()

        pooled_acc = float(np.mean(p_pred == p_true))
        pooled_prob_metrics = compute_probability_metrics(p_true, p_probs, n_bins=10)

        summary_records.append({
            "condition": cond,
            "channel_count": ch_count,
            "method": method,
            "mean_temperature": temp_mean,
            "pooled_accuracy": pooled_acc,
            "subject_accuracy_mean": acc_mean,
            "subject_accuracy_std": acc_std,
            "subject_macro_f1_mean": f1_mean,
            "subject_macro_f1_std": f1_std,
            "pooled_brier_score": pooled_prob_metrics["brier_score"],
            "subject_brier_mean": brier_mean,
            "subject_brier_std": brier_std,
            "pooled_log_loss": pooled_prob_metrics["log_loss"],
            "subject_log_loss_mean": logloss_mean,
            "subject_log_loss_std": logloss_std,
            "pooled_top_label_ece": pooled_prob_metrics["top_label_ece"],
            "subject_top_label_ece_mean": ece_mean,
            "subject_top_label_ece_std": ece_std,
            "pooled_classwise_ece": pooled_prob_metrics["classwise_ece"],
            "subject_classwise_ece_mean": class_ece_mean,
            "subject_classwise_ece_std": class_ece_std,
        })

    summary_df = pd.DataFrame(summary_records).sort_values(
        by=["channel_count", "condition", "method"], ascending=[False, True, False]
    ).reset_index(drop=True)
    summary_path = tables_dir / "phase_d_calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Error detection summary table
    err_records = []
    for (cond, method), grp in fold_metrics_df.groupby(["condition", "method"]):
        ch_count = grp["channel_count"].iloc[0]
        # Pooled AUROC/AUPRC
        sub_preds = predictions_df[
            (predictions_df["condition"] == cond) & (predictions_df["method"] == method)
        ]
        p_true = sub_preds["true_label"].to_numpy()
        p_pred = sub_preds["predicted_label"].to_numpy()
        p_umax = sub_preds["u_max"].to_numpy()
        p_ent = sub_preds["entropy"].to_numpy()

        p_err_umax = compute_error_detection_metrics(p_true, p_pred, p_umax)
        p_err_ent = compute_error_detection_metrics(p_true, p_pred, p_ent)

        err_records.append({
            "condition": cond,
            "channel_count": ch_count,
            "method": method,
            "error_rate": float(np.mean(p_pred != p_true)),
            "pooled_auroc_umax": p_err_umax["auroc_error"],
            "subject_auroc_umax_mean": grp["auroc_error_umax"].mean(),
            "subject_auroc_umax_std": grp["auroc_error_umax"].std(),
            "pooled_auprc_umax": p_err_umax["auprc_error"],
            "subject_auprc_umax_mean": grp["auprc_error_umax"].mean(),
            "subject_auprc_umax_std": grp["auprc_error_umax"].std(),
            "pooled_auroc_entropy": p_err_ent["auroc_error"],
            "subject_auroc_entropy_mean": grp["auroc_error_entropy"].mean(),
            "subject_auroc_entropy_std": grp["auroc_error_entropy"].std(),
            "pooled_auprc_entropy": p_err_ent["auprc_error"],
            "subject_auprc_entropy_mean": grp["auprc_error_entropy"].mean(),
            "subject_auprc_entropy_std": grp["auprc_error_entropy"].std(),
        })

    err_df = pd.DataFrame(err_records).sort_values(
        by=["channel_count", "condition", "method"], ascending=[False, True, False]
    ).reset_index(drop=True)
    err_path = tables_dir / "phase_d_error_detection_summary.csv"
    err_df.to_csv(err_path, index=False)
    print(f"Saved: {err_path}")

    # Generate Figures
    generate_phase_d_figures(predictions_df, figures_dir)

    # Machine-readable run report
    report_dict = {
        "timestamp": datetime.now().isoformat(),
        "total_runtime_seconds": round(time.time() - t0, 2),
        "n_workers": n_jobs,
        "n_samples": len(predictions_df) // len(summary_df),
        "conditions": summary_df["condition"].unique().tolist(),
        "methods": ["uncalibrated_softmax", "group_safe_temperature", "group_safe_platt"],
        "leakage_checks_passed": True,
        "calibration_summary": summary_df.to_dict(orient="records"),
        "error_detection_summary": err_df.to_dict(orient="records"),
    }
    report_path = metrics_dir / "phase_d_run_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report_dict, fp, indent=2)
    print(f"Saved: {report_path}")

    # Print summary
    print("\n" + "=" * 105)
    print("PHASE D: PROBABILITY CALIBRATION SUMMARY (Pooled & Subject Mean ± Std)")
    print("=" * 105)
    print(f"{'Condition':<12} {'Chs':<5} {'Method':<25} {'Pooled Acc':<12} {'Brier Score':<14} {'Top ECE':<12} {'Log Loss':<12}")
    print("-" * 105)
    for _, r in summary_df.iterrows():
        print(f"{r['condition']:<12} {int(r['channel_count']):<5} {r['method']:<25} "
              f"{r['pooled_accuracy']*100:6.2f}%      "
              f"{r['pooled_brier_score']:<14.4f} "
              f"{r['pooled_top_label_ece']*100:5.2f}%      "
              f"{r['pooled_log_loss']:<12.4f}")
    print("=" * 105)

    print("\n" + "=" * 105)
    print("PHASE D: UNCERTAINTY ERROR-DETECTION SUMMARY (AUROC & AUPRC)")
    print("=" * 105)
    print(f"{'Condition':<12} {'Chs':<5} {'Method':<25} {'Err Rate':<10} {'AUROC (u_max)':<16} {'AUPRC (u_max)':<16} {'AUROC (Entropy)':<16}")
    print("-" * 105)
    for _, r in err_df.iterrows():
        print(f"{r['condition']:<12} {int(r['channel_count']):<5} {r['method']:<25} "
              f"{r['error_rate']*100:5.2f}%     "
              f"{r['pooled_auroc_umax']:<16.4f} "
              f"{r['pooled_auprc_umax']:<16.4f} "
              f"{r['pooled_auroc_entropy']:<16.4f}")
    print("=" * 105)

    return {
        "summary": summary_df,
        "error_detection": err_df,
        "fold_metrics": fold_metrics_df,
        "predictions": predictions_df,
    }


def generate_phase_d_figures(predictions_df: pd.DataFrame, figures_dir: Path):
    """
    Generate publication-quality reliability diagrams and error-detection curves.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Reliability Diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, dpi=300)
    
    for ax_idx, cond in enumerate(["all_62", "pred_32"]):
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

        for method, color, marker, label in [
            ("uncalibrated_softmax", "#d95f02", "s", "Uncalibrated Softmax"),
            ("group_safe_temperature", "#377eb8", "^", "Group-Safe Temp Scaling"),
            ("group_safe_platt", "#1b9e77", "o", "Group-Safe Platt Scaling"),
        ]:
            sub_p = predictions_df[
                (predictions_df["condition"] == cond) & (predictions_df["method"] == method)
            ]
            y_t = sub_p["true_label"].to_numpy()
            probs = sub_p[["prob_neutral", "prob_sad", "prob_fear", "prob_happy"]].to_numpy()
            top_ece, _, details = compute_expected_calibration_error(y_t, probs, n_bins=10)

            mask = details["bin_counts"] > 0
            ax.plot(
                details["bin_confs"][mask],
                details["bin_accs"][mask],
                marker=marker,
                color=color,
                linewidth=2,
                label=f"{label} (ECE={top_ece*100:.2f}%)",
            )

        cond_title = "All 62 Channels" if cond == "all_62" else "32 Channels (Predictive Importance)"
        ax.set_title(f"Reliability Diagram: {cond_title}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean Confidence", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Empirical Accuracy", fontsize=11)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    rel_path = figures_dir / "phase_d_reliability_diagram.png"
    plt.savefig(rel_path, dpi=300)
    plt.close()
    print(f"Saved: {rel_path}")

    # 2. Error Detection ROC and PR Curves
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    colors = {
        "all_62": "#1f77b4",
        "pred_32": "#2ca02c",
        "stab_32": "#ff7f0e",
        "pred_12": "#9467bd",
    }
    labels = {
        "all_62": "All 62 Channels",
        "pred_32": "32 Ch (Predictive)",
        "stab_32": "32 Ch (Stability-Aware)",
        "pred_12": "12 Ch (Predictive)",
    }

    for cond, color in colors.items():
        sub_p = predictions_df[
            (predictions_df["condition"] == cond) & (predictions_df["method"] == "group_safe_temperature")
        ]
        is_err = sub_p["is_error"].to_numpy()
        umax = sub_p["u_max"].to_numpy()

        fpr, tpr, _ = roc_curve(is_err, umax)
        precision, recall, _ = precision_recall_curve(is_err, umax)
        err_metrics = compute_error_detection_metrics(sub_p["true_label"].to_numpy(), sub_p["predicted_label"].to_numpy(), umax)

        ax_roc.plot(fpr, tpr, color=color, linewidth=2, label=f"{labels[cond]} (AUROC={err_metrics['auroc_error']:.3f})")
        ax_pr.plot(recall, precision, color=color, linewidth=2, label=f"{labels[cond]} (AUPRC={err_metrics['auprc_error']:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (0.500)")
    ax_roc.set_title("Error Detection ROC Curves (u_max)", fontsize=13, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Error Detection)", fontsize=11)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=10)

    base_err_rate = np.mean(predictions_df["is_error"])
    ax_pr.axhline(base_err_rate, color="k", linestyle="--", alpha=0.6, label=f"Random Guess ({base_err_rate:.3f})")
    ax_pr.set_title("Error Detection Precision-Recall Curves (u_max)", fontsize=13, fontweight="bold")
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.legend(loc="lower left", frameon=True, fontsize=10)

    plt.tight_layout()
    det_path = figures_dir / "phase_d_uncertainty_error_detection.png"
    plt.savefig(det_path, dpi=300)
    plt.close()
    print(f"Saved: {det_path}")


if __name__ == "__main__":
    run_phase_d_calibration(n_jobs=4)
