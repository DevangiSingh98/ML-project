"""
Evaluate Classifier Performance of Corrected Stability-Aware Channel Selection.

Compares 3 conditions across all 15 outer Leave-One-Subject-Out (LOSO) folds on genuine SEED-IV data:
1. all_62: All 62 physical EEG channels (310 features) - Upper reference baseline
2. corr_stab_32: 32 channels selected by Corrected Stability-Aware criterion (160 features) - Primary condition
3. pred_32: 32 channels selected by Predictive Importance criterion (160 features) - Feature selection reference

Methodology guarantees:
- 15 outer LOSO folds strictly isolating the held-out subject.
- StandardScaler fitted on outer-training subjects only.
- Channel selection uses only outer-training subjects.
- Physical channel grouping strictly preserved (5 frequency bands per channel).
- Dual prediction rules recorded: LibSVM discrete pairwise majority voting (SVC.predict) and continuous margin argmax.
- Checkpoint / resume capability saving progress at the fold level.

Outputs:
- outputs/predictions/corrected_channel_performance_predictions.csv
- outputs/metrics/corrected_channel_performance_fold_metrics.csv
- outputs/tables/corrected_channel_performance_comparison.csv
- outputs/tables/corrected_channel_performance_subject_summary.csv
- outputs/figures/corrected_channel_performance_comparison.png
- outputs/metrics/corrected_channel_performance_run_report.json
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data_loader import load_seed_iv_dataset, load_config, LABEL_NAMES
from src.subject_split import get_loso_splits
from src.channel_features import slice_channel_features, get_channel_names
from src.channel_selection import (
    compute_inner_grouped_permutation_importance,
    compute_channel_stability_scores,
)
from src.preprocessing import create_baseline_pipeline
from src.metrics import compute_classification_metrics


def load_or_compute_fold_channels(
    fold_idx: int,
    test_sub: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_meta: pd.DataFrame,
    all_channel_names: List[str],
    selections_df: Optional[pd.DataFrame] = None,
    budget: int = 32,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Obtain the 32 channel indices for corrected stability and predictive importance.
    Uses verified cached selections from Stage 2 if available, or computes fresh.
    """
    ch_to_idx = {name: i for i, name in enumerate(all_channel_names)}

    if selections_df is not None:
        # Match by fold or test_subject and budget
        match = selections_df[
            (selections_df["test_subject"] == test_sub) & (selections_df["budget"] == budget)
        ]
        if len(match) > 0:
            row = match.iloc[0]
            corr_names = [c.strip() for c in row["corrected_stability_channels"].split(",")]
            pred_names = [c.strip() for c in row["predictive_importance_channels"].split(",")]
            if len(corr_names) == budget and len(pred_names) == budget:
                corr_indices = sorted([ch_to_idx[c] for c in corr_names])
                pred_indices = sorted([ch_to_idx[c] for c in pred_names])
                return corr_indices, pred_indices

    # Fresh computation on training data only
    importance_mat, ranks_mat, avg_imp = compute_inner_grouped_permutation_importance(
        X_train=X_train,
        y_train=y_train,
        train_metadata=train_meta,
        n_splits=3,
        seed=seed + fold_idx,
    )
    scores = compute_channel_stability_scores(
        avg_importance=avg_imp,
        ranks_matrix=ranks_mat,
        total_channels=len(all_channel_names),
        method="multiplicative",
    )
    corr_indices = sorted(scores["rank_order"][:budget].tolist())
    pred_indices = sorted(np.argsort(-avg_imp)[:budget].tolist())

    return corr_indices, pred_indices


def evaluate_corrected_channel_performance(
    config_path: str = "config.yaml",
    selections_csv_path: str = "outputs/tables/corrected_channel_selections.csv",
    predictions_output_path: str = "outputs/predictions/corrected_channel_performance_predictions.csv",
    fold_metrics_output_path: str = "outputs/metrics/corrected_channel_performance_fold_metrics.csv",
    comparison_table_path: str = "outputs/tables/corrected_channel_performance_comparison.csv",
    subject_summary_path: str = "outputs/tables/corrected_channel_performance_subject_summary.csv",
    figure_output_path: str = "outputs/figures/corrected_channel_performance_comparison.png",
    report_json_path: str = "outputs/metrics/corrected_channel_performance_run_report.json",
    cache_size: int = 1000,
) -> Dict[str, Any]:
    """
    Main evaluation pipeline across all 15 outer LOSO folds.
    """
    start_total_time = time.time()
    config = load_config(config_path)

    print("=" * 80)
    print("STAGE 2.5: EVALUATING CORRECTED STABILITY-AWARE CHANNEL PERFORMANCE")
    print("Conditions: 1. all_62 | 2. corr_stab_32 | 3. pred_32")
    print("Evaluation Protocol: 15-Fold Outer LOSO Cross-Validation on Genuine SEED-IV")
    print("=" * 80)

    dataset = load_seed_iv_dataset(config=config)
    X = dataset.X
    y = dataset.y
    metadata = dataset.metadata
    all_channel_names = dataset.channels
    total_samples = len(X)
    print(f"Loaded dataset: {total_samples} windows, {X.shape[1]} features, {len(all_channel_names)} physical channels.")

    # Load verified Stage 2 selections if available
    selections_df = None
    if os.path.exists(selections_csv_path):
        selections_df = pd.read_csv(selections_csv_path)
        print(f"Loaded verified Stage 2 channel selections from: {selections_csv_path}")

    # Check for existing checkpoint results to allow resumption if interrupted
    completed_folds = set()
    existing_pred_dfs = []
    existing_fold_metrics = []

    if os.path.exists(fold_metrics_output_path) and os.path.exists(predictions_output_path):
        try:
            m_chk = pd.read_csv(fold_metrics_output_path)
            # A fold is complete if all 3 conditions are recorded
            for f, grp in m_chk.groupby("fold"):
                if set(grp["condition"].unique()) == {"all_62", "corr_stab_32", "pred_32"}:
                    completed_folds.add(f)
            if completed_folds:
                print(f"Found existing checkpoint with {len(completed_folds)} completed folds: {sorted(completed_folds)}")
                existing_fold_metrics = m_chk[m_chk["fold"].isin(completed_folds)].to_dict("records")
                p_chk = pd.read_csv(predictions_output_path)
                existing_pred_dfs = [p_chk[p_chk["fold"].isin(completed_folds)]]
        except Exception as e:
            print(f"Warning: Could not parse checkpoint ({e}). Starting fresh.")
            completed_folds = set()

    fold_metrics_records = list(existing_fold_metrics)
    all_pred_dfs = list(existing_pred_dfs)

    os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(fold_metrics_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(comparison_table_path), exist_ok=True)
    os.makedirs(os.path.dirname(figure_output_path), exist_ok=True)

    for fold_idx, test_sub, train_idx, test_idx in get_loso_splits(metadata, subject_key="subject_id"):
        if fold_idx in completed_folds:
            print(f"Fold {fold_idx:2d}/15 (Test Subject {test_sub:2d}) already completed in checkpoint. Skipping.")
            continue

        fold_start = time.time()
        print(f"\n--- Fold {fold_idx:2d}/15 (Held-out Subject {test_sub:2d}) ---")

        train_meta = metadata.iloc[train_idx].reset_index(drop=True)
        test_meta = metadata.iloc[test_idx].copy().reset_index(drop=True)

        # Strict leakage assertion
        assert test_sub not in train_meta["subject_id"].unique(), f"Leakage: subject {test_sub} in train!"

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Determine channels for this outer fold
        corr_32_idx, pred_32_idx = load_or_compute_fold_channels(
            fold_idx=fold_idx,
            test_sub=test_sub,
            X_train=X_train,
            y_train=y_train,
            train_meta=train_meta,
            all_channel_names=all_channel_names,
            selections_df=selections_df,
            budget=32,
            seed=42,
        )

        conditions: Dict[str, List[int]] = {
            "all_62": list(range(62)),
            "corr_stab_32": corr_32_idx,
            "pred_32": pred_32_idx,
        }

        fold_preds = []

        for cond_name, ch_indices in conditions.items():
            t_cond_start = time.time()
            n_ch = len(ch_indices)

            # Slicing indivisible 5 frequency bands per channel
            X_tr_sub, _ = slice_channel_features(X_train, ch_indices)
            X_te_sub, _ = slice_channel_features(X_test, ch_indices)

            # Baseline RBF-SVM Pipeline (StandardScaler fitted on outer-training subjects only)
            pipe = create_baseline_pipeline(C=1.0, gamma="scale", cache_size=cache_size)
            t_fit_start = time.time()
            pipe.fit(X_tr_sub, y_train)
            fit_time = time.time() - t_fit_start

            # Predictions: Primary rule (SVC.predict via LibSVM voting)
            t_pred_start = time.time()
            y_pred_svc = pipe.predict(X_te_sub)
            pred_time = time.time() - t_pred_start

            # Predictions: Continuous decision margins (one-vs-rest / pairwise margins)
            dec_scores = pipe.decision_function(X_te_sub)
            y_pred_margin = np.argmax(dec_scores, axis=1)

            # Compute classification metrics for primary rule
            m_svc = compute_classification_metrics(y_test, y_pred_svc)
            # Compute classification metrics for margin argmax rule
            m_margin = compute_classification_metrics(y_test, y_pred_margin)

            ch_names_str = ",".join([all_channel_names[c] for c in ch_indices])

            fold_record = {
                "fold": fold_idx,
                "test_subject": test_sub,
                "condition": cond_name,
                "channels_count": n_ch,
                "accuracy": m_svc["accuracy"],
                "balanced_accuracy": m_svc["balanced_accuracy"],
                "macro_f1": m_svc["macro_f1"],
                "margin_accuracy": m_margin["accuracy"],
                "margin_balanced_accuracy": m_margin["balanced_accuracy"],
                "margin_macro_f1": m_margin["macro_f1"],
                "decision_rule_agreement": float(np.mean(y_pred_svc == y_pred_margin)),
                "recall_neutral": round(m_svc.get("recall_neutral", 0.0), 6),
                "recall_sad": round(m_svc.get("recall_sad", 0.0), 6),
                "recall_fear": round(m_svc.get("recall_fear", 0.0), 6),
                "recall_happy": round(m_svc.get("recall_happy", 0.0), 6),
                "fit_time_seconds": round(fit_time, 3),
                "predict_time_seconds": round(pred_time, 3),
                "selected_channels": ch_names_str,
            }
            fold_metrics_records.append(fold_record)

            # Build prediction dataframe
            pred_df = test_meta.copy()
            pred_df["fold"] = fold_idx
            pred_df["condition"] = cond_name
            pred_df["channels_count"] = n_ch
            pred_df["true_label"] = y_test
            pred_df["true_label_name"] = [LABEL_NAMES[y_i] for y_i in y_test]
            pred_df["predicted_label"] = y_pred_svc
            pred_df["predicted_label_name"] = [LABEL_NAMES[p_i] for p_i in y_pred_svc]
            pred_df["is_correct"] = (y_pred_svc == y_test).astype(int)
            pred_df["margin_predicted_label"] = y_pred_margin
            pred_df["margin_is_correct"] = (y_pred_margin == y_test).astype(int)

            for c_i in range(4):
                pred_df[f"decision_score_{c_i}"] = dec_scores[:, c_i]

            fold_preds.append(pred_df)

            cond_elapsed = time.time() - t_cond_start
            print(
                f"  [{cond_name:12s} ({n_ch:2d} ch)] Acc: {m_svc['accuracy']*100:5.2f}% | "
                f"BalAcc: {m_svc['balanced_accuracy']*100:5.2f}% | F1: {m_svc['macro_f1']*100:5.2f}% | "
                f"(fit: {fit_time:5.1f}s, total: {cond_elapsed:5.1f}s)",
                flush=True,
            )

        all_pred_dfs.extend(fold_preds)

        # Checkpoint: Save after each fold to prevent data loss
        pd.DataFrame(fold_metrics_records).to_csv(fold_metrics_output_path, index=False)
        pd.concat(all_pred_dfs, ignore_index=True).to_csv(predictions_output_path, index=False)

        fold_elapsed = time.time() - fold_start
        print(f"Fold {fold_idx:2d} complete in {fold_elapsed:.1f}s. Checkpoint saved.", flush=True)

    total_pipeline_time = time.time() - start_total_time
    print(f"\nAll 15 LOSO folds complete in {total_pipeline_time:.1f}s ({total_pipeline_time/60:.2f} min).")

    # Combine all predictions and fold metrics
    full_pred_df = pd.concat(all_pred_dfs, ignore_index=True)
    full_metrics_df = pd.DataFrame(fold_metrics_records)

    # 1. Summary Comparison Across Conditions
    conditions_order = ["all_62", "corr_stab_32", "pred_32"]
    comparison_records = []
    baseline_acc = None
    baseline_bal = None
    baseline_f1 = None

    for cond in conditions_order:
        cond_preds = full_pred_df[full_pred_df["condition"] == cond]
        cond_folds = full_metrics_df[full_metrics_df["condition"] == cond]

        # Pooled metrics across all 37,575 windows
        y_true_all = cond_preds["true_label"].to_numpy()
        y_pred_all = cond_preds["predicted_label"].to_numpy()
        pooled_m = compute_classification_metrics(y_true_all, y_pred_all)

        # Margin argmax pooled accuracy
        y_margin_all = cond_preds["margin_predicted_label"].to_numpy()
        pooled_margin_acc = float(np.mean(y_margin_all == y_true_all))

        # Subject-level mean and std
        acc_mean = float(cond_folds["accuracy"].mean())
        acc_std = float(cond_folds["accuracy"].std())
        bal_mean = float(cond_folds["balanced_accuracy"].mean())
        bal_std = float(cond_folds["balanced_accuracy"].std())
        f1_mean = float(cond_folds["macro_f1"].mean())
        f1_std = float(cond_folds["macro_f1"].std())

        rule_agreement = float(np.mean(cond_folds["decision_rule_agreement"]))
        n_ch = 62 if cond == "all_62" else 32

        if cond == "all_62":
            baseline_acc = pooled_m["accuracy"]
            baseline_bal = pooled_m["balanced_accuracy"]
            baseline_f1 = pooled_m["macro_f1"]
            delta_acc = 0.0
            delta_f1 = 0.0
        else:
            delta_acc = (pooled_m["accuracy"] - baseline_acc) * 100.0
            delta_f1 = (pooled_m["macro_f1"] - baseline_f1) * 100.0

        comparison_records.append({
            "condition": cond,
            "condition_label": "All 62 Channels (Baseline)" if cond == "all_62" else (
                "32 Corrected Stability Channels" if cond == "corr_stab_32" else "32 Predictive Importance Channels"
            ),
            "channels_retained": n_ch,
            "pct_channels_retained": round(n_ch / 62.0 * 100.0, 2),
            "pct_channels_removed": round((62 - n_ch) / 62.0 * 100.0, 2),
            "features_count": n_ch * 5,
            "pooled_accuracy": round(pooled_m["accuracy"], 6),
            "pooled_balanced_accuracy": round(pooled_m["balanced_accuracy"], 6),
            "pooled_macro_f1": round(pooled_m["macro_f1"], 6),
            "pooled_margin_accuracy": round(pooled_margin_acc, 6),
            "subject_acc_mean": round(acc_mean, 6),
            "subject_acc_std": round(acc_std, 6),
            "subject_bal_acc_mean": round(bal_mean, 6),
            "subject_bal_acc_std": round(bal_std, 6),
            "subject_f1_mean": round(f1_mean, 6),
            "subject_f1_std": round(f1_std, 6),
            "delta_pooled_acc_pp": round(delta_acc, 3),
            "delta_pooled_f1_pp": round(delta_f1, 3),
            "rule_agreement": round(rule_agreement, 4),
        })

    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(comparison_table_path, index=False)

    # 2. Per-Subject Summary Table
    sub_records = []
    unique_subs = sorted(metadata["subject_id"].unique())
    for s_id in unique_subs:
        row = {"subject_id": s_id}
        for cond in conditions_order:
            sub_f = full_metrics_df[
                (full_metrics_df["test_subject"] == s_id) & (full_metrics_df["condition"] == cond)
            ]
            if len(sub_f) > 0:
                row[f"{cond}_accuracy"] = round(sub_f.iloc[0]["accuracy"], 6)
                row[f"{cond}_macro_f1"] = round(sub_f.iloc[0]["macro_f1"], 6)
        row["corr_stab_vs_all62_diff"] = round(
            row["corr_stab_32_accuracy"] - row["all_62_accuracy"], 6
        )
        row["corr_stab_vs_pred32_diff"] = round(
            row["corr_stab_32_accuracy"] - row["pred_32_accuracy"], 6
        )
        sub_records.append(row)

    subject_summary_df = pd.DataFrame(sub_records)
    subject_summary_df.to_csv(subject_summary_path, index=False)

    # 3. Generate Publication Figure
    generate_comparison_figure(
        comparison_df=comparison_df,
        subject_summary_df=subject_summary_df,
        full_pred_df=full_pred_df,
        output_path=figure_output_path,
    )

    # 4. Generate Machine-Readable Run Report
    run_report = {
        "title": "Stage 2.5: Corrected Stability-Aware Channel Performance Evaluation",
        "dataset": "SEED-IV",
        "total_windows": total_samples,
        "loso_folds": 15,
        "total_runtime_seconds": round(total_pipeline_time, 2),
        "total_runtime_minutes": round(total_pipeline_time / 60.0, 2),
        "conditions_evaluated": conditions_order,
        "model_parameters": {
            "classifier": "SVC",
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "scaler": "StandardScaler (training subjects only)",
        },
        "comparison_results": comparison_df.to_dict("records"),
        "per_subject_summary": subject_summary_df.to_dict("records"),
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 2.5 PERFORMANCE SUMMARY:")
    print("=" * 80)
    print(comparison_df[[
        "condition", "channels_retained", "pct_channels_removed",
        "pooled_accuracy", "pooled_macro_f1", "subject_acc_mean", "subject_acc_std", "delta_pooled_acc_pp"
    ]].to_string(index=False))

    return run_report


def generate_comparison_figure(
    comparison_df: pd.DataFrame,
    subject_summary_df: pd.DataFrame,
    full_pred_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Generate high-resolution 3-panel comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Panel 1: Bar chart of Pooled Performance Metrics
    ax1 = axes[0]
    conds = ["all_62", "corr_stab_32", "pred_32"]
    labels = ["All 62 Ch\n(100% Ch)", "32 Corrected\nStability (51.6%)", "32 Predictive\nImportance (51.6%)"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    accs = [comparison_df.loc[comparison_df["condition"] == c, "pooled_accuracy"].values[0] * 100 for c in conds]
    bals = [comparison_df.loc[comparison_df["condition"] == c, "pooled_balanced_accuracy"].values[0] * 100 for c in conds]
    f1s = [comparison_df.loc[comparison_df["condition"] == c, "pooled_macro_f1"].values[0] * 100 for c in conds]

    x = np.arange(len(conds))
    width = 0.25

    rects1 = ax1.bar(x - width, accs, width, label="Accuracy", color=colors[0], alpha=0.9)
    rects2 = ax1.bar(x, bals, width, label="Balanced Acc", color=colors[1], alpha=0.9)
    rects3 = ax1.bar(x + width, f1s, width, label="Macro-F1", color=colors[2], alpha=0.9)

    ax1.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Pooled Performance Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(25, 45)
    ax1.axhline(25.0, color="gray", linestyle="--", linewidth=0.8, label="Chance (25%)")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Add value annotations
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Panel 2: Per-Subject Paired Accuracy Trajectories
    ax2 = axes[1]
    subs = subject_summary_df["subject_id"].values
    sub_acc_62 = subject_summary_df["all_62_accuracy"].values * 100
    sub_acc_corr = subject_summary_df["corr_stab_32_accuracy"].values * 100
    sub_acc_pred = subject_summary_df["pred_32_accuracy"].values * 100

    for i in range(len(subs)):
        ax2.plot([0, 1, 2], [sub_acc_62[i], sub_acc_corr[i], sub_acc_pred[i]],
                 color="gray", alpha=0.4, linewidth=1.0)

    # Plot subject points
    ax2.scatter(np.zeros(len(subs)), sub_acc_62, color=colors[0], s=35, zorder=3, label="All 62")
    ax2.scatter(np.ones(len(subs)), sub_acc_corr, color=colors[1], s=35, zorder=3, label="Corr Stab 32")
    ax2.scatter(np.full(len(subs), 2), sub_acc_pred, color=colors[2], s=35, zorder=3, label="Pred 32")

    # Means
    mean_62 = np.mean(sub_acc_62)
    mean_corr = np.mean(sub_acc_corr)
    mean_pred = np.mean(sub_acc_pred)
    ax2.plot([0, 1, 2], [mean_62, mean_corr, mean_pred], color="black", marker="D",
             markersize=8, linewidth=2.5, zorder=4, label="Subject Mean")

    ax2.set_title("B. Subject-Level Accuracy Trajectories", fontsize=12, fontweight="bold")
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["All 62", "Corr Stab 32", "Pred 32"], fontsize=10)
    ax2.set_ylabel("Subject Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Panel 3: Confusion Matrix for 32 Corrected Stability Channels
    ax3 = axes[2]
    corr_preds = full_pred_df[full_pred_df["condition"] == "corr_stab_32"]
    cm = confusion_matrix(corr_preds["true_label"], corr_preds["predicted_label"], labels=[0, 1, 2, 3])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    im = ax3.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=0.6)
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    classes = [LABEL_NAMES[i] for i in range(4)]
    tick_marks = np.arange(len(classes))
    ax3.set_xticks(tick_marks)
    ax3.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax3.set_yticks(tick_marks)
    ax3.set_yticklabels(classes, fontsize=9)
    ax3.set_title("C. 32 Corrected Stability Confusion Matrix", fontsize=12, fontweight="bold")
    ax3.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")

    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax3.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})",
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved publication comparison figure to: {output_path}")


if __name__ == "__main__":
    evaluate_corrected_channel_performance()
