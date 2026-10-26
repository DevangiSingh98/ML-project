"""
Trial-Level Prediction Aggregation Evaluation for SEED-IV EEG Emotion Recognition.

This module evaluates whole-trial emotion classification by aggregating window-level
predictions from existing Phase D Leave-One-Subject-Out models across each trial.

Evaluates:
- Method A: Mean-probability aggregation (soft voting)
- Method B: Majority voting (hard voting with deterministic probability tie-breaking)
Across 4 channel conditions ('all_62', 'pred_32', 'stab_32', 'pred_12') and 2 calibration
regimes ('group_safe_temperature', 'uncalibrated_softmax').
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL_NAMES: Dict[int, str] = {
    0: "neutral",
    1: "sad",
    2: "fear",
    3: "happy",
}

PROB_COLS: List[str] = ["prob_neutral", "prob_sad", "prob_fear", "prob_happy"]


def validate_trial_structure(df: pd.DataFrame) -> None:
    """
    Validate that every trial can be mapped unambiguously and satisfies all consistency rules.
    
    Checks:
    - Required columns present.
    - Exactly 1,080 unique trial presentations per (condition, method).
    - Exactly 72 trials per subject per (condition, method).
    - Zero contradictory ground-truth labels within any trial.
    - Every trial has at least one window (W >= 1).
    - No duplicate windows within any trial.
    """
    required_cols = [
        "subject_id", "session_id", "trial_id", "window_id",
        "condition", "method", "true_label", "predicted_label",
        "prob_neutral", "prob_sad", "prob_fear", "prob_happy"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Phase D predictions missing required columns: {missing}")

    # Check for duplicate windows
    dup_windows = df.duplicated(subset=["condition", "method", "subject_id", "session_id", "trial_id", "window_id"]).sum()
    if dup_windows > 0:
        raise ValueError(f"Found {dup_windows} duplicate window entries in Phase D predictions!")

    # Check trial counts and label consistency for each condition and method
    for (cond, meth), grp in df.groupby(["condition", "method"]):
        trial_grp = grp.groupby(["subject_id", "session_id", "trial_id"])
        n_trials = trial_grp.ngroups
        if n_trials != 1080:
            raise ValueError(f"Condition {cond} ({meth}) has {n_trials} trials; expected exactly 1,080!")

        # Subject trial counts
        sub_trials = grp.groupby("subject_id")["trial_id"].nunique()  # Note: trial_id repeats across sessions
        sub_total_trials = grp.drop_duplicates(subset=["subject_id", "session_id", "trial_id"]).groupby("subject_id").size()
        if not (sub_total_trials == 72).all():
            raise ValueError(f"Condition {cond} ({meth}) has subjects with != 72 trials: {sub_total_trials.to_dict()}")

        # Check label consistency
        label_inconsistencies = trial_grp["true_label"].nunique()
        if (label_inconsistencies != 1).any():
            bad_trials = label_inconsistencies[label_inconsistencies != 1]
            raise ValueError(f"Condition {cond} ({meth}) has trials with conflicting true labels: {bad_trials}")

        # Check window counts >= 1
        win_counts = trial_grp.size()
        if (win_counts < 1).any():
            raise ValueError(f"Condition {cond} ({meth}) has trials with 0 windows!")


def aggregate_single_trial(
    tr_df: pd.DataFrame
) -> Tuple[int, List[float], int, bool]:
    """
    Perform both Mean Probability Aggregation and Majority Voting for a single trial's windows.
    
    Returns:
        pred_mean: Predicted class from mean probabilities.
        mean_probs: Averaged 4-class probability vector.
        pred_vote: Predicted class from majority vote (with deterministic tie-break).
        had_vote_tie: True if majority voting required tie-breaking.
    """
    # Method A: Mean probability
    mean_probs = tr_df[PROB_COLS].mean().values.tolist()
    pred_mean = int(np.argmax(mean_probs))

    # Method B: Majority voting
    counts = tr_df["predicted_label"].value_counts()
    top_count = counts.iloc[0]
    tied_classes = counts[counts == top_count].index.tolist()

    if len(tied_classes) == 1:
        pred_vote = int(tied_classes[0])
        had_vote_tie = False
    else:
        # Deterministic tie-breaking: highest cumulative probability among tied classes
        best_cls = tied_classes[0]
        best_sum = -1.0
        for c in tied_classes:
            c_col = PROB_COLS[c]
            p_sum = tr_df[c_col].sum()
            if p_sum > best_sum:
                best_sum = p_sum
                best_cls = c
        pred_vote = int(best_cls)
        had_vote_tie = True

    return pred_mean, mean_probs, pred_vote, had_vote_tie


def generate_trial_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trial-level predictions for all conditions, methods, and aggregation strategies.
    
    Returns:
        DataFrame with exactly 1,080 rows per (condition, calibration_method, aggregation_method).
    """
    records = []

    # Filter for relevant conditions and methods
    target_conditions = ["all_62", "pred_32", "stab_32", "pred_12"]
    target_methods = ["group_safe_temperature", "uncalibrated_softmax"]

    filtered_df = df[df["condition"].isin(target_conditions) & df["method"].isin(target_methods)].copy()

    grouped = filtered_df.groupby(["condition", "method", "subject_id", "session_id", "trial_id"])

    for (cond, meth, sub_id, sess_id, tr_id), tr_df in grouped:
        y_true = int(tr_df["true_label"].iloc[0])
        y_true_name = LABEL_NAMES[y_true]
        win_count = len(tr_df)

        pred_mean, mean_probs, pred_vote, had_vote_tie = aggregate_single_trial(tr_df)

        # Record Method A: mean_probability
        records.append({
            "subject_id": sub_id,
            "session_id": sess_id,
            "trial_id": tr_id,
            "condition": cond,
            "calibration_method": meth,
            "aggregation_method": "mean_probability",
            "window_count": win_count,
            "true_label": y_true,
            "true_label_name": y_true_name,
            "predicted_label": pred_mean,
            "predicted_label_name": LABEL_NAMES[pred_mean],
            "prob_neutral": mean_probs[0],
            "prob_sad": mean_probs[1],
            "prob_fear": mean_probs[2],
            "prob_happy": mean_probs[3],
            "is_correct": int(pred_mean == y_true),
            "had_tie_break": False,
        })

        # Record Method B: majority_vote
        records.append({
            "subject_id": sub_id,
            "session_id": sess_id,
            "trial_id": tr_id,
            "condition": cond,
            "calibration_method": meth,
            "aggregation_method": "majority_vote",
            "window_count": win_count,
            "true_label": y_true,
            "true_label_name": y_true_name,
            "predicted_label": pred_vote,
            "predicted_label_name": LABEL_NAMES[pred_vote],
            "prob_neutral": mean_probs[0],
            "prob_sad": mean_probs[1],
            "prob_fear": mean_probs[2],
            "prob_happy": mean_probs[3],
            "is_correct": int(pred_vote == y_true),
            "had_tie_break": had_vote_tie,
        })

    trial_df = pd.DataFrame(records)
    return trial_df


def compute_metrics_and_comparisons(
    trial_df: pd.DataFrame,
    window_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute comprehensive pooled and subject-level trial metrics, class-level metrics,
    and paired statistical comparisons between Mean Probability and Majority Vote.
    """
    summary_rows = []
    subject_rows = []
    class_rows = []

    # Map window-level accuracies for context
    win_acc_map = {}
    for (cond, meth), grp in window_df.groupby(["condition", "method"]):
        win_acc_map[(cond, meth)] = grp["is_correct"].mean() * 100.0

    grouped_trials = trial_df.groupby(["condition", "calibration_method", "aggregation_method"])

    for (cond, cal_meth, agg_meth), grp in grouped_trials:
        y_true = grp["true_label"].values
        y_pred = grp["predicted_label"].values
        subjects = grp["subject_id"].values

        total_trials = len(grp)
        correct_trials = int((y_true == y_pred).sum())
        incorrect_trials = total_trials - correct_trials
        pooled_acc = (correct_trials / total_trials) * 100.0

        balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
        macro_f1 = f1_score(y_true, y_pred, average="macro") * 100.0

        # Subject-level accuracy distribution (N = 15)
        sub_accs = []
        for s in range(1, 16):
            s_mask = subjects == s
            s_acc = np.mean(y_true[s_mask] == y_pred[s_mask]) * 100.0
            sub_accs.append(s_acc)
            subject_rows.append({
                "condition": cond,
                "calibration_method": cal_meth,
                "aggregation_method": agg_meth,
                "subject_id": s,
                "trial_count": int(s_mask.sum()),
                "correct_trials": int((y_true[s_mask] == y_pred[s_mask]).sum()),
                "accuracy": s_acc
            })

        sub_accs = np.array(sub_accs)
        mean_acc = float(np.mean(sub_accs))
        std_acc = float(np.std(sub_accs, ddof=1))
        median_acc = float(np.median(sub_accs))
        min_acc = float(np.min(sub_accs))
        max_acc = float(np.max(sub_accs))

        # Student-t 95% CI
        se_acc = std_acc / np.sqrt(15)
        t_crit = float(stats.t.ppf(0.975, df=14))
        ci95_lower_t = mean_acc - t_crit * se_acc
        ci95_upper_t = mean_acc + t_crit * se_acc

        # Bootstrap 95% CI (subject-level resampling, B=2000, seed=42)
        np.random.seed(42)
        boot_means = []
        for _ in range(2000):
            resample = np.random.choice(sub_accs, size=15, replace=True)
            boot_means.append(np.mean(resample))
        ci95_lower_boot = float(np.percentile(boot_means, 2.5))
        ci95_upper_boot = float(np.percentile(boot_means, 97.5))

        # Contextual window accuracy
        win_acc = win_acc_map.get((cond, cal_meth), np.nan)

        summary_rows.append({
            "condition": cond,
            "calibration_method": cal_meth,
            "aggregation_method": agg_meth,
            "total_trials": total_trials,
            "correct_trials": correct_trials,
            "incorrect_trials": incorrect_trials,
            "window_level_accuracy": win_acc,
            "pooled_trial_accuracy": pooled_acc,
            "subject_mean_accuracy": mean_acc,
            "subject_std_accuracy": std_acc,
            "subject_median_accuracy": median_acc,
            "subject_min_accuracy": min_acc,
            "subject_max_accuracy": max_acc,
            "ci95_lower_t": ci95_lower_t,
            "ci95_upper_t": ci95_upper_t,
            "ci95_lower_boot": ci95_lower_boot,
            "ci95_upper_boot": ci95_upper_boot,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
        })

        # Class-level metrics
        prec = precision_score(y_true, y_pred, average=None) * 100.0
        rec = recall_score(y_true, y_pred, average=None) * 100.0
        f1 = f1_score(y_true, y_pred, average=None) * 100.0
        cm = confusion_matrix(y_true, y_pred)

        for c in range(4):
            class_rows.append({
                "condition": cond,
                "calibration_method": cal_meth,
                "aggregation_method": agg_meth,
                "class_id": c,
                "class_name": LABEL_NAMES[c],
                "trial_support": int((y_true == c).sum()),
                "precision": prec[c],
                "recall": rec[c],
                "f1_score": f1[c],
            })

    summary_df = pd.DataFrame(summary_rows)
    subject_df = pd.DataFrame(subject_rows)
    class_df = pd.DataFrame(class_rows)

    return summary_df, subject_df, class_df


def compute_paired_aggregation_tests(subject_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform paired statistical comparison between Mean Probability Aggregation and Majority Voting
    across the 15 subjects for each (condition, calibration_method).
    """
    paired_rows = []

    for (cond, cal_meth), grp in subject_df.groupby(["condition", "calibration_method"]):
        mean_sub = grp[grp["aggregation_method"] == "mean_probability"].sort_values("subject_id")["accuracy"].values
        vote_sub = grp[grp["aggregation_method"] == "majority_vote"].sort_values("subject_id")["accuracy"].values

        deltas = mean_sub - vote_sub  # Mean Prob - Majority Vote
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1))
        se_delta = std_delta / np.sqrt(15)

        t_crit = float(stats.t.ppf(0.975, df=14))
        ci95_lower_t = mean_delta - t_crit * se_delta
        ci95_upper_t = mean_delta + t_crit * se_delta

        # Bootstrap CI (B=2000, seed=42)
        np.random.seed(42)
        boot_means = []
        for _ in range(2000):
            resample = np.random.choice(deltas, size=15, replace=True)
            boot_means.append(np.mean(resample))
        ci95_lower_boot = float(np.percentile(boot_means, 2.5))
        ci95_upper_boot = float(np.percentile(boot_means, 97.5))

        # Paired t-test
        if std_delta > 1e-8:
            t_res = stats.ttest_rel(mean_sub, vote_sub)
            t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)
        else:
            t_stat, t_pval = 0.0, 1.0

        # Wilcoxon signed-rank test
        non_zeros = deltas[np.abs(deltas) > 1e-8]
        if len(non_zeros) > 0:
            try:
                w_res = stats.wilcoxon(deltas, zero_method="pratt", alternative="two-sided")
                w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                w_stat, w_pval = np.nan, 1.0
        else:
            w_stat, w_pval = 0.0, 1.0

        n_pos = int(np.sum(deltas > 1e-5))
        n_neg = int(np.sum(deltas < -1e-5))
        n_zero = 15 - n_pos - n_neg

        paired_rows.append({
            "condition": cond,
            "calibration_method": cal_meth,
            "mean_prob_acc_mean": float(np.mean(mean_sub)),
            "maj_vote_acc_mean": float(np.mean(vote_sub)),
            "mean_delta_acc_pp": mean_delta,
            "std_delta_acc_pp": std_delta,
            "ci95_lower_t": ci95_lower_t,
            "ci95_upper_t": ci95_upper_t,
            "ci95_lower_boot": ci95_lower_boot,
            "ci95_upper_boot": ci95_upper_boot,
            "t_stat": t_stat,
            "t_pval": t_pval,
            "wilcoxon_stat": w_stat,
            "wilcoxon_pval": w_pval,
            "n_pos_subjects": n_pos,
            "n_neg_subjects": n_neg,
            "n_zero_subjects": n_zero,
        })

    return pd.DataFrame(paired_rows)


def plot_trial_figures(
    summary_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    figures_dir: Path
) -> None:
    """Generate and save publication-quality comparison charts and confusion matrices."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comparison Figure: Window vs Mean-Prob vs Maj-Vote across conditions
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Filter for primary regime: group_safe_temperature
    prim = summary_df[summary_df["calibration_method"] == "group_safe_temperature"].copy()
    conditions = ["all_62", "pred_32", "stab_32", "pred_12"]
    x = np.arange(len(conditions))
    width = 0.25

    win_accs = [prim[prim["condition"] == c]["window_level_accuracy"].iloc[0] for c in conditions]
    mean_accs = [prim[(prim["condition"] == c) & (prim["aggregation_method"] == "mean_probability")]["subject_mean_accuracy"].iloc[0] for c in conditions]
    mean_errs = [prim[(prim["condition"] == c) & (prim["aggregation_method"] == "mean_probability")]["subject_std_accuracy"].iloc[0] / np.sqrt(15) for c in conditions]
    vote_accs = [prim[(prim["condition"] == c) & (prim["aggregation_method"] == "majority_vote")]["subject_mean_accuracy"].iloc[0] for c in conditions]
    vote_errs = [prim[(prim["condition"] == c) & (prim["aggregation_method"] == "majority_vote")]["subject_std_accuracy"].iloc[0] / np.sqrt(15) for c in conditions]

    r1 = ax.bar(x - width, win_accs, width, label="Window-Level Baseline (Phase A/D)", color="#34495e", alpha=0.85)
    r2 = ax.bar(x, mean_accs, width, yerr=mean_errs, capsize=4, label="Trial Mean-Probability Aggregation", color="#2980b9", alpha=0.90)
    r3 = ax.bar(x + width, vote_accs, width, yerr=vote_errs, capsize=4, label="Trial Majority Voting", color="#27ae60", alpha=0.90)

    # Reference chance level
    ax.axhline(25.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Chance Level (25.0%)")

    ax.set_ylabel("Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("SEED-IV Subject-Independent Emotion Recognition:\nWindow-Level vs. Whole-Trial Aggregation", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["All 62 Channels", "Predictive 32 (`pred_32`)", "Stable 32 (`stab_32`)", "Predictive 12 (`pred_12`)"], fontsize=11)
    ax.set_ylim(0, 55)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Value labels
    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="semibold")

    fig.tight_layout()
    comp_path = figures_dir / "trial_level_aggregation_comparison.png"
    fig.savefig(comp_path)
    plt.close(fig)

    # 2. Confusion Matrices for Primary Conditions under Mean Probability Aggregation
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=300)
    class_names = ["Neutral", "Sad", "Fear", "Happy"]

    for ax, cond, title in zip(
        axes,
        conditions,
        ["All 62 Channels", "Predictive 32", "Stable 32", "Predictive 12"]
    ):
        sub = trial_df[
            (trial_df["condition"] == cond) &
            (trial_df["calibration_method"] == "group_safe_temperature") &
            (trial_df["aggregation_method"] == "mean_probability")
        ]
        cm = confusion_matrix(sub["true_label"], sub["predicted_label"])
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100.0

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=60)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9.5)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=9.5)

        for i in range(4):
            for j in range(4):
                val = cm_norm[i, j]
                count = cm[i, j]
                text_color = "white" if val > 30 else "black"
                ax.text(j, i, f"{val:.1f}%\n({count})",
                        ha="center", va="center", color=text_color, fontsize=8.5)

        ax.set_ylabel("True Emotion", fontsize=10, fontweight="semibold")
        ax.set_xlabel("Predicted Emotion", fontsize=10, fontweight="semibold")

    fig.suptitle("Trial-Level Normalized Confusion Matrices (Mean-Probability Aggregation, Temperature-Scaled)",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    cm_path = figures_dir / "trial_level_aggregation_confusion_matrix.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)


def run_experiment_a() -> None:
    """Main execution orchestrator for Trial-Level Experiment A."""
    print("=== STARTING TRIAL-LEVEL EXPERIMENT A EVALUATION ===")
    
    preds_path = Path("outputs/predictions/phase_d_calibrated_predictions.csv")
    if not preds_path.is_file():
        raise FileNotFoundError(f"Missing required input file: {preds_path}")

    print(f"Loading {preds_path}...")
    df = pd.read_csv(preds_path)
    print(f"Loaded {len(df)} window prediction records.")

    print("Step 1 & 3: Validating trial structure and composite keys...")
    validate_trial_structure(df)
    print("Trial validation passed successfully: exactly 1,080 trials per condition/method, 72 per subject, 0 label contradictions.")

    print("Step 2: Generating whole-trial aggregated predictions (Methods A and B)...")
    trial_preds_df = generate_trial_predictions(df)
    print(f"Generated {len(trial_preds_df)} trial-level prediction records.")

    # Save trial predictions
    out_preds_path = Path("outputs/predictions/trial_level_aggregation_predictions.csv")
    out_preds_path.parent.mkdir(parents=True, exist_ok=True)
    trial_preds_df.to_csv(out_preds_path, index=False)
    print(f"Saved: {out_preds_path}")

    print("Step 4: Computing comprehensive metrics, CIs, and paired tests...")
    summary_df, subject_df, class_df = compute_metrics_and_comparisons(trial_preds_df, df)
    paired_df = compute_paired_aggregation_tests(subject_df)

    # Save tables
    out_tables_dir = Path("outputs/tables")
    out_tables_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_tables_dir / "trial_level_aggregation_summary.csv"
    subject_path = out_tables_dir / "trial_level_aggregation_subject_metrics.csv"
    class_path = out_tables_dir / "trial_level_aggregation_class_metrics.csv"

    summary_df.to_csv(summary_path, index=False)
    subject_df.to_csv(subject_path, index=False)
    class_df.to_csv(class_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {subject_path}")
    print(f"Saved: {class_path}")

    print("Step 6: Generating publication-quality figures...")
    figures_dir = Path("outputs/figures")
    plot_trial_figures(summary_df, trial_preds_df, figures_dir)
    print("Generated figures: trial_level_aggregation_comparison.png, trial_level_aggregation_confusion_matrix.png")

    print("\n=== SUMMARY OF EXPERIMENT A RESULTS ===")
    display_cols = [
        "condition", "calibration_method", "aggregation_method",
        "window_level_accuracy", "pooled_trial_accuracy", "subject_mean_accuracy",
        "subject_std_accuracy", "ci95_lower_t", "ci95_upper_t", "macro_f1"
    ]
    print(summary_df[display_cols].to_string(index=False))

    print("\n=== PAIRED COMPARISON: MEAN PROBABILITY VS MAJORITY VOTE ===")
    paired_display = [
        "condition", "calibration_method", "mean_prob_acc_mean", "maj_vote_acc_mean",
        "mean_delta_acc_pp", "ci95_lower_t", "ci95_upper_t", "t_pval", "wilcoxon_pval",
        "n_pos_subjects", "n_neg_subjects", "n_zero_subjects"
    ]
    print(paired_df[paired_display].to_string(index=False))


if __name__ == "__main__":
    run_experiment_a()
