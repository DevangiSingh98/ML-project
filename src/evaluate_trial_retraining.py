"""
Trial-Level Feature Aggregation and Retraining Pipeline (Experiment B).

This module aggregates SEED-IV window-level Differential Entropy (de_LDS) features
into whole-trial feature vectors and trains subject-independent classifiers using
strict Leave-One-Subject-Out (LOSO) cross-validation.

Evaluates:
- Fixed RBF SVM Baseline (C=1.0, gamma='scale')
- Tuned Linear SVM (C in [0.01, 0.1, 1.0, 10.0]) via inner GroupKFold
- Tuned RBF SVM (C in [0.1, 1.0, 10.0], gamma in ['scale', 0.001, 0.01, 0.1]) via inner GroupKFold
Across 4 channel configurations: 'all_62', 'pred_32', 'stab_32', 'pred_12',
using strictly nested training-only channel selections from Stage 2.5.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_seed_iv_dataset, DatasetContainer, LABEL_NAMES



def build_trial_feature_dataset(
    dataset: Optional[DatasetContainer] = None,
    output_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Construct whole-trial feature dataset by averaging 4-second window features.
    
    Returns:
        X_trial: Shape (1080, 310) float32 array of trial-averaged DE features.
        y_trial: Shape (1080,) int64 array of trial ground-truth emotion labels.
        trial_meta: DataFrame with 1,080 rows and columns:
                    ['trial_key', 'subject_id', 'session_id', 'trial_id', 'label', 'label_name', 'window_count']
    """
    if dataset is None:
        dataset = load_seed_iv_dataset()

    X = dataset.X
    meta = dataset.metadata.copy()

    # Construct unambiguous composite trial key
    meta["trial_key"] = (
        meta["subject_id"].astype(str) + "_" +
        meta["session_id"].astype(str) + "_" +
        meta["trial_id"].astype(str)
    )

    # Validate trial integrity
    trial_groups = meta.groupby("trial_key", sort=False)
    n_trials = trial_groups.ngroups
    if n_trials != 1080:
        raise ValueError(f"Expected exactly 1,080 trials, found {n_trials}")

    # Check label consistency within trial
    label_nunique = trial_groups["label"].nunique()
    if (label_nunique != 1).any():
        raise ValueError("Found trials with conflicting window labels!")

    # Check window counts >= 1
    win_counts = trial_groups.size()
    if (win_counts < 1).any():
        raise ValueError("Found trials with 0 windows!")

    # Check subject balance
    sub_counts = meta.drop_duplicates(subset=["subject_id", "session_id", "trial_id"]).groupby("subject_id").size()
    if not (sub_counts == 72).all():
        raise ValueError(f"Subjects do not have exactly 72 trials: {sub_counts.to_dict()}")

    # Compute mean features per trial
    group_indices = trial_groups.indices
    trial_keys_ordered = list(trial_groups.groups.keys())

    X_trial = np.zeros((n_trials, X.shape[1]), dtype=np.float32)
    meta_records = []

    for i, key in enumerate(trial_keys_ordered):
        idxs = group_indices[key]
        X_trial[i] = X[idxs].mean(axis=0)

        first_row = meta.iloc[idxs[0]]
        meta_records.append({
            "trial_key": key,
            "subject_id": int(first_row["subject_id"]),
            "session_id": int(first_row["session_id"]),
            "trial_id": int(first_row["trial_id"]),
            "label": int(first_row["label"]),
            "label_name": str(first_row["label_name"]),
            "window_count": len(idxs),
        })

    trial_meta = pd.DataFrame(meta_records)
    y_trial = trial_meta["label"].to_numpy(dtype=np.int64)

    # Save reproducible intermediate artifact if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Combine metadata and feature columns for intermediate CSV
        feat_cols = dataset.feature_names
        feat_df = pd.DataFrame(X_trial, columns=feat_cols)
        full_artifact = pd.concat([trial_meta, feat_df], axis=1)
        full_artifact.to_csv(output_path, index=False)
        print(f"Saved trial-level feature artifact to: {output_path}")

    return X_trial, y_trial, trial_meta


def get_fold_channel_feature_indices(
    test_subject: int,
    condition: str,
    channel_to_indices: Dict[str, List[int]],
    selections_df: Optional[pd.DataFrame] = None,
) -> List[int]:
    """
    Extract strictly nested training-only feature column indices for a given fold and condition.
    
    Conditions:
    - 'all_62': All 310 features (62 channels x 5 bands).
    - 'stab_32': 32 stability-aware channels from Stage 2.5 selections for this test_subject.
    - 'pred_32': 32 predictive-importance channels from Stage 2.5 selections for this test_subject.
    - 'pred_12': 12 predictive-importance channels from Stage 2.5 selections for this test_subject.
    """
    if condition == "all_62":
        return list(range(310))

    if selections_df is None:
        raise ValueError("selections_df must be provided for reduced channel conditions.")

    if condition == "stab_32":
        row = selections_df[(selections_df["test_subject"] == test_subject) & (selections_df["budget"] == 32)]
        if len(row) == 0:
            raise KeyError(f"No stab_32 channel selection for test_subject {test_subject}")
        ch_names = [c.strip() for c in row.iloc[0]["corrected_stability_channels"].split(",")]
    elif condition == "pred_32":
        row = selections_df[(selections_df["test_subject"] == test_subject) & (selections_df["budget"] == 32)]
        if len(row) == 0:
            raise KeyError(f"No pred_32 channel selection for test_subject {test_subject}")
        ch_names = [c.strip() for c in row.iloc[0]["predictive_importance_channels"].split(",")]
    elif condition == "pred_12":
        row = selections_df[(selections_df["test_subject"] == test_subject) & (selections_df["budget"] == 12)]
        if len(row) == 0:
            raise KeyError(f"No pred_12 channel selection for test_subject {test_subject}")
        ch_names = [c.strip() for c in row.iloc[0]["predictive_importance_channels"].split(",")]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    indices = []
    for ch in ch_names:
        indices.extend(channel_to_indices[ch])
    return sorted(indices)


def train_and_evaluate_loso_models(
    X_trial: np.ndarray,
    y_trial: np.ndarray,
    trial_meta: pd.DataFrame,
    channel_to_indices: Dict[str, List[int]],
    selections_df: pd.DataFrame,
    conditions: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Execute 15-fold outer LOSO evaluation for Fixed RBF, Tuned Linear, and Tuned RBF models.
    
    Returns:
        DataFrame with trial predictions (1,080 trials x conditions x 3 models).
    """
    if conditions is None:
        conditions = ["all_62", "pred_32", "stab_32", "pred_12"]

    subjects = trial_meta["subject_id"].values
    n_subjects = len(np.unique(subjects))

    pred_records = []

    print(f"Starting 15-fold outer LOSO evaluation across {len(conditions)} conditions and 3 model types...")

    for fold_idx, test_sub in enumerate(sorted(np.unique(subjects))):
        t_fold_start = time.time()
        train_mask = subjects != test_sub
        test_mask = subjects == test_sub

        train_subs = subjects[train_mask]
        test_meta = trial_meta[test_mask].reset_index(drop=True)

        print(f"--- Outer Fold {fold_idx + 1}/15: Test Subject {test_sub} (Train: {train_mask.sum()} trials, Test: {test_mask.sum()} trials) ---")

        for cond in conditions:
            # 1. Extract training-only feature indices
            feat_idx = get_fold_channel_feature_indices(
                test_subject=test_sub,
                condition=cond,
                channel_to_indices=channel_to_indices,
                selections_df=selections_df,
            )

            X_tr = X_trial[train_mask][:, feat_idx]
            X_te = X_trial[test_mask][:, feat_idx]
            y_tr = y_trial[train_mask]
            y_te = y_trial[test_mask]

            # 2. Fit StandardScaler strictly on training trials
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            # 3. Model 1: Fixed RBF SVM Baseline (C=1.0, gamma='scale')
            clf_rbf_fixed = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
            clf_rbf_fixed.fit(X_tr_scaled, y_tr)
            preds_rbf_fixed = clf_rbf_fixed.predict(X_te_scaled)

            for i in range(len(test_meta)):
                pred_records.append({
                    "subject_id": test_sub,
                    "session_id": int(test_meta.loc[i, "session_id"]),
                    "trial_id": int(test_meta.loc[i, "trial_id"]),
                    "trial_key": test_meta.loc[i, "trial_key"],
                    "condition": cond,
                    "model_type": "rbf_svm_fixed",
                    "best_params": "C=1.0,gamma=scale",
                    "true_label": int(y_te[i]),
                    "true_label_name": LABEL_NAMES[int(y_te[i])],
                    "predicted_label": int(preds_rbf_fixed[i]),
                    "predicted_label_name": LABEL_NAMES[int(preds_rbf_fixed[i])],
                    "is_correct": int(preds_rbf_fixed[i] == y_te[i]),
                })

            # 4. Model 2: Tuned Linear SVM via inner GroupKFold
            inner_gkf = GroupKFold(n_splits=5)
            param_grid_lin = {"C": [0.01, 0.1, 1.0, 10.0]}
            cv_lin = GridSearchCV(
                estimator=LinearSVC(dual=False, random_state=seed, max_iter=2000),
                param_grid=param_grid_lin,
                cv=inner_gkf,
                scoring="accuracy",
                n_jobs=-1,
            )
            cv_lin.fit(X_tr_scaled, y_tr, groups=train_subs)
            best_lin_clf = cv_lin.best_estimator_
            best_lin_params = str(cv_lin.best_params_)
            preds_lin_tuned = best_lin_clf.predict(X_te_scaled)

            for i in range(len(test_meta)):
                pred_records.append({
                    "subject_id": test_sub,
                    "session_id": int(test_meta.loc[i, "session_id"]),
                    "trial_id": int(test_meta.loc[i, "trial_id"]),
                    "trial_key": test_meta.loc[i, "trial_key"],
                    "condition": cond,
                    "model_type": "linear_svm_tuned",
                    "best_params": best_lin_params,
                    "true_label": int(y_te[i]),
                    "true_label_name": LABEL_NAMES[int(y_te[i])],
                    "predicted_label": int(preds_lin_tuned[i]),
                    "predicted_label_name": LABEL_NAMES[int(preds_lin_tuned[i])],
                    "is_correct": int(preds_lin_tuned[i] == y_te[i]),
                })

            # 5. Model 3: Tuned RBF SVM via inner GroupKFold
            param_grid_rbf = {
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", 0.001, 0.01, 0.1],
            }
            cv_rbf = GridSearchCV(
                estimator=SVC(kernel="rbf", random_state=seed),
                param_grid=param_grid_rbf,
                cv=inner_gkf,
                scoring="accuracy",
                n_jobs=-1,
            )
            cv_rbf.fit(X_tr_scaled, y_tr, groups=train_subs)
            best_rbf_clf = cv_rbf.best_estimator_
            best_rbf_params = str(cv_rbf.best_params_)
            preds_rbf_tuned = best_rbf_clf.predict(X_te_scaled)

            for i in range(len(test_meta)):
                pred_records.append({
                    "subject_id": test_sub,
                    "session_id": int(test_meta.loc[i, "session_id"]),
                    "trial_id": int(test_meta.loc[i, "trial_id"]),
                    "trial_key": test_meta.loc[i, "trial_key"],
                    "condition": cond,
                    "model_type": "rbf_svm_tuned",
                    "best_params": best_rbf_params,
                    "true_label": int(y_te[i]),
                    "true_label_name": LABEL_NAMES[int(y_te[i])],
                    "predicted_label": int(preds_rbf_tuned[i]),
                    "predicted_label_name": LABEL_NAMES[int(preds_rbf_tuned[i])],
                    "is_correct": int(preds_rbf_tuned[i] == y_te[i]),
                })

        fold_dur = time.time() - t_fold_start
        print(f"Fold {test_sub} completed in {fold_dur:.1f}s.")

    pred_df = pd.DataFrame(pred_records)
    return pred_df


def compute_retraining_metrics(
    pred_df: pd.DataFrame,
    exp_a_summary_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute pooled and subject-level metrics, class metrics, and comparisons against Experiment A.
    """
    summary_rows = []
    subject_rows = []
    class_rows = []

    grouped = pred_df.groupby(["condition", "model_type"])

    for (cond, model_type), grp in grouped:
        y_true = grp["true_label"].values
        y_pred = grp["predicted_label"].values
        subjects = grp["subject_id"].values

        total_trials = len(grp)
        correct_trials = int((y_true == y_pred).sum())
        incorrect_trials = total_trials - correct_trials
        pooled_acc = (correct_trials / total_trials) * 100.0

        balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
        macro_f1 = f1_score(y_true, y_pred, average="macro") * 100.0

        # Subject-level accuracy distribution
        sub_accs = []
        for s in range(1, 16):
            s_mask = subjects == s
            s_acc = np.mean(y_true[s_mask] == y_pred[s_mask]) * 100.0
            sub_accs.append(s_acc)
            subject_rows.append({
                "condition": cond,
                "model_type": model_type,
                "subject_id": s,
                "trial_count": int(s_mask.sum()),
                "correct_trials": int((y_true[s_mask] == y_pred[s_mask]).sum()),
                "accuracy": s_acc,
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

        summary_rows.append({
            "condition": cond,
            "model_type": model_type,
            "total_trials": total_trials,
            "correct_trials": correct_trials,
            "incorrect_trials": incorrect_trials,
            "pooled_accuracy": pooled_acc,
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

        for c in range(4):
            class_rows.append({
                "condition": cond,
                "model_type": model_type,
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


def compare_with_experiment_a(
    exp_b_subject_df: pd.DataFrame,
    exp_a_subject_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform paired statistical comparison across the 15 subjects between each Experiment B model
    and Experiment A's Mean-Probability and Majority-Vote predictions.
    """
    comp_rows = []

    # Filter Exp A for primary regime: group_safe_temperature
    a_prim = exp_a_subject_df[exp_a_subject_df["calibration_method"] == "group_safe_temperature"].copy()

    for (cond, model_type), b_grp in exp_b_subject_df.groupby(["condition", "model_type"]):
        b_accs = b_grp.sort_values("subject_id")["accuracy"].values

        for a_agg in ["mean_probability", "majority_vote"]:
            a_sub = a_prim[(a_prim["condition"] == cond) & (a_prim["aggregation_method"] == a_agg)]
            if len(a_sub) != 15:
                continue
            a_accs = a_sub.sort_values("subject_id")["accuracy"].values

            deltas = b_accs - a_accs  # Exp B - Exp A
            mean_delta = float(np.mean(deltas))
            std_delta = float(np.std(deltas, ddof=1))
            se_delta = std_delta / np.sqrt(15)

            t_crit = float(stats.t.ppf(0.975, df=14))
            ci_lower_t = mean_delta - t_crit * se_delta
            ci_upper_t = mean_delta + t_crit * se_delta

            # Bootstrap CI (B=2000, seed=42)
            np.random.seed(42)
            boot_means = []
            for _ in range(2000):
                resample = np.random.choice(deltas, size=15, replace=True)
                boot_means.append(np.mean(resample))
            ci_lower_boot = float(np.percentile(boot_means, 2.5))
            ci_upper_boot = float(np.percentile(boot_means, 97.5))

            if std_delta > 1e-8:
                t_res = stats.ttest_rel(b_accs, a_accs)
                t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)
            else:
                t_stat, t_pval = 0.0, 1.0

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

            comp_rows.append({
                "condition": cond,
                "exp_b_model": model_type,
                "exp_a_baseline": f"exp_a_{a_agg}",
                "exp_b_mean_acc": float(np.mean(b_accs)),
                "exp_a_mean_acc": float(np.mean(a_accs)),
                "mean_delta_acc_pp": mean_delta,
                "std_delta_acc_pp": std_delta,
                "ci95_lower_t": ci_lower_t,
                "ci95_upper_t": ci_upper_t,
                "ci95_lower_boot": ci_lower_boot,
                "ci95_upper_boot": ci_upper_boot,
                "t_stat": t_stat,
                "t_pval": t_pval,
                "wilcoxon_stat": w_stat,
                "wilcoxon_pval": w_pval,
                "n_pos_subjects": n_pos,
                "n_neg_subjects": n_neg,
                "n_zero_subjects": n_zero,
            })

    return pd.DataFrame(comp_rows)


def plot_retraining_figures(
    summary_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    exp_a_summary_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Generate publication-quality comparison charts and confusion matrices."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comparison Figure: Window Baseline vs Exp A vs Exp B models
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    conditions = ["all_62", "pred_32", "stab_32", "pred_12"]
    x = np.arange(len(conditions))
    width = 0.16

    # Extract accuracies
    # Window baseline
    win_accs = [39.53, 38.90, 38.56, 38.04]

    # Exp A Mean Prob
    a_prim = exp_a_summary_df[exp_a_summary_df["calibration_method"] == "group_safe_temperature"]
    a_mean_accs = [a_prim[(a_prim["condition"] == c) & (a_prim["aggregation_method"] == "mean_probability")]["subject_mean_accuracy"].iloc[0] for c in conditions]

    # Exp B models
    b_rbf_fixed = [summary_df[(summary_df["condition"] == c) & (summary_df["model_type"] == "rbf_svm_fixed")]["subject_mean_accuracy"].iloc[0] for c in conditions]
    b_rbf_fixed_err = [summary_df[(summary_df["condition"] == c) & (summary_df["model_type"] == "rbf_svm_fixed")]["subject_std_accuracy"].iloc[0] / np.sqrt(15) for c in conditions]

    b_lin_tuned = [summary_df[(summary_df["condition"] == c) & (summary_df["model_type"] == "linear_svm_tuned")]["subject_mean_accuracy"].iloc[0] for c in conditions]
    b_lin_tuned_err = [summary_df[(summary_df["condition"] == c) & (summary_df["model_type"] == "linear_svm_tuned")]["subject_std_accuracy"].iloc[0] / np.sqrt(15) for c in conditions]

    b_rbf_tuned = [summary_df[(summary_df["condition"] == c) & (summary_df["model_type"] == "rbf_svm_tuned")]["subject_mean_accuracy"].iloc[0] for c in conditions]
    b_rbf_tuned_err = [summary_df[(summary_df["condition"] == c) & (summary_df["model_type"] == "rbf_svm_tuned")]["subject_std_accuracy"].iloc[0] / np.sqrt(15) for c in conditions]

    r0 = ax.bar(x - 2 * width, win_accs, width, label="Window Baseline (Linear)", color="#7f8c8d", alpha=0.85)
    r1 = ax.bar(x - width, a_mean_accs, width, label="Exp A: Trial Prediction Agg (Soft Vote)", color="#3498db", alpha=0.85)
    r2 = ax.bar(x, b_rbf_fixed, width, yerr=b_rbf_fixed_err, capsize=3, label="Exp B: Trial Fixed RBF (C=1, scale)", color="#e67e22", alpha=0.85)
    r3 = ax.bar(x + width, b_lin_tuned, width, yerr=b_lin_tuned_err, capsize=3, label="Exp B: Trial Tuned Linear SVM", color="#9b59b6", alpha=0.85)
    r4 = ax.bar(x + 2 * width, b_rbf_tuned, width, yerr=b_rbf_tuned_err, capsize=3, label="Exp B: Trial Tuned RBF SVM", color="#27ae60", alpha=0.85)

    ax.axhline(25.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Chance Level (25.0%)")

    ax.set_ylabel("Subject-Independent Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("SEED-IV Whole-Trial Emotion Classification: Experiment B Model Comparison\n(15-Fold Outer LOSO Evaluation Across 4 Channel Montages)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["All 62 Channels", "Predictive 32 (`pred_32`)", "Stable 32 (`stab_32`)", "Predictive 12 (`pred_12`)"], fontsize=11)
    ax.set_ylim(0, 52)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    fig.tight_layout()
    comp_path = figures_dir / "trial_level_retraining_comparison.png"
    fig.savefig(comp_path)
    plt.close(fig)

    # 2. Confusion Matrices for all_62 across the 3 models
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    class_names = ["Neutral", "Sad", "Fear", "Happy"]

    models = ["rbf_svm_fixed", "linear_svm_tuned", "rbf_svm_tuned"]
    titles = ["Fixed RBF SVM (C=1, scale)", "Tuned Linear SVM", "Tuned RBF SVM"]

    for ax, m, t in zip(axes, models, titles):
        sub = pred_df[(pred_df["condition"] == "all_62") & (pred_df["model_type"] == m)]
        cm = confusion_matrix(sub["true_label"], sub["predicted_label"])
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100.0

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Purples", vmin=0, vmax=60)
        ax.set_title(f"All 62 Channels: {t}", fontsize=11, fontweight="bold", pad=8)
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

    fig.suptitle("Whole-Trial Retrained Normalized Confusion Matrices (All 62 Channels)", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    cm_path = figures_dir / "trial_level_retraining_confusion_matrix.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)


def run_experiment_b() -> None:
    """Main execution orchestrator for Trial-Level Experiment B."""
    print("=== STARTING TRIAL-LEVEL EXPERIMENT B EVALUATION ===")
    start_time = time.time()

    # 1. Load dataset & build trial features
    print("Step 1: Loading SEED-IV dataset and aggregating features per trial...")
    dataset = load_seed_iv_dataset()
    ch_to_indices = dataset.channel_to_indices

    features_csv_path = Path("outputs/tables/trial_level_features.csv")
    X_trial, y_trial, trial_meta = build_trial_feature_dataset(
        dataset=dataset,
        output_path=features_csv_path,
    )
    print(f"Constructed trial feature matrix: {X_trial.shape}, labels: {y_trial.shape}")

    # 2. Load verified Stage 2.5 channel selections
    selections_path = Path("outputs/tables/corrected_channel_selections.csv")
    if not selections_path.is_file():
        raise FileNotFoundError(f"Missing required channel selections file: {selections_path}")
    selections_df = pd.read_csv(selections_path)
    print(f"Loaded verified channel selections from: {selections_path}")

    # 3. Execute LOSO evaluation across models and conditions
    print("Step 2: Executing 15-fold outer LOSO evaluation...")
    conditions = ["all_62", "pred_32", "stab_32", "pred_12"]
    pred_df = train_and_evaluate_loso_models(
        X_trial=X_trial,
        y_trial=y_trial,
        trial_meta=trial_meta,
        channel_to_indices=ch_to_indices,
        selections_df=selections_df,
        conditions=conditions,
    )

    # Save trial retraining predictions
    out_preds_path = Path("outputs/predictions/trial_level_retraining_predictions.csv")
    out_preds_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_preds_path, index=False)
    print(f"Saved: {out_preds_path} ({len(pred_df)} rows)")

    # 4. Compute metrics and comparisons
    print("Step 3: Computing metrics and paired comparisons against Experiment A...")
    exp_a_summary_path = Path("outputs/tables/trial_level_aggregation_summary.csv")
    exp_a_subject_path = Path("outputs/tables/trial_level_aggregation_subject_metrics.csv")
    exp_a_summary_df = pd.read_csv(exp_a_summary_path) if exp_a_summary_path.is_file() else None
    exp_a_subject_df = pd.read_csv(exp_a_subject_path) if exp_a_subject_path.is_file() else None

    summary_df, subject_df, class_df = compute_retraining_metrics(pred_df, exp_a_summary_df)

    out_tables_dir = Path("outputs/tables")
    summary_path = out_tables_dir / "trial_level_retraining_summary.csv"
    subject_path = out_tables_dir / "trial_level_retraining_subject_metrics.csv"
    class_path = out_tables_dir / "trial_level_retraining_class_metrics.csv"

    summary_df.to_csv(summary_path, index=False)
    subject_df.to_csv(subject_path, index=False)
    class_df.to_csv(class_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {subject_path}")
    print(f"Saved: {class_path}")

    # 5. Paired comparisons with Experiment A
    if exp_a_subject_df is not None:
        comp_df = compare_with_experiment_a(subject_df, exp_a_subject_df)
        comp_path = out_tables_dir / "trial_level_retraining_vs_exp_a_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"Saved: {comp_path}")

        print("\n=== PAIRED COMPARISONS WITH EXPERIMENT A ===")
        print(comp_df[["condition", "exp_b_model", "exp_a_baseline", "mean_delta_acc_pp", "ci95_lower_t", "ci95_upper_t", "t_pval", "wilcoxon_pval", "n_pos_subjects", "n_neg_subjects", "n_zero_subjects"]].to_string(index=False))

    # 6. Figures
    print("Step 4: Generating publication-quality figures...")
    figures_dir = Path("outputs/figures")
    if exp_a_summary_df is not None:
        plot_retraining_figures(summary_df, pred_df, exp_a_summary_df, figures_dir)
        print("Generated figures: trial_level_retraining_comparison.png, trial_level_retraining_confusion_matrix.png")

    print("\n=== SUMMARY OF EXPERIMENT B RETRAINING RESULTS ===")
    display_cols = [
        "condition", "model_type", "total_trials", "pooled_accuracy",
        "subject_mean_accuracy", "subject_std_accuracy", "ci95_lower_t", "ci95_upper_t", "macro_f1"
    ]
    print(summary_df[display_cols].to_string(index=False))

    total_time = time.time() - start_time
    print(f"=== EXPERIMENT B PIPELINE COMPLETED IN {total_time:.1f}s ===")


if __name__ == "__main__":
    run_experiment_b()
