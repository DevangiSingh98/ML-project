"""
src/evaluate_few_shot_calibration.py

Experiment D3: Four-Shot Subject Calibration Combined with CORAL.

Protocol:
1. Feature baseline: SEED-IV trial-level de_LDS features (1,080 trials, 72 per subject).
2. Outer evaluation: 15 outer Leave-One-Subject-Out (LOSO) folds.
3. Calibration Protocol:
   - In each outer fold, 14 subjects form the labelled source training set (1,008 trials).
   - For the held-out subject (72 trials), exactly 4 calibration trials are selected:
     exactly 1 per emotion class (Neutral=0, Sad=1, Fear=2, Happy=3).
   - Evaluation is conducted exclusively on the remaining 68 trials.
   - Repeated across 5 pre-specified calibration seeds: [42, 100, 2024, 7, 999].
   - Calibration trials are strictly excluded from the test set.
4. Methods compared on identical 68 test trials:
   - Method A: source_only_baseline (Source-trained classifier without target adaptation)
   - Method B: coral_unsupervised (CORAL adaptation without labelled target calibration)
   - Method C: few_shot_calibration_only (Four-shot calibration without CORAL)
   - Method D: coral_plus_four_shot (CORAL combined with four-shot calibration)
5. Leakage Safeguards & Nested Validation:
   - Zero test-trial emotion labels are ever used.
   - Calibration weight (w_cal) and classifier C are selected strictly within the 14 training
     subjects via simulated 4-shot validation folds.
6. Channel configurations: all_62, stab_32 (primary baseline), pred_32, pred_12.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import (
    LABEL_NAMES,
    load_seed_iv_dataset,
)
from src.evaluate_subject_adaptation import (
    compute_coral_transformation,
    get_channel_feature_indices,
)

OUTPUTS_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

CALIBRATION_SEEDS = [42, 100, 2024, 7, 999]


def select_four_shot_calibration_split(
    y_target: np.ndarray,
    seed: int,
    n_classes: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select exactly 4 calibration trials: exactly one per emotion class.
    
    Args:
        y_target: 1D array of 72 true emotion labels for the held-out subject.
        seed: Random seed for reproducible selection.
        n_classes: Number of emotion classes (4 for SEED-IV).
        
    Returns:
        calib_indices: 1D array of 4 indices (one per class).
        test_indices: 1D array of 68 test indices.
    """
    rng = np.random.default_rng(seed)
    calib_indices = []

    for c in range(n_classes):
        cls_idx = np.where(y_target == c)[0]
        if len(cls_idx) == 0:
            raise ValueError(f"Class {c} not found in target labels.")
        chosen = rng.choice(cls_idx)
        calib_indices.append(chosen)

    calib_indices = np.array(calib_indices, dtype=int)
    calib_set = set(calib_indices)
    test_indices = np.array([i for i in range(len(y_target)) if i not in calib_set], dtype=int)

    # Invariants
    assert len(calib_indices) == n_classes, f"Expected {n_classes} calibration trials, got {len(calib_indices)}"
    assert len(test_indices) == len(y_target) - n_classes, f"Expected {len(y_target) - n_classes} test trials, got {len(test_indices)}"
    assert len(set(calib_indices).intersection(set(test_indices))) == 0, "Calibration and test sets overlap!"
    assert len(np.unique(y_target[calib_indices])) == n_classes, "Calibration set must have exactly 1 trial per class!"

    return calib_indices, test_indices


def tune_calibration_hyperparameters(
    Xs: np.ndarray,
    ys: np.ndarray,
    source_subjects: np.ndarray,
    w_cal_grid: List[float] = [2.0, 5.0, 10.0, 20.0],
    c_grid: List[float] = [0.005, 0.01, 0.02],
    calib_seeds: List[int] = [42, 100, 2024],
    reg: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Tune calibration sample weight w_cal and classifier C using simulated 4-shot
    calibration folds strictly within the 14 training subjects.
    
    Zero outer test subject data is ever seen.
    """
    unique_subs = sorted(np.unique(source_subjects))
    val_subs = unique_subs[::3]  # 4-5 representative validation subjects

    # Precompute inner scaling and CORAL for each validation subject
    val_cache = []
    for v_sub in val_subs:
        in_tr_m = source_subjects != v_sub
        in_val_m = source_subjects == v_sub

        in_Xs = Xs[in_tr_m]
        in_ys = ys[in_tr_m]
        in_Xv = Xs[in_val_m]
        in_yv = ys[in_val_m]

        scaler = StandardScaler()
        in_Xs_s = scaler.fit_transform(in_Xs)
        in_Xv_s = scaler.transform(in_Xv)

        _, A_ts, mu_s, mu_t = compute_coral_transformation(in_Xs_s, in_Xv_s, reg=reg)
        in_Xv_coral = (in_Xv_s - mu_t) @ A_ts + mu_s

        val_cache.append({
            "in_Xs_s": in_Xs_s,
            "in_ys": in_ys,
            "in_Xv_s": in_Xv_s,
            "in_Xv_coral": in_Xv_coral,
            "in_yv": in_yv,
        })

    grid_results = []
    for w_cal in w_cal_grid:
        for c in c_grid:
            val_accs = []
            for cache in val_cache:
                in_Xs_s = cache["in_Xs_s"]
                in_ys = cache["in_ys"]
                in_Xv_coral = cache["in_Xv_coral"]
                in_yv = cache["in_yv"]

                seed_accs = []
                for s in calib_seeds:
                    calib_idx, test_idx = select_four_shot_calibration_split(in_yv, seed=s, n_classes=4)
                    X_aug = np.vstack([in_Xs_s, in_Xv_coral[calib_idx]])
                    y_aug = np.concatenate([in_ys, in_yv[calib_idx]])
                    weights = np.concatenate([np.ones(len(in_ys)), np.full(len(calib_idx), w_cal)])

                    clf = LinearSVC(dual=False, C=c, random_state=seed, max_iter=2000)
                    clf.fit(X_aug, y_aug, sample_weight=weights)
                    pred = clf.predict(in_Xv_coral[test_idx])
                    seed_accs.append(accuracy_score(in_yv[test_idx], pred))

                val_accs.append(np.mean(seed_accs))

            grid_results.append({
                "w_cal": w_cal,
                "c": c,
                "val_acc": float(np.mean(val_accs)),
            })

    best_config = max(grid_results, key=lambda r: r["val_acc"])
    return best_config


def run_experiment_d3_evaluation(
    X_trial: np.ndarray,
    y_trial: np.ndarray,
    trial_meta: pd.DataFrame,
    channel_to_indices: Dict[str, List[int]],
    selections_df: pd.DataFrame,
    conditions: List[str] = ["stab_32", "all_62", "pred_32", "pred_12"],
    calib_seeds: List[int] = CALIBRATION_SEEDS,
    reg: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Execute 15-fold outer LOSO evaluation comparing:
    - Method A: source_only_baseline
    - Method B: coral_unsupervised
    - Method C: few_shot_calibration_only
    - Method D: coral_plus_four_shot
    across 5 pre-specified calibration seeds, scored strictly on 68 test trials.
    """
    subjects = trial_meta["subject_id"].to_numpy()
    unique_subs = sorted(np.unique(subjects))

    pred_records = []
    print(f"\nStarting Experiment D3: 15-fold outer LOSO evaluation across {len(conditions)} conditions and {len(calib_seeds)} calibration seeds...", flush=True)

    for fold_idx, test_sub in enumerate(unique_subs):
        t_fold = time.time()
        train_mask = subjects != test_sub
        test_mask = subjects == test_sub

        test_meta = trial_meta[test_mask].reset_index(drop=True)
        train_subs = subjects[train_mask]

        print(f"--- Outer Fold {fold_idx + 1}/15: Test Subject {test_sub} (Train: {train_mask.sum()}, Test: {test_mask.sum()}) ---", flush=True)

        for cond in conditions:
            feat_idx = get_channel_feature_indices(
                test_subject=test_sub,
                condition=cond,
                channel_to_indices=channel_to_indices,
                selections_df=selections_df,
            )

            Xs = X_trial[train_mask][:, feat_idx]
            ys = y_trial[train_mask]
            Xt = X_trial[test_mask][:, feat_idx]
            yt = y_trial[test_mask]

            # 1. Source-only preprocessing
            scaler = StandardScaler()
            Xs_s = scaler.fit_transform(Xs)
            Xt_s = scaler.transform(Xt)

            # 2. Transductive CORAL Alignment (Target-to-Source)
            # Uses target feature vectors without emotion labels
            _, A_ts, mu_s, mu_t = compute_coral_transformation(Xs_s, Xt_s, reg=reg)
            Xt_coral = (Xt_s - mu_t) @ A_ts + mu_s

            # 3. Inner Hyperparameter Tuning strictly within 14 training subjects
            best_config = tune_calibration_hyperparameters(
                Xs=Xs,
                ys=ys,
                source_subjects=train_subs,
                w_cal_grid=[5.0, 10.0, 15.0],
                c_grid=[0.005, 0.01, 0.02],
                calib_seeds=[42, 100, 2024],
                reg=reg,
                seed=seed,
            )
            w_cal = best_config["w_cal"]
            best_c = best_config["c"]

            # 4. Fit Baseline Models on Source Data
            clf_source = LinearSVC(dual=False, C=best_c, max_iter=2000, tol=1e-3, random_state=seed)
            clf_source.fit(Xs_s, ys)

            # 5. Evaluate across 5 calibration splits
            for calib_seed in calib_seeds:
                calib_idx, test_idx = select_four_shot_calibration_split(yt, seed=calib_seed, n_classes=4)

                # Ground truth for test trials
                yt_test = yt[test_idx]
                test_meta_split = test_meta.iloc[test_idx].reset_index(drop=True)

                # Method A: Source only (evaluated on 68 test trials)
                preds_a = clf_source.predict(Xt_s[test_idx])

                # Method B: CORAL only (evaluated on 68 aligned test trials)
                preds_b = clf_source.predict(Xt_coral[test_idx])

                # Method C: Few-shot calibration only (raw space, source + 4 calibration trials)
                X_aug_c = np.vstack([Xs_s, Xt_s[calib_idx]])
                y_aug_c = np.concatenate([ys, yt[calib_idx]])
                weights_c = np.concatenate([np.ones(len(ys)), np.full(len(calib_idx), w_cal)])
                clf_c = LinearSVC(dual=False, C=best_c, max_iter=2000, tol=1e-3, random_state=seed)
                clf_c.fit(X_aug_c, y_aug_c, sample_weight=weights_c)
                preds_c = clf_c.predict(Xt_s[test_idx])

                # Method D: CORAL + Few-shot calibration (aligned space, source + 4 aligned calibration trials)
                X_aug_d = np.vstack([Xs_s, Xt_coral[calib_idx]])
                y_aug_d = np.concatenate([ys, yt[calib_idx]])
                weights_d = np.concatenate([np.ones(len(ys)), np.full(len(calib_idx), w_cal)])
                clf_d = LinearSVC(dual=False, C=best_c, max_iter=2000, tol=1e-3, random_state=seed)
                clf_d.fit(X_aug_d, y_aug_d, sample_weight=weights_d)
                preds_d = clf_d.predict(Xt_coral[test_idx])

                # Record 68 test predictions per method
                methods_data = [
                    ("source_only_baseline", "none", preds_a),
                    ("coral_unsupervised", "unsupervised_covariance_alignment", preds_b),
                    ("few_shot_calibration_only", "four_shot_calibration_raw", preds_c),
                    ("coral_plus_four_shot", "coral_plus_four_shot_calibration", preds_d),
                ]

                for m_name, adapt_type, preds in methods_data:
                    for i in range(len(test_idx)):
                        orig_trial_idx = test_idx[i]
                        pred_records.append({
                            "subject_id": test_sub,
                            "session_id": int(test_meta.loc[orig_trial_idx, "session_id"]),
                            "trial_id": int(test_meta.loc[orig_trial_idx, "trial_id"]),
                            "trial_key": test_meta.loc[orig_trial_idx, "trial_key"],
                            "condition": cond,
                            "calibration_seed": calib_seed,
                            "method": m_name,
                            "adaptation_type": adapt_type,
                            "best_c": best_c,
                            "w_cal": w_cal if "calibration" in m_name else np.nan,
                            "n_features": len(feat_idx),
                            "n_calibration_trials": 4,
                            "n_test_trials": 68,
                            "true_label": int(yt[orig_trial_idx]),
                            "true_label_name": LABEL_NAMES[int(yt[orig_trial_idx])],
                            "predicted_label": int(preds[i]),
                            "predicted_label_name": LABEL_NAMES[int(preds[i])],
                            "is_correct": int(preds[i] == yt[orig_trial_idx]),
                        })

        print(f"  Fold {fold_idx + 1} completed in {time.time() - t_fold:.1f}s", flush=True)

    pred_df = pd.DataFrame(pred_records)
    return pred_df


def compute_experiment_d3_metrics(
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute comprehensive metrics across conditions and methods:
    1. Subject-level metrics averaged over the 5 calibration splits first.
    2. Overall summary table (mean over 15 subjects, std, CIs, balanced acc, macro-F1).
    3. Class-level metrics table.
    4. Paired comparisons across the 15 subjects (paired t-test, Wilcoxon, CIs).
    """
    # 1. Subject-level metrics (First aggregate over calibration seeds per subject)
    # A subject has 68 trials x 5 seeds = 340 test predictions per (condition, method)
    sub_records = []
    for (cond, method, sub), grp in pred_df.groupby(["condition", "method", "subject_id"]):
        # Accuracy across repeated seeds
        # Calculate accuracy within each seed first, then take the subject mean
        seed_accs = []
        seed_bal_accs = []
        seed_f1s = []
        for s, s_grp in grp.groupby("calibration_seed"):
            acc = accuracy_score(s_grp["true_label"], s_grp["predicted_label"]) * 100.0
            bal = balanced_accuracy_score(s_grp["true_label"], s_grp["predicted_label"]) * 100.0
            f1 = f1_score(s_grp["true_label"], s_grp["predicted_label"], average="macro") * 100.0
            seed_accs.append(acc)
            seed_bal_accs.append(bal)
            seed_f1s.append(f1)

        sub_records.append({
            "condition": cond,
            "method": method,
            "subject_id": sub,
            "accuracy": float(np.mean(seed_accs)),
            "balanced_accuracy": float(np.mean(seed_bal_accs)),
            "macro_f1": float(np.mean(seed_f1s)),
            "seed_acc_std": float(np.std(seed_accs, ddof=1)) if len(seed_accs) > 1 else 0.0,
        })
    subject_df = pd.DataFrame(sub_records)

    # 2. Overall summary metrics across the 15 subjects
    sum_records = []
    for (cond, method), grp in pred_df.groupby(["condition", "method"]):
        n_features = grp["n_features"].iloc[0]
        tot_test_trials = len(grp)  # 15 subjects x 68 trials x 5 seeds = 5,100
        corr_test_trials = grp["is_correct"].sum()
        pooled_acc = corr_test_trials / tot_test_trials * 100.0

        sub_accs = subject_df[(subject_df["condition"] == cond) & (subject_df["method"] == method)]["accuracy"].to_numpy()
        mean_acc = float(np.mean(sub_accs))
        std_acc = float(np.std(sub_accs, ddof=1))
        med_acc = float(np.median(sub_accs))
        min_acc = float(np.min(sub_accs))
        max_acc = float(np.max(sub_accs))

        # Student-t 95% CI across 15 subjects
        n = len(sub_accs)
        se = std_acc / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_t = (mean_acc - t_crit * se, mean_acc + t_crit * se)

        # Bootstrap 95% CI across 15 subjects
        rng = np.random.default_rng(42)
        boot_means = [np.mean(rng.choice(sub_accs, size=n, replace=True)) for _ in range(2000)]
        ci_boot = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))

        sub_bal_accs = subject_df[(subject_df["condition"] == cond) & (subject_df["method"] == method)]["balanced_accuracy"].to_numpy()
        sub_f1s = subject_df[(subject_df["condition"] == cond) & (subject_df["method"] == method)]["macro_f1"].to_numpy()

        sum_records.append({
            "condition": cond,
            "method": method,
            "n_features": n_features,
            "n_subjects": n,
            "total_test_evaluations": tot_test_trials,
            "pooled_accuracy": pooled_acc,
            "subject_mean_accuracy": mean_acc,
            "subject_std_accuracy": std_acc,
            "subject_median_accuracy": med_acc,
            "subject_min_accuracy": min_acc,
            "subject_max_accuracy": max_acc,
            "ci95_lower_t": ci_t[0],
            "ci95_upper_t": ci_t[1],
            "ci95_lower_boot": ci_boot[0],
            "ci95_upper_boot": ci_boot[1],
            "balanced_accuracy": float(np.mean(sub_bal_accs)),
            "macro_f1": float(np.mean(sub_f1s)),
        })
    summary_df = pd.DataFrame(sum_records)

    # 3. Class metrics (computed on all test predictions across 5 seeds)
    cls_records = []
    for (cond, method), grp in pred_df.groupby(["condition", "method"]):
        y_true = grp["true_label"].to_numpy()
        y_pred = grp["predicted_label"].to_numpy()
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        for c in range(4):
            cls_records.append({
                "condition": cond,
                "method": method,
                "class_id": c,
                "class_name": LABEL_NAMES[c],
                "support": int(s[c]),
                "precision": float(p[c] * 100.0),
                "recall": float(r[c] * 100.0),
                "f1_score": float(f[c] * 100.0),
            })
    class_df = pd.DataFrame(cls_records)

    # 4. Paired comparisons across the 15 subjects (aggregated subject-level accuracy)
    comp_records = []
    conditions = summary_df["condition"].unique()

    for cond in conditions:
        sub_source = subject_df[(subject_df["condition"] == cond) & (subject_df["method"] == "source_only_baseline")].set_index("subject_id")["accuracy"]
        sub_coral = subject_df[(subject_df["condition"] == cond) & (subject_df["method"] == "coral_unsupervised")].set_index("subject_id")["accuracy"]
        sub_calib = subject_df[(subject_df["condition"] == cond) & (subject_df["method"] == "few_shot_calibration_only")].set_index("subject_id")["accuracy"]
        sub_coral_calib = subject_df[(subject_df["condition"] == cond) & (subject_df["method"] == "coral_plus_four_shot")].set_index("subject_id")["accuracy"]

        comparisons = [
            ("coral_plus_four_shot", "source_only_baseline", sub_coral_calib, sub_source),
            ("coral_plus_four_shot", "coral_unsupervised", sub_coral_calib, sub_coral),
            ("coral_plus_four_shot", "few_shot_calibration_only", sub_coral_calib, sub_calib),
            ("coral_unsupervised", "source_only_baseline", sub_coral, sub_source),
            ("few_shot_calibration_only", "source_only_baseline", sub_calib, sub_source),
        ]

        for eval_m, ref_m, s_eval, s_ref in comparisons:
            diff = (s_eval - s_ref).dropna()
            n = len(diff)
            mean_d = float(np.mean(diff))
            std_d = float(np.std(diff, ddof=1))
            se_d = std_d / np.sqrt(n)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            ci_t = (mean_d - t_crit * se_d, mean_d + t_crit * se_d)

            rng = np.random.default_rng(42)
            boot_d = [np.mean(rng.choice(diff.values, size=n, replace=True)) for _ in range(2000)]
            ci_boot = (float(np.percentile(boot_d, 2.5)), float(np.percentile(boot_d, 97.5)))

            t_res = stats.ttest_rel(s_eval, s_ref)
            try:
                w_res = stats.wilcoxon(s_eval, s_ref)
                w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                w_stat, w_pval = np.nan, np.nan

            comp_records.append({
                "condition": cond,
                "evaluated_method": eval_m,
                "reference_baseline": ref_m,
                "evaluated_mean_acc": float(np.mean(s_eval)),
                "reference_mean_acc": float(np.mean(s_ref)),
                "mean_delta_acc_pp": mean_d,
                "std_delta_acc_pp": std_d,
                "ci95_lower_t": ci_t[0],
                "ci95_upper_t": ci_t[1],
                "ci95_lower_boot": ci_boot[0],
                "ci95_upper_boot": ci_boot[1],
                "t_stat": float(t_res.statistic),
                "t_pval": float(t_res.pvalue),
                "wilcoxon_stat": w_stat,
                "wilcoxon_pval": w_pval,
                "n_pos_subjects": int((diff > 0).sum()),
                "n_neg_subjects": int((diff < 0).sum()),
                "n_zero_subjects": int((diff == 0).sum()),
            })

    comparison_df = pd.DataFrame(comp_records)
    return summary_df, subject_df, class_df, comparison_df


def plot_experiment_d3_figures(
    summary_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> None:
    """
    Generate publication figures for Experiment D3:
    1. Grouped bar chart comparing the 4 methods across channel sets with 80% line.
    2. Confusion matrix comparison heatmaps on stab_32.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Comparison Bar Chart
    conditions = ["all_62", "stab_32", "pred_32", "pred_12"]
    methods = [
        ("source_only_baseline", "Method A: Source Only", "#718096"),
        ("few_shot_calibration_only", "Method C: 4-Shot Calib Only", "#ED8936"),
        ("coral_unsupervised", "Method B: CORAL Only", "#3182CE"),
        ("coral_plus_four_shot", "Method D: CORAL + 4-Shot", "#38A169"),
    ]

    x = np.arange(len(conditions))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (m_id, m_label, color) in enumerate(methods):
        sub_df = summary_df[summary_df["method"] == m_id].set_index("condition").reindex(conditions)
        means = sub_df["subject_mean_accuracy"].values
        stds = sub_df["subject_std_accuracy"].values
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=m_label, color=color, alpha=0.9, edgecolor="black")
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.2,
                    f"{mean:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold"
                )

    ax.axhline(80.0, color="#E53E3E", linestyle="--", linewidth=2.0, label="Aspirational Target (80%)")
    ax.axhline(25.0, color="#A0AEC0", linestyle=":", linewidth=1.5, label="Chance Level (25%)")

    ax.set_ylabel("Subject-Level Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Experiment D3: Four-Shot Calibration Combined with CORAL\n15-Fold Outer LOSO on SEED-IV (68 Test Trials per Split x 5 Seeds)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(["All 62 Channels", "Stable 32 Channels\n(Primary)", "Predictive 32 Channels", "Predictive 12 Channels"], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 90)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    chart_path = FIGURES_DIR / "experiment_d3_calibration_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved D3 calibration comparison figure to: {chart_path}")

    # 2. Confusion Matrices (stab_32)
    stab_preds = pred_df[pred_df["condition"] == "stab_32"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    class_labels = ["Neutral", "Sad", "Fear", "Happy"]

    for idx, (m_id, m_label, _) in enumerate(methods):
        sub_p = stab_preds[stab_preds["method"] == m_id]
        cm = confusion_matrix(sub_p["true_label"], sub_p["predicted_label"])
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100.0

        ax = axes[idx]
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=80)
        ax.set_title(f"{m_label}\n(Acc: {accuracy_score(sub_p['true_label'], sub_p['predicted_label'])*100:.1f}%)", fontsize=10, fontweight="bold")

        for r in range(4):
            for c in range(4):
                val = cm_norm[r, c]
                cnt = cm[r, c]
                txt_color = "white" if val > 45 else "black"
                ax.text(c, r, f"{val:.1f}%\n({cnt})", ha="center", va="center", color=txt_color, fontsize=8)

        ax.set_xticks(range(4))
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(class_labels, fontsize=9)
        ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("True Class", fontsize=10, fontweight="bold")

    fig.suptitle("Experiment D3 Confusion Matrices on stab_32 (Scored on 68 Test Trials x 5 Seeds = 5,100 Tests)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    cm_path = FIGURES_DIR / "experiment_d3_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved D3 confusion matrix figure to: {cm_path}")


def main() -> None:
    print("=" * 80)
    print("EXPERIMENT D3: FOUR-SHOT SUBJECT CALIBRATION COMBINED WITH CORAL")
    print("=" * 80)

    # 1. Load Trial Features & Metadata
    trial_feat_path = TABLES_DIR / "experiment_c_trial_features.npz"
    trial_meta_path = TABLES_DIR / "experiment_c_trial_metadata.csv"

    if not trial_feat_path.is_file() or not trial_meta_path.is_file():
        print("Constructing trial features from dataset...")
        from src.evaluate_trial_retraining import build_trial_feature_dataset
        X_trial, y_trial, trial_meta = build_trial_feature_dataset()
    else:
        print(f"Loading trial features from: {trial_feat_path}")
        feat_dict = np.load(trial_feat_path)
        X_trial = feat_dict["mean"]
        trial_meta = pd.read_csv(trial_meta_path)
        y_trial = trial_meta["label"].to_numpy()

    selections_path = TABLES_DIR / "corrected_channel_selections.csv"
    if not selections_path.is_file():
        raise FileNotFoundError(f"Missing channel selections: {selections_path}")
    selections_df = pd.read_csv(selections_path)

    dataset = load_seed_iv_dataset()
    channel_to_indices = dataset.channel_to_indices

    # 2. Run Experiment D3 evaluation
    t0 = time.time()
    pred_df = run_experiment_d3_evaluation(
        X_trial=X_trial,
        y_trial=y_trial,
        trial_meta=trial_meta,
        channel_to_indices=channel_to_indices,
        selections_df=selections_df,
        conditions=["stab_32", "all_62", "pred_32", "pred_12"],
        calib_seeds=CALIBRATION_SEEDS,
        reg=1.0,
        seed=42,
    )

    pred_path = PREDICTIONS_DIR / "experiment_d3_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved Experiment D3 predictions ({len(pred_df)} rows) to: {pred_path}")

    # 3. Compute metrics & comparisons
    print("\nComputing comprehensive summary, subject, class, and comparison metrics...")
    summary_df, subject_df, class_df, comparison_df = compute_experiment_d3_metrics(pred_df)

    sum_path = TABLES_DIR / "experiment_d3_summary.csv"
    sub_path = TABLES_DIR / "experiment_d3_subject_metrics.csv"
    cls_path = TABLES_DIR / "experiment_d3_class_metrics.csv"
    comp_path = TABLES_DIR / "experiment_d3_vs_baselines_comparison.csv"

    summary_df.to_csv(sum_path, index=False)
    subject_df.to_csv(sub_path, index=False)
    class_df.to_csv(cls_path, index=False)
    comparison_df.to_csv(comp_path, index=False)

    print(f"Saved summary metrics to: {sum_path}")
    print(f"Saved subject metrics to: {sub_path}")
    print(f"Saved class metrics to: {cls_path}")
    print(f"Saved comparison metrics to: {comp_path}")

    print("\nSummary Results (Subject-Level Averages across 15 Subjects):")
    print(summary_df[["condition", "method", "pooled_accuracy", "subject_mean_accuracy", "subject_std_accuracy", "macro_f1"]].to_string(index=False))

    print("\nKey Comparisons (Paired Differences across 15 Subjects):")
    print(comparison_df[["condition", "evaluated_method", "reference_baseline", "evaluated_mean_acc", "reference_mean_acc", "mean_delta_acc_pp", "t_pval", "wilcoxon_pval", "n_pos_subjects"]].to_string(index=False))

    # 4. Generate Figures
    print("\nGenerating publication figures...")
    plot_experiment_d3_figures(summary_df, pred_df)

    total_time = time.time() - t0
    print(f"\nExperiment D3 Pipeline successfully completed in {total_time:.1f}s ({total_time/60:.2f} min).")


if __name__ == "__main__":
    main()
