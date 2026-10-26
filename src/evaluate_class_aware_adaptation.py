"""
src/evaluate_class_aware_adaptation.py

Experiment D2: Class-Aware Unsupervised Domain Adaptation for SEED-IV.

Protocol:
1. Feature baseline: SEED-IV trial-level de_LDS features (1,080 trials, 72 per subject).
2. Outer evaluation: 15 outer Leave-One-Subject-Out (LOSO) folds.
3. Methods compared:
   - Strict LOSO Baseline (no target data)
   - D1 CORAL Target-to-Source Alignment (unsupervised marginal covariance alignment)
   - Class-Balanced Pseudo-Label Refinement (CB-PLR) (class-aware adaptation over CORAL)
4. Class-Aware Adaptation Logic:
   - Base classifier with probability calibration fitted on source domain.
   - Target trials mapped to source domain via Tikhonov-regularized CORAL.
   - Target posterior probabilities and confidence scores generated on aligned target features.
   - Conservative class-balanced selection: for each of the 4 classes, select up to k_max trials
     with confidence >= tau.
   - Retrain classifier on augmented source + weighted pseudo-labeled target data.
   - Evaluate on aligned target trials.
5. Leakage Safeguards & Nested Validation:
   - Zero target emotion labels are ever used for alignment, pseudo-label selection, tuning, or stopping.
   - Hyperparameters (tau, k_max, w_pseudo, C) are tuned strictly within the 14 training subjects
     via simulated source-to-validation subject folds.
   - Pseudo-label quality (precision, recall, selection rate) is audited exclusively in inner validation folds.
6. Channel configurations: stab_32 (primary baseline), all_62, pred_32, pred_12.
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
from sklearn.calibration import CalibratedClassifierCV
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


def select_conservative_pseudo_labels(
    probs: np.ndarray,
    tau: float = 0.45,
    k_max: int = 4,
    n_classes: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select conservative, class-balanced pseudo-labels from predicted class probabilities.
    
    Rule:
    For each class c in {0, ..., n_classes - 1}:
      Find target samples where argmax(probs) == c and max(probs) >= tau.
      Sort candidates by confidence descending.
      Select up to k_max top candidates per class to preserve class balance.
      
    Args:
        probs: Array of shape (n_samples, n_classes) of predicted class probabilities.
        tau: Confidence threshold.
        k_max: Maximum number of pseudo-labeled samples permitted per class.
        n_classes: Number of emotion classes (4 for SEED-IV).
        
    Returns:
        selected_indices: 1D array of selected sample indices.
        selected_labels: 1D array of assigned pseudo-labels.
        selected_confidences: 1D array of prediction confidence for selected samples.
    """
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)

    selected_indices = []
    selected_labels = []
    selected_confidences = []

    for c in range(n_classes):
        cand_mask = (preds == c) & (confs >= tau)
        cand_indices = np.where(cand_mask)[0]
        if len(cand_indices) > 0:
            # Sort by confidence descending
            sorted_idx = cand_indices[np.argsort(-confs[cand_indices])]
            top_k = sorted_idx[:k_max]
            selected_indices.extend(top_k)
            selected_labels.extend([c] * len(top_k))
            selected_confidences.extend(confs[top_k])

    if len(selected_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    selected_indices = np.array(selected_indices, dtype=int)
    selected_labels = np.array(selected_labels, dtype=int)
    selected_confidences = np.array(selected_confidences, dtype=float)

    return selected_indices, selected_labels, selected_confidences


def tune_cb_plr_hyperparameters(
    Xs: np.ndarray,
    ys: np.ndarray,
    source_subjects: np.ndarray,
    tau_grid: List[float] = [0.35, 0.45, 0.55],
    k_max_grid: List[int] = [2, 4, 6],
    w_pseudo_grid: List[float] = [0.1, 0.25, 0.5],
    c_grid: List[float] = [0.01],
    reg: float = 1.0,
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Tune CB-PLR hyperparameters using simulated source-to-validation subject folds
    strictly within the 14 training subjects.
    
    Zero outer held-out test subject data is ever seen or used.
    Also audits pseudo-label precision on inner validation subjects where true labels exist.
    """
    unique_subs = sorted(np.unique(source_subjects))
    val_subs = unique_subs[::3]  # 4-5 representative validation subjects

    # Precompute base models, CORAL features, and predicted probabilities for each validation subject
    val_cache = []
    for v_sub in val_subs:
        in_tr_m = source_subjects != v_sub
        in_val_m = source_subjects == v_sub

        in_Xs = Xs[in_tr_m]
        in_ys = ys[in_tr_m]
        in_Xval = Xs[in_val_m]
        in_yval = ys[in_val_m]

        scaler = StandardScaler()
        in_Xs_s = scaler.fit_transform(in_Xs)
        in_Xval_s = scaler.transform(in_Xval)

        base_clf = LinearSVC(dual=False, C=c_grid[0], random_state=seed, max_iter=2000)
        base_clf.fit(in_Xs_s, in_ys)
        b_acc = accuracy_score(in_yval, base_clf.predict(in_Xval_s))

        _, A_ts, mu_s, mu_t = compute_coral_transformation(in_Xs_s, in_Xval_s, reg=reg)
        in_Xval_coral = (in_Xval_s - mu_t) @ A_ts + mu_s
        c_acc = accuracy_score(in_yval, base_clf.predict(in_Xval_coral))

        cal = CalibratedClassifierCV(LinearSVC(dual=False, C=c_grid[0], random_state=seed, max_iter=2000), cv=3)
        cal.fit(in_Xs_s, in_ys)
        probs = cal.predict_proba(in_Xval_coral)

        val_cache.append({
            "v_sub": v_sub,
            "in_Xs_s": in_Xs_s,
            "in_ys": in_ys,
            "in_Xval_coral": in_Xval_coral,
            "in_yval": in_yval,
            "base_acc": b_acc,
            "coral_acc": c_acc,
            "probs": probs,
        })

    grid_results = []
    for tau in tau_grid:
        for k_max in k_max_grid:
            for w_pseudo in w_pseudo_grid:
                val_accs = []
                val_precs = []
                val_counts = []
                base_accs = [cache["base_acc"] for cache in val_cache]
                coral_accs = [cache["coral_acc"] for cache in val_cache]

                for cache in val_cache:
                    in_Xs_s = cache["in_Xs_s"]
                    in_ys = cache["in_ys"]
                    in_Xval_coral = cache["in_Xval_coral"]
                    in_yval = cache["in_yval"]
                    probs = cache["probs"]
                    c_acc = cache["coral_acc"]

                    sel_idx, sel_labels, sel_confs = select_conservative_pseudo_labels(
                        probs=probs, tau=tau, k_max=k_max, n_classes=4
                    )

                    if len(sel_idx) > 0:
                        pseudo_prec = accuracy_score(in_yval[sel_idx], sel_labels)
                        val_precs.append(pseudo_prec)
                        val_counts.append(len(sel_idx))

                        X_aug = np.vstack([in_Xs_s, in_Xval_coral[sel_idx]])
                        y_aug = np.concatenate([in_ys, sel_labels])
                        weights = np.concatenate([np.ones(len(in_ys)), np.full(len(sel_idx), w_pseudo)])

                        clf_retrained = LinearSVC(dual=False, C=c_grid[0], random_state=seed, max_iter=2000)
                        clf_retrained.fit(X_aug, y_aug, sample_weight=weights)
                        adapted_acc = accuracy_score(in_yval, clf_retrained.predict(in_Xval_coral))
                    else:
                        val_precs.append(0.0)
                        val_counts.append(0)
                        adapted_acc = c_acc

                    val_accs.append(adapted_acc)

                grid_results.append({
                    "tau": tau,
                    "k_max": k_max,
                    "w_pseudo": w_pseudo,
                    "c": c_grid[0],
                    "val_acc": float(np.mean(val_accs)),
                    "val_precision": float(np.mean(val_precs)),
                    "n_pseudo_selected": float(np.mean(val_counts)),
                    "base_val_acc": float(np.mean(base_accs)),
                    "coral_val_acc": float(np.mean(coral_accs)),
                })

    # Pick best configuration by inner validation accuracy
    best_config = max(grid_results, key=lambda r: r["val_acc"])
    return best_config, grid_results


def run_experiment_d2_evaluation(
    X_trial: np.ndarray,
    y_trial: np.ndarray,
    trial_meta: pd.DataFrame,
    channel_to_indices: Dict[str, List[int]],
    selections_df: pd.DataFrame,
    conditions: List[str] = ["stab_32", "all_62", "pred_32", "pred_12"],
    reg: float = 1.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute 15-fold outer LOSO evaluation comparing:
    1. Strict LOSO Baseline (no adaptation)
    2. D1 CORAL Target-to-Source (unsupervised covariance alignment)
    3. Experiment D2 Class-Balanced Pseudo-Label Refinement (CB-PLR)
    """
    subjects = trial_meta["subject_id"].to_numpy()
    unique_subs = sorted(np.unique(subjects))

    pred_records = []
    inner_audit_records = []

    print(f"\nStarting Experiment D2: 15-fold outer LOSO evaluation across {len(conditions)} conditions...", flush=True)

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

            # 1. Strict source-only preprocessing
            scaler = StandardScaler()
            Xs_s = scaler.fit_transform(Xs)
            Xt_s = scaler.transform(Xt)

            # 2. Strict LOSO Baseline (Linear SVC, C=0.01)
            clf_base = LinearSVC(dual=False, C=0.01, max_iter=2000, tol=1e-3, random_state=seed)
            clf_base.fit(Xs_s, ys)
            preds_base = clf_base.predict(Xt_s)

            # 3. D1 CORAL Alignment (Target-to-Source)
            _, A_ts, mu_s, mu_t = compute_coral_transformation(Xs_s, Xt_s, reg=reg)
            Xt_coral = (Xt_s - mu_t) @ A_ts + mu_s
            preds_coral = clf_base.predict(Xt_coral)

            # 4. Experiment D2: Nested Inner Validation Tuning for CB-PLR
            # Strictly within the 14 training subjects
            best_config, inner_grid = tune_cb_plr_hyperparameters(
                Xs=Xs,
                ys=ys,
                source_subjects=train_subs,
                tau_grid=[0.35, 0.45, 0.55],
                k_max_grid=[2, 4, 6],
                w_pseudo_grid=[0.1, 0.25, 0.5],
                c_grid=[0.01],
                reg=reg,
                seed=seed,
            )

            # Record inner audit results for verification
            for r in inner_grid:
                inner_audit_records.append({
                    "outer_test_subject": test_sub,
                    "condition": cond,
                    **r
                })

            # 5. Fit Calibrated Model on Source to Generate Target Pseudo-Labels
            cal = CalibratedClassifierCV(
                LinearSVC(dual=False, C=best_config["c"], max_iter=2000, tol=1e-3, random_state=seed),
                cv=3,
            )
            cal.fit(Xs_s, ys)
            probs_t = cal.predict_proba(Xt_coral)
            confs_t = probs_t.max(axis=1)

            # Select conservative pseudo-labels using inner-selected hyperparameters
            sel_idx, sel_labels, sel_confs = select_conservative_pseudo_labels(
                probs=probs_t,
                tau=best_config["tau"],
                k_max=best_config["k_max"],
                n_classes=4,
            )

            # Train adapted classifier with source + pseudo-labeled target data
            if len(sel_idx) > 0:
                X_aug = np.vstack([Xs_s, Xt_coral[sel_idx]])
                y_aug = np.concatenate([ys, sel_labels])
                weights = np.concatenate([
                    np.ones(len(ys)),
                    np.full(len(sel_idx), best_config["w_pseudo"]),
                ])

                clf_adapted = LinearSVC(
                    dual=False, C=best_config["c"], max_iter=2000, tol=1e-3, random_state=seed
                )
                clf_adapted.fit(X_aug, y_aug, sample_weight=weights)
                preds_adapted = clf_adapted.predict(Xt_coral)
            else:
                preds_adapted = preds_coral

            # Record predictions for all 72 trials
            for i in range(len(test_meta)):
                # Baseline
                pred_records.append({
                    "subject_id": test_sub,
                    "session_id": int(test_meta.loc[i, "session_id"]),
                    "trial_id": int(test_meta.loc[i, "trial_id"]),
                    "trial_key": test_meta.loc[i, "trial_key"],
                    "condition": cond,
                    "method": "strict_loso_baseline",
                    "adaptation_type": "none",
                    "best_c": 0.01,
                    "tau": np.nan,
                    "k_max": np.nan,
                    "w_pseudo": np.nan,
                    "n_features": len(feat_idx),
                    "n_pseudo_selected": 0,
                    "true_label": int(yt[i]),
                    "true_label_name": LABEL_NAMES[int(yt[i])],
                    "predicted_label": int(preds_base[i]),
                    "predicted_label_name": LABEL_NAMES[int(preds_base[i])],
                    "confidence": np.nan,
                    "is_correct": int(preds_base[i] == yt[i]),
                })
                # D1 CORAL (T->S)
                pred_records.append({
                    "subject_id": test_sub,
                    "session_id": int(test_meta.loc[i, "session_id"]),
                    "trial_id": int(test_meta.loc[i, "trial_id"]),
                    "trial_key": test_meta.loc[i, "trial_key"],
                    "condition": cond,
                    "method": "coral_target_to_source",
                    "adaptation_type": "unsupervised_covariance_alignment",
                    "best_c": 0.01,
                    "tau": np.nan,
                    "k_max": np.nan,
                    "w_pseudo": np.nan,
                    "n_features": len(feat_idx),
                    "n_pseudo_selected": 0,
                    "true_label": int(yt[i]),
                    "true_label_name": LABEL_NAMES[int(yt[i])],
                    "predicted_label": int(preds_coral[i]),
                    "predicted_label_name": LABEL_NAMES[int(preds_coral[i])],
                    "confidence": float(confs_t[i]),
                    "is_correct": int(preds_coral[i] == yt[i]),
                })
                # D2 Class-Balanced Pseudo-Label Refinement (CB-PLR)
                pred_records.append({
                    "subject_id": test_sub,
                    "session_id": int(test_meta.loc[i, "session_id"]),
                    "trial_id": int(test_meta.loc[i, "trial_id"]),
                    "trial_key": test_meta.loc[i, "trial_key"],
                    "condition": cond,
                    "method": "class_balanced_pseudo_label_refinement",
                    "adaptation_type": "class_aware_pseudo_label_adaptation",
                    "best_c": best_config["c"],
                    "tau": best_config["tau"],
                    "k_max": best_config["k_max"],
                    "w_pseudo": best_config["w_pseudo"],
                    "n_features": len(feat_idx),
                    "n_pseudo_selected": len(sel_idx),
                    "true_label": int(yt[i]),
                    "true_label_name": LABEL_NAMES[int(yt[i])],
                    "predicted_label": int(preds_adapted[i]),
                    "predicted_label_name": LABEL_NAMES[int(preds_adapted[i])],
                    "confidence": float(confs_t[i]),
                    "is_correct": int(preds_adapted[i] == yt[i]),
                })

        print(f"  Fold {fold_idx + 1} completed in {time.time() - t_fold:.1f}s", flush=True)

    pred_df = pd.DataFrame(pred_records)
    inner_audit_df = pd.DataFrame(inner_audit_records)
    return pred_df, inner_audit_df


def compute_experiment_d2_metrics(
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute comprehensive metrics across conditions and methods:
    1. Overall summary table (pooled acc, subject mean/std/median/CIs, balanced acc, macro-F1)
    2. Subject-level accuracy table
    3. Per-class metrics table (precision, recall, f1)
    4. Paired comparisons vs both Strict LOSO Baseline and D1 CORAL Baseline
    """
    # 1. Subject-level metrics
    sub_records = []
    for (cond, method, sub), grp in pred_df.groupby(["condition", "method", "subject_id"]):
        acc = accuracy_score(grp["true_label"], grp["predicted_label"]) * 100.0
        bal_acc = balanced_accuracy_score(grp["true_label"], grp["predicted_label"]) * 100.0
        f1 = f1_score(grp["true_label"], grp["predicted_label"], average="macro") * 100.0
        sub_records.append({
            "condition": cond,
            "method": method,
            "subject_id": sub,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": f1,
        })
    subject_df = pd.DataFrame(sub_records)

    # 2. Overall summary metrics
    sum_records = []
    for (cond, method), grp in pred_df.groupby(["condition", "method"]):
        n_features = grp["n_features"].iloc[0]
        tot = len(grp)
        corr = grp["is_correct"].sum()
        pooled_acc = corr / tot * 100.0

        sub_accs = subject_df[(subject_df["condition"] == cond) & (subject_df["method"] == method)]["accuracy"].to_numpy()
        mean_acc = np.mean(sub_accs)
        std_acc = np.std(sub_accs, ddof=1)
        med_acc = np.median(sub_accs)
        min_acc = np.min(sub_accs)
        max_acc = np.max(sub_accs)

        # Student-t 95% CI
        n = len(sub_accs)
        se = std_acc / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_t = (mean_acc - t_crit * se, mean_acc + t_crit * se)

        # Bootstrap 95% CI
        rng = np.random.default_rng(42)
        boot_means = [np.mean(rng.choice(sub_accs, size=n, replace=True)) for _ in range(2000)]
        ci_boot = (np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5))

        bal_acc = balanced_accuracy_score(grp["true_label"], grp["predicted_label"]) * 100.0
        f1 = f1_score(grp["true_label"], grp["predicted_label"], average="macro") * 100.0

        sum_records.append({
            "condition": cond,
            "method": method,
            "n_features": n_features,
            "total_trials": tot,
            "correct_trials": corr,
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
            "balanced_accuracy": bal_acc,
            "macro_f1": f1,
        })
    summary_df = pd.DataFrame(sum_records)

    # 3. Class metrics
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

    # 4. Paired comparisons vs Strict LOSO Baseline and vs D1 CORAL Baseline
    comp_records = []
    conditions = summary_df["condition"].unique()

    for cond in conditions:
        sub_base = subject_df[(subject_df["condition"] == cond) & (subject_df["method"] == "strict_loso_baseline")].set_index("subject_id")["accuracy"]
        sub_coral = subject_df[(subject_df["condition"] == cond) & (subject_df["method"] == "coral_target_to_source")].set_index("subject_id")["accuracy"]
        sub_cb_plr = subject_df[(subject_df["condition"] == cond) & (subject_df["method"] == "class_balanced_pseudo_label_refinement")].set_index("subject_id")["accuracy"]

        # Comparison 1: CB-PLR vs Strict Baseline
        diff_vs_base = (sub_cb_plr - sub_base).dropna()
        n = len(diff_vs_base)
        mean_d = float(np.mean(diff_vs_base))
        std_d = float(np.std(diff_vs_base, ddof=1))
        se_d = std_d / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_t = (mean_d - t_crit * se_d, mean_d + t_crit * se_d)

        rng = np.random.default_rng(42)
        boot_d = [np.mean(rng.choice(diff_vs_base.values, size=n, replace=True)) for _ in range(2000)]
        ci_boot = (float(np.percentile(boot_d, 2.5)), float(np.percentile(boot_d, 97.5)))

        t_res = stats.ttest_rel(sub_cb_plr, sub_base)
        try:
            w_res = stats.wilcoxon(sub_cb_plr, sub_base)
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_pval = np.nan, np.nan

        comp_records.append({
            "condition": cond,
            "evaluated_method": "class_balanced_pseudo_label_refinement",
            "reference_baseline": "strict_loso_baseline",
            "evaluated_mean_acc": float(np.mean(sub_cb_plr)),
            "reference_mean_acc": float(np.mean(sub_base)),
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
            "n_pos_subjects": int((diff_vs_base > 0).sum()),
            "n_neg_subjects": int((diff_vs_base < 0).sum()),
            "n_zero_subjects": int((diff_vs_base == 0).sum()),
        })

        # Comparison 2: CB-PLR vs D1 CORAL Baseline
        diff_vs_coral = (sub_cb_plr - sub_coral).dropna()
        mean_dc = float(np.mean(diff_vs_coral))
        std_dc = float(np.std(diff_vs_coral, ddof=1))
        se_dc = std_dc / np.sqrt(n)
        ci_tc = (mean_dc - t_crit * se_dc, mean_dc + t_crit * se_dc)
        boot_dc = [np.mean(rng.choice(diff_vs_coral.values, size=n, replace=True)) for _ in range(2000)]
        ci_bootc = (float(np.percentile(boot_dc, 2.5)), float(np.percentile(boot_dc, 97.5)))

        t_resc = stats.ttest_rel(sub_cb_plr, sub_coral)
        try:
            w_resc = stats.wilcoxon(sub_cb_plr, sub_coral)
            w_statc, w_pvalc = float(w_resc.statistic), float(w_resc.pvalue)
        except Exception:
            w_statc, w_pvalc = np.nan, np.nan

        comp_records.append({
            "condition": cond,
            "evaluated_method": "class_balanced_pseudo_label_refinement",
            "reference_baseline": "coral_target_to_source",
            "evaluated_mean_acc": float(np.mean(sub_cb_plr)),
            "reference_mean_acc": float(np.mean(sub_coral)),
            "mean_delta_acc_pp": mean_dc,
            "std_delta_acc_pp": std_dc,
            "ci95_lower_t": ci_tc[0],
            "ci95_upper_t": ci_tc[1],
            "ci95_lower_boot": ci_bootc[0],
            "ci95_upper_boot": ci_bootc[1],
            "t_stat": float(t_resc.statistic),
            "t_pval": float(t_resc.pvalue),
            "wilcoxon_stat": w_statc,
            "wilcoxon_pval": w_pvalc,
            "n_pos_subjects": int((diff_vs_coral > 0).sum()),
            "n_neg_subjects": int((diff_vs_coral < 0).sum()),
            "n_zero_subjects": int((diff_vs_coral == 0).sum()),
        })

    comparison_df = pd.DataFrame(comp_records)
    return summary_df, subject_df, class_df, comparison_df


def plot_experiment_d2_figures(
    summary_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> None:
    """
    Generate publication figures for Experiment D2:
    1. Grouped bar chart comparing Strict Baseline, D1 CORAL, and D2 CB-PLR with 80% line.
    2. Confusion matrix comparison heatmaps on stab_32.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Comparison Bar Chart
    conditions = ["all_62", "stab_32", "pred_32", "pred_12"]
    methods = [
        ("strict_loso_baseline", "Strict LOSO Baseline", "#718096"),
        ("coral_target_to_source", "D1: CORAL (T->S)", "#3182CE"),
        ("class_balanced_pseudo_label_refinement", "D2: Class-Balanced Pseudo-Label (CB-PLR)", "#38A169"),
    ]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (m_id, m_label, color) in enumerate(methods):
        sub_df = summary_df[summary_df["method"] == m_id].set_index("condition").reindex(conditions)
        means = sub_df["subject_mean_accuracy"].values
        stds = sub_df["subject_std_accuracy"].values
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=4, label=m_label, color=color, alpha=0.9, edgecolor="black")
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.2,
                    f"{mean:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold"
                )

    ax.axhline(80.0, color="#E53E3E", linestyle="--", linewidth=2.0, label="Aspirational Target (80%)")
    ax.axhline(25.0, color="#A0AEC0", linestyle=":", linewidth=1.5, label="Chance Level (25%)")

    ax.set_ylabel("Subject-Level Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Experiment D2: Class-Aware Pseudo-Label Adaptation vs Baselines\n15-Fold Outer LOSO on SEED-IV (1,080 Trials)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["All 62 Channels", "Stable 32 Channels\n(Primary)", "Predictive 32 Channels", "Predictive 12 Channels"], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 90)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    chart_path = FIGURES_DIR / "experiment_d2_adaptation_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved D2 adaptation comparison figure to: {chart_path}")

    # 2. Confusion Matrices (stab_32)
    stab_preds = pred_df[pred_df["condition"] == "stab_32"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    class_labels = ["Neutral", "Sad", "Fear", "Happy"]

    for idx, (m_id, m_label, _) in enumerate(methods):
        sub_p = stab_preds[stab_preds["method"] == m_id]
        cm = confusion_matrix(sub_p["true_label"], sub_p["predicted_label"])
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100.0

        ax = axes[idx]
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=80)
        ax.set_title(f"{m_label}\n(Acc: {accuracy_score(sub_p['true_label'], sub_p['predicted_label'])*100:.1f}%)", fontsize=11, fontweight="bold")

        for r in range(4):
            for c in range(4):
                val = cm_norm[r, c]
                cnt = cm[r, c]
                txt_color = "white" if val > 45 else "black"
                ax.text(c, r, f"{val:.1f}%\n({cnt})", ha="center", va="center", color=txt_color, fontsize=9)

        ax.set_xticks(range(4))
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(class_labels, fontsize=9)
        ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("True Class", fontsize=10, fontweight="bold")

    fig.suptitle("Experiment D2 Confusion Matrices on stab_32 (270 Trials per Class)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    cm_path = FIGURES_DIR / "experiment_d2_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved D2 confusion matrix figure to: {cm_path}")


def main() -> None:
    print("=" * 80)
    print("EXPERIMENT D2: CLASS-AWARE UNSUPERVISED DOMAIN ADAPTATION (CB-PLR)")
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

    # 2. Run Experiment D2 evaluation
    t0 = time.time()
    pred_df, inner_audit_df = run_experiment_d2_evaluation(
        X_trial=X_trial,
        y_trial=y_trial,
        trial_meta=trial_meta,
        channel_to_indices=channel_to_indices,
        selections_df=selections_df,
        conditions=["stab_32", "all_62", "pred_32", "pred_12"],
        reg=1.0,
        seed=42,
    )

    pred_path = PREDICTIONS_DIR / "experiment_d2_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved Experiment D2 predictions ({len(pred_df)} rows) to: {pred_path}")

    inner_audit_path = TABLES_DIR / "experiment_d2_inner_pseudo_audit.csv"
    inner_audit_df.to_csv(inner_audit_path, index=False)
    print(f"Saved inner validation pseudo-label audit to: {inner_audit_path}")

    # 3. Compute metrics & comparisons
    print("\nComputing comprehensive summary, subject, class, and comparison metrics...")
    summary_df, subject_df, class_df, comparison_df = compute_experiment_d2_metrics(pred_df)

    sum_path = TABLES_DIR / "experiment_d2_summary.csv"
    sub_path = TABLES_DIR / "experiment_d2_subject_metrics.csv"
    cls_path = TABLES_DIR / "experiment_d2_class_metrics.csv"
    comp_path = TABLES_DIR / "experiment_d2_vs_baselines_comparison.csv"

    summary_df.to_csv(sum_path, index=False)
    subject_df.to_csv(sub_path, index=False)
    class_df.to_csv(cls_path, index=False)
    comparison_df.to_csv(comp_path, index=False)

    print(f"Saved summary metrics to: {sum_path}")
    print(f"Saved subject metrics to: {sub_path}")
    print(f"Saved class metrics to: {cls_path}")
    print(f"Saved comparison metrics to: {comp_path}")

    print("\nSummary Results:")
    print(summary_df[["condition", "method", "pooled_accuracy", "subject_mean_accuracy", "subject_std_accuracy", "macro_f1"]].to_string(index=False))

    print("\nComparison Results (Paired Differences vs Strict LOSO Baseline and D1 CORAL):")
    print(comparison_df[["condition", "reference_baseline", "evaluated_mean_acc", "reference_mean_acc", "mean_delta_acc_pp", "t_pval", "wilcoxon_pval", "n_pos_subjects"]].to_string(index=False))

    # 4. Generate Figures
    print("\nGenerating publication figures...")
    plot_experiment_d2_figures(summary_df, pred_df)

    total_time = time.time() - t0
    print(f"\nExperiment D2 Pipeline successfully completed in {total_time:.1f}s ({total_time/60:.2f} min).")


if __name__ == "__main__":
    main()
