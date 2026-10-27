"""
src/evaluate_experiment_g.py

Experiment G: Pushing Window-Level CORAL Accuracy Toward 65-70%
While Preserving Selective Prediction.

Strict 15-Fold Outer Leave-One-Subject-Out (LOSO) Cross-Validation on SEED-IV
Evaluating all 37,575 4-Second EEG Windows on stab_32.

Configurations Evaluated:
1. Reference Model: stab_32, LinearSVC (C=0.001), CORAL (T->S, lambda=1.0) [Experiment C Benchmark]
2. Experiment G1: CORAL with Non-Linear RBF-SVM (Nystroem Kernel Approximation, C=0.001, gamma=0.01)
3. Experiment G2: Target-Similarity-Weighted Source Training (LinearSVC C=0.001, Distribution Distance Weights)
4. Experiment G3: Conservative Class-Balanced Pseudo-Label Refinement (LinearSVC C=0.001, Top-Quotas per Class)
5. Experiment G4: Literature-Inspired Conditional Subdomain Adaptation (Soft Class-Centroid Residual Alignment)
6. Experiment G5: Ultra-Wide Margin Adaptive Refinement (Inner-Tuned C in [0.0003, 0.0005, 0.001] + Conditional Alignment)

Evaluation Modes:
A. Operational Selective Prediction: Thresholds frozen from inner validation percentiles.
B. Diagnostic Exact Coverage: Evaluated at exact coverage levels 20%, 25%, 30%, 50%, 70%, 80%, 90%, 100%.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_seed_iv_dataset
from src.selective_prediction import (
    compute_max_probability_uncertainty,
    _trapezoid_integration,
)
from src.evaluate_window_coral_optimisation import (
    compute_coral_projection,
    compute_cov_matrices,
    FastTemperatureScaler,
)

OUTPUTS_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

EXACT_COVERAGE_LEVELS = [0.20, 0.25, 0.30, 0.50, 0.70, 0.80, 0.90, 1.00]
NOMINAL_COVERAGE_LEVELS = [0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 1.00]


def run_inner_experiment_g_tuning(
    Xs_s: np.ndarray,
    ys: np.ndarray,
    source_subjects: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run 3-fold subject-grouped validation on the 14 training subjects to:
    1. Select best C for G5 from [0.0003, 0.0005, 0.001].
    2. Select best shrinkage eta for G4 from [0.10, 0.25].
    3. Generate out-of-fold validation scores for temperature scaling and threshold selection.
    
    Zero outer test subject data is ever seen.
    """
    gkf = GroupKFold(n_splits=3)
    c_candidates = [0.0003, 0.0005, 0.001]
    eta_candidates = [0.10, 0.25]

    splits_data = []
    for inner_tr_idx, inner_val_idx in gkf.split(Xs_s, ys, groups=source_subjects):
        in_Xs = Xs_s[inner_tr_idx]
        in_ys = ys[inner_tr_idx]
        in_Xv = Xs_s[inner_val_idx]
        in_yv = ys[inner_val_idx]

        scaler = StandardScaler()
        in_Xs_s = scaler.fit_transform(in_Xs)
        in_Xv_s = scaler.transform(in_Xv)

        clfs = {}
        for c_val in c_candidates:
            clf = LinearSVC(dual=False, C=c_val, random_state=seed, max_iter=1500, tol=1e-2)
            clf.fit(in_Xs_s, in_ys)
            clfs[c_val] = clf

        d = in_Xs_s.shape[1]
        mu_s = in_Xs_s.mean(axis=0)
        mu_v = in_Xv_s.mean(axis=0)
        cov_s = np.cov(in_Xs_s - mu_s, rowvar=False) + 1.0 * np.eye(d)
        cov_v = np.cov(in_Xv_s - mu_v, rowvar=False) + 1.0 * np.eye(d)
        _, A_ts = compute_coral_projection(cov_s, cov_v)
        in_Xv_coral = (in_Xv_s - mu_v) @ A_ts + mu_s

        splits_data.append({
            "in_ys": in_ys,
            "in_yv": in_yv,
            "in_Xs_s": in_Xs_s,
            "in_Xv_coral": in_Xv_coral,
            "clfs": clfs,
            "val_idx": inner_val_idx,
            "d": d,
        })

    # Evaluate C candidates
    c_scores = {}
    for c_val in c_candidates:
        accs = []
        for sp in splits_data:
            preds = sp["clfs"][c_val].predict(sp["in_Xv_coral"])
            accs.append(accuracy_score(sp["in_yv"], preds))
        c_scores[c_val] = float(np.mean(accs))
    best_c = max(c_scores.keys(), key=lambda c: c_scores[c])

    # Evaluate conditional eta on inner splits
    eta_scores = {}
    for eta in eta_candidates:
        accs = []
        for sp in splits_data:
            # Source centroids
            s_cents = np.zeros((4, sp["d"]))
            for k in range(4):
                s_cents[k] = sp["in_Xs_s"][sp["in_ys"] == k].mean(axis=0)

            # Target probabilities from base clf
            probs_v = FastTemperatureScaler().fit(
                sp["clfs"][0.001].decision_function(sp["in_Xs_s"][:3000]), sp["in_ys"][:3000]
            ).predict_proba(sp["clfs"][0.001].decision_function(sp["in_Xv_coral"]))

            t_cents = np.zeros((4, sp["d"]))
            for k in range(4):
                wk = probs_v[:, k:k+1]
                t_cents[k] = np.sum(sp["in_Xv_coral"] * wk, axis=0) / (np.sum(wk) + 1e-6)

            delta = np.zeros_like(sp["in_Xv_coral"])
            for k in range(4):
                delta += probs_v[:, k:k+1] * (s_cents[k] - t_cents[k])

            in_Xv_cond = sp["in_Xv_coral"] + eta * delta
            preds = sp["clfs"][0.001].predict(in_Xv_cond)
            accs.append(accuracy_score(sp["in_yv"], preds))
        eta_scores[eta] = float(np.mean(accs))
    best_eta = max(eta_scores.keys(), key=lambda e: eta_scores[e])

    # Out-of-fold calibration for reference and winner
    z_oof_ref = np.zeros((len(Xs_s), 4), dtype=np.float64)
    for sp in splits_data:
        z_oof_ref[sp["val_idx"]] = sp["clfs"][0.001].decision_function(sp["in_Xv_coral"])

    scaler_temp = FastTemperatureScaler().fit(z_oof_ref, ys)
    probs_oof = scaler_temp.predict_proba(z_oof_ref)
    u_max_val = compute_max_probability_uncertainty(probs_oof)

    thresholds = {}
    for cov in NOMINAL_COVERAGE_LEVELS:
        if cov == 1.0:
            thresholds[cov] = 1.0
        else:
            thresholds[cov] = float(np.percentile(u_max_val, cov * 100.0))

    return {
        "best_c": best_c,
        "best_eta": best_eta,
        "scaler_temp": scaler_temp,
        "thresholds": thresholds,
        "c_scores": c_scores,
        "eta_scores": eta_scores,
    }


def evaluate_experiment_g_pipeline(
    X_stab: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute strict 15-fold outer LOSO evaluation for Experiment G candidates.
    """
    subjects = metadata["subject_id"].to_numpy()
    unique_subs = sorted(np.unique(subjects))
    n_windows = len(X_stab)
    assert n_windows == 37575, f"Expected 37,575 windows, got {n_windows}"

    configs = [
        {"id": "ref_exp_c", "name": "Reference Model (stab_32, C=0.001, lam=1.0)"},
        {"id": "exp_g1_rbf_svm", "name": "Exp G1: Non-Linear RBF-SVM (Nystroem RBF on CORAL)"},
        {"id": "exp_g2_sim_weighted", "name": "Exp G2: Target-Similarity-Weighted Source Training"},
        {"id": "exp_g3_pseudo_refined", "name": "Exp G3: Conservative Class-Balanced Pseudo-Refinement"},
        {"id": "exp_g4_conditional_coral", "name": "Exp G4: Subdomain Conditional Residual Alignment"},
        {"id": "exp_g5_adaptive_wide", "name": "Exp G5: Adaptive Ultra-Wide Margin + Conditional CORAL"},
    ]

    all_predictions = []
    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT G: 15-FOLD OUTER LOSO OPTIMISATION ({len(configs)} CANDIDATES)")
    print(f"{'='*80}\n", flush=True)

    t_global = time.time()

    for fold_idx, test_sub in enumerate(unique_subs, 1):
        t_fold = time.time()
        print(f"--- Outer Fold {fold_idx}/15: Subject {test_sub} ---", flush=True)

        train_mask = (subjects != test_sub)
        test_mask = (subjects == test_sub)

        Xs = X_stab[train_mask]
        ys = y[train_mask]
        source_subs = subjects[train_mask]

        Xt = X_stab[test_mask]
        yt = y[test_mask]
        test_indices = np.where(test_mask)[0]
        n_test = len(Xt)

        # 1. Standardize using source data only
        scaler = StandardScaler()
        Xs_s = scaler.fit_transform(Xs)
        Xt_s = scaler.transform(Xt)

        # 2. Inner Validation on 14 training subjects
        inner_tuning = run_inner_experiment_g_tuning(Xs_s, ys, source_subs, seed=seed)
        best_c = inner_tuning["best_c"]
        best_eta = inner_tuning["best_eta"]
        scaler_temp = inner_tuning["scaler_temp"]
        thresholds = inner_tuning["thresholds"]

        # 3. Target-to-Source CORAL Alignment (T -> S, lambda = 1.0)
        Cs, Ct, mu_s, mu_t = compute_cov_matrices(Xs_s, Xt_s, method="tikhonov", reg=1.0)
        _, A_ts = compute_coral_projection(Cs, Ct)
        Xt_coral = (Xt_s - mu_t) @ A_ts + mu_s

        # ---------------------------------------------------------------------
        # Fit Candidates
        # ---------------------------------------------------------------------
        # Candidate 1: Reference Model (LinearSVC C=0.001)
        clf_ref = LinearSVC(dual=False, C=0.001, random_state=seed, max_iter=1500, tol=1e-2)
        clf_ref.fit(Xs_s, ys)

        # Candidate 2: Exp G1 (Nystroem RBF Approximation + LinearSVC)
        nystroem = Nystroem(kernel='rbf', gamma=0.01, n_components=300, random_state=seed)
        Xs_rbf = nystroem.fit_transform(Xs_s)
        Xt_rbf = nystroem.transform(Xt_coral)
        clf_g1 = LinearSVC(dual=False, C=0.001, random_state=seed, max_iter=1500, tol=1e-2)
        clf_g1.fit(Xs_rbf, ys)

        # Candidate 3: Exp G2 (Target-Similarity-Weighted Source Training)
        sub_dists = {}
        for s_id in np.unique(source_subs):
            s_mask = (source_subs == s_id)
            mu_s_i = Xs_s[s_mask].mean(axis=0)
            cov_s_i = np.cov(Xs_s[s_mask] - mu_s_i, rowvar=False) + 1.0 * np.eye(Xs_s.shape[1])
            d_i = np.linalg.norm(mu_s_i - mu_t) + 0.05 * np.linalg.norm(cov_s_i - Ct, 'fro')
            sub_dists[s_id] = d_i

        d_vals = np.array([sub_dists[s] for s in np.unique(source_subs)])
        weights_sub = np.exp(-(d_vals - np.mean(d_vals)) / (np.std(d_vals) + 1e-5))
        weights_sub = np.clip(weights_sub / weights_sub.sum(), 0.03, 0.20)
        weights_sub = weights_sub / weights_sub.sum()
        sub_weight_map = dict(zip(np.unique(source_subs), weights_sub))
        sample_weights = np.array([sub_weight_map[s] for s in source_subs])

        clf_g2 = LinearSVC(dual=False, C=0.001, random_state=seed, max_iter=1500, tol=1e-2)
        clf_g2.fit(Xs_s, ys, sample_weight=sample_weights)

        # Candidate 4: Exp G3 (Conservative Class-Balanced Pseudo-Label Refinement)
        z_t_ref = clf_ref.decision_function(Xt_coral)
        probs_t_ref = scaler_temp.predict_proba(z_t_ref)
        max_p_ref = np.max(probs_t_ref, axis=1)
        pseudo_labels_ref = np.argmax(probs_t_ref, axis=1)

        pseudo_idx = []
        for k in range(4):
            k_indices = np.where(pseudo_labels_ref == k)[0]
            if len(k_indices) > 0:
                top_k = k_indices[np.argsort(-max_p_ref[k_indices])[:40]]  # Top 40 per class
                pseudo_idx.extend(top_k)
        pseudo_idx = np.array(pseudo_idx)

        X_aug = np.vstack([Xs_s, Xt_coral[pseudo_idx]])
        y_aug = np.concatenate([ys, pseudo_labels_ref[pseudo_idx]])
        w_aug = np.concatenate([np.ones(len(Xs_s)), np.ones(len(pseudo_idx)) * 0.5])

        clf_g3 = LinearSVC(dual=False, C=0.001, random_state=seed, max_iter=1500, tol=1e-2)
        clf_g3.fit(X_aug, y_aug, sample_weight=w_aug)

        # Candidate 5: Exp G4 (Subdomain Conditional Residual Alignment)
        source_centroids = np.zeros((4, Xs_s.shape[1]))
        for k in range(4):
            source_centroids[k] = Xs_s[ys == k].mean(axis=0)

        target_centroids = np.zeros((4, Xt_coral.shape[1]))
        for k in range(4):
            weights_k = probs_t_ref[:, k:k+1]
            sum_w = np.sum(weights_k) + 1e-6
            target_centroids[k] = np.sum(Xt_coral * weights_k, axis=0) / sum_w

        delta_shift_g4 = np.zeros_like(Xt_coral)
        for k in range(4):
            delta_shift_g4 += probs_t_ref[:, k:k+1] * (source_centroids[k] - target_centroids[k])

        Xt_cond_g4 = Xt_coral + best_eta * delta_shift_g4

        # Candidate 6: Exp G5 (Adaptive Ultra-Wide Margin + Conditional Alignment)
        clf_g5 = LinearSVC(dual=False, C=best_c, random_state=seed, max_iter=1500, tol=1e-2)
        clf_g5.fit(Xs_s, ys)

        # ---------------------------------------------------------------------
        # Generate Predictions & Uncertainties for All Candidates
        # ---------------------------------------------------------------------
        candidate_evals = [
            ("ref_exp_c", clf_ref, Xt_coral),
            ("exp_g1_rbf_svm", clf_g1, Xt_rbf),
            ("exp_g2_sim_weighted", clf_g2, Xt_coral),
            ("exp_g3_pseudo_refined", clf_g3, Xt_coral),
            ("exp_g4_conditional_coral", clf_ref, Xt_cond_g4),
            ("exp_g5_adaptive_wide", clf_g5, Xt_cond_g4),
        ]

        for cid, model, eval_features in candidate_evals:
            z_eval = model.decision_function(eval_features)
            probs_eval = scaler_temp.predict_proba(z_eval)
            preds_eval = np.argmax(probs_eval, axis=1)
            u_max_eval = compute_max_probability_uncertainty(probs_eval)
            is_error_eval = (preds_eval != yt).astype(int)

            for i in range(n_test):
                rec = {
                    "config_id": cid,
                    "subject_id": test_sub,
                    "window_index": test_indices[i],
                    "true_label": int(yt[i]),
                    "predicted_label": int(preds_eval[i]),
                    "is_error": int(is_error_eval[i]),
                    "confidence": float(np.max(probs_eval[i])),
                    "u_max": float(u_max_eval[i]),
                    "prob_neutral": float(probs_eval[i, 0]),
                    "prob_sad": float(probs_eval[i, 1]),
                    "prob_fear": float(probs_eval[i, 2]),
                    "prob_happy": float(probs_eval[i, 3]),
                }
                for cov in NOMINAL_COVERAGE_LEVELS:
                    rec[f"op_reliable_cov_{int(cov*100)}"] = 1 if u_max_eval[i] <= thresholds[cov] else 0

                all_predictions.append(rec)

        print(f"  Fold {fold_idx} completed in {time.time()-t_fold:.1f}s (Inner: Best C={best_c}, Best Eta={best_eta})", flush=True)

    pred_df = pd.DataFrame(all_predictions)
    print(f"\nAll 15 folds completed. Total records: {len(pred_df):,} in {time.time()-t_global:.1f}s.", flush=True)

    # -------------------------------------------------------------------------
    # Aggregate Metrics across Evaluated Configurations
    # -------------------------------------------------------------------------
    summary_records = []
    unreliability_records = []
    operational_records = []
    exact_coverage_records = []
    per_subject_records = []

    for cfg in configs:
        cid = cfg["id"]
        grp = pred_df[pred_df["config_id"] == cid].copy()
        n_samples = len(grp)
        y_true = grp["true_label"].to_numpy()
        y_pred = grp["predicted_label"].to_numpy()
        is_error = grp["is_error"].to_numpy()
        u_max = grp["u_max"].to_numpy()
        probs = grp[["prob_neutral", "prob_sad", "prob_fear", "prob_happy"]].to_numpy()

        # Classification Metrics
        pooled_acc = float(accuracy_score(y_true, y_pred)) * 100.0
        bal_acc = float(balanced_accuracy_score(y_true, y_pred)) * 100.0
        macro_f1 = float(f1_score(y_true, y_pred, average="macro")) * 100.0

        sub_accs = {}
        for sub_id, sg in grp.groupby("subject_id"):
            s_acc = float(accuracy_score(sg["true_label"], sg["predicted_label"])) * 100.0
            sub_accs[sub_id] = s_acc
            per_subject_records.append({
                "config_id": cid,
                "config_name": cfg["name"],
                "subject_id": sub_id,
                "accuracy": s_acc,
            })

        sub_acc_vals = list(sub_accs.values())
        mean_sub_acc = float(np.mean(sub_acc_vals))
        std_sub_acc = float(np.std(sub_acc_vals))
        min_sub_acc = float(np.min(sub_acc_vals))
        max_sub_acc = float(np.max(sub_acc_vals))

        # Uncertainty Metrics
        err_auroc = float(roc_auc_score(is_error, u_max))
        err_auprc = float(average_precision_score(is_error, u_max))
        brier = float(brier_score_loss(is_error, u_max))
        try:
            nll = float(log_loss(y_true, probs, labels=[0, 1, 2, 3]))
        except Exception:
            nll = np.nan

        # Unreliability detailed stats
        err_binary_pred = (u_max >= 0.5).astype(int)
        err_prec, err_rec, err_f1, _ = precision_recall_fscore_support(
            is_error, err_binary_pred, average="binary", zero_division=0
        )
        err_bal_acc = float(balanced_accuracy_score(is_error, err_binary_pred)) * 100.0

        unreliability_records.append({
            "config_id": cid,
            "config_name": cfg["name"],
            "auroc": err_auroc,
            "auprc": err_auprc,
            "brier_score": brier,
            "log_loss": nll,
            "error_precision": float(err_prec) * 100.0,
            "error_recall": float(err_rec) * 100.0,
            "error_f1": float(err_f1) * 100.0,
            "error_balanced_acc": err_bal_acc,
        })

        # Operational Selective Prediction Metrics (Mode A)
        for cov in NOMINAL_COVERAGE_LEVELS:
            flag_col = f"op_reliable_cov_{int(cov*100)}"
            accepted = grp[grp[flag_col] == 1]
            abstained = grp[grp[flag_col] == 0]
            n_acc = len(accepted)
            act_cov = (n_acc / n_samples) * 100.0
            sel_acc = float(accuracy_score(accepted["true_label"], accepted["predicted_label"])) * 100.0 if n_acc > 0 else np.nan
            sel_risk = 100.0 - sel_acc if not np.isnan(sel_acc) else np.nan

            operational_records.append({
                "config_id": cid,
                "config_name": cfg["name"],
                "nominal_coverage_pct": int(cov * 100),
                "actual_coverage_pct": act_cov,
                "selective_accuracy": sel_acc,
                "selective_risk": sel_risk,
                "accepted_windows": n_acc,
                "abstained_windows": len(abstained),
                "total_windows": n_samples,
            })

        # Diagnostic Exact-Coverage Curve (Mode B)
        sort_order = np.lexsort((np.arange(len(grp)), u_max))
        exact_accs = {}
        achieved_c_list = []
        achieved_risk_list = []

        for cov in EXACT_COVERAGE_LEVELS:
            k = int(np.round(cov * n_samples))
            k = max(k, 1)
            accepted_idx = sort_order[:k]
            acc_exact = float(accuracy_score(y_true[accepted_idx], y_pred[accepted_idx])) * 100.0
            risk_exact = 100.0 - acc_exact
            exact_accs[int(cov * 100)] = acc_exact
            achieved_c_list.append(cov)
            achieved_risk_list.append(risk_exact / 100.0)

            exact_coverage_records.append({
                "config_id": cid,
                "config_name": cfg["name"],
                "exact_coverage_pct": int(cov * 100),
                "retained_windows": k,
                "selective_accuracy": acc_exact,
                "selective_risk": risk_exact,
            })

        aurc = _trapezoid_integration(np.array(achieved_risk_list), np.array(achieved_c_list))

        summary_records.append({
            "config_id": cid,
            "config_name": cfg["name"],
            "pooled_accuracy": pooled_acc,
            "subject_mean_accuracy": mean_sub_acc,
            "subject_std_accuracy": std_sub_acc,
            "subject_min_accuracy": min_sub_acc,
            "subject_max_accuracy": max_sub_acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "auroc_error": err_auroc,
            "auprc_error": err_auprc,
            "aurc": aurc,
            "brier_score": brier,
            "log_loss": nll,
            "exact_acc_20": exact_accs[20],
            "exact_acc_25": exact_accs[25],
            "exact_acc_30": exact_accs[30],
            "exact_acc_50": exact_accs[50],
            "exact_acc_70": exact_accs[70],
            "exact_acc_80": exact_accs[80],
            "exact_acc_90": exact_accs[90],
            "exact_acc_100": exact_accs[100],
        })

    summary_df = pd.DataFrame(summary_records)
    unreliability_df = pd.DataFrame(unreliability_records)
    op_df = pd.DataFrame(operational_records)
    exact_df = pd.DataFrame(exact_coverage_records)
    sub_df = pd.DataFrame(per_subject_records)

    return pred_df, summary_df, unreliability_df, op_df, exact_df, sub_df


def plot_experiment_g_risk_coverage(exact_df: pd.DataFrame, output_path: Path):
    """Plot risk-coverage curves for Experiment G models against targets."""
    plt.figure(figsize=(10, 6), dpi=300)
    palette = {
        "ref_exp_c": ("#2ca02c", "D", "-", "Reference Model (LinearSVC C=0.001)"),
        "exp_g1_rbf_svm": ("#d62728", "s", "--", "Exp G1: Non-Linear RBF-SVM"),
        "exp_g2_sim_weighted": ("#ff7f0e", "v", "-.", "Exp G2: Similarity-Weighted"),
        "exp_g3_pseudo_refined": ("#9467bd", "P", "-", "Exp G3: Pseudo-Refined"),
        "exp_g4_conditional_coral": ("#1f77b4", "^", "-", "Exp G4: Conditional Subdomain CORAL"),
        "exp_g5_adaptive_wide": ("#17becf", "*", "-", "Exp G5: Adaptive Ultra-Wide Margin"),
    }

    for cid, (color, marker, ls, label) in palette.items():
        subset = exact_df[exact_df["config_id"] == cid].sort_values("exact_coverage_pct")
        if len(subset) > 0:
            plt.plot(
                subset["exact_coverage_pct"],
                subset["selective_accuracy"],
                label=label,
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=2.0,
                markersize=8,
            )

    plt.axhline(65.0, color="#d62728", linestyle=":", linewidth=1.8, label="Primary Target (65%)")
    plt.axhline(70.0, color="#2ca02c", linestyle="--", linewidth=1.8, label="Stretch Target (70%)")
    plt.axhline(80.0, color="#9467bd", linestyle="-.", linewidth=1.8, label="Aspirational (80%)")

    plt.title("Diagnostic Risk–Coverage Curves: Experiment G Candidates", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Exact Coverage (%) [Retained Windows]", fontsize=12, fontweight="bold")
    plt.ylabel("Selective Accuracy (%)", fontsize=12, fontweight="bold")
    plt.xlim(15, 105)
    plt.ylim(30, 95)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=9, framealpha=0.95)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Experiment G risk-coverage plot to: {output_path}")


def main():
    t_start = time.time()
    print("=" * 80)
    print("EXPERIMENT G: PUSH WINDOW-LEVEL CORAL ACCURACY TOWARD 65-70%")
    print("=" * 80)

    # 1. Load Dataset
    print("\nLoading SEED-IV dataset...")
    dataset = load_seed_iv_dataset()
    X = dataset.X
    y = dataset.y
    metadata = dataset.metadata.copy()

    selections_df = pd.read_csv(TABLES_DIR / "corrected_channel_selections.csv")
    stab_32_row = selections_df[selections_df["budget"] == 32].iloc[0]
    stab_32_channels = [c.strip() for c in stab_32_row["corrected_stability_channels"].split(",")]

    ch_names = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
                "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
                "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7",
                "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
                "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
    ch_to_idx = {ch: list(range(i * 5, (i + 1) * 5)) for i, ch in enumerate(ch_names)}
    stab_32_indices = sorted([idx for ch in stab_32_channels for idx in ch_to_idx[ch]])

    X_stab = X[:, stab_32_indices]

    # 2. Run Pipeline
    pred_df, summary_df, unreliability_df, op_df, exact_df, sub_df = evaluate_experiment_g_pipeline(
        X_stab=X_stab,
        y=y,
        metadata=metadata,
        seed=42,
    )

    # 3. Save Deliverables
    print("\nSaving deliverables...")

    # Deliverable 1: Predictions CSV
    pred_path = PREDICTIONS_DIR / "experiment_g_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"1. Saved predictions to: {pred_path} ({len(pred_df):,} rows)")

    # Deliverable 2: Comparison CSV
    comp_path = TABLES_DIR / "experiment_g_comparison.csv"
    summary_df.to_csv(comp_path, index=False)
    print(f"2. Saved summary comparison to: {comp_path}")

    # Deliverable 3: Unreliability Metrics CSV
    unrel_path = TABLES_DIR / "experiment_g_unreliability_metrics.csv"
    unreliability_df.to_csv(unrel_path, index=False)
    print(f"3. Saved unreliability metrics to: {unrel_path}")

    # Deliverable 4: Selective Comparison CSV
    op_exact_df = pd.merge(
        op_df,
        exact_df.rename(columns={"exact_coverage_pct": "nominal_coverage_pct", "selective_accuracy": "exact_selective_accuracy", "selective_risk": "exact_selective_risk"}),
        on=["config_id", "config_name", "nominal_coverage_pct"],
        how="outer"
    )
    sel_path = TABLES_DIR / "experiment_g_selective_comparison.csv"
    op_exact_df.to_csv(sel_path, index=False)
    print(f"4. Saved selective comparison to: {sel_path}")

    # Deliverable 5: Per-Subject CSV
    sub_pivot = sub_df.pivot(index="subject_id", columns="config_name", values="accuracy")
    sub_path = TABLES_DIR / "experiment_g_per_subject.csv"
    sub_pivot.to_csv(sub_path)
    print(f"5. Saved per-subject comparison to: {sub_path}")

    # Deliverable 6: Updated Risk-Coverage Plot
    fig_path = FIGURES_DIR / "experiment_g_risk_coverage.png"
    plot_experiment_g_risk_coverage(exact_df, fig_path)

    print(f"\nExperiment G execution completed successfully in {time.time()-t_start:.1f}s.")
    return summary_df


if __name__ == "__main__":
    main()
