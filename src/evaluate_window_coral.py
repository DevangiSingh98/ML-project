"""
src/evaluate_window_coral.py

Window-Level CORAL Domain Adaptation & Uncertainty-Aware Unreliability Prediction
for SEED-IV EEG Emotion Recognition.

Protocol:
1. Feature Representation: SEED-IV 4-second EEG windows (37,575 windows total).
   - 15 subjects, 3 sessions, 72 movie trials (2,505 windows per subject).
   - all_62: 62 channels x 5 frequency bands = 310 features.
   - stab_32: 32 stability-selected channels x 5 bands = 160 features.
2. Outer Evaluation: Strict 15-fold Leave-One-Subject-Out (LOSO) cross-validation.
   - 14 source subjects per outer fold (35,070 training windows).
   - 1 held-out target subject (2,505 test windows).
   - Every window evaluated exactly once.
3. Methods Compared:
   - Method A: LinearSVC Baseline (strict LOSO, no adaptation).
   - Method B: CORAL with fixed Tikhonov ridge (lambda = 1.0).
   - Method C: CORAL with Ledoit-Wolf analytical covariance shrinkage.
   - Method D: CORAL with Oracle Approximating Shrinkage (OAS).
   Reference baseline: Phase B RBF-SVM (LibSVM voting: 39.19%, margin argmax: 39.53%).
4. Domain Adaptation Procedure:
   - Feature scaling (StandardScaler) fitted strictly on source-subject training windows.
   - Unsupervised target adaptation: held-out subject's UNLABELED windows used to estimate
     target mean and covariance matrix.
   - Zero target emotion labels are used for covariance estimation, model fitting,
     calibration, or threshold selection.
5. Uncertainty Estimation & Unreliability Prediction:
   - Inner 3-fold GroupKFold across the 14 training subjects simulates the adaptation pipeline.
   - Temperature scaling calibrator fitted strictly on out-of-fold validation decision scores.
   - Unreliability metrics:
     * u_max = 1.0 - max_k p_k
     * Shannon entropy H(p) = -sum p_k ln(p_k)
     * Decision margin = z_(1) - z_(2)
   - Coverage thresholds tau_c for c in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5] computed strictly
     from inner out-of-fold validation uncertainty percentiles.
   - Frozen decision rule: outer predictions and unreliability flags are frozen BEFORE
     evaluating accuracy or error-detection metrics against true labels.
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
from scipy import linalg, stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, OAS
from sklearn.exceptions import ConvergenceWarning
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

from src.data_loader import (
    DEFAULT_FREQUENCY_BANDS,
    LABEL_NAMES,
    DatasetContainer,
    load_seed_iv_dataset,
)
from src.selective_prediction import (
    compute_predictive_entropy,
    compute_max_probability_uncertainty,
    _trapezoid_integration,
)

OUTPUTS_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

COVERAGE_LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def compute_coral_projection(
    Cs: np.ndarray,
    Ct: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CORAL transformation matrices from symmetric positive definite covariances.
    
    A_st = Cs^(-1/2) @ Ct^(1/2) (Source -> Target)
    A_ts = Ct^(-1/2) @ Cs^(1/2) (Target -> Source)
    """
    # Enforce exact symmetry
    Cs = 0.5 * (Cs + Cs.T)
    Ct = 0.5 * (Ct + Ct.T)

    vals_s, vecs_s = np.linalg.eigh(Cs)
    vals_t, vecs_t = np.linalg.eigh(Ct)

    vals_s = np.clip(vals_s, a_min=1e-8, a_max=None)
    vals_t = np.clip(vals_t, a_min=1e-8, a_max=None)

    sqrt_Cs = vecs_s @ np.diag(np.sqrt(vals_s)) @ vecs_s.T
    inv_sqrt_Cs = vecs_s @ np.diag(1.0 / np.sqrt(vals_s)) @ vecs_s.T

    sqrt_Ct = vecs_t @ np.diag(np.sqrt(vals_t)) @ vecs_t.T
    inv_sqrt_Ct = vecs_t @ np.diag(1.0 / np.sqrt(vals_t)) @ vecs_t.T

    A_st = (inv_sqrt_Cs @ sqrt_Ct).real
    A_ts = (inv_sqrt_Ct @ sqrt_Cs).real
    return A_st, A_ts


def compute_covariance_estimates(
    Xs: np.ndarray,
    Xt: np.ndarray,
    method: str = "oas",
    reg: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Estimate source and target mean and covariance matrices under specified method.
    
    Methods:
    - 'tikhonov': Sample covariance + reg * Identity
    - 'ledoit_wolf': Analytical Ledoit-Wolf optimal shrinkage
    - 'oas': Oracle Approximating Shrinkage
    
    Returns:
        (Cs, Ct, mu_s, mu_t, rho_s, rho_t)
    """
    d = Xs.shape[1]

    if method == "tikhonov":
        mu_s = Xs.mean(axis=0)
        mu_t = Xt.mean(axis=0)
        Cs = np.cov(Xs - mu_s, rowvar=False) + reg * np.eye(d)
        Ct = np.cov(Xt - mu_t, rowvar=False) + reg * np.eye(d)
        rho_s, rho_t = 0.0, 0.0
    elif method == "ledoit_wolf":
        lw_s = LedoitWolf().fit(Xs)
        lw_t = LedoitWolf().fit(Xt)
        Cs = lw_s.covariance_
        Ct = lw_t.covariance_
        mu_s = lw_s.location_
        mu_t = lw_t.location_
        rho_s = float(lw_s.shrinkage_)
        rho_t = float(lw_t.shrinkage_)
    elif method == "oas":
        oas_s = OAS().fit(Xs)
        oas_t = OAS().fit(Xt)
        Cs = oas_s.covariance_
        Ct = oas_t.covariance_
        mu_s = oas_s.location_
        mu_t = oas_t.location_
        rho_s = float(oas_s.shrinkage_)
        rho_t = float(oas_t.shrinkage_)
    else:
        raise ValueError(f"Unknown covariance method: {method}")

    return Cs, Ct, mu_s, mu_t, rho_s, rho_t


class TemperatureScaler:
    """
    Fit temperature parameter T > 0 strictly on out-of-fold validation decision scores.
    Preserves decision margin ranking and argmax while calibrating probabilities.
    """
    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, z_val: np.ndarray, y_val: np.ndarray) -> "TemperatureScaler":
        n_classes = z_val.shape[1]
        y_val = np.asarray(y_val, dtype=int)

        def nll_loss(t_param: np.ndarray) -> float:
            t = max(t_param[0], 1e-3)
            scaled_z = z_val / t
            max_z = np.max(scaled_z, axis=1, keepdims=True)
            exp_z = np.exp(scaled_z - max_z)
            probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            probs = np.clip(probs, 1e-15, 1.0)
            log_p = np.log(probs)
            return float(-np.mean(log_p[np.arange(len(y_val)), y_val]))

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.05, 20.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0]) if res.success else 1.0
        return self

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        t = max(self.temperature, 1e-3)
        scaled_z = z / t
        max_z = np.max(scaled_z, axis=1, keepdims=True)
        exp_z = np.exp(scaled_z - max_z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return probs


def compute_uncertainty_metrics_dict(probs: np.ndarray, z: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute multiple complementary uncertainty metrics:
    - u_max: 1.0 - max_k p_k
    - entropy: Shannon predictive entropy
    - margin: difference between top-1 and top-2 decision scores
    - prob_margin: difference between top-1 and top-2 calibrated probabilities
    """
    u_max = compute_max_probability_uncertainty(probs)
    entropy = compute_predictive_entropy(probs)

    # Decision score margin
    sorted_z = np.sort(z, axis=1)
    margin = sorted_z[:, -1] - sorted_z[:, -2]

    # Prob margin
    sorted_p = np.sort(probs, axis=1)
    prob_margin = sorted_p[:, -1] - sorted_p[:, -2]

    return {
        "u_max": u_max,
        "entropy": entropy,
        "margin": margin,
        "prob_margin": prob_margin,
    }


def run_inner_validation_for_all_methods(
    Xs: np.ndarray,
    ys: np.ndarray,
    source_subjects: np.ndarray,
    methods: List[str] = ["baseline", "tikhonov", "ledoit_wolf", "oas"],
    C: float = 0.01,
    n_splits: int = 3,
    reg: float = 1.0,
    seed: int = 42,
) -> Dict[str, Tuple[TemperatureScaler, Dict[float, float], float]]:
    """
    Run inner GroupKFold across the 14 training subjects ONCE for all methods:
    1. Fits inner LinearSVC once per split on inner source training subjects.
    2. Generates out-of-fold decision scores under simulated adaptation for each method.
    3. Fits temperature scaling on out-of-fold validation scores.
    4. Determines coverage thresholds tau_c for u_max strictly from validation percentiles.
    
    Zero outer test subject data is ever seen.
    """
    gkf = GroupKFold(n_splits=n_splits)
    z_oofs = {m: np.zeros((len(Xs), 4), dtype=np.float64) for m in methods}

    for inner_tr_idx, inner_val_idx in gkf.split(Xs, ys, groups=source_subjects):
        in_Xs = Xs[inner_tr_idx]
        in_ys = ys[inner_tr_idx]
        in_Xv = Xs[inner_val_idx]
        in_yv = ys[inner_val_idx]

        scaler = StandardScaler()
        in_Xs_s = scaler.fit_transform(in_Xs)
        in_Xv_s = scaler.transform(in_Xv)

        # Fit classifier ONCE per inner split
        clf = LinearSVC(dual=False, C=C, random_state=seed, max_iter=1000, tol=1e-2)
        clf.fit(in_Xs_s, in_ys)

        for m in methods:
            if m == "baseline":
                z_oofs[m][inner_val_idx] = clf.decision_function(in_Xv_s)
            else:
                Cs, Ct, mu_s, mu_t, _, _ = compute_covariance_estimates(
                    in_Xs_s, in_Xv_s, method=m, reg=reg
                )
                _, A_ts = compute_coral_projection(Cs, Ct)
                in_Xv_coral = (in_Xv_s - mu_t) @ A_ts + mu_s
                z_oofs[m][inner_val_idx] = clf.decision_function(in_Xv_coral)

    results = {}
    for m in methods:
        z_oof = z_oofs[m]
        scaler_temp = TemperatureScaler().fit(z_oof, ys)
        probs_oof = scaler_temp.predict_proba(z_oof)
        u_max_val = compute_max_probability_uncertainty(probs_oof)

        thresholds = {}
        for cov in COVERAGE_LEVELS:
            if cov == 1.0:
                thresholds[cov] = 1.0
            else:
                thresholds[cov] = float(np.percentile(u_max_val, cov * 100.0))

        val_preds = np.argmax(z_oof, axis=1)
        val_acc = float(accuracy_score(ys, val_preds))
        results[m] = (scaler_temp, thresholds, val_acc)

    return results


def run_window_coral_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    feature_indices: List[int],
    condition_name: str = "all_62",
    methods: List[str] = ["baseline", "tikhonov", "ledoit_wolf", "oas"],
    C: float = 0.01,
    reg: float = 1.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute 15-fold outer LOSO evaluation for window-level emotion recognition & unreliability prediction.
    
    Returns:
        pred_df: All 37,575 window predictions per method with probabilities, uncertainty, and flags.
        summary_df: Summary metrics (accuracy, balanced acc, macro-F1, AUROC, AUPRC, AURC).
        risk_coverage_df: Risk-coverage curve coordinates across coverage levels.
    """
    subjects = metadata["subject_id"].to_numpy()
    unique_subs = sorted(np.unique(subjects))
    n_windows_total = len(X)
    assert n_windows_total == 37575, f"Expected 37,575 windows, got {n_windows_total}"

    print(f"\n{'='*80}")
    print(f"RUNNING WINDOW-LEVEL EVALUATION: Condition '{condition_name}' ({len(feature_indices)} features)")
    print(f"Methods: {methods}")
    print(f"{'='*80}\n", flush=True)

    all_pred_records = []
    outer_fold_metrics = []

    for fold_idx, test_sub in enumerate(unique_subs):
        t_fold = time.time()
        tr_m = subjects != test_sub
        te_m = subjects == test_sub

        Xs = X[tr_m][:, feature_indices]
        ys = y[tr_m]
        Xt = X[te_m][:, feature_indices]
        yt = y[te_m]

        meta_test = metadata[te_m].reset_index(drop=True)
        source_subs = subjects[tr_m]

        print(f"--- Outer Fold {fold_idx+1}/15: Subject {test_sub} (Train: {len(Xs)}, Test: {len(Xt)}) ---", flush=True)

        # 1. Fit scaler strictly on source training windows
        scaler = StandardScaler()
        Xs_s = scaler.fit_transform(Xs)
        Xt_s = scaler.transform(Xt)

        # 2. Fit outer source classifier ONCE for all methods (Target-to-Source symmetry)
        t_train = time.time()
        clf_source = LinearSVC(dual=False, C=C, random_state=seed, max_iter=2000, tol=1e-2)
        clf_source.fit(Xs_s, ys)
        train_time = time.time() - t_train

        # 3. Inner Group-Safe Temperature Calibration & Threshold Selection for all methods
        inner_calib_results = run_inner_validation_for_all_methods(
            Xs=Xs,
            ys=ys,
            source_subjects=source_subs,
            methods=methods,
            C=C,
            n_splits=3,
            reg=reg,
            seed=seed,
        )

        # 4. Evaluate each method on outer test subject
        for method in methods:
            t_method = time.time()
            scaler_temp, thresholds, val_acc = inner_calib_results[method]

            # B. Target Domain Adaptation & Prediction
            rho_s_val, rho_t_val = 0.0, 0.0
            if method == "baseline":
                # Unadapted
                z_test = clf_source.decision_function(Xt_s)
                adaptation_desc = "none_inductive"
            else:
                # Transductive CORAL alignment
                Cs, Ct, mu_s, mu_t, rho_s_val, rho_t_val = compute_covariance_estimates(
                    Xs_s, Xt_s, method=method, reg=reg
                )
                _, A_ts = compute_coral_projection(Cs, Ct)
                Xt_coral = (Xt_s - mu_t) @ A_ts + mu_s
                z_test = clf_source.decision_function(Xt_coral)
                adaptation_desc = f"transductive_coral_{method}"

            # C. Calibrated Probabilities & Uncertainty Metrics
            probs_test = scaler_temp.predict_proba(z_test)
            preds_test = np.argmax(z_test, axis=1)

            u_dict = compute_uncertainty_metrics_dict(probs_test, z_test)
            u_max_test = u_dict["u_max"]
            entropy_test = u_dict["entropy"]
            margin_test = u_dict["margin"]

            # Predicted error probability via calibrated 1 - max_prob
            prob_error_test = u_max_test.copy()

            # Record predictions for each window
            for i in range(len(meta_test)):
                y_true = int(yt[i])
                y_pred = int(preds_test[i])
                is_correct = int(y_pred == y_true)
                is_error = int(y_pred != y_true)

                rec = {
                    "subject_id": test_sub,
                    "session_id": int(meta_test.loc[i, "session_id"]),
                    "trial_id": int(meta_test.loc[i, "trial_id"]),
                    "window_id": int(meta_test.loc[i, "window_id"]),
                    "condition": condition_name,
                    "method": method,
                    "adaptation_type": adaptation_desc,
                    "temperature": scaler_temp.temperature,
                    "shrinkage_source": rho_s_val,
                    "shrinkage_target": rho_t_val,
                    "true_label": y_true,
                    "true_label_name": LABEL_NAMES[y_true],
                    "predicted_label": y_pred,
                    "predicted_label_name": LABEL_NAMES[y_pred],
                    "is_correct": is_correct,
                    "is_error": is_error,
                    "prob_neutral": float(probs_test[i, 0]),
                    "prob_sad": float(probs_test[i, 1]),
                    "prob_fear": float(probs_test[i, 2]),
                    "prob_happy": float(probs_test[i, 3]),
                    "confidence": float(np.max(probs_test[i])),
                    "u_max": float(u_max_test[i]),
                    "predictive_entropy": float(entropy_test[i]),
                    "decision_margin": float(margin_test[i]),
                    "predicted_error_prob": float(prob_error_test[i]),
                }

                # Threshold decision flags for each coverage level
                for cov in COVERAGE_LEVELS:
                    tau = thresholds[cov]
                    # Reliable if uncertainty <= threshold
                    is_rel = int(u_max_test[i] <= tau)
                    rec[f"reliable_cov_{int(cov*100)}"] = is_rel

                all_pred_records.append(rec)

        print(f"  Fold {fold_idx+1} completed in {time.time()-t_fold:.1f}s", flush=True)

    pred_df = pd.DataFrame(all_pred_records)
    print(f"\nAll 15 folds completed. Total prediction records: {len(pred_df):,}", flush=True)

    # ---------------------------------------------------------
    # Compute Comprehensive Evaluation Metrics
    # ---------------------------------------------------------
    summary_records = []
    risk_cov_records = []

    for (cond, meth), grp in pred_df.groupby(["condition", "method"]):
        n_samples = len(grp)
        y_true = grp["true_label"].to_numpy()
        y_pred = grp["predicted_label"].to_numpy()
        is_error = grp["is_error"].to_numpy()
        u_max = grp["u_max"].to_numpy()
        entropy = grp["predictive_entropy"].to_numpy()
        probs = grp[["prob_neutral", "prob_sad", "prob_fear", "prob_happy"]].to_numpy()

        # 1. Overall Emotion Recognition Metrics
        pooled_acc = float(accuracy_score(y_true, y_pred)) * 100.0
        bal_acc = float(balanced_accuracy_score(y_true, y_pred)) * 100.0
        macro_f1 = float(f1_score(y_true, y_pred, average="macro")) * 100.0

        # Per-subject metrics
        sub_accs = grp.groupby("subject_id").apply(lambda sg: accuracy_score(sg["true_label"], sg["predicted_label"]) * 100.0)
        mean_sub_acc = float(sub_accs.mean())
        std_sub_acc = float(sub_accs.std())
        min_sub_acc = float(sub_accs.min())
        max_sub_acc = float(sub_accs.max())

        # 2. Probability Calibration Metrics
        brier = float(brier_score_loss(is_error, u_max))  # error probability brier score
        # Multiclass log loss
        try:
            nll = float(log_loss(y_true, probs, labels=[0, 1, 2, 3]))
        except Exception:
            nll = np.nan

        # 3. Unreliability Prediction (Error Detection AUROC / AUPRC)
        auroc_umax = float(roc_auc_score(is_error, u_max))
        auprc_umax = float(average_precision_score(is_error, u_max))
        auroc_entropy = float(roc_auc_score(is_error, entropy))
        auprc_entropy = float(average_precision_score(is_error, entropy))

        # Error detection classification metrics at 80% coverage threshold
        # (Flagging bottom 20% most uncertain as errors)
        pred_error_flag = (grp["reliable_cov_80"] == 0).astype(int)  # 1 = predicted error
        err_prec, err_rec, err_f1, _ = precision_recall_fscore_support(is_error, pred_error_flag, average="binary", zero_division=0)
        err_bal_acc = float(balanced_accuracy_score(is_error, pred_error_flag)) * 100.0
        cm = confusion_matrix(is_error, pred_error_flag)

        # 4. Risk-Coverage Curve & Selective Accuracy
        # Sort by uncertainty ascending (most confident first)
        sort_idx = np.argsort(u_max)
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        correct_sorted = (y_pred_sorted == y_true_sorted).astype(float)

        cov_grid = np.linspace(0.1, 1.0, 91)  # 10% to 100%
        achieved_covs = []
        achieved_risks = []
        achieved_accs = []

        for c in cov_grid:
            n_retain = int(np.round(c * n_samples))
            n_retain = max(n_retain, 1)
            sel_acc = float(np.mean(correct_sorted[:n_retain])) * 100.0
            sel_risk = 100.0 - sel_acc
            act_cov = (n_retain / n_samples) * 100.0
            achieved_covs.append(act_cov)
            achieved_risks.append(sel_risk)
            achieved_accs.append(sel_acc)

        # AURC (Area Under Risk-Coverage curve) in [0, 1]
        aurc = _trapezoid_integration(np.array(achieved_risks) / 100.0, np.array(achieved_covs) / 100.0)

        # Selective accuracy at standard pre-declared coverage levels
        sel_acc_dict = {}
        for target_c in COVERAGE_LEVELS:
            flag_col = f"reliable_cov_{int(target_c * 100)}"
            retained = grp[grp[flag_col] == 1]
            if len(retained) > 0:
                acc_c = float(accuracy_score(retained["true_label"], retained["predicted_label"])) * 100.0
                act_cov_pct = (len(retained) / n_samples) * 100.0
            else:
                acc_c = np.nan
                act_cov_pct = 0.0
            sel_acc_dict[int(target_c * 100)] = (acc_c, act_cov_pct)

            risk_cov_records.append({
                "condition": cond,
                "method": meth,
                "nominal_coverage_pct": int(target_c * 100),
                "achieved_coverage_pct": act_cov_pct,
                "selective_accuracy": acc_c,
                "selective_risk": 100.0 - acc_c,
                "retained_samples": len(retained),
                "total_samples": n_samples,
            })

        summary_records.append({
            "condition": cond,
            "method": meth,
            "n_windows": n_samples,
            "pooled_accuracy": pooled_acc,
            "subject_mean_accuracy": mean_sub_acc,
            "subject_std_accuracy": std_sub_acc,
            "subject_min_accuracy": min_sub_acc,
            "subject_max_accuracy": max_sub_acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "auroc_error_umax": auroc_umax,
            "auprc_error_umax": auprc_umax,
            "auroc_error_entropy": auroc_entropy,
            "auprc_error_entropy": auprc_entropy,
            "error_bal_acc_cov80": err_bal_acc,
            "error_precision_cov80": float(err_prec) * 100.0,
            "error_recall_cov80": float(err_rec) * 100.0,
            "error_f1_cov80": float(err_f1) * 100.0,
            "aurc": aurc,
            "sel_acc_100": sel_acc_dict[100][0],
            "sel_acc_90": sel_acc_dict[90][0],
            "sel_acc_80": sel_acc_dict[80][0],
            "sel_acc_70": sel_acc_dict[70][0],
            "sel_acc_60": sel_acc_dict[60][0],
            "sel_acc_50": sel_acc_dict[50][0],
            "brier_score": brier,
            "log_loss": nll,
            "cm_tn": int(cm[0, 0]),
            "cm_fp": int(cm[0, 1]),
            "cm_fn": int(cm[1, 0]),
            "cm_tp": int(cm[1, 1]),
        })

    summary_df = pd.DataFrame(summary_records)
    risk_cov_df = pd.DataFrame(risk_cov_records)

    return pred_df, summary_df, risk_cov_df


def plot_risk_coverage_curves(
    risk_cov_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate publication-ready Risk-Coverage Curves comparing all evaluated methods.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    colors = {
        "baseline": "#7f7f7f",
        "tikhonov": "#1f77b4",
        "ledoit_wolf": "#2ca02c",
        "oas": "#d62728",
    }
    labels = {
        "baseline": "LinearSVM Baseline (No Adapt)",
        "tikhonov": "CORAL (Tikhonov reg=1.0)",
        "ledoit_wolf": "CORAL (Ledoit-Wolf Shrinkage)",
        "oas": "CORAL (OAS Shrinkage)",
    }

    # Plot 1: all_62
    df_62 = risk_cov_df[risk_cov_df["condition"] == "all_62"]
    for meth in ["baseline", "tikhonov", "ledoit_wolf", "oas"]:
        sub_df = df_62[df_62["method"] == meth].sort_values("achieved_coverage_pct")
        if len(sub_df) > 0:
            ax1.plot(
                sub_df["achieved_coverage_pct"],
                sub_df["selective_accuracy"],
                marker="o",
                linewidth=2.2,
                color=colors.get(meth, "#333"),
                label=labels.get(meth, meth),
            )

    ax1.axhline(80.0, color="#9467bd", linestyle="--", linewidth=1.5, alpha=0.8, label="Aspirational Goal (80%)")
    ax1.set_xlabel("Coverage (%) [Retained Windows]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Selective Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Full Montage (all_62 Channels)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=10)
    ax1.set_xlim(45, 105)

    # Plot 2: stab_32
    df_32 = risk_cov_df[risk_cov_df["condition"] == "stab_32"]
    if len(df_32) > 0:
        for meth in ["baseline", "tikhonov", "ledoit_wolf", "oas"]:
            sub_df = df_32[df_32["method"] == meth].sort_values("achieved_coverage_pct")
            if len(sub_df) > 0:
                ax2.plot(
                    sub_df["achieved_coverage_pct"],
                    sub_df["selective_accuracy"],
                    marker="s",
                    linewidth=2.2,
                    color=colors.get(meth, "#333"),
                    label=labels.get(meth, meth),
                )
        ax2.axhline(80.0, color="#9467bd", linestyle="--", linewidth=1.5, alpha=0.8, label="Aspirational Goal (80%)")
        ax2.set_xlabel("Coverage (%) [Retained Windows]", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Selective Accuracy (%)", fontsize=12, fontweight="bold")
        ax2.set_title("Reduced Montage (stab_32 Channels)", fontsize=13, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower left", fontsize=10)
        ax2.set_xlim(45, 105)

    plt.suptitle("Risk-Coverage Curves: Window-Level Selective Accuracy vs Retention Rate", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved risk-coverage plot to: {output_path}")


def main():
    t_start = time.time()
    print("=" * 80)
    print("STARTING WINDOW-LEVEL CORAL EXPERIMENT & UNRELIABILITY PREDICTION")
    print("=" * 80)

    # 1. Load SEED-IV Dataset
    print("\n1. Loading SEED-IV Dataset...")
    dataset = load_seed_iv_dataset()
    X = dataset.X
    y = dataset.y
    metadata = dataset.metadata.copy()

    # Load corrected channel selections for stab_32
    selections_path = TABLES_DIR / "corrected_channel_selections.csv"
    selections_df = pd.read_csv(selections_path)

    # Channel feature mapping
    ch_names = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
                "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
                "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7",
                "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
                "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
    ch_to_idx = {ch: list(range(i * 5, (i + 1) * 5)) for i, ch in enumerate(ch_names)}

    # Evaluate all_62 first
    all_62_indices = list(range(310))
    pred_62, sum_62, rc_62 = run_window_coral_evaluation(
        X=X,
        y=y,
        metadata=metadata,
        feature_indices=all_62_indices,
        condition_name="all_62",
        methods=["baseline", "tikhonov", "ledoit_wolf", "oas"],
        C=0.01,
        reg=1.0,
        seed=42,
    )

    # Also evaluate stab_32 (fixed across outer folds using consensus stable channels)
    # The primary stab_32 consensus channels from Stage 2.5
    stab_32_row = selections_df[selections_df["budget"] == 32].iloc[0]
    stab_32_channels = [c.strip() for c in stab_32_row["corrected_stability_channels"].split(",")]
    stab_32_indices = []
    for ch in stab_32_channels:
        stab_32_indices.extend(ch_to_idx[ch])
    stab_32_indices = sorted(stab_32_indices)

    pred_32, sum_32, rc_32 = run_window_coral_evaluation(
        X=X,
        y=y,
        metadata=metadata,
        feature_indices=stab_32_indices,
        condition_name="stab_32",
        methods=["baseline", "tikhonov", "ledoit_wolf", "oas"],
        C=0.01,
        reg=1.0,
        seed=42,
    )

    # Combine results
    all_preds_df = pd.concat([pred_62, pred_32], ignore_index=True)
    all_summary_df = pd.concat([sum_62, sum_32], ignore_index=True)
    all_rc_df = pd.concat([rc_62, rc_32], ignore_index=True)

    # Add reference Phase B RBF-SVM baseline into comparison table
    base_preds = pd.read_csv(PREDICTIONS_DIR / "baseline_predictions.csv")
    rbf_acc = float(accuracy_score(base_preds["label"], base_preds["predicted_label"])) * 100.0
    rbf_bal = float(balanced_accuracy_score(base_preds["label"], base_preds["predicted_label"])) * 100.0
    rbf_f1 = float(f1_score(base_preds["label"], base_preds["predicted_label"], average="macro")) * 100.0
    rbf_sub_acc = base_preds.groupby("subject_id").apply(lambda sg: accuracy_score(sg["label"], sg["predicted_label"]) * 100.0)

    # Save output artifacts
    print("\nSaving output artifacts...")

    # 1. Predictions
    pred_path = PREDICTIONS_DIR / "window_coral_predictions.csv"
    all_preds_df.to_csv(pred_path, index=False)
    print(f"1. Saved predictions to: {pred_path} ({len(all_preds_df):,} rows)")

    # 2. Comparison Table
    comp_path = TABLES_DIR / "window_coral_comparison.csv"
    all_summary_df.to_csv(comp_path, index=False)
    print(f"2. Saved comparison table to: {comp_path}")

    # 3. Unreliability Metrics Table
    unrel_cols = [
        "condition", "method", "n_windows", "pooled_accuracy", "subject_mean_accuracy",
        "auroc_error_umax", "auprc_error_umax", "auroc_error_entropy", "auprc_error_entropy",
        "error_bal_acc_cov80", "error_precision_cov80", "error_recall_cov80", "error_f1_cov80",
        "aurc", "brier_score", "log_loss", "cm_tn", "cm_fp", "cm_fn", "cm_tp"
    ]
    unrel_path = TABLES_DIR / "window_coral_unreliability_metrics.csv"
    all_summary_df[unrel_cols].to_csv(unrel_path, index=False)
    print(f"3. Saved unreliability metrics to: {unrel_path}")

    # 4. Risk Coverage Table
    rc_path = TABLES_DIR / "window_coral_risk_coverage.csv"
    all_rc_df.to_csv(rc_path, index=False)
    print(f"4. Saved risk-coverage table to: {rc_path}")

    # 5. Risk Coverage Figure
    fig_path = FIGURES_DIR / "window_coral_risk_coverage.png"
    plot_risk_coverage_curves(all_rc_df, fig_path)

    print(f"\nExecution finished in {time.time()-t_start:.1f}s.")
    return all_summary_df


if __name__ == "__main__":
    main()
