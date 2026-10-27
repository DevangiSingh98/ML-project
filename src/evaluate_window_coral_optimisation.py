"""
src/evaluate_window_coral_optimisation.py

Phase 3: Systematic Accuracy & Reliability Optimisation for Window-Level CORAL.
Strictly preserves outer 15-fold Leave-One-Subject-Out (LOSO) cross-validation,
evaluating all 37,575 4-second EEG windows on SEED-IV.

Systematic Candidates Evaluated:
1. Reference Model: stab_32, LinearSVC (C=0.01), CORAL Tikhonov (lambda=1.0, T->S).
2. Experiment A (Lambda Grid): Inner validation selection of lambda in [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0].
3. Experiment B (Direction): Source-to-Target (S->T) CORAL alignment vs Target-to-Source (T->S).
4. Experiment C (Classifier Penalty): Wide-margin LinearSVC with C=0.001 vs C=0.01.
5. Experiment D (Covariance Estimator): Analytical OAS Shrinkage with tuned C.
6. Experiment E (Channel Montage): all_62 full montage (310 features) with tuned C vs stab_32 (160 features).
7. Experiment F (Joint Validation Optimised): Inner 3-fold subject-grouped validation selecting
   both C in [0.0005, 0.001, 0.005, 0.01] and lambda in [0.25, 0.5, 1.0, 2.0] per outer fold.

Two Distinct Selective Prediction Modes:
A. Operational Selective Prediction: Thresholds tau_c frozen from inner validation percentiles.
B. Diagnostic Exact Coverage: Exact coverage levels at 20%, 25%, 30%, 50%, 70%, 80%, 90%, 100%
   using deterministic tie-breaking.
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
from scipy import linalg
from scipy.optimize import minimize
from sklearn.covariance import OAS, LedoitWolf
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

EXACT_COVERAGE_LEVELS = [0.20, 0.25, 0.30, 0.50, 0.70, 0.80, 0.90, 1.00]
NOMINAL_COVERAGE_LEVELS = [0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 1.00]


def compute_coral_projection(Cs: np.ndarray, Ct: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute symmetric positive definite CORAL projection matrices A_st and A_ts."""
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


def compute_cov_matrices(Xs: np.ndarray, Xt: np.ndarray, method: str = "tikhonov", reg: float = 1.0):
    """Compute source and target covariance matrices under specified method."""
    d = Xs.shape[1]
    if method == "tikhonov":
        mu_s = Xs.mean(axis=0)
        mu_t = Xt.mean(axis=0)
        reg_val = max(reg, 1e-5)
        Cs = np.cov(Xs - mu_s, rowvar=False) + reg_val * np.eye(d)
        Ct = np.cov(Xt - mu_t, rowvar=False) + reg_val * np.eye(d)
        return Cs, Ct, mu_s, mu_t
    elif method == "oas":
        oas_s = OAS().fit(Xs)
        oas_t = OAS().fit(Xt)
        return oas_s.covariance_, oas_t.covariance_, oas_s.location_, oas_t.location_
    elif method == "ledoit_wolf":
        lw_s = LedoitWolf().fit(Xs)
        lw_t = LedoitWolf().fit(Xt)
        return lw_s.covariance_, lw_t.covariance_, lw_s.location_, lw_t.location_
    else:
        raise ValueError(f"Unknown method: {method}")


class FastTemperatureScaler:
    """Group-safe temperature scaling for probability calibration."""
    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, z_val: np.ndarray, y_val: np.ndarray) -> "FastTemperatureScaler":
        y_val = np.asarray(y_val, dtype=int)
        def nll_loss(t_param: np.ndarray) -> float:
            t = max(t_param[0], 1e-3)
            scaled_z = z_val / t
            max_z = np.max(scaled_z, axis=1, keepdims=True)
            exp_z = np.exp(scaled_z - max_z)
            probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            probs = np.clip(probs, 1e-15, 1.0)
            return float(-np.mean(np.log(probs)[np.arange(len(y_val)), y_val]))

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.05, 20.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0]) if res.success else 1.0
        return self

    def predict_proba(self, z: np.ndarray) -> np.ndarray:
        t = max(self.temperature, 1e-3)
        scaled_z = z / t
        max_z = np.max(scaled_z, axis=1, keepdims=True)
        exp_z = np.exp(scaled_z - max_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def run_inner_model_selection_and_calibration(
    Xs: np.ndarray,
    ys: np.ndarray,
    source_subjects: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run 3-fold subject-grouped validation on the 14 training subjects to:
    1. Select best lambda among [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0] for C=0.01.
    2. Select best (C, lambda) jointly from C in [0.0005, 0.001, 0.005, 0.01] and lambda in [0.25, 0.5, 1.0, 2.0].
    3. Generate out-of-fold validation scores for temperature scaling and threshold selection.
    
    Zero outer test subject data is ever seen.
    """
    gkf = GroupKFold(n_splits=3)
    c_candidates = [0.0005, 0.001, 0.005, 0.01]
    lambda_candidates = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

    # Pre-split and fit inner classifiers
    splits_data = []
    for inner_tr_idx, inner_val_idx in gkf.split(Xs, ys, groups=source_subjects):
        in_Xs = Xs[inner_tr_idx]
        in_ys = ys[inner_tr_idx]
        in_Xv = Xs[inner_val_idx]
        in_yv = ys[inner_val_idx]

        scaler = StandardScaler()
        in_Xs_s = scaler.fit_transform(in_Xs)
        in_Xv_s = scaler.transform(in_Xv)

        # Fit inner classifiers for each C candidate
        clfs = {}
        for c_val in c_candidates:
            clf = LinearSVC(dual=False, C=c_val, random_state=seed, max_iter=1500, tol=1e-2)
            clf.fit(in_Xs_s, in_ys)
            clfs[c_val] = clf

        # Precompute uncentered covariances
        d = in_Xs.shape[1]
        mu_s = in_Xs_s.mean(axis=0)
        mu_v = in_Xv_s.mean(axis=0)
        cov_s_raw = np.cov(in_Xs_s - mu_s, rowvar=False)
        cov_v_raw = np.cov(in_Xv_s - mu_v, rowvar=False)

        splits_data.append({
            "in_yv": in_yv,
            "in_Xv_s": in_Xv_s,
            "clfs": clfs,
            "cov_s_raw": cov_s_raw,
            "cov_v_raw": cov_v_raw,
            "mu_s": mu_s,
            "mu_v": mu_v,
            "val_idx": inner_val_idx,
            "d": d,
        })

    # Evaluate grid of (C, lambda)
    grid_scores = {}
    for c_val in c_candidates:
        for lmb in lambda_candidates:
            accs = []
            for sp in splits_data:
                d = sp["d"]
                Cs = sp["cov_s_raw"] + lmb * np.eye(d)
                Ct = sp["cov_v_raw"] + lmb * np.eye(d)
                _, A_ts = compute_coral_projection(Cs, Ct)
                Xv_coral = (sp["in_Xv_s"] - sp["mu_v"]) @ A_ts + sp["mu_s"]
                preds = sp["clfs"][c_val].predict(Xv_coral)
                accs.append(accuracy_score(sp["in_yv"], preds))
            grid_scores[(c_val, lmb)] = float(np.mean(accs))

    # 1. Best lambda for fixed C=0.01 (Exp A)
    best_lambda_c01 = max(lambda_candidates, key=lambda l: grid_scores[(0.01, l)])

    # 2. Best joint (C, lambda) (Exp F)
    best_c_joint, best_lambda_joint = max(grid_scores.keys(), key=lambda k: grid_scores[k])

    # 3. Compute OOF decision scores for calibration for selected configurations
    oof_scores = {}
    for conf_name, (c_sel, lmb_sel) in [
        ("ref_stab32_c01_lam1", (0.01, 1.0)),
        ("exp_a_tuned_lambda", (0.01, best_lambda_c01)),
        ("exp_c_wide_margin_c001", (0.001, 1.0)),
        ("exp_f_joint_tuned", (best_c_joint, best_lambda_joint)),
    ]:
        z_oof = np.zeros((len(Xs), 4), dtype=np.float64)
        for sp in splits_data:
            d = sp["d"]
            Cs = sp["cov_s_raw"] + lmb_sel * np.eye(d)
            Ct = sp["cov_v_raw"] + lmb_sel * np.eye(d)
            _, A_ts = compute_coral_projection(Cs, Ct)
            Xv_coral = (sp["in_Xv_s"] - sp["mu_v"]) @ A_ts + sp["mu_s"]
            z_oof[sp["val_idx"]] = sp["clfs"][c_sel].decision_function(Xv_coral)

        scaler_temp = FastTemperatureScaler().fit(z_oof, ys)
        probs_oof = scaler_temp.predict_proba(z_oof)
        u_max_val = compute_max_probability_uncertainty(probs_oof)

        thresholds = {}
        for cov in NOMINAL_COVERAGE_LEVELS:
            if cov == 1.0:
                thresholds[cov] = 1.0
            else:
                thresholds[cov] = float(np.percentile(u_max_val, cov * 100.0))

        oof_scores[conf_name] = {
            "scaler_temp": scaler_temp,
            "thresholds": thresholds,
            "val_acc": float(grid_scores[(c_sel, lmb_sel)]),
        }

    return {
        "best_lambda_c01": best_lambda_c01,
        "best_c_joint": best_c_joint,
        "best_lambda_joint": best_lambda_joint,
        "oof_calib": oof_scores,
        "grid_scores": grid_scores,
    }


def evaluate_window_coral_optimisation_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    selections_df: pd.DataFrame,
    ch_to_idx: Dict[str, List[int]],
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute full 15-fold outer Leave-One-Subject-Out evaluation across candidate models.
    """
    subjects = metadata["subject_id"].to_numpy()
    unique_subs = sorted(np.unique(subjects))
    n_windows = len(X)
    assert n_windows == 37575, f"Expected 37,575 windows, got {n_windows}"

    # Extract stab_32 indices
    stab_32_row = selections_df[selections_df["budget"] == 32].iloc[0]
    stab_32_channels = [c.strip() for c in stab_32_row["corrected_stability_channels"].split(",")]
    stab_32_indices = sorted([idx for ch in stab_32_channels for idx in ch_to_idx[ch]])
    all_62_indices = list(range(310))

    configs = [
        # 1. Reference
        {"id": "ref_stab32_c01_lam1", "name": "Reference Model (stab_32, C=0.01, lam=1.0)", "montage": "stab_32", "indices": stab_32_indices, "C": 0.01, "lambda": 1.0, "dir": "T->S", "cov": "tikhonov"},
        # 2. Experiment A: Tuned Lambda
        {"id": "exp_a_tuned_lambda", "name": "Exp A: Inner-Tuned Lambda (stab_32, C=0.01)", "montage": "stab_32", "indices": stab_32_indices, "C": 0.01, "lambda": "inner_tuned", "dir": "T->S", "cov": "tikhonov"},
        # 3. Experiment B: Source-to-Target Direction
        {"id": "exp_b_dir_st", "name": "Exp B: Source-to-Target CORAL (stab_32, C=0.01, lam=1.0)", "montage": "stab_32", "indices": stab_32_indices, "C": 0.01, "lambda": 1.0, "dir": "S->T", "cov": "tikhonov"},
        # 4. Experiment C: Wide-Margin Linear Classifier
        {"id": "exp_c_wide_margin_c001", "name": "Exp C: Wide-Margin Classifier (stab_32, C=0.001, lam=1.0)", "montage": "stab_32", "indices": stab_32_indices, "C": 0.001, "lambda": 1.0, "dir": "T->S", "cov": "tikhonov"},
        # 5. Experiment D: Analytical OAS Covariance Shrinkage
        {"id": "exp_d_oas_c001", "name": "Exp D: OAS Covariance Shrinkage (stab_32, C=0.001)", "montage": "stab_32", "indices": stab_32_indices, "C": 0.001, "lambda": "oas", "dir": "T->S", "cov": "oas"},
        # 6. Experiment E: Full Montage (all_62) Wide-Margin
        {"id": "exp_e_all62_c001", "name": "Exp E: Full Montage all_62 (C=0.001, lam=1.0)", "montage": "all_62", "indices": all_62_indices, "C": 0.001, "lambda": 1.0, "dir": "T->S", "cov": "tikhonov"},
        # 7. Experiment F: Joint Inner-Validation Optimised (C and lambda)
        {"id": "exp_f_joint_tuned", "name": "Exp F: Joint Inner-Validated (stab_32, Tuned C & Lam)", "montage": "stab_32", "indices": stab_32_indices, "C": "inner_tuned", "lambda": "inner_tuned", "dir": "T->S", "cov": "tikhonov"},
    ]

    all_predictions = []
    print(f"\n{'='*80}")
    print(f"STARTING 15-FOLD OUTER LOSO OPTIMISATION EXPERIMENT ({len(configs)} CANDIDATES)")
    print(f"{'='*80}\n", flush=True)

    t_global = time.time()

    for fold_idx, test_sub in enumerate(unique_subs):
        t_fold = time.time()
        tr_m = subjects != test_sub
        te_m = subjects == test_sub

        source_subs = subjects[tr_m]
        meta_test = metadata[te_m].reset_index(drop=True)
        yt = y[te_m]

        print(f"--- Outer Fold {fold_idx+1}/15: Subject {test_sub} (Train: {tr_m.sum()}, Test: {te_m.sum()}) ---", flush=True)

        # 1. Inner Subject-Grouped Validation for Model Selection on stab_32
        Xs_stab = X[tr_m][:, stab_32_indices]
        ys_tr = y[tr_m]
        val_res = run_inner_model_selection_and_calibration(
            Xs=Xs_stab,
            ys=ys_tr,
            source_subjects=source_subs,
            seed=seed,
        )
        best_lam_c01 = val_res["best_lambda_c01"]
        best_c_joint = val_res["best_c_joint"]
        best_lam_joint = val_res["best_lambda_joint"]
        oof_calib = val_res["oof_calib"]

        print(f"  Inner Val Choices: Best Lambda (C=0.01) = {best_lam_c01}, Best Joint (C, Lam) = ({best_c_joint}, {best_lam_joint})")

        # 2. Evaluate Each Candidate Configuration
        for cfg in configs:
            feat_idx = cfg["indices"]
            Xs = X[tr_m][:, feat_idx]
            Xt = X[te_m][:, feat_idx]
            d = len(feat_idx)

            # Fit scaler on outer source training windows only
            scaler = StandardScaler()
            Xs_s = scaler.fit_transform(Xs)
            Xt_s = scaler.transform(Xt)

            # Determine C and lambda
            c_val = cfg["C"]
            if c_val == "inner_tuned":
                c_val = best_c_joint

            lam_val = cfg["lambda"]
            if lam_val == "inner_tuned":
                lam_val = best_lam_joint if cfg["C"] == "inner_tuned" else best_lam_c01

            # Covariance estimation & CORAL alignment
            if cfg["cov"] == "oas":
                Cs, Ct, mu_s, mu_t = compute_cov_matrices(Xs_s, Xt_s, method="oas")
            else:
                Cs, Ct, mu_s, mu_t = compute_cov_matrices(Xs_s, Xt_s, method="tikhonov", reg=float(lam_val))

            A_st, A_ts = compute_coral_projection(Cs, Ct)

            # Direction
            if cfg["dir"] == "S->T":
                # Align source to target, train on aligned source, predict on raw target
                Xs_aligned = (Xs_s - mu_s) @ A_st + mu_t
                clf = LinearSVC(dual=False, C=c_val, random_state=seed, max_iter=2000, tol=1e-2)
                clf.fit(Xs_aligned, ys_tr)
                z_test = clf.decision_function(Xt_s)
            else:
                # Align target to source, train on source, predict on aligned target
                clf = LinearSVC(dual=False, C=c_val, random_state=seed, max_iter=2000, tol=1e-2)
                clf.fit(Xs_s, ys_tr)
                Xt_aligned = (Xt_s - mu_t) @ A_ts + mu_s
                z_test = clf.decision_function(Xt_aligned)

            # Calibration and threshold selection
            calib_key = cfg["id"]
            if calib_key not in oof_calib:
                calib_key = "ref_stab32_c01_lam1"

            scaler_temp = oof_calib[calib_key]["scaler_temp"]
            thresholds = oof_calib[calib_key]["thresholds"]

            probs_test = scaler_temp.predict_proba(z_test)
            preds_test = np.argmax(z_test, axis=1)
            u_max_test = compute_max_probability_uncertainty(probs_test)
            entropy_test = compute_predictive_entropy(probs_test)

            # Record sample predictions
            for i in range(len(meta_test)):
                y_true = int(yt[i])
                y_pred = int(preds_test[i])
                rec = {
                    "subject_id": test_sub,
                    "session_id": int(meta_test.loc[i, "session_id"]),
                    "trial_id": int(meta_test.loc[i, "trial_id"]),
                    "window_id": int(meta_test.loc[i, "window_id"]),
                    "config_id": cfg["id"],
                    "config_name": cfg["name"],
                    "montage": cfg["montage"],
                    "selected_C": float(c_val),
                    "selected_lambda": str(lam_val),
                    "direction": cfg["dir"],
                    "true_label": y_true,
                    "true_label_name": LABEL_NAMES[y_true],
                    "predicted_label": y_pred,
                    "predicted_label_name": LABEL_NAMES[y_pred],
                    "is_correct": int(y_pred == y_true),
                    "is_error": int(y_pred != y_true),
                    "prob_neutral": float(probs_test[i, 0]),
                    "prob_sad": float(probs_test[i, 1]),
                    "prob_fear": float(probs_test[i, 2]),
                    "prob_happy": float(probs_test[i, 3]),
                    "u_max": float(u_max_test[i]),
                    "predictive_entropy": float(entropy_test[i]),
                }

                # Operational threshold flags
                for cov in NOMINAL_COVERAGE_LEVELS:
                    tau = thresholds[cov]
                    rec[f"op_reliable_cov_{int(cov*100)}"] = int(u_max_test[i] <= tau)

                all_predictions.append(rec)

        print(f"  Fold {fold_idx+1} completed in {time.time()-t_fold:.1f}s", flush=True)

    pred_df = pd.DataFrame(all_predictions)
    print(f"\nAll 15 folds completed. Total records: {len(pred_df):,} in {time.time()-t_global:.1f}s.")

    # -------------------------------------------------------------
    # 3. Compute Summary Classification & Reliability Metrics
    # -------------------------------------------------------------
    summary_records = []
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

        # Overall Classification
        pooled_acc = float(accuracy_score(y_true, y_pred)) * 100.0
        bal_acc = float(balanced_accuracy_score(y_true, y_pred)) * 100.0
        macro_f1 = float(f1_score(y_true, y_pred, average="macro")) * 100.0

        sub_accs = grp.groupby("subject_id").apply(lambda sg: accuracy_score(sg["true_label"], sg["predicted_label"]) * 100.0)
        mean_sub_acc = float(sub_accs.mean())
        std_sub_acc = float(sub_accs.std())
        min_sub_acc = float(sub_accs.min())
        max_sub_acc = float(sub_accs.max())

        for sub_id, s_acc in sub_accs.items():
            per_subject_records.append({
                "config_id": cid,
                "config_name": cfg["name"],
                "subject_id": sub_id,
                "accuracy": s_acc,
            })

        # Uncertainty Metrics
        err_auroc = float(roc_auc_score(is_error, u_max))
        err_auprc = float(average_precision_score(is_error, u_max))
        brier = float(brier_score_loss(is_error, u_max))
        try:
            nll = float(log_loss(y_true, probs, labels=[0, 1, 2, 3]))
        except Exception:
            nll = np.nan

        # Operational Selective Prediction Metrics
        op_acc_dict = {}
        op_cov_dict = {}
        for cov in NOMINAL_COVERAGE_LEVELS:
            flag_col = f"op_reliable_cov_{int(cov*100)}"
            accepted = grp[grp[flag_col] == 1]
            abstained = grp[grp[flag_col] == 0]
            n_acc = len(accepted)
            act_cov = (n_acc / n_samples) * 100.0
            sel_acc = float(accuracy_score(accepted["true_label"], accepted["predicted_label"])) * 100.0 if n_acc > 0 else np.nan
            sel_risk = 100.0 - sel_acc if not np.isnan(sel_acc) else np.nan

            op_acc_dict[int(cov*100)] = sel_acc
            op_cov_dict[int(cov*100)] = act_cov

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

        # Diagnostic Exact-Coverage Curve (Deterministic tie-breaking by window index)
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

        # Calculate AURC across exact coverages
        aurc = _trapezoid_integration(np.array(achieved_risk_list), np.array(achieved_c_list))

        summary_records.append({
            "config_id": cid,
            "config_name": cfg["name"],
            "montage": cfg["montage"],
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
            # Exact diagnostic coverage
            "exact_acc_20": exact_accs[20],
            "exact_acc_25": exact_accs[25],
            "exact_acc_30": exact_accs[30],
            "exact_acc_50": exact_accs[50],
            "exact_acc_70": exact_accs[70],
            "exact_acc_80": exact_accs[80],
            "exact_acc_90": exact_accs[90],
            "exact_acc_100": exact_accs[100],
            # Operational coverage at nominal 50%
            "op_acc_cov50": op_acc_dict[50],
            "op_act_cov50": op_cov_dict[50],
            "op_acc_cov80": op_acc_dict[80],
            "op_act_cov80": op_cov_dict[80],
        })

    summary_df = pd.DataFrame(summary_records)
    op_df = pd.DataFrame(operational_records)
    exact_df = pd.DataFrame(exact_coverage_records)
    sub_df = pd.DataFrame(per_subject_records)

    return pred_df, summary_df, op_df, exact_df, sub_df


def plot_updated_risk_coverage_curves(exact_df: pd.DataFrame, output_path: Path):
    """Plot diagnostic exact-coverage curves across candidate models."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    colors = {
        "ref_stab32_c01_lam1": "#7f7f7f",
        "exp_a_tuned_lambda": "#1f77b4",
        "exp_b_dir_st": "#ff7f0e",
        "exp_c_wide_margin_c001": "#2ca02c",
        "exp_d_oas_c001": "#d62728",
        "exp_e_all62_c001": "#9467bd",
        "exp_f_joint_tuned": "#17becf",
    }
    markers = {
        "ref_stab32_c01_lam1": "o",
        "exp_a_tuned_lambda": "s",
        "exp_b_dir_st": "v",
        "exp_c_wide_margin_c001": "D",
        "exp_d_oas_c001": "^",
        "exp_e_all62_c001": "<",
        "exp_f_joint_tuned": "*",
    }

    for cid, grp in exact_df.groupby("config_id"):
        grp_sorted = grp.sort_values("exact_coverage_pct")
        cname = grp_sorted.iloc[0]["config_name"]
        ax.plot(
            grp_sorted["exact_coverage_pct"],
            grp_sorted["selective_accuracy"],
            marker=markers.get(cid, "o"),
            linewidth=2.0,
            markersize=7,
            color=colors.get(cid, "#333"),
            label=cname,
        )

    ax.axhline(65.0, color="#d62728", linestyle=":", linewidth=1.5, alpha=0.8, label="Primary Target (65%)")
    ax.axhline(70.0, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8, label="Stretch Target (70%)")
    ax.axhline(80.0, color="#9467bd", linestyle="-.", linewidth=1.5, alpha=0.8, label="Aspirational (80%)")

    ax.set_xlabel("Exact Coverage (%) [Retained Windows]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Selective Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Diagnostic Risk-Coverage Curves Across Optimized Window-Level CORAL Models", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(15, 105)
    ax.set_ylim(45, 95)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved updated risk-coverage plot to: {output_path}")


def main():
    t_start = time.time()
    print("=" * 80)
    print("PHASE 3: WINDOW-LEVEL CORAL ACCURACY & RELIABILITY OPTIMISATION")
    print("=" * 80)

    # 1. Load Data
    print("Loading SEED-IV dataset...")
    dataset = load_seed_iv_dataset()
    X = dataset.X
    y = dataset.y
    metadata = dataset.metadata.copy()

    selections_df = pd.read_csv(TABLES_DIR / "corrected_channel_selections.csv")

    ch_names = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
                "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
                "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7",
                "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
                "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
    ch_to_idx = {ch: list(range(i * 5, (i + 1) * 5)) for i, ch in enumerate(ch_names)}

    # 2. Run Pipeline
    pred_df, summary_df, op_df, exact_df, sub_df = evaluate_window_coral_optimisation_pipeline(
        X=X,
        y=y,
        metadata=metadata,
        selections_df=selections_df,
        ch_to_idx=ch_to_idx,
        seed=42,
    )

    # 3. Save Deliverables
    print("\nSaving deliverables...")

    # Deliverable 1: Predictions CSV
    pred_path = PREDICTIONS_DIR / "window_coral_accuracy_optimisation_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"1. Saved predictions to: {pred_path} ({len(pred_df):,} rows)")

    # Deliverable 2: Optimisation Summary CSV
    sum_path = TABLES_DIR / "window_coral_accuracy_optimisation.csv"
    summary_df.to_csv(sum_path, index=False)
    print(f"2. Saved summary metrics to: {sum_path}")

    # Deliverable 3: Selective Comparison CSV (Combining Operational & Exact)
    op_exact_df = pd.merge(
        op_df,
        exact_df.rename(columns={"exact_coverage_pct": "nominal_coverage_pct", "selective_accuracy": "exact_selective_accuracy", "selective_risk": "exact_selective_risk"}),
        on=["config_id", "config_name", "nominal_coverage_pct"],
        how="outer"
    )
    sel_path = TABLES_DIR / "window_coral_selective_comparison.csv"
    op_exact_df.to_csv(sel_path, index=False)
    print(f"3. Saved selective comparison to: {sel_path}")

    # Deliverable 4: Per-Subject Comparison Table
    sub_pivot = sub_df.pivot(index="subject_id", columns="config_name", values="accuracy")
    sub_path = TABLES_DIR / "window_coral_per_subject_comparison.csv"
    sub_pivot.to_csv(sub_path)
    print(f"4. Saved per-subject comparison to: {sub_path}")

    # Deliverable 5: Updated Risk-Coverage Plot
    fig_path = FIGURES_DIR / "window_coral_accuracy_optimisation_risk_coverage.png"
    plot_updated_risk_coverage_curves(exact_df, fig_path)

    print(f"\nPhase 3 execution completed successfully in {time.time()-t_start:.1f}s.")
    return summary_df


if __name__ == "__main__":
    main()
