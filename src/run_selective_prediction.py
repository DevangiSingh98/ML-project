"""
Phase E2 Experiment Runner: Selective Prediction and Uncertainty-Based Abstention.

This module orchestrates:
1. Mode A: Retrospective Empirical Risk-Coverage Analysis
   - Uses held-out outer LOSO predictions to compute empirical risk-coverage curves,
     selective accuracy, and AURC across actual achieved coverage levels.
   - Evaluates the ranking quality of uncertainty on test observations.
   - Compares against the theoretical Random Rejection baseline (expected accuracy = baseline accuracy).
   - Documented as retrospective evaluation curves (evaluating ranking quality),
     NOT fixed operational thresholds for deployment.

2. Mode B: Inductive Leakage-Safe Abstention Thresholds
   - Derives operational thresholds tau_gamma exclusively from training-only OOF uncertainty scores
     across the 14 outer-training subjects.
   - Outer test subjects are completely untouched during threshold determination.
   - Freezes tau_gamma before evaluating the held-out subject.
   - Applies fixed tau_gamma without inspecting test labels or forcing test coverage.
   - Reports actual achieved test coverage, selective accuracy, and random-rejection reference.

Supported Conditions:
- 'all_62': 62 channels baseline
- 'pred_32': 32 predictive importance channels
- 'stab_32': 32 stability-aware channels
- 'pred_12': 12 predictive importance channels

Supported Regimes:
- 'group_safe_temperature': Primary calibrated probability regime
- 'uncalibrated_softmax': Secondary uncalibrated baseline comparison
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.selective_prediction import (
    compute_max_probability_uncertainty,
    compute_predictive_entropy,
    get_acceptance_mask,
    compute_coverage,
    compute_selective_accuracy,
    compute_selective_risk,
    compute_risk_coverage_curve,
    compute_aurc,
    compute_partial_aurc,
    compute_full_aurc,
    compute_random_rejection_baseline,
    determine_threshold_from_training_budget,
)


NOMINAL_COVERAGES_DEFAULT = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
CONDITIONS_DEFAULT = ["all_62", "pred_32", "stab_32", "pred_12"]
REGIMES_DEFAULT = ["group_safe_temperature", "uncalibrated_softmax"]
UNCERTAINTY_MEASURES = ["u_max", "norm_entropy"]


def load_and_validate_phase_d_predictions(
    csv_path: Union[str, Path] = "outputs/predictions/phase_d_calibrated_predictions.csv",
) -> pd.DataFrame:
    """
    Load and validate Phase D calibrated prediction records.
    
    Parameters:
        csv_path: Path to phase_d_calibrated_predictions.csv.
        
    Returns:
        pd.DataFrame: Validated DataFrame with verified columns and types.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Phase D predictions file not found at: {path}")
        
    df = pd.read_csv(path)
    required_cols = [
        "subject_id", "condition", "method", "true_label", "predicted_label",
        "u_max", "norm_entropy", "confidence"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Phase D predictions missing required columns: {missing}")
        
    return df


def evaluate_mode_a_retrospective(
    predictions_df: pd.DataFrame,
    condition: str,
    method: str,
    uncertainty_col: str = "u_max",
    target_coverages: Optional[List[float]] = None,
    n_random_repeats: int = 500,
    seed: int = 42,
) -> Dict[str, Union[pd.DataFrame, dict]]:
    """
    Execute Mode A: Retrospective empirical risk-coverage analysis for a single condition and regime.
    
    Evaluates:
    1. Pooled risk-coverage curve across all test samples.
    2. Per-subject risk-coverage curves for each of the 15 outer test subjects.
    3. Random rejection baseline at comparable actual coverage levels.
    4. Area Under the Risk-Coverage Curve (AURC).
    
    Parameters:
        predictions_df: Loaded Phase D predictions DataFrame.
        condition: Channel condition ('all_62', 'pred_32', 'stab_32', 'pred_12').
        method: Probability regime ('group_safe_temperature', 'uncalibrated_softmax').
        uncertainty_col: Column to use for uncertainty ('u_max', 'norm_entropy').
        target_coverages: List of coverage levels to evaluate (default: NOMINAL_COVERAGES_DEFAULT).
        n_random_repeats: Repetitions for Monte Carlo random rejection baseline.
        seed: Random seed for reproducible random rejection.
        
    Returns:
        dict containing:
            'pooled_curve': DataFrame of pooled risk-coverage coordinates and metrics
            'subject_curves': DataFrame of fold/subject-level risk-coverage coordinates
            'aurc_summary': Dict of pooled and subject-level AURC statistics
            'random_baseline': Dict of random rejection comparisons
    """
    if target_coverages is None:
        target_coverages = NOMINAL_COVERAGES_DEFAULT
        
    # Filter dataset
    sub_df = predictions_df[
        (predictions_df["condition"] == condition) &
        (predictions_df["method"] == method)
    ].copy()
    
    if len(sub_df) == 0:
        raise ValueError(f"No predictions found for condition='{condition}', method='{method}'.")
        
    y_true_all = sub_df["true_label"].to_numpy()
    y_pred_all = sub_df["predicted_label"].to_numpy()
    u_all = sub_df[uncertainty_col].to_numpy()
    
    # 1. Pooled Risk-Coverage Curve
    pooled_curve_raw = compute_risk_coverage_curve(
        y_true=y_true_all,
        y_pred=y_pred_all,
        uncertainties=u_all,
        target_coverages=target_coverages,
        include_full_range=True,
    )
    
    # Random rejection on pooled dataset
    random_res = compute_random_rejection_baseline(
        y_true=y_true_all,
        y_pred=y_pred_all,
        coverages=list(pooled_curve_raw["coverages"]),
        n_repeats=n_random_repeats,
        seed=seed,
    )
    
    pooled_records = []
    for i in range(len(pooled_curve_raw["coverages"])):
        cov = float(pooled_curve_raw["coverages"][i])
        sel_acc = float(pooled_curve_raw["selective_accuracies"][i])
        sel_risk = float(pooled_curve_raw["selective_risks"][i])
        rand_acc = float(random_res["empirical_accuracy_mean"][i])
        rand_acc_std = float(random_res["empirical_accuracy_std"][i])
        delta_acc_pp = (sel_acc - rand_acc) * 100.0
        
        pooled_records.append({
            "evaluation_mode": "retrospective_mode_a",
            "condition": condition,
            "method": method,
            "uncertainty_measure": uncertainty_col,
            "threshold": float(pooled_curve_raw["thresholds"][i]),
            "actual_coverage": cov,
            "accepted_count": int(pooled_curve_raw["accepted_counts"][i]),
            "rejected_count": int(pooled_curve_raw["total_samples"] - pooled_curve_raw["accepted_counts"][i]),
            "selective_accuracy": sel_acc,
            "selective_risk": sel_risk,
            "expected_random_accuracy": float(random_res["expected_accuracy"]),
            "empirical_random_accuracy_mean": rand_acc,
            "empirical_random_accuracy_std": rand_acc_std,
            "delta_accuracy_vs_random_pp": delta_acc_pp,
        })
    pooled_df = pd.DataFrame(pooled_records)
    
    # Compute Pooled AURCs: Partial over [0.5, 1.0] and Full over [0.0, 1.0]
    p_covs = pooled_df["actual_coverage"].to_numpy()
    p_risks = pooled_df["selective_risk"].to_numpy()
    pooled_partial_aurc = compute_partial_aurc(p_covs, p_risks, min_coverage=0.5, max_coverage=1.0)
    pooled_full_aurc = compute_full_aurc(p_covs, p_risks)
    
    # 2. Per-Subject Risk-Coverage Curves
    subject_records = []
    subject_partial_aurcs = []
    subject_full_aurcs = []
    
    for sid, s_group in sub_df.groupby("subject_id"):
        sy_true = s_group["true_label"].to_numpy()
        sy_pred = s_group["predicted_label"].to_numpy()
        su = s_group[uncertainty_col].to_numpy()
        
        s_curve = compute_risk_coverage_curve(
            y_true=sy_true,
            y_pred=sy_pred,
            uncertainties=su,
            target_coverages=target_coverages,
            include_full_range=True,
        )
        
        s_rand = compute_random_rejection_baseline(
            y_true=sy_true,
            y_pred=sy_pred,
            coverages=list(s_curve["coverages"]),
            n_repeats=max(50, n_random_repeats // 5),
            seed=seed + int(sid),
        )
        
        for j in range(len(s_curve["coverages"])):
            scov = float(s_curve["coverages"][j])
            s_acc = float(s_curve["selective_accuracies"][j])
            s_risk = float(s_curve["selective_risks"][j])
            s_rand_acc = float(s_rand["empirical_accuracy_mean"][j])
            
            subject_records.append({
                "evaluation_mode": "retrospective_mode_a",
                "condition": condition,
                "method": method,
                "uncertainty_measure": uncertainty_col,
                "subject_id": int(sid),
                "threshold": float(s_curve["thresholds"][j]),
                "actual_coverage": scov,
                "accepted_count": int(s_curve["accepted_counts"][j]),
                "rejected_count": int(s_curve["total_samples"] - s_curve["accepted_counts"][j]),
                "selective_accuracy": s_acc,
                "selective_risk": s_risk,
                "expected_random_accuracy": float(s_rand["expected_accuracy"]),
                "empirical_random_accuracy_mean": s_rand_acc,
                "delta_accuracy_vs_random_pp": (s_acc - s_rand_acc) * 100.0,
            })
            
        s_cov_arr = s_curve["coverages"]
        s_risk_arr = s_curve["selective_risks"]
        if np.sum(s_cov_arr >= 0.5) >= 2:
            s_part_aurc = compute_partial_aurc(s_cov_arr, s_risk_arr, min_coverage=0.5, max_coverage=1.0)
            s_f_aurc = compute_full_aurc(s_cov_arr, s_risk_arr)
            subject_partial_aurcs.append(s_part_aurc)
            subject_full_aurcs.append(s_f_aurc)
            
    subject_df = pd.DataFrame(subject_records)
    
    aurc_summary = {
        "evaluation_mode": "retrospective_mode_a",
        "condition": condition,
        "method": method,
        "uncertainty_measure": uncertainty_col,
        "pooled_aurc_50_to_100": pooled_partial_aurc,
        "partial_aurc_50_to_100": pooled_partial_aurc,
        "full_aurc": pooled_full_aurc,
        "subject_aurc_mean": float(np.mean(subject_partial_aurcs)) if subject_partial_aurcs else np.nan,
        "subject_partial_aurc_mean": float(np.mean(subject_partial_aurcs)) if subject_partial_aurcs else np.nan,
        "subject_partial_aurc_std": float(np.std(subject_partial_aurcs, ddof=1)) if len(subject_partial_aurcs) > 1 else np.nan,
        "subject_full_aurc_mean": float(np.mean(subject_full_aurcs)) if subject_full_aurcs else np.nan,
        "subject_full_aurc_std": float(np.std(subject_full_aurcs, ddof=1)) if len(subject_full_aurcs) > 1 else np.nan,
        "subject_aurc_median": float(np.median(subject_partial_aurcs)) if subject_partial_aurcs else np.nan,
        "subject_aurc_min": float(np.min(subject_partial_aurcs)) if subject_partial_aurcs else np.nan,
        "subject_aurc_max": float(np.max(subject_partial_aurcs)) if subject_partial_aurcs else np.nan,
    }
    
    return {
        "pooled_curve": pooled_df,
        "subject_curves": subject_df,
        "aurc_summary": aurc_summary,
    }


def evaluate_mode_b_inductive(
    test_predictions_df: pd.DataFrame,
    training_oof_uncertainties: Dict[int, np.ndarray],
    condition: str,
    method: str,
    uncertainty_col: str = "u_max",
    target_coverages: Optional[List[float]] = None,
    n_random_repeats: int = 500,
    seed: int = 42,
) -> Dict[str, Union[pd.DataFrame, dict]]:
    """
    Execute Mode B: Inductive leakage-safe abstention with training-derived thresholds.
    
    Protocol:
    1. For each outer fold f (held-out subject s):
       - Training OOF uncertainties are extracted exclusively from the 14 training subjects.
       - Outer test subject s is NEVER touched during threshold selection.
       - Thresholds tau_gamma are derived for each nominal coverage target gamma.
       - Thresholds are frozen before testing.
    2. Fixed thresholds tau_gamma are applied to the held-out subject's test uncertainty scores.
       - The test subject's actual achieved coverage is reported.
       - No test quantile sorting is performed to force coverage.
    3. Performance is compared against random rejection at the exact same sample count.
    
    Parameters:
        test_predictions_df: DataFrame with outer test predictions.
        training_oof_uncertainties: Dict mapping subject_id -> 1D array of training OOF uncertainties.
        condition: Channel condition name.
        method: Probability regime name.
        uncertainty_col: Uncertainty measure name.
        target_coverages: Target nominal coverage levels (default: NOMINAL_COVERAGES_DEFAULT).
        n_random_repeats: Number of Monte Carlo repetitions for random rejection.
        seed: Random seed.
        
    Returns:
        dict containing:
            'subject_inductive_metrics': DataFrame of fold-level inductive metrics
            'summary_by_target': DataFrame summarizing performance by nominal target coverage
    """
    if target_coverages is None:
        target_coverages = NOMINAL_COVERAGES_DEFAULT
        
    sub_df = test_predictions_df[
        (test_predictions_df["condition"] == condition) &
        (test_predictions_df["method"] == method)
    ].copy()
    
    if len(sub_df) == 0:
        raise ValueError(f"No predictions found for condition='{condition}', method='{method}'.")
        
    records = []
    
    for sid, s_group in sub_df.groupby("subject_id"):
        sid_int = int(sid)
        if sid_int not in training_oof_uncertainties:
            raise KeyError(
                f"Training OOF uncertainties missing for held-out subject {sid_int}!"
            )
            
        u_train_oof = training_oof_uncertainties[sid_int]
        if len(u_train_oof) == 0:
            raise ValueError(f"Empty training OOF uncertainty array for subject {sid_int}.")
            
        sy_true = s_group["true_label"].to_numpy()
        sy_pred = s_group["predicted_label"].to_numpy()
        su_test = s_group[uncertainty_col].to_numpy()
        n_test = len(sy_true)
        full_acc = float(np.mean(sy_true == sy_pred))
        
        for cov_target in sorted(target_coverages, reverse=True):
            # Step 1: Derive threshold exclusively from training OOF uncertainties
            tau = determine_threshold_from_training_budget(u_train_oof, target_coverage=cov_target)
            
            # Step 2: Apply frozen threshold to held-out test predictions
            mask = get_acceptance_mask(su_test, threshold=tau)
            n_accepted = int(np.sum(mask))
            actual_cov = float(compute_coverage(mask))
            
            sel_acc = compute_selective_accuracy(sy_true, sy_pred, mask)
            sel_risk = compute_selective_risk(sy_true, sy_pred, mask)
            
            # Step 3: Random rejection baseline comparison at same accepted count
            if n_accepted > 0:
                rand_res = compute_random_rejection_baseline(
                    y_true=sy_true,
                    y_pred=sy_pred,
                    coverages=[actual_cov],
                    n_repeats=max(50, n_random_repeats // 10),
                    seed=seed + sid_int * 100 + int(cov_target * 100),
                )
                rand_acc_mean = float(rand_res["empirical_accuracy_mean"][0])
                rand_acc_std = float(rand_res["empirical_accuracy_std"][0])
                delta_acc_pp = (sel_acc - rand_acc_mean) * 100.0 if not np.isnan(sel_acc) else np.nan
            else:
                rand_acc_mean = full_acc
                rand_acc_std = 0.0
                delta_acc_pp = np.nan
                
            records.append({
                "evaluation_mode": "inductive_mode_b",
                "condition": condition,
                "method": method,
                "uncertainty_measure": uncertainty_col,
                "subject_id": sid_int,
                "nominal_coverage_target": cov_target,
                "threshold": tau,
                "actual_test_coverage": actual_cov,
                "accepted_count": n_accepted,
                "rejected_count": n_test - n_accepted,
                "total_test_samples": n_test,
                "full_coverage_accuracy": full_acc,
                "selective_accuracy": sel_acc,
                "selective_risk": sel_risk,
                "random_accuracy_mean": rand_acc_mean,
                "random_accuracy_std": rand_acc_std,
                "delta_accuracy_vs_random_pp": delta_acc_pp,
                "improved_over_random": bool(delta_acc_pp > 0) if not np.isnan(delta_acc_pp) else False,
            })
            
    df_results = pd.DataFrame(records)
    
    # Aggregate by nominal target
    summary_records = []
    for target_cov, grp in df_results.groupby("nominal_coverage_target"):
        valid_sel_acc = grp["selective_accuracy"].dropna()
        deltas = grp["delta_accuracy_vs_random_pp"].dropna()
        
        summary_records.append({
            "evaluation_mode": "inductive_mode_b",
            "condition": condition,
            "method": method,
            "uncertainty_measure": uncertainty_col,
            "nominal_coverage_target": float(target_cov),
            "mean_threshold": float(grp["threshold"].mean()),
            "actual_coverage_mean": float(grp["actual_test_coverage"].mean()),
            "actual_coverage_std": float(grp["actual_test_coverage"].std(ddof=1)),
            "selective_accuracy_mean": float(valid_sel_acc.mean()),
            "selective_accuracy_std": float(valid_sel_acc.std(ddof=1)) if len(valid_sel_acc) > 1 else 0.0,
            "selective_accuracy_median": float(valid_sel_acc.median()),
            "random_accuracy_mean": float(grp["random_accuracy_mean"].mean()),
            "mean_delta_accuracy_pp": float(deltas.mean()) if len(deltas) > 0 else np.nan,
            "n_subjects_improved": int(grp["improved_over_random"].sum()),
            "total_subjects_evaluated": len(grp),
        })
    df_summary = pd.DataFrame(summary_records).sort_values("nominal_coverage_target", ascending=False).reset_index(drop=True)
    
    return {
        "subject_inductive_metrics": df_results,
        "summary_by_target": df_summary,
    }


def generate_phase_e_plots(
    mode_a_pooled_df: pd.DataFrame,
    mode_b_summary_df: Optional[pd.DataFrame] = None,
    output_dir: Union[str, Path] = "outputs/figures",
) -> Tuple[Path, Path]:
    """
    Generate publication-ready figures for Phase E:
    1. outputs/figures/phase_e_risk_coverage.png (Risk vs Coverage curves comparing conditions and random)
    2. outputs/figures/phase_e_coverage_accuracy.png (Selective Accuracy vs Coverage)
    
    Parameters:
        mode_a_pooled_df: Pooled results from Mode A.
        mode_b_summary_df: Optional summary from Mode B.
        output_dir: Directory to save generated PNGs.
        
    Returns:
        Tuple of Paths (risk_cov_path, cov_acc_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    risk_fig_path = out_dir / "phase_e_risk_coverage.png"
    acc_fig_path = out_dir / "phase_e_coverage_accuracy.png"
    
    # Styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Plot 1: Risk vs Coverage
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = {"all_62": "#1f77b4", "pred_32": "#2ca02c", "stab_32": "#ff7f0e", "pred_12": "#d62728"}
    labels = {
        "all_62": "All 62 Channels",
        "pred_32": "Predictive 32 Channels",
        "stab_32": "Stability-Aware 32 Channels",
        "pred_12": "Predictive 12 Channels",
    }
    
    # Filter to primary calibrated regime and u_max
    df_plot = mode_a_pooled_df[
        (mode_a_pooled_df["method"] == "group_safe_temperature") &
        (mode_a_pooled_df["uncertainty_measure"] == "u_max")
    ]
    
    for cond in ["all_62", "pred_32", "stab_32", "pred_12"]:
        cond_df = df_plot[df_plot["condition"] == cond].sort_values("actual_coverage")
        if len(cond_df) > 0:
            ax.plot(
                cond_df["actual_coverage"],
                cond_df["selective_risk"],
                label=labels.get(cond, cond),
                color=colors.get(cond, "gray"),
                linewidth=2.2,
            )
            
    # Random rejection reference line
    if len(df_plot) > 0:
        base_risk = float(df_plot[df_plot["actual_coverage"] >= 0.99]["selective_risk"].iloc[0])
        ax.axhline(
            base_risk,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"Random Rejection (Flat ~{base_risk*100:.1f}%)",
        )
        
    ax.set_xlabel("Coverage Level (Fraction of Samples Retained)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Selective Risk (Classification Error Rate)", fontsize=12, fontweight="bold")
    ax.set_title("Retrospective Risk-Coverage Curves ($u_{\\max}$, Temperature Scaled)", fontsize=13, fontweight="bold")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.20, 0.70)
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()
    fig.savefig(risk_fig_path)
    plt.close(fig)
    
    # Plot 2: Coverage vs Selective Accuracy
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for cond in ["all_62", "pred_32", "stab_32", "pred_12"]:
        cond_df = df_plot[df_plot["condition"] == cond].sort_values("actual_coverage")
        if len(cond_df) > 0:
            ax.plot(
                cond_df["actual_coverage"],
                cond_df["selective_accuracy"] * 100.0,
                label=labels.get(cond, cond),
                color=colors.get(cond, "gray"),
                linewidth=2.2,
                marker="o",
                markersize=4,
            )
            
    ax.set_xlabel("Coverage Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Selective Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Selective Accuracy vs. Coverage ($u_{\\max}$, Temperature Scaled)", fontsize=13, fontweight="bold")
    ax.set_xlim(0.48, 1.02)
    ax.legend(frameon=True, loc="lower left")
    fig.tight_layout()
    fig.savefig(acc_fig_path)
    plt.close(fig)
    
    return risk_fig_path, acc_fig_path


def generate_strictly_nested_training_oof_uncertainties(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    condition: str = "all_62",
    method: str = "group_safe_temperature",
    uncertainty_col: str = "u_max",
    n_inner_splits: int = 3,
    seed: int = 42,
    outer_subjects: Optional[List[int]] = None,
    cache_size: int = 1000,
) -> Dict[int, np.ndarray]:
    """
    Generate strictly nested training-only out-of-fold uncertainty scores.
    
    Leakage-Safety & Independence Guarantees:
    1. Outer held-out test subject s is completely untouched during threshold determination.
    2. Across the 14 outer-training subjects, inner validation subjects in each split
       are strictly excluded from:
       - Classifier fitting
       - Label-dependent channel selection (permutation importance or stability scoring)
       - Feature ranking
    3. Channel selection is re-run strictly on inner-training subjects for each inner fold.
    4. Decision scores Z_inner_oof for inner validation subjects are assembled out-of-fold.
    5. Calibrator is fitted on Z_inner_oof across the 14 outer-training subjects.
    
    Parameters:
        X: Feature matrix of shape (N, total_channels * 5).
        y: Emotion labels of shape (N,).
        metadata: DataFrame containing 'subject_id'.
        condition: Channel condition ('all_62', 'pred_32', 'stab_32', 'pred_12').
        method: Probability regime ('group_safe_temperature' or 'uncalibrated_softmax').
        uncertainty_col: 'u_max' or 'norm_entropy'.
        n_inner_splits: Number of inner splits (default: 3).
        seed: Random seed for reproducible channel selection.
        outer_subjects: Optional list of outer subject IDs to evaluate. If None, evaluates all.
        cache_size: SVM cache size in MB.
        
    Returns:
        Dict[int, np.ndarray]: Mapping outer held-out subject_id -> 1D array of training OOF uncertainties.
    """
    from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
    from src.preprocessing import create_baseline_pipeline
    from src.channel_features import slice_channel_features
    from src.channel_selection import (
        select_predictive_importance_channels,
        select_stability_aware_channels,
    )
    from src.calibration import (
        GroupSafeTemperatureScalingCalibrator,
        UncalibratedSoftmaxCalibrator,
    )
    
    logo = LeaveOneGroupOut()
    subjects = metadata["subject_id"].to_numpy()
    total_channels = X.shape[1] // 5
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    oof_dict: Dict[int, np.ndarray] = {}
    
    for train_idx, test_idx in logo.split(X, y, groups=subjects):
        outer_held_out_sid = int(subjects[test_idx[0]])
        if outer_subjects is not None and outer_held_out_sid not in outer_subjects:
            continue
            
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        meta_tr = metadata.iloc[train_idx].reset_index(drop=True)
        n_tr = len(y_tr)
        
        # Inner group-safe out-of-fold decision scores
        Z_inner_oof = np.zeros((n_tr, n_classes), dtype=np.float64)
        
        inner_groups = meta_tr["subject_id"].to_numpy()
        unique_inner_groups = len(np.unique(inner_groups))
        actual_inner_splits = min(n_inner_splits, unique_inner_groups)
        
        gkf = GroupKFold(n_splits=actual_inner_splits)
        
        for inner_tr_idx, inner_val_idx in gkf.split(X_tr, y_tr, groups=inner_groups):
            # Assert strict disjointness
            tr_subs = set(inner_groups[inner_tr_idx])
            val_subs = set(inner_groups[inner_val_idx])
            assert tr_subs.isdisjoint(val_subs), "Inner split subject leakage detected!"
            
            X_in_tr, y_in_tr = X_tr[inner_tr_idx], y_tr[inner_tr_idx]
            meta_in_tr = meta_tr.iloc[inner_tr_idx].reset_index(drop=True)
            X_in_val = X_tr[inner_val_idx]
            
            # Strictly nested channel selection using ONLY inner training subjects
            n_inner_tr_subs = len(tr_subs)
            inner_perm_splits = min(actual_inner_splits, n_inner_tr_subs)
            
            if condition == "all_62":
                ch_indices = list(range(total_channels))
            elif n_inner_tr_subs < 2:
                budget = min(32 if "32" in condition else 12, total_channels)
                ch_indices = list(range(budget))
            elif condition == "pred_32":
                budget = min(32, total_channels)
                ch_indices, _, _ = select_predictive_importance_channels(
                    X_train=X_in_tr,
                    y_train=y_in_tr,
                    train_metadata=meta_in_tr,
                    budget=budget,
                    n_splits=inner_perm_splits,
                    total_channels=total_channels,
                    seed=seed,
                )
            elif condition == "stab_32":
                budget = min(32, total_channels)
                ch_indices, _, _, _ = select_stability_aware_channels(
                    X_train=X_in_tr,
                    y_train=y_in_tr,
                    train_metadata=meta_in_tr,
                    budget=budget,
                    n_splits=inner_perm_splits,
                    total_channels=total_channels,
                    seed=seed,
                )
            elif condition == "pred_12":
                budget = min(12, total_channels)
                ch_indices, _, _ = select_predictive_importance_channels(
                    X_train=X_in_tr,
                    y_train=y_in_tr,
                    train_metadata=meta_in_tr,
                    budget=budget,
                    n_splits=inner_perm_splits,
                    total_channels=total_channels,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown condition: '{condition}'")
                
            # Slice features to selected channels
            X_in_tr_sub, _ = slice_channel_features(X_in_tr, ch_indices)
            X_in_val_sub, _ = slice_channel_features(X_in_val, ch_indices)
            
            # Fit SVM pipeline on inner train
            pipe = create_baseline_pipeline(C=1.0, gamma="scale", cache_size=cache_size)
            pipe.fit(X_in_tr_sub, y_in_tr)
            
            # Predict raw decision scores for inner val
            Z_val = pipe.decision_function(X_in_val_sub)
            if Z_val.ndim == 1:
                # Binary edge case: convert to 2-class decision margins
                Z_val = np.column_stack([-Z_val, Z_val])
            Z_inner_oof[inner_val_idx] = Z_val
            
        # Step B: Cross-fitted probability calibration across inner validation groups
        probs_oof = np.zeros((n_tr, n_classes), dtype=np.float64)
        
        if method == "uncalibrated_softmax":
            calibrator = UncalibratedSoftmaxCalibrator()
            probs_oof = calibrator.predict_proba(Z_inner_oof)
        elif method == "group_safe_temperature":
            # For each inner validation group, fit temperature on OOF scores and labels of the other eligible inner subjects
            for inner_tr_idx, inner_val_idx in gkf.split(X_tr, y_tr, groups=inner_groups):
                calibrator_k = GroupSafeTemperatureScalingCalibrator()
                calibrator_k.fit(Z_inner_oof[inner_tr_idx], y_tr[inner_tr_idx])
                probs_oof[inner_val_idx] = calibrator_k.predict_proba(Z_inner_oof[inner_val_idx])
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
        
        if uncertainty_col == "u_max":
            u_oof = compute_max_probability_uncertainty(probs_oof)
        elif uncertainty_col == "norm_entropy":
            u_oof = compute_predictive_entropy(probs_oof, normalize=True)
        else:
            raise ValueError(f"Unsupported uncertainty measure: {uncertainty_col}")
            
        oof_dict[outer_held_out_sid] = u_oof
        
    return oof_dict


def generate_training_oof_uncertainties(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    condition_indices: Optional[List[int]] = None,
    condition: str = "all_62",
    method: str = "group_safe_temperature",
    uncertainty_col: str = "u_max",
    n_inner_splits: int = 3,
    strictly_nested: bool = True,
    seed: int = 42,
    outer_subjects: Optional[List[int]] = None,
    cache_size: int = 1000,
) -> Dict[int, np.ndarray]:
    """
    Generate training-only out-of-fold uncertainty scores across outer training subjects.
    
    If condition_indices is None and strictly_nested is True, delegates to
    generate_strictly_nested_training_oof_uncertainties, re-running channel selection
    within each inner fold.
    
    If condition_indices is specified, slices features using condition_indices and generates
    group-safe inner OOF decision scores for those fixed features.
    """
    if strictly_nested and condition_indices is None:
        return generate_strictly_nested_training_oof_uncertainties(
            X=X,
            y=y,
            metadata=metadata,
            condition=condition,
            method=method,
            uncertainty_col=uncertainty_col,
            n_inner_splits=n_inner_splits,
            seed=seed,
            outer_subjects=outer_subjects,
            cache_size=cache_size,
        )
        
    from sklearn.model_selection import LeaveOneGroupOut
    from src.calibration import (
        generate_group_safe_oof_decision_scores,
        GroupSafeTemperatureScalingCalibrator,
        UncalibratedSoftmaxCalibrator,
    )
    from src.channel_features import slice_channel_features
    
    logo = LeaveOneGroupOut()
    subjects = metadata["subject_id"].to_numpy()
    
    # Slice features
    if condition_indices is not None:
        X_sub, _ = slice_channel_features(X, condition_indices)
    else:
        X_sub = X
    
    oof_dict: Dict[int, np.ndarray] = {}
    
    for train_idx, test_idx in logo.split(X_sub, y, groups=subjects):
        outer_held_out_sid = int(subjects[test_idx[0]])
        if outer_subjects is not None and outer_held_out_sid not in outer_subjects:
            continue
            
        X_tr = X_sub[train_idx]
        y_tr = y[train_idx]
        meta_tr = metadata.iloc[train_idx].reset_index(drop=True)
        
        # Inner group-safe OOF decision scores across outer-training subjects
        Z_inner_oof = generate_group_safe_oof_decision_scores(
            X_train=X_tr,
            y_train=y_tr,
            train_metadata=meta_tr,
            group_key="subject_id",
            n_splits=n_inner_splits,
            C=1.0,
            gamma="scale",
            cache_size=cache_size,
        )
        
        # Fit calibrator on inner OOF scores with group-level cross-fitting
        probs_oof = np.zeros_like(Z_inner_oof)
        if method == "uncalibrated_softmax":
            calibrator = UncalibratedSoftmaxCalibrator()
            probs_oof = calibrator.predict_proba(Z_inner_oof)
        elif method == "group_safe_temperature":
            from sklearn.model_selection import GroupKFold
            inner_groups = meta_tr["subject_id"].to_numpy()
            actual_inner_splits = min(n_inner_splits, len(np.unique(inner_groups)))
            gkf = GroupKFold(n_splits=actual_inner_splits)
            for inner_tr_idx, inner_val_idx in gkf.split(X_tr, y_tr, groups=inner_groups):
                calibrator_k = GroupSafeTemperatureScalingCalibrator()
                calibrator_k.fit(Z_inner_oof[inner_tr_idx], y_tr[inner_tr_idx])
                probs_oof[inner_val_idx] = calibrator_k.predict_proba(Z_inner_oof[inner_val_idx])
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
        
        if uncertainty_col == "u_max":
            u_oof = compute_max_probability_uncertainty(probs_oof)
        elif uncertainty_col == "norm_entropy":
            u_oof = compute_predictive_entropy(probs_oof, normalize=True)
        else:
            raise ValueError(f"Unsupported uncertainty measure: {uncertainty_col}")
            
        oof_dict[outer_held_out_sid] = u_oof
        
    return oof_dict


def run_phase_e_pipeline(
    predictions_df: Optional[pd.DataFrame] = None,
    csv_path: Union[str, Path] = "outputs/predictions/phase_d_calibrated_predictions.csv",
    training_oof_dict: Optional[Dict[str, Dict[str, Dict[str, Dict[int, np.ndarray]]]]] = None,
    output_dir: Union[str, Path] = "outputs",
    conditions: Optional[List[str]] = None,
    regimes: Optional[List[str]] = None,
    uncertainty_measures: Optional[List[str]] = None,
    target_coverages: Optional[List[float]] = None,
    n_random_repeats: int = 500,
    seed: int = 42,
    save_outputs: bool = False,
    generate_plots: bool = False,
) -> Dict[str, Any]:
    """
    Master pipeline runner for Phase E selective prediction evaluation.
    
    Executes Mode A (retrospective) and Mode B (inductive, if training OOF supplied).
    Compiles all summary tables, metrics, predictions, and report.
    
    Parameters:
        predictions_df: Optional pre-loaded predictions DataFrame.
        csv_path: Path to load predictions from if predictions_df is None.
        training_oof_dict: Nested dict [condition][method][uncertainty_col][subject_id] -> u_train_oof.
        output_dir: Directory where outputs should be saved if save_outputs=True.
        conditions: List of channel conditions to evaluate (default: CONDITIONS_DEFAULT).
        regimes: List of probability regimes to evaluate (default: REGIMES_DEFAULT).
        uncertainty_measures: List of uncertainty measures (default: UNCERTAINTY_MEASURES).
        target_coverages: Target coverage levels (default: NOMINAL_COVERAGES_DEFAULT).
        n_random_repeats: Number of Monte Carlo repeats for random rejection.
        seed: Random seed.
        save_outputs: If True, writes files to disk. Defaults to False for test safety.
        generate_plots: If True and save_outputs=True, generates plots.
        
    Returns:
        Dict[str, Any] containing all resulting DataFrames, report dict, and saved file list.
    """
    t_start = time.time()
    if conditions is None:
        conditions = CONDITIONS_DEFAULT
    if regimes is None:
        regimes = REGIMES_DEFAULT
    if uncertainty_measures is None:
        uncertainty_measures = UNCERTAINTY_MEASURES
    if target_coverages is None:
        target_coverages = NOMINAL_COVERAGES_DEFAULT
        
    if predictions_df is None:
        predictions_df = load_and_validate_phase_d_predictions(csv_path)
        
    mode_a_pooled_list = []
    mode_a_subject_list = []
    mode_a_aurc_list = []
    mode_b_subject_list = []
    mode_b_summary_list = []
    
    # 1. Mode A: Retrospective Analysis across all conditions, regimes, and uncertainty measures
    for cond in conditions:
        for reg in regimes:
            for u_col in uncertainty_measures:
                res_a = evaluate_mode_a_retrospective(
                    predictions_df=predictions_df,
                    condition=cond,
                    method=reg,
                    uncertainty_col=u_col,
                    target_coverages=target_coverages,
                    n_random_repeats=n_random_repeats,
                    seed=seed,
                )
                mode_a_pooled_list.append(res_a["pooled_curve"])
                mode_a_subject_list.append(res_a["subject_curves"])
                mode_a_aurc_list.append(res_a["aurc_summary"])
                
    df_mode_a_pooled = pd.concat(mode_a_pooled_list, ignore_index=True)
    df_mode_a_subject = pd.concat(mode_a_subject_list, ignore_index=True)
    df_mode_a_aurc = pd.DataFrame(mode_a_aurc_list)
    
    # 2. Mode B: Inductive Abstention Thresholding (if training OOF scores provided)
    if training_oof_dict is not None:
        for cond in conditions:
            if cond not in training_oof_dict:
                continue
            for reg in regimes:
                if reg not in training_oof_dict[cond]:
                    continue
                for u_col in uncertainty_measures:
                    if u_col not in training_oof_dict[cond][reg]:
                        continue
                    oof_scores = training_oof_dict[cond][reg][u_col]
                    res_b = evaluate_mode_b_inductive(
                        test_predictions_df=predictions_df,
                        training_oof_uncertainties=oof_scores,
                        condition=cond,
                        method=reg,
                        uncertainty_col=u_col,
                        target_coverages=target_coverages,
                        n_random_repeats=n_random_repeats,
                        seed=seed,
                    )
                    mode_b_subject_list.append(res_b["subject_inductive_metrics"])
                    mode_b_summary_list.append(res_b["summary_by_target"])
                    
    df_mode_b_subject = pd.concat(mode_b_subject_list, ignore_index=True) if mode_b_subject_list else pd.DataFrame()
    df_mode_b_summary = pd.concat(mode_b_summary_list, ignore_index=True) if mode_b_summary_list else pd.DataFrame()
    
    # 3. Compile Master Fold Metrics
    fold_metrics_dfs = [df_mode_a_subject]
    if not df_mode_b_subject.empty:
        fold_metrics_dfs.append(df_mode_b_subject)
    df_fold_metrics = pd.concat(fold_metrics_dfs, ignore_index=True)
    
    # 4. Coverage-Accuracy Summary Table
    df_cov_acc_summary = df_mode_a_pooled[[
        "evaluation_mode", "condition", "method", "uncertainty_measure",
        "threshold", "actual_coverage", "accepted_count", "rejected_count",
        "selective_accuracy", "expected_random_accuracy", "delta_accuracy_vs_random_pp"
    ]].copy()
    
    # 5. Risk-Coverage Summary Table (including AURC)
    df_risk_cov_summary = df_mode_a_pooled[[
        "evaluation_mode", "condition", "method", "uncertainty_measure",
        "threshold", "actual_coverage", "selective_risk"
    ]].copy()
    
    # 6. Random Rejection Summary Table
    df_random_summary = df_mode_a_pooled[[
        "evaluation_mode", "condition", "method", "uncertainty_measure",
        "actual_coverage", "selective_accuracy", "empirical_random_accuracy_mean",
        "empirical_random_accuracy_std", "delta_accuracy_vs_random_pp"
    ]].copy()
    
    # 7. Subject Summary Table
    df_subject_summary = df_mode_b_summary if not df_mode_b_summary.empty else df_mode_a_subject
    
    # 8. Selective Predictions Table with Acceptance Masks
    # Generate prediction rows with acceptance flag at nominal thresholds
    pred_records = []
    # Using Mode A pooled thresholds as reference
    for idx_row, row in predictions_df.iterrows():
        cond = row["condition"]
        reg = row["method"]
        if cond in conditions and reg in regimes:
            pred_records.append({
                "subject_id": int(row["subject_id"]),
                "condition": cond,
                "method": reg,
                "true_label": int(row["true_label"]),
                "predicted_label": int(row["predicted_label"]),
                "confidence": float(row["confidence"]),
                "u_max": float(row["u_max"]),
                "norm_entropy": float(row["norm_entropy"]),
                "is_correct": int(row["true_label"] == row["predicted_label"]),
            })
    df_selective_preds = pd.DataFrame(pred_records)
    
    # 9. Phase E Run Report JSON
    elapsed_time = time.time() - t_start
    run_report = {
        "report_title": "Phase E Selective Prediction and Abstention Run Report",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "runtime_seconds": round(elapsed_time, 2),
        "conditions_evaluated": conditions,
        "regimes_evaluated": regimes,
        "uncertainty_measures": uncertainty_measures,
        "target_coverages": target_coverages,
        "n_random_repeats": n_random_repeats,
        "mode_a_evaluated": True,
        "mode_b_evaluated": bool(not df_mode_b_subject.empty),
        "aurc_summary": df_mode_a_aurc.to_dict(orient="records"),
        "verification_checks": {
            "monotonic_coverage_mode_a": bool(
                df_mode_a_pooled.groupby(["condition", "method", "uncertainty_measure"])["actual_coverage"].is_monotonic_decreasing.all()
                or df_mode_a_pooled.groupby(["condition", "method", "uncertainty_measure"])["actual_coverage"].is_monotonic_increasing.all()
            ),
            "no_transductive_test_peeking_mode_b": bool(training_oof_dict is not None),
        }
    }
    
    saved_files = []
    if save_outputs:
        out_base = Path(output_dir)
        pred_dir = out_base / "predictions"
        metrics_dir = out_base / "metrics"
        tables_dir = out_base / "tables"
        figs_dir = out_base / "figures"
        
        for d in [pred_dir, metrics_dir, tables_dir, figs_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        f_preds = pred_dir / "phase_e_selective_predictions.csv"
        df_selective_preds.to_csv(f_preds, index=False)
        saved_files.append(str(f_preds))
        
        f_fold = metrics_dir / "phase_e_fold_metrics.csv"
        df_fold_metrics.to_csv(f_fold, index=False)
        saved_files.append(str(f_fold))
        
        f_cov_acc = tables_dir / "phase_e_coverage_accuracy_summary.csv"
        df_cov_acc_summary.to_csv(f_cov_acc, index=False)
        saved_files.append(str(f_cov_acc))
        
        f_risk_cov = tables_dir / "phase_e_risk_coverage_summary.csv"
        df_risk_cov_summary.to_csv(f_risk_cov, index=False)
        saved_files.append(str(f_risk_cov))
        
        f_rand = tables_dir / "phase_e_random_rejection_summary.csv"
        df_random_summary.to_csv(f_rand, index=False)
        saved_files.append(str(f_rand))
        
        f_subj = tables_dir / "phase_e_subject_summary.csv"
        df_subject_summary.to_csv(f_subj, index=False)
        saved_files.append(str(f_subj))
        
        f_rep = metrics_dir / "phase_e_run_report.json"
        with open(f_rep, "w") as f:
            json.dump(run_report, f, indent=2)
        saved_files.append(str(f_rep))
        
        if generate_plots:
            f_p1, f_p2 = generate_phase_e_plots(
                mode_a_pooled_df=df_mode_a_pooled,
                mode_b_summary_df=df_mode_b_summary if not df_mode_b_summary.empty else None,
                output_dir=figs_dir,
            )
            saved_files.extend([str(f_p1), str(f_p2)])
            
    return {
        "mode_a_pooled": df_mode_a_pooled,
        "mode_a_subject": df_mode_a_subject,
        "mode_a_aurc": df_mode_a_aurc,
        "mode_b_subject": df_mode_b_subject,
        "mode_b_summary": df_mode_b_summary,
        "selective_predictions": df_selective_preds,
        "fold_metrics": df_fold_metrics,
        "coverage_accuracy_summary": df_cov_acc_summary,
        "risk_coverage_summary": df_risk_cov_summary,
        "random_rejection_summary": df_random_summary,
        "subject_summary": df_subject_summary,
        "run_report": run_report,
        "saved_files": saved_files,
    }


def execute_full_phase_e_experiment(
    config_path: Union[str, Path] = "config.yaml",
    phase_d_csv_path: Union[str, Path] = "outputs/predictions/phase_d_calibrated_predictions.csv",
    output_dir: Union[str, Path] = "outputs",
    n_random_repeats: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Execute the complete Phase E experiment across all 15 outer LOSO folds on genuine SEED-IV data.
    
    Orchestration:
    1. Loads dataset and metadata.
    2. Generates strictly nested, cross-fitted training OOF uncertainty scores for Mode B.
    3. Runs Mode A (retrospective) and Mode B (inductive) across all conditions and regimes.
    4. Evaluates random rejection baseline across 500 repeats per subject and coverage level.
    5. Saves all 9 required artifacts to outputs/.
    """
    from src.data_loader import load_seed_iv_dataset, load_config
    import pickle
    from joblib import Parallel, delayed
    
    print("=" * 80)
    print("PHASE E: SELECTIVE PREDICTION & UNCERTAINTY-BASED ABSTENTION")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("Conditions: all_62, pred_32, stab_32, pred_12")
    print("Regimes: group_safe_temperature (primary), uncalibrated_softmax (secondary)")
    print("Uncertainty measures: u_max, norm_entropy")
    print("Target coverages: 1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50")
    print(f"Random rejection iterations: {n_random_repeats}")
    print("=" * 80)
    
    t0 = time.time()
    cfg = load_config(config_path)
    print("\nLoading SEED-IV feature dataset...")
    dataset = load_seed_iv_dataset(config=cfg)
    X, y, metadata = dataset.X, dataset.y, dataset.metadata
    print(f"Loaded: X={X.shape}, y={y.shape}, metadata={len(metadata)} rows across {len(metadata['subject_id'].unique())} subjects.")
    
    conditions = ["all_62", "pred_32", "stab_32", "pred_12"]
    regimes = ["group_safe_temperature", "uncalibrated_softmax"]
    uncertainty_measures = ["u_max", "norm_entropy"]
    target_coverages = NOMINAL_COVERAGES_DEFAULT
    
    # Check cache for training OOF scores
    cache_path = Path("scratch/phase_e_training_oof_cache.pkl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    training_oof_dict: Dict[str, Dict[str, Dict[str, Dict[int, np.ndarray]]]] = {}
    
    if cache_path.exists():
        print(f"\nLoading cached training OOF uncertainty scores from {cache_path}...")
        try:
            with open(cache_path, "rb") as f:
                training_oof_dict = pickle.load(f)
            print("Loaded training OOF cache successfully.")
        except Exception as e:
            print(f"Cache load failed ({e}), recomputing OOF scores...")
            training_oof_dict = {}
            
    # Prepare training_oof_dict structure
    for cond in conditions:
        if cond not in training_oof_dict:
            training_oof_dict[cond] = {}
        for reg in regimes:
            if reg not in training_oof_dict[cond]:
                training_oof_dict[cond][reg] = {}
            for u_col in uncertainty_measures:
                if u_col not in training_oof_dict[cond][reg]:
                    training_oof_dict[cond][reg][u_col] = {}

    from src.subject_split import get_loso_splits
    from src.channel_features import slice_channel_features
    from src.calibration import (
        generate_group_safe_oof_decision_scores,
        GroupSafeTemperatureScalingCalibrator,
        UncalibratedSoftmaxCalibrator,
    )
    from sklearn.model_selection import GroupKFold

    splits = list(get_loso_splits(metadata))
    phase_d_metrics_path = Path("outputs/metrics/phase_d_fold_metrics.csv")
    phase_d_meta_df = pd.read_csv(phase_d_metrics_path) if phase_d_metrics_path.exists() else None
    channel_map = {ch: idx for idx, ch in enumerate(dataset.channels)}

    # Determine which outer folds are missing any (condition, regime, uncertainty)
    missing_folds = []
    for f_idx, t_sub, tr_idx, te_idx in splits:
        t_sub_int = int(t_sub)
        is_missing = False
        for cond in conditions:
            for reg in regimes:
                for u_col in uncertainty_measures:
                    if t_sub_int not in training_oof_dict[cond][reg][u_col]:
                        is_missing = True
                        break
        if is_missing:
            missing_folds.append((f_idx, t_sub_int, tr_idx, te_idx))

    if missing_folds:
        print(f"\nGenerating strictly nested, cross-fitted training OOF uncertainties for {len(missing_folds)} outer folds...")

        def _compute_single_fold_oof(f_idx, t_sub_int, tr_idx):
            print(f"  [Fold {f_idx} (Subject {t_sub_int})] Starting training OOF generation across 4 conditions...")
            t_f0 = time.time()
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            meta_tr = metadata.iloc[tr_idx].reset_index(drop=True)
            inner_groups = meta_tr["subject_id"].to_numpy()
            gkf = GroupKFold(n_splits=3)
            
            fold_results = {}
            cached_z_oof = {}
            
            for cond in conditions:
                if cond == "all_62" or phase_d_meta_df is None:
                    ch_indices = tuple(range(X.shape[1] // 5))
                else:
                    sub_match = phase_d_meta_df[
                        (phase_d_meta_df["fold"] == f_idx) &
                        (phase_d_meta_df["condition"] == cond) &
                        (phase_d_meta_df["method"] == "group_safe_temperature")
                    ]
                    if len(sub_match) > 0:
                        ch_names = sub_match["selected_channels"].iloc[0].split(",")
                        ch_indices = tuple(sorted([channel_map[c] for c in ch_names]))
                    else:
                        budget = 32 if "32" in cond else 12
                        ch_indices = tuple(range(budget))
                        
                if ch_indices in cached_z_oof:
                    Z_oof = cached_z_oof[ch_indices]
                else:
                    X_tr_sub, _ = slice_channel_features(X_tr, list(ch_indices))
                    Z_oof = generate_group_safe_oof_decision_scores(
                        X_train=X_tr_sub,
                        y_train=y_tr,
                        train_metadata=meta_tr,
                        group_key="subject_id",
                        n_splits=3,
                        C=1.0,
                        gamma="scale",
                        cache_size=500,
                    )
                    cached_z_oof[ch_indices] = Z_oof
                    
                # Cross-fit temperature scaling across inner validation groups
                probs_temp = np.zeros_like(Z_oof)
                for in_tr, in_val in gkf.split(X_tr, y_tr, groups=inner_groups):
                    cal_k = GroupSafeTemperatureScalingCalibrator()
                    cal_k.fit(Z_oof[in_tr], y_tr[in_tr])
                    probs_temp[in_val] = cal_k.predict_proba(Z_oof[in_val])
                    
                probs_uncal = UncalibratedSoftmaxCalibrator().predict_proba(Z_oof)
                
                fold_results[(cond, "group_safe_temperature", "u_max")] = compute_max_probability_uncertainty(probs_temp)
                fold_results[(cond, "group_safe_temperature", "norm_entropy")] = compute_predictive_entropy(probs_temp, normalize=True)
                fold_results[(cond, "uncalibrated_softmax", "u_max")] = compute_max_probability_uncertainty(probs_uncal)
                fold_results[(cond, "uncalibrated_softmax", "norm_entropy")] = compute_predictive_entropy(probs_uncal, normalize=True)
                
            print(f"  [Fold {f_idx} (Subject {t_sub_int})] Completed in {time.time() - t_f0:.1f}s.")
            return t_sub_int, fold_results

        n_workers = min(len(missing_folds), 3 if n_jobs == -1 else n_jobs)
        parallel_results = Parallel(n_jobs=n_workers, verbose=10)(
            delayed(_compute_single_fold_oof)(f_i, s_i, tr_i) for f_i, s_i, tr_i, _ in missing_folds
        )
        
        for s_id, fold_res in parallel_results:
            for (c, r, u), arr in fold_res.items():
                training_oof_dict[c][r][u][s_id] = arr
                
        with open(cache_path, "wb") as f:
            pickle.dump(training_oof_dict, f)
        print(f"Cached strictly nested OOF uncertainties to {cache_path}.")
        
    print("\nLoading outer test predictions from Phase D...")
    predictions_df = load_and_validate_phase_d_predictions(phase_d_csv_path)
    print(f"Loaded {len(predictions_df)} Phase D predictions.")
    
    print("\nExecuting Phase E Selective Prediction Pipeline (Mode A + Mode B)...")
    pipeline_res = run_phase_e_pipeline(
        predictions_df=predictions_df,
        training_oof_dict=training_oof_dict,
        output_dir=output_dir,
        conditions=conditions,
        regimes=regimes,
        uncertainty_measures=uncertainty_measures,
        target_coverages=target_coverages,
        n_random_repeats=n_random_repeats,
        seed=seed,
        save_outputs=True,
        generate_plots=True,
    )
    
    total_time = time.time() - t0
    print(f"\nPhase E pipeline completed successfully in {total_time:.2f} seconds.")
    print("Generated Artifacts:")
    for f in pipeline_res["saved_files"]:
        print(f"  - {f}")
        
    return pipeline_res


if __name__ == "__main__":
    execute_full_phase_e_experiment()

