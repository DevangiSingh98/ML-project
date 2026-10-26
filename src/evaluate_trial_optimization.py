"""
src/evaluate_trial_optimization.py

Experiment C: Trial-Level Feature and Classifier Optimization under Strict LOSO.

Evaluates:
1. Feature representations:
   A. Mean across windows (310 features)
   B. Mean + Standard Deviation across windows (620 features)
   C. Mean + Median Absolute Deviation across windows (620 features)
   D. Early, Middle, Late temporal window summaries (930 features)
2. Alternative SEED-IV features inventory (de_movingAve, psd_movingAve, psd_LDS)
3. Classifiers with inner subject-grouped CV and strict Pipeline scaling:
   - LinearSVC (broad log C grid)
   - Regularized Logistic Regression (C grid)
   - Shrinkage Linear Discriminant Analysis (shrinkage grid)
   - Tuned RBF SVM reference
4. Reduced channel sets (pred_32, stab_32, pred_12) with fold-specific training-only selection.
5. Paired comparisons against Experiment B baseline (stab_32 Linear SVM, 41.94%).
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import (
    DEFAULT_FREQUENCY_BANDS,
    LABEL_NAMES,
    DatasetContainer,
    load_seed_iv_dataset,
)

SEED_IV_DIR = ROOT_DIR / "SEED_IV"
OUTPUTS_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

EXPERIMENT_B_BASELINE_ACC = 41.94444444444444


def inventory_seed_iv_alternative_features() -> Dict[str, Any]:
    """
    Inventory available alternative feature representations in the SEED-IV dataset.
    Reports paths, file counts, and verified shapes.
    """
    smooth_dir = SEED_IV_DIR / "eeg_feature_smooth"
    inventory = {
        "feature_directory": str(smooth_dir),
        "available_sessions": [],
        "feature_types": {},
    }
    
    if not smooth_dir.is_dir():
        return inventory

    sessions = [d.name for d in smooth_dir.iterdir() if d.is_dir()]
    inventory["available_sessions"] = sorted(sessions)

    # Check session 1 sample file
    sample_file = smooth_dir / "1" / "1_20160518.mat"
    if sample_file.is_file():
        import scipy.io as sio
        mat = sio.loadmat(str(sample_file))
        keys = [k for k in mat.keys() if not k.startswith("__")]
        # Group keys by feature prefix
        prefixes = set()
        for k in keys:
            for p in ["de_movingAve", "de_LDS", "psd_movingAve", "psd_LDS"]:
                if k.startswith(p):
                    prefixes.add(p)
        for p in sorted(prefixes):
            k_sample = f"{p}1"
            shape = mat[k_sample].shape if k_sample in mat else None
            inventory["feature_types"][p] = {
                "sample_key": k_sample,
                "shape": shape,
                "channels": shape[0] if shape else None,
                "bands": shape[2] if shape else None,
                "description": f"{p} Differential Entropy / Power Spectral Density",
            }

    return inventory


def build_all_trial_feature_representations(
    dataset: Optional[DatasetContainer] = None,
    output_npz_path: Optional[Path] = None,
    output_meta_path: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """
    Construct whole-trial feature representations across all 1,080 trials.
    
    Representations:
    - 'mean' (Representation A): Temporal average across all windows (310 features)
    - 'mean_std' (Representation B): Temporal mean + standard deviation (620 features)
    - 'mean_mad' (Representation C): Temporal mean + median absolute deviation (620 features)
    - 'early_mid_late' (Representation D): Means across early, mid, late window thirds (930 features)
    
    Returns:
        representations: Dict mapping rep_name -> array of shape (1080, d)
        y_trial: Shape (1080,) array of integer emotion labels (0..3)
        trial_meta: DataFrame of shape (1080, 7) metadata
    """
    if dataset is None:
        dataset = load_seed_iv_dataset()

    X = dataset.X
    meta = dataset.metadata.copy()

    meta["trial_key"] = (
        meta["subject_id"].astype(str) + "_" +
        meta["session_id"].astype(str) + "_" +
        meta["trial_id"].astype(str)
    )

    trial_groups = meta.groupby("trial_key", sort=False)
    n_trials = trial_groups.ngroups
    if n_trials != 1080:
        raise ValueError(f"Expected exactly 1,080 trials, found {n_trials}")

    # Validate label consistency
    label_nunique = trial_groups["label"].nunique()
    if (label_nunique != 1).any():
        raise ValueError("Conflicting labels found within trial!")

    # Validate window count
    win_counts = trial_groups.size()
    if (win_counts < 10).any():
        raise ValueError("Found trials with fewer than 10 windows!")

    # Validate subject counts
    sub_counts = meta.drop_duplicates(subset=["subject_id", "session_id", "trial_id"]).groupby("subject_id").size()
    if not (sub_counts == 72).all():
        raise ValueError("Subjects do not have exactly 72 trials!")

    group_indices = trial_groups.indices
    trial_keys_ordered = list(trial_groups.groups.keys())

    X_mean = np.zeros((n_trials, 310), dtype=np.float32)
    X_mean_std = np.zeros((n_trials, 620), dtype=np.float32)
    X_mean_mad = np.zeros((n_trials, 620), dtype=np.float32)
    X_early_mid_late = np.zeros((n_trials, 930), dtype=np.float32)

    meta_records = []

    for i, key in enumerate(trial_keys_ordered):
        idxs = group_indices[key]
        W = len(idxs)
        X_win = X[idxs]  # Shape: (W, 310)

        # Rep A: Mean
        m_vec = X_win.mean(axis=0)
        X_mean[i] = m_vec

        # Rep B: Mean + Std
        s_vec = X_win.std(axis=0, ddof=0)
        X_mean_std[i] = np.concatenate([m_vec, s_vec])

        # Rep C: Mean + MAD
        med_vec = np.median(X_win, axis=0)
        mad_vec = np.median(np.abs(X_win - med_vec), axis=0)
        X_mean_mad[i] = np.concatenate([m_vec, mad_vec])

        # Rep D: Early, Middle, Late thirds
        # Safe partitioning for any W >= 10
        c1 = W // 3
        c2 = 2 * (W // 3)
        m_early = X_win[:c1].mean(axis=0)
        m_mid = X_win[c1:c2].mean(axis=0)
        m_late = X_win[c2:].mean(axis=0)
        X_early_mid_late[i] = np.concatenate([m_early, m_mid, m_late])

        first_row = meta.iloc[idxs[0]]
        meta_records.append({
            "trial_key": key,
            "subject_id": int(first_row["subject_id"]),
            "session_id": int(first_row["session_id"]),
            "trial_id": int(first_row["trial_id"]),
            "label": int(first_row["label"]),
            "label_name": str(first_row["label_name"]),
            "window_count": W,
        })

    trial_meta = pd.DataFrame(meta_records)
    y_trial = trial_meta["label"].to_numpy(dtype=np.int64)

    representations = {
        "mean": X_mean,
        "mean_std": X_mean_std,
        "mean_mad": X_mean_mad,
        "early_mid_late": X_early_mid_late,
    }

    # Save intermediate artifacts if requested
    if output_npz_path is not None:
        output_npz_path = Path(output_npz_path)
        output_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz_path,
            mean=X_mean,
            mean_std=X_mean_std,
            mean_mad=X_mean_mad,
            early_mid_late=X_early_mid_late,
            y_trial=y_trial,
        )
        print(f"Saved trial feature representations to: {output_npz_path}")

    if output_meta_path is not None:
        output_meta_path = Path(output_meta_path)
        output_meta_path.parent.mkdir(parents=True, exist_ok=True)
        trial_meta.to_csv(output_meta_path, index=False)
        print(f"Saved trial metadata to: {output_meta_path}")

    return representations, y_trial, trial_meta


def get_channel_feature_indices(
    test_subject: int,
    condition: str,
    channel_to_indices: Dict[str, List[int]],
    selections_df: Optional[pd.DataFrame],
    rep_type: str,
) -> List[int]:
    """
    Extract strictly nested training-only feature column indices for a given fold,
    condition, and representation type.
    """
    rep_dims = {
        "mean": (310, 1),
        "mean_std": (620, 2),
        "mean_mad": (620, 2),
        "early_mid_late": (930, 3),
    }
    if rep_type not in rep_dims:
        raise ValueError(f"Unknown rep_type: {rep_type}")

    total_dim, n_blocks = rep_dims[rep_type]

    if condition == "all_62":
        return list(range(total_dim))

    if selections_df is None:
        raise ValueError("selections_df required for reduced channel sets.")

    if condition == "stab_32":
        row = selections_df[(selections_df["test_subject"] == test_subject) & (selections_df["budget"] == 32)]
        if len(row) == 0:
            raise KeyError(f"No stab_32 channel selection for test_subject {test_subject}")
        ch_names = [c.strip() for c in row.iloc[0]["corrected_stability_channels"].split(",")]
    elif condition in ("pred_32", "pred_12"):
        b = 32 if condition == "pred_32" else 12
        row = selections_df[(selections_df["test_subject"] == test_subject) & (selections_df["budget"] == b)]
        if len(row) == 0:
            raise KeyError(f"No {condition} channel selection for test_subject {test_subject}")
        pred_col = "predictive_importance_channels" if "predictive_importance_channels" in row.columns else "predictive_channels"
        ch_names = [c.strip() for c in row.iloc[0][pred_col].split(",")]
    else:
        raise ValueError(f"Unknown channel condition: {condition}")

    # Base 310 indices
    base_indices = []
    for ch in ch_names:
        if ch not in channel_to_indices:
            raise KeyError(f"Channel '{ch}' not recognized in channel_to_indices")
        base_indices.extend(channel_to_indices[ch])

    # Extend across feature blocks
    expanded_indices = []
    for b in range(n_blocks):
        offset = b * 310
        expanded_indices.extend([idx + offset for idx in base_indices])

    return sorted(expanded_indices)


def get_model_pipeline_and_grid(model_name: str, seed: int = 42) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Build a leakage-safe Pipeline and parameter grid for inner GroupKFold tuning.
    StandardScaler is always inside the Pipeline so inner validation groups
    are never used to fit the scaler.
    """
    if model_name == "linear_svc":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual=False, max_iter=1000, tol=1e-3, random_state=seed)),
        ])
        grid = {
            "clf__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        }
    elif model_name == "logistic_regression":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, tol=1e-3, solver="lbfgs", random_state=seed)),
        ])
        grid = {
            "clf__C": [1e-3, 1e-2, 1e-1, 1.0],
        }
    elif model_name == "shrinkage_lda":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr")),
        ])
        grid = {
            "clf__shrinkage": ["auto", 0.1, 0.3, 0.5, 0.7, 0.9],
        }
    elif model_name == "rbf_svm":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=seed)),
        ])
        grid = {
            "clf__C": [0.5, 1.0, 5.0],
            "clf__gamma": ["scale", 0.01],
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return pipe, grid


def evaluate_loso_configuration(
    X_feat: np.ndarray,
    y_trial: np.ndarray,
    trial_meta: pd.DataFrame,
    condition: str,
    rep_name: str,
    model_name: str,
    channel_to_indices: Dict[str, List[int]],
    selections_df: Optional[pd.DataFrame],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Run 15-fold outer LOSO evaluation for a single (representation, condition, model) tuple.
    """
    subjects = trial_meta["subject_id"].to_numpy()
    unique_subs = sorted(np.unique(subjects))
    pipe, param_grid = get_model_pipeline_and_grid(model_name, seed=seed)

    records = []

    for test_sub in unique_subs:
        train_mask = subjects != test_sub
        test_mask = subjects == test_sub

        test_meta = trial_meta[test_mask].reset_index(drop=True)
        train_subs = subjects[train_mask]

        feat_idx = get_channel_feature_indices(
            test_subject=test_sub,
            condition=condition,
            channel_to_indices=channel_to_indices,
            selections_df=selections_df,
            rep_type=rep_name,
        )

        X_tr = X_feat[train_mask][:, feat_idx]
        X_te = X_feat[test_mask][:, feat_idx]
        y_tr = y_trial[train_mask]
        y_te = y_trial[test_mask]

        # Inner GroupKFold strictly on the 14 training subjects
        inner_gkf = GroupKFold(n_splits=5)
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_gkf,
            scoring="accuracy",
            n_jobs=1,
        )
        grid_search.fit(X_tr, y_tr, groups=train_subs)

        best_pipe = grid_search.best_estimator_
        best_params_str = str(grid_search.best_params_)
        preds = best_pipe.predict(X_te)

        for i in range(len(test_meta)):
            records.append({
                "subject_id": int(test_sub),
                "session_id": int(test_meta.loc[i, "session_id"]),
                "trial_id": int(test_meta.loc[i, "trial_id"]),
                "trial_key": str(test_meta.loc[i, "trial_key"]),
                "representation": rep_name,
                "condition": condition,
                "model_name": model_name,
                "n_features": len(feat_idx),
                "best_params": best_params_str,
                "true_label": int(y_te[i]),
                "true_label_name": LABEL_NAMES[int(y_te[i])],
                "predicted_label": int(preds[i]),
                "predicted_label_name": LABEL_NAMES[int(preds[i])],
                "is_correct": int(preds[i] == y_te[i]),
            })

    return records


def run_baseline_reproduction(
    X_mean: np.ndarray,
    y_trial: np.ndarray,
    trial_meta: pd.DataFrame,
    channel_to_indices: Dict[str, List[int]],
    selections_df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 2: Reproduce the Experiment B baseline:
    - Trial-level mean de_LDS features (310 features)
    - Tuned Linear SVM
    - Existing fold-specific stab_32 channel sets
    - Evaluates both:
      a) Exp B original (outer-scaled before inner CV)
      b) Strict Pipeline (inner-scaled inside inner CV)
    """
    print("\n[Step 2] Reproducing Experiment B Baseline on stab_32...")
    subjects = trial_meta["subject_id"].to_numpy()
    unique_subs = sorted(np.unique(subjects))

    pred_records = []

    for test_sub in unique_subs:
        train_mask = subjects != test_sub
        test_mask = subjects == test_sub
        test_meta = trial_meta[test_mask].reset_index(drop=True)
        train_subs = subjects[train_mask]

        feat_idx = get_channel_feature_indices(
            test_subject=test_sub,
            condition="stab_32",
            channel_to_indices=channel_to_indices,
            selections_df=selections_df,
            rep_type="mean",
        )

        X_tr = X_mean[train_mask][:, feat_idx]
        X_te = X_mean[test_mask][:, feat_idx]
        y_tr = y_trial[train_mask]
        y_te = y_trial[test_mask]

        # Mode A: Original Exp B (outer-scaled before GridSearchCV)
        scaler_outer = StandardScaler()
        X_tr_s = scaler_outer.fit_transform(X_tr)
        X_te_s = scaler_outer.transform(X_te)
        cv_orig = GridSearchCV(
            estimator=LinearSVC(dual=False, random_state=seed, max_iter=2000),
            param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
            cv=GroupKFold(n_splits=5),
            scoring="accuracy",
            n_jobs=-1,
        )
        cv_orig.fit(X_tr_s, y_tr, groups=train_subs)
        preds_orig = cv_orig.predict(X_te_s)

        # Mode B: Strict Pipeline (inner-scaled inside GridSearchCV)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual="auto", max_iter=5000, random_state=seed)),
        ])
        cv_pipe = GridSearchCV(
            estimator=pipe,
            param_grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
            cv=GroupKFold(n_splits=5),
            scoring="accuracy",
            n_jobs=-1,
        )
        cv_pipe.fit(X_tr, y_tr, groups=train_subs)
        preds_pipe = cv_pipe.predict(X_te)

        for i in range(len(test_meta)):
            # Record original
            pred_records.append({
                "subject_id": test_sub,
                "session_id": int(test_meta.loc[i, "session_id"]),
                "trial_id": int(test_meta.loc[i, "trial_id"]),
                "trial_key": test_meta.loc[i, "trial_key"],
                "mode": "exp_b_original_outer_scaled",
                "condition": "stab_32",
                "model_name": "linear_svc_tuned_exp_b",
                "true_label": int(y_te[i]),
                "predicted_label": int(preds_orig[i]),
                "is_correct": int(preds_orig[i] == y_te[i]),
            })
            # Record strict pipeline
            pred_records.append({
                "subject_id": test_sub,
                "session_id": int(test_meta.loc[i, "session_id"]),
                "trial_id": int(test_meta.loc[i, "trial_id"]),
                "trial_key": test_meta.loc[i, "trial_key"],
                "mode": "strict_pipeline_inner_scaled",
                "condition": "stab_32",
                "model_name": "linear_svc_tuned_pipeline",
                "true_label": int(y_te[i]),
                "predicted_label": int(preds_pipe[i]),
                "is_correct": int(preds_pipe[i] == y_te[i]),
            })

    pred_df = pd.DataFrame(pred_records)
    pred_path = PREDICTIONS_DIR / "experiment_c_baseline_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved baseline predictions to: {pred_path}")

    # Summary table
    summary_rows = []
    for mode in ["exp_b_original_outer_scaled", "strict_pipeline_inner_scaled"]:
        sub_df = pred_df[pred_df["mode"] == mode]
        tot = len(sub_df)
        corr = sub_df["is_correct"].sum()
        pooled_acc = (corr / tot) * 100.0
        by_sub = sub_df.groupby("subject_id")["is_correct"].mean() * 100.0
        sub_mean = by_sub.mean()
        sub_std = by_sub.std(ddof=1)
        summary_rows.append({
            "mode": mode,
            "condition": "stab_32",
            "model_name": "Linear SVM",
            "total_trials": tot,
            "correct_trials": corr,
            "pooled_accuracy": pooled_acc,
            "subject_mean_accuracy": sub_mean,
            "subject_std_accuracy": sub_std,
            "subject_min_accuracy": by_sub.min(),
            "subject_max_accuracy": by_sub.max(),
        })

    summary_df = pd.DataFrame(summary_rows)
    sum_path = TABLES_DIR / "experiment_c_baseline_summary.csv"
    summary_df.to_csv(sum_path, index=False)
    print(f"Saved baseline summary to: {sum_path}")
    print(summary_df.to_string(index=False))

    return pred_df, summary_df


def compute_comprehensive_metrics(
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute summary metrics, subject metrics, class metrics, and paired differences
    against the Experiment B baseline (Linear SVM stab_32, 41.94%).
    """
    combos = pred_df.groupby(["representation", "condition", "model_name"]).size().reset_index()

    summary_records = []
    subject_records = []
    class_records = []

    # Get baseline subject accuracies for stab_32 Linear SVM
    base_sub_accs = None
    # Check if baseline is present in pred_df or calculate from Exp B reference
    exp_b_path = ROOT_DIR / "outputs" / "predictions" / "trial_level_retraining_predictions.csv"
    if exp_b_path.is_file():
        exp_b_df = pd.read_csv(exp_b_path)
        base_subset = exp_b_df[(exp_b_df["condition"] == "stab_32") & (exp_b_df["model_type"] == "linear_svm_tuned")]
        base_sub_accs = base_subset.groupby("subject_id")["is_correct"].mean() * 100.0

    comparison_records = []

    for _, row in combos.iterrows():
        rep = row["representation"]
        cond = row["condition"]
        mod = row["model_name"]

        sub_df = pred_df[
            (pred_df["representation"] == rep) &
            (pred_df["condition"] == cond) &
            (pred_df["model_name"] == mod)
        ]

        total = len(sub_df)
        correct = int(sub_df["is_correct"].sum())
        pooled_acc = (correct / total) * 100.0

        y_true = sub_df["true_label"].to_numpy()
        y_pred = sub_df["predicted_label"].to_numpy()

        bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
        m_f1 = f1_score(y_true, y_pred, average="macro") * 100.0

        # Per subject
        by_sub = sub_df.groupby("subject_id")["is_correct"].mean() * 100.0
        for s_id, s_acc in by_sub.items():
            subject_records.append({
                "representation": rep,
                "condition": cond,
                "model_name": mod,
                "subject_id": s_id,
                "accuracy": s_acc,
            })

        mean_acc = by_sub.mean()
        std_acc = by_sub.std(ddof=1)
        med_acc = by_sub.median()
        min_acc = by_sub.min()
        max_acc = by_sub.max()

        # 95% Confidence Intervals
        sem = stats.sem(by_sub)
        ci_t = stats.t.interval(0.95, df=len(by_sub) - 1, loc=mean_acc, scale=sem)

        # Bootstrap CI
        rng = np.random.default_rng(42)
        boot_means = [rng.choice(by_sub.values, size=len(by_sub), replace=True).mean() for _ in range(2000)]
        ci_boot = (np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5))

        summary_records.append({
            "representation": rep,
            "condition": cond,
            "model_name": mod,
            "n_features": int(sub_df["n_features"].iloc[0]),
            "total_trials": total,
            "correct_trials": correct,
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
            "macro_f1": m_f1,
        })

        # Class metrics
        prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
        for c in range(4):
            class_records.append({
                "representation": rep,
                "condition": cond,
                "model_name": mod,
                "class_id": c,
                "class_name": LABEL_NAMES[c],
                "support": supp[c],
                "precision": prec[c] * 100.0,
                "recall": rec[c] * 100.0,
                "f1_score": f1[c] * 100.0,
            })

        # Paired comparison with Exp B baseline
        if base_sub_accs is not None:
            deltas = by_sub - base_sub_accs
            mean_delta = deltas.mean()
            std_delta = deltas.std(ddof=1)
            delta_sem = stats.sem(deltas)
            ci_delta_t = stats.t.interval(0.95, df=len(deltas) - 1, loc=mean_delta, scale=delta_sem)
            boot_deltas = [rng.choice(deltas.values, size=len(deltas), replace=True).mean() for _ in range(2000)]
            ci_delta_boot = (np.percentile(boot_deltas, 2.5), np.percentile(boot_deltas, 97.5))

            t_stat, t_pval = stats.ttest_1samp(deltas, popmean=0.0)
            if (deltas == 0).all():
                w_stat, w_pval = 0.0, 1.0
            else:
                try:
                    w_res = stats.wilcoxon(deltas, zero_method="wilcox")
                    w_stat, w_pval = w_res.statistic, w_res.pvalue
                except Exception:
                    w_stat, w_pval = np.nan, np.nan

            n_pos = int((deltas > 0).sum())
            n_neg = int((deltas < 0).sum())
            n_zero = int((deltas == 0).sum())

            comparison_records.append({
                "representation": rep,
                "condition": cond,
                "model_name": mod,
                "candidate_mean_acc": mean_acc,
                "exp_b_baseline_acc": base_sub_accs.mean(),
                "mean_delta_acc_pp": mean_delta,
                "std_delta_acc_pp": std_delta,
                "ci95_lower_t": ci_delta_t[0],
                "ci95_upper_t": ci_delta_t[1],
                "ci95_lower_boot": ci_delta_boot[0],
                "ci95_upper_boot": ci_delta_boot[1],
                "t_stat": t_stat,
                "t_pval": t_pval,
                "wilcoxon_stat": w_stat,
                "wilcoxon_pval": w_pval,
                "n_pos_subjects": n_pos,
                "n_neg_subjects": n_neg,
                "n_zero_subjects": n_zero,
            })

    summary_df = pd.DataFrame(summary_records)
    subject_df = pd.DataFrame(subject_records)
    class_df = pd.DataFrame(class_records)
    comparison_df = pd.DataFrame(comparison_records)

    return summary_df, subject_df, class_df, comparison_df


def plot_experiment_c_figures(
    summary_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    output_fig_comparison: Path,
    output_fig_cm: Path,
):
    """
    Generate publication-quality figures:
    1. Grouped bar chart comparing representations across classifiers.
    2. Confusion matrices for top-performing models.
    """
    output_fig_comparison.parent.mkdir(parents=True, exist_ok=True)
    output_fig_cm.parent.mkdir(parents=True, exist_ok=True)

    # Figure 1: Representation Comparison on all_62
    all_62_summary = summary_df[summary_df["condition"] == "all_62"].copy()
    if len(all_62_summary) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        reps = ["mean", "mean_std", "mean_mad", "early_mid_late"]
        rep_labels = ["Mean (310)", "Mean+Std (620)", "Mean+MAD (620)", "Early/Mid/Late (930)"]
        models = ["linear_svc", "logistic_regression", "shrinkage_lda", "rbf_svm"]
        model_labels = ["Linear SVC", "Logistic Reg", "Shrinkage LDA", "RBF SVM"]
        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

        x = np.arange(len(reps))
        width = 0.18

        for m_idx, (m, m_lbl, col) in enumerate(zip(models, model_labels, colors)):
            means = []
            sems = []
            for r in reps:
                match = all_62_summary[(all_62_summary["representation"] == r) & (all_62_summary["model_name"] == m)]
                if len(match) > 0:
                    means.append(match.iloc[0]["subject_mean_accuracy"])
                    sems.append(match.iloc[0]["subject_std_accuracy"] / np.sqrt(15))
                else:
                    means.append(0.0)
                    sems.append(0.0)

            ax.bar(x + m_idx * width - (len(models) - 1) * width / 2, means, width, yerr=sems, label=m_lbl, color=col, capsize=3, alpha=0.9)

        # Baseline reference line
        ax.axhline(EXPERIMENT_B_BASELINE_ACC, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Exp B Best Baseline (stab_32 Linear SVM: {EXPERIMENT_B_BASELINE_ACC:.2f}%)")
        ax.axhline(25.0, color="gray", linestyle=":", linewidth=1.5, label="Chance Level (25.0%)")

        ax.set_ylabel("Subject-Level Mean Accuracy (%)", fontsize=12)
        ax.set_title("Experiment C: Whole-Trial Feature Representation & Classifier Optimization (all_62)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(rep_labels, fontsize=11)
        ax.set_ylim(20, 52)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

        plt.tight_layout()
        plt.savefig(output_fig_comparison, dpi=300)
        plt.close()
        print(f"Saved representation comparison figure to: {output_fig_comparison}")

    # Figure 2: Confusion Matrices for Best Models
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    class_names = ["Neutral", "Sad", "Fear", "Happy"]

    # Select 3 key models to plot
    # 1. Linear SVC on all_62 (mean)
    # 2. Best representation on all_62
    # 3. Best representation on stab_32
    configs = [
        ("mean", "all_62", "linear_svc", "Linear SVC - all_62 (Mean)"),
        ("early_mid_late", "all_62", "linear_svc", "Linear SVC - all_62 (Early/Mid/Late)"),
        ("mean", "stab_32", "linear_svc", "Linear SVC - stab_32 (Mean)"),
    ]

    for ax_idx, (r, c, m, title) in enumerate(configs):
        sub_df = pred_df[(pred_df["representation"] == r) & (pred_df["condition"] == c) & (pred_df["model_name"] == m)]
        ax = axes[ax_idx]
        if len(sub_df) > 0:
            cm = confusion_matrix(sub_df["true_label"], sub_df["predicted_label"], labels=[0, 1, 2, 3])
            cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100.0

            im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=15, vmax=60)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(class_names, fontsize=10)
            ax.set_yticklabels(class_names, fontsize=10)
            ax.set_xlabel("Predicted Emotion", fontsize=10)
            if ax_idx == 0:
                ax.set_ylabel("True Emotion", fontsize=10)

            for i in range(4):
                for j in range(4):
                    val = cm_norm[i, j]
                    txt_color = "white" if val > 38 else "black"
                    ax.text(j, i, f"{val:.1f}%\n({cm[i, j]})", ha="center", va="center", color=txt_color, fontsize=9)
        else:
            ax.set_title(f"{title} (N/A)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_fig_cm, dpi=300)
    plt.close()
    print(f"Saved confusion matrix figure to: {output_fig_cm}")


def main():
    t_start = time.time()
    print("=" * 80)
    print("EXPERIMENT C: TRIAL-LEVEL FEATURE AND CLASSIFIER OPTIMIZATION")
    print("=" * 80)

    # Step 1: Inventory SEED-IV features
    print("\n[Step 1] Auditing SEED-IV Alternative Features...")
    inventory = inventory_seed_iv_alternative_features()
    print(f"Found features in: {inventory['feature_directory']}")
    for k, v in inventory["feature_types"].items():
        print(f"  - {k}: shape {v['shape']}, description: {v['description']}")

    # Load dataset and channel mappings
    npz_path = TABLES_DIR / "experiment_c_trial_features.npz"
    meta_path = TABLES_DIR / "experiment_c_trial_metadata.csv"
    selections_path = TABLES_DIR / "corrected_channel_selections.csv"
    if not selections_path.is_file():
        raise FileNotFoundError(f"Missing channel selections: {selections_path}")
    selections_df = pd.read_csv(selections_path)

    channels = [
        "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
        "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
        "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
        "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
    ]
    channel_to_indices = {ch: list(range(i * 5, (i + 1) * 5)) for i, ch in enumerate(channels)}

    if npz_path.is_file() and meta_path.is_file():
        print(f"\nLoading existing trial features from: {npz_path}")
        npz_data = np.load(npz_path)
        reps = {
            "mean": npz_data["mean"],
            "mean_std": npz_data["mean_std"],
            "mean_mad": npz_data["mean_mad"],
            "early_mid_late": npz_data["early_mid_late"],
        }
        y_trial = npz_data["y_trial"]
        trial_meta = pd.read_csv(meta_path)
    else:
        print("\nLoading dataset and building feature representations...")
        dataset = load_seed_iv_dataset()
        reps, y_trial, trial_meta = build_all_trial_feature_representations(
            dataset=dataset,
            output_npz_path=npz_path,
            output_meta_path=meta_path,
        )

    # Step 2: Establish reproducible baseline
    run_baseline_reproduction(
        X_mean=reps["mean"],
        y_trial=y_trial,
        trial_meta=trial_meta,
        channel_to_indices=channel_to_indices,
        selections_df=selections_df,
    )

    # Step 3 & 4: Evaluate Feature Representations and Classifiers
    # Stage 1: all_62 across all 4 representations x 4 classifiers (16 conditions)
    print("\n[Step 3 & 4 - Stage 1] Evaluating 4 representations x 4 classifiers on all_62...")
    all_pred_records = []
    
    rep_names = ["mean", "mean_std", "mean_mad", "early_mid_late"]
    model_names = ["linear_svc", "logistic_regression", "shrinkage_lda", "rbf_svm"]

    for r_idx, rep_name in enumerate(rep_names):
        for m_idx, mod_name in enumerate(model_names):
            t0 = time.time()
            print(f"  Evaluating [all_62] Rep: {rep_name:14s} | Model: {mod_name:19s} ...", end="", flush=True)
            recs = evaluate_loso_configuration(
                X_feat=reps[rep_name],
                y_trial=y_trial,
                trial_meta=trial_meta,
                condition="all_62",
                rep_name=rep_name,
                model_name=mod_name,
                channel_to_indices=channel_to_indices,
                selections_df=selections_df,
            )
            all_pred_records.extend(recs)
            cur_acc = np.mean([r["is_correct"] for r in recs]) * 100.0
            print(f" Done in {time.time() - t0:5.1f}s | Acc: {cur_acc:5.2f}%")

    # Stage 2: Evaluate top representations on reduced channel sets
    # We evaluate 'mean' and 'early_mid_late' (temporal dynamics) across pred_32, stab_32, pred_12
    print("\n[Step 4 - Stage 2] Evaluating Channel Subsets (pred_32, stab_32, pred_12) for Strongest Representations...")
    subset_conditions = ["stab_32", "pred_32", "pred_12"]
    subset_reps = ["mean", "early_mid_late"]
    subset_models = ["linear_svc", "shrinkage_lda", "logistic_regression"]

    for cond in subset_conditions:
        for rep_name in subset_reps:
            for mod_name in subset_models:
                t0 = time.time()
                print(f"  Evaluating [{cond:7s}] Rep: {rep_name:14s} | Model: {mod_name:19s} ...", end="", flush=True)
                recs = evaluate_loso_configuration(
                    X_feat=reps[rep_name],
                    y_trial=y_trial,
                    trial_meta=trial_meta,
                    condition=cond,
                    rep_name=rep_name,
                    model_name=mod_name,
                    channel_to_indices=channel_to_indices,
                    selections_df=selections_df,
                )
                all_pred_records.extend(recs)
                cur_acc = np.mean([r["is_correct"] for r in recs]) * 100.0
                print(f" Done in {time.time() - t0:5.1f}s | Acc: {cur_acc:5.2f}%")

    # Save full predictions table
    full_pred_df = pd.DataFrame(all_pred_records)
    pred_path = PREDICTIONS_DIR / "experiment_c_predictions.csv"
    full_pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved full predictions table ({len(full_pred_df)} rows) to: {pred_path}")

    # Step 5: Metrics and Comparisons
    print("\n[Step 5] Computing Summary, Subject, Class, and Comparison Metrics...")
    summary_df, subject_df, class_df, comparison_df = compute_comprehensive_metrics(full_pred_df)

    sum_path = TABLES_DIR / "experiment_c_summary.csv"
    sub_path = TABLES_DIR / "experiment_c_subject_metrics.csv"
    cls_path = TABLES_DIR / "experiment_c_class_metrics.csv"
    cmp_path = TABLES_DIR / "experiment_c_vs_exp_b_comparison.csv"

    summary_df.to_csv(sum_path, index=False)
    subject_df.to_csv(sub_path, index=False)
    class_df.to_csv(cls_path, index=False)
    comparison_df.to_csv(cmp_path, index=False)

    print(f"Saved summary metrics to: {sum_path}")
    print(f"Saved subject metrics to: {sub_path}")
    print(f"Saved class metrics to: {cls_path}")
    print(f"Saved comparison metrics to: {cmp_path}")

    # Step 6: Publication Figures
    print("\n[Step 6] Generating Publication Figures...")
    fig_comp_path = FIGURES_DIR / "experiment_c_representations_comparison.png"
    fig_cm_path = FIGURES_DIR / "experiment_c_confusion_matrix.png"
    plot_experiment_c_figures(summary_df, full_pred_df, fig_comp_path, fig_cm_path)

    elapsed = time.time() - t_start
    print(f"\nExperiment C Pipeline successfully completed in {elapsed:.1f}s ({elapsed / 60:.2f} min).")


if __name__ == "__main__":
    main()
