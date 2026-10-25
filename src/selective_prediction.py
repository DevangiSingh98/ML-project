"""
Phase E: Selective Prediction and Uncertainty-Based Abstention for Multiclass Emotion Recognition.

Mathematical Foundations & Guarantees:
1. Maximum Probability Uncertainty:
       u_max(p) = 1.0 - max_{k} p_k,   range: [0, 1 - 1/K]
2. Predictive Shannon Entropy:
       H(p) = - sum_{k} p_k * ln(p_k), range: [0, ln(K)]
   Safely evaluates 0 * ln(0) -> 0 without producing NaN or Inf.
3. Monotonic Abstention:
       accept(x) <=> uncertainty(x) <= tau
   Higher uncertainty strictly decreases or maintains acceptance probability;
   higher uncertainty never makes a prediction more likely to be accepted.
4. Selective Metrics:
       Coverage = N_accepted / N_total
       Selective Risk = N_incorrect_accepted / N_accepted
       Selective Accuracy = N_correct_accepted / N_accepted
   For any non-empty accepted set: Selective Accuracy == 1.0 - Selective Risk.
   For an empty accepted set: returns np.nan (explicitly undefined, never 0 or 1).
5. Area Under the Risk-Coverage Curve (AURC):
       AURC = int_{c_min}^{1} Risk(c) dc
   Evaluated using the trapezoidal rule over actual achieved coverages.
6. Random Rejection Baseline:
       E[Acc_random(c)] = Baseline Accuracy (constant across all coverage levels).
       Simulated via label-agnostic uniform random draws without replacement.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def _trapezoid_integration(y: np.ndarray, x: np.ndarray) -> float:
    """Internal trapezoidal integration compatible across NumPy 1.x and 2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    # Fallback for older scipy/numpy environments
    from scipy.integrate import trapezoid
    return float(trapezoid(y, x))


def validate_probability_array(probs: np.ndarray, atol: float = 1e-3) -> np.ndarray:
    """
    Validate that an input array represents a valid 2D multiclass probability distribution.
    
    Parameters:
        probs: Array of shape (N, K).
        atol: Absolute tolerance for row-sum normalization check.
        
    Returns:
        np.ndarray: Validated 2D float64 array.
        
    Raises:
        ValueError: If probs is not 2D, contains negative values, or rows do not sum to 1.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Probability array must be 2-dimensional (N, K), got shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] < 2:
        raise ValueError(f"Probability array must have at least 1 sample and 2 classes, got shape {arr.shape}.")
    if np.any(arr < -1e-7):
        raise ValueError("Probability array contains negative values.")
    
    # Clip tiny negative numerical artifacts to 0
    arr = np.clip(arr, 0.0, 1.0)
    row_sums = np.sum(arr, axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(f"Rows of probability array do not sum to 1.0 within tolerance {atol}.")
    
    # Re-normalize rows to ensure exact sum == 1.0
    arr = arr / np.sum(arr, axis=1, keepdims=True)
    return arr


def compute_max_probability_uncertainty(probs: np.ndarray) -> np.ndarray:
    """
    Compute maximum-probability uncertainty: u_max = 1.0 - max_k p_k.
    
    Parameters:
        probs: Array of shape (N, K) containing probabilities summing to 1.0.
        
    Returns:
        u_max: 1D array of shape (N,), with values in [0.0, 1.0 - 1/K].
               0.0 corresponds to complete certainty (one-hot probability).
    """
    valid_probs = validate_probability_array(probs)
    max_p = np.max(valid_probs, axis=1)
    return 1.0 - max_p


def compute_predictive_entropy(
    probs: np.ndarray,
    base: float = np.e,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute Shannon predictive entropy: H(p) = - sum_{k=1}^K p_k * log_base(p_k).
    
    Safely evaluates zero probabilities using the continuous limit: lim_{p->0} p * log(p) = 0.
    Produces zero NaNs and zero infinities for valid probability inputs.
    
    Parameters:
        probs: Array of shape (N, K) containing probabilities summing to 1.0.
        base: Logarithm base (default: e for natural logarithm in nats).
        normalize: If True, divides by log_base(K) to scale entropy strictly into [0.0, 1.0].
        
    Returns:
        entropy: 1D array of shape (N,).
    """
    if base <= 1.0:
        raise ValueError(f"Logarithm base must be strictly greater than 1.0, got {base}.")
    
    valid_probs = validate_probability_array(probs)
    n_classes = valid_probs.shape[1]
    
    # Safe computation: evaluate only where p > 0
    log_probs = np.zeros_like(valid_probs)
    positive_mask = valid_probs > 0.0
    log_probs[positive_mask] = np.log(valid_probs[positive_mask]) / np.log(base)
    
    entropy = -np.sum(valid_probs * log_probs, axis=1)
    # Clip tiny negative numerical noise
    entropy = np.maximum(entropy, 0.0)
    
    if normalize and n_classes > 1:
        max_entropy = np.log(n_classes) / np.log(base)
        entropy = np.clip(entropy / max_entropy, 0.0, 1.0)
        
    return entropy


def get_acceptance_mask(
    uncertainties: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Determine which predictions are accepted given an uncertainty score and an acceptance threshold.
    
    Rule:
        accept(x) = True  <=>  uncertainty(x) <= threshold
        
    Properties:
    - Lower uncertainty implies higher model confidence, so it is accepted first.
    - Higher uncertainty strictly reduces or maintains acceptance.
    - Samples with uncertainty equal to threshold are accepted (standard threshold inclusion).
    
    Parameters:
        uncertainties: 1D array of shape (N,) containing non-negative uncertainty scores.
        threshold: Real-valued scalar acceptance threshold.
        
    Returns:
        accepted_mask: 1D boolean array of shape (N,), True for accepted samples.
    """
    u_arr = np.asarray(uncertainties, dtype=np.float64)
    if u_arr.ndim != 1:
        raise ValueError(f"Uncertainties must be a 1D array, got shape {u_arr.shape}.")
    if np.isposinf(threshold) or threshold >= 1.0:
        # Explicit 100% coverage rule: unconditionally accept all eligible predictions
        return np.ones(len(u_arr), dtype=bool)
    if np.isneginf(threshold) or threshold < 0.0:
        return np.zeros(len(u_arr), dtype=bool)
    return u_arr <= threshold


def compute_coverage(accepted_mask: np.ndarray) -> float:
    """
    Compute empirical coverage: fraction of samples accepted by the abstention policy.
    
    Coverage = N_accepted / N_total
    
    Parameters:
        accepted_mask: 1D boolean array indicating accepted predictions.
        
    Returns:
        coverage: Float in [0.0, 1.0].
    """
    mask = np.asarray(accepted_mask, dtype=bool)
    if len(mask) == 0:
        raise ValueError("Cannot compute coverage on an empty mask.")
    return float(np.mean(mask))


def compute_selective_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accepted_mask: np.ndarray,
) -> float:
    """
    Compute selective classification accuracy on the subset of accepted predictions.
    
    Selective Accuracy = N_correct_accepted / N_accepted
    
    Defined behavior for empty accepted set:
    - If accepted_count == 0: Returns np.nan.
      (Never silently reports 0.0 or 1.0 when zero predictions were made).
      
    Parameters:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        accepted_mask: Boolean mask of accepted predictions, shape (N,).
        
    Returns:
        float: Accuracy on accepted samples in [0.0, 1.0], or np.nan if N_accepted == 0.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    mask = np.asarray(accepted_mask, dtype=bool)
    
    if len(y_t) != len(y_p) or len(y_t) != len(mask):
        raise ValueError(
            f"Shape mismatch: len(y_true)={len(y_t)}, len(y_pred)={len(y_p)}, len(mask)={len(mask)}."
        )
    
    n_accepted = int(np.sum(mask))
    if n_accepted == 0:
        return float(np.nan)
    
    n_correct_accepted = int(np.sum((y_p[mask] == y_t[mask])))
    return float(n_correct_accepted / n_accepted)


def compute_selective_risk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accepted_mask: np.ndarray,
) -> float:
    """
    Compute selective classification risk (error rate) on accepted predictions.
    
    Selective Risk = N_incorrect_accepted / N_accepted
    
    For any non-empty accepted set: Selective Risk == 1.0 - Selective Accuracy.
    For an empty accepted set: returns np.nan.
    
    Parameters:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        accepted_mask: Boolean mask of accepted predictions, shape (N,).
        
    Returns:
        float: Error rate on accepted samples in [0.0, 1.0], or np.nan if N_accepted == 0.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    mask = np.asarray(accepted_mask, dtype=bool)
    
    if len(y_t) != len(y_p) or len(y_t) != len(mask):
        raise ValueError(
            f"Shape mismatch: len(y_true)={len(y_t)}, len(y_pred)={len(y_p)}, len(mask)={len(mask)}."
        )
    
    n_accepted = int(np.sum(mask))
    if n_accepted == 0:
        return float(np.nan)
    
    n_incorrect_accepted = int(np.sum((y_p[mask] != y_t[mask])))
    return float(n_incorrect_accepted / n_accepted)


def compute_risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    target_coverages: Optional[List[float]] = None,
    include_full_range: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute the empirical risk-coverage curve by sweeping uncertainty acceptance thresholds.
    
    Requirements & Handling of Tied Uncertainty Scores:
    - When multiple samples have tied uncertainty scores, thresholding at that score accepts all
      tied samples simultaneously. Samples with identical scores are never arbitrarily split.
    - If target_coverages is specified, actual achieved coverage is reported. The policy does NOT
      fabricate artificial intermediate coverage points when ties prevent exact matching.
    - Guaranteed full-coverage endpoint (coverage == 1.0, threshold == max(uncertainties) or np.inf).
    - Monotonically non-decreasing coverage array sorted ascendingly.
    
    Parameters:
        y_true: Ground truth class labels, shape (N,).
        y_pred: Model predictions, shape (N,).
        uncertainties: Uncertainty scores (higher = more uncertain), shape (N,).
        target_coverages: Optional list of target coverage levels (e.g. [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]).
                          If None, sweeps all unique uncertainty thresholds present in the data.
        include_full_range: If True, ensures the 100% full-coverage baseline is included.
        
    Returns:
        dict containing:
            'thresholds': Threshold values used, shape (M,)
            'coverages': Actual achieved coverage values in [0.0, 1.0], shape (M,)
            'selective_risks': Selective error rates on accepted predictions, shape (M,)
            'selective_accuracies': Selective accuracy on accepted predictions, shape (M,)
            'accepted_counts': Absolute count of accepted samples, shape (M,)
            'total_samples': Total number of samples N
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    u = np.asarray(uncertainties, dtype=np.float64)
    n_total = len(y_t)
    
    if len(y_p) != n_total or len(u) != n_total:
        raise ValueError(f"Array length mismatch: y_true={n_total}, y_pred={len(y_p)}, u={len(u)}.")
    if n_total == 0:
        raise ValueError("Cannot compute risk-coverage curve on an empty sample set.")
        
    # Determine thresholds to evaluate
    if target_coverages is not None:
        # Determine empirical quantile thresholds corresponding to target coverages
        sorted_targets = sorted(list(set(target_coverages)), reverse=True)
        thresholds_to_eval = []
        for cov in sorted_targets:
            if cov <= 0.0 or cov > 1.0:
                continue
            if cov >= 1.0:
                thresholds_to_eval.append(float(np.inf))
            else:
                # Quantile threshold: accept lowest (cov * 100)% of uncertainty
                th = float(np.percentile(u, cov * 100.0, method="linear"))
                thresholds_to_eval.append(th)
    else:
        # Sweep all unique uncertainty thresholds
        unique_u = np.unique(u)
        thresholds_to_eval = [float(val) for val in unique_u]
        
    if include_full_range:
        if float(np.inf) not in thresholds_to_eval:
            thresholds_to_eval.append(float(np.inf))
            
    # Sort thresholds ascendingly so coverage increases monotonically
    thresholds_to_eval = sorted(list(set(thresholds_to_eval)))
    
    res_thresh = []
    res_cov = []
    res_risk = []
    res_acc = []
    res_count = []
    
    for th in thresholds_to_eval:
        mask = get_acceptance_mask(u, th)
        n_acc = int(np.sum(mask))
        if n_acc == 0:
            continue
        cov = n_acc / n_total
        risk = compute_selective_risk(y_t, y_p, mask)
        acc = compute_selective_accuracy(y_t, y_p, mask)
        
        res_thresh.append(th)
        res_cov.append(cov)
        res_risk.append(risk)
        res_acc.append(acc)
        res_count.append(n_acc)
        
    # Ensure points are ordered by coverage ascending
    cov_arr = np.array(res_cov, dtype=np.float64)
    sort_idx = np.argsort(cov_arr)
    
    return {
        "thresholds": np.array(res_thresh, dtype=np.float64)[sort_idx],
        "coverages": cov_arr[sort_idx],
        "selective_risks": np.array(res_risk, dtype=np.float64)[sort_idx],
        "selective_accuracies": np.array(res_acc, dtype=np.float64)[sort_idx],
        "accepted_counts": np.array(res_count, dtype=np.int64)[sort_idx],
        "total_samples": n_total,
    }


def compute_partial_aurc(
    coverages: np.ndarray,
    selective_risks: np.ndarray,
    min_coverage: float = 0.5,
    max_coverage: float = 1.0,
) -> float:
    """
    Compute Partial Area Under the Risk-Coverage Curve (Partial-AURC) over [min_coverage, max_coverage].
    
    Formula:
        Partial_AURC = int_{min_coverage}^{max_coverage} Selective_Risk(c) dc
        
    Integration Convention:
    - Filters valid finite points.
    - Sorts strictly by coverage ascending, averaging duplicate coverages.
    - Clamps/interpolates to exact integration bounds [min_coverage, max_coverage]:
      If the minimum observed coverage c_obs_min > min_coverage, risk is extended horizontally
      from c_obs_min down to min_coverage: r(c) = r(c_obs_min) for c in [min_coverage, c_obs_min].
      If the maximum observed coverage c_obs_max < max_coverage, risk is extended horizontally
      up to max_coverage: r(c) = r(c_obs_max) for c in [c_obs_max, max_coverage].
    - Evaluates the trapezoidal integral over the exact interval [min_coverage, max_coverage].
    - Total interval width is strictly (max_coverage - min_coverage), e.g., 0.5 for [0.5, 1.0].
    
    Parameters:
        coverages: 1D array of coverage values in [0.0, 1.0].
        selective_risks: 1D array of selective risks in [0.0, 1.0].
        min_coverage: Lower integration bound (default: 0.5).
        max_coverage: Upper integration bound (default: 1.0).
        
    Returns:
        float: Partial AURC over the specified coverage interval.
    """
    if min_coverage >= max_coverage:
        raise ValueError(f"min_coverage ({min_coverage}) must be strictly less than max_coverage ({max_coverage}).")
        
    c_arr = np.asarray(coverages, dtype=np.float64)
    r_arr = np.asarray(selective_risks, dtype=np.float64)
    
    if len(c_arr) != len(r_arr):
        raise ValueError(f"Array length mismatch: coverages={len(c_arr)}, risks={len(r_arr)}.")
    if len(c_arr) < 1:
        raise ValueError("At least 1 point is required to compute partial AURC.")
        
    valid_mask = np.isfinite(c_arr) & np.isfinite(r_arr)
    if np.sum(valid_mask) < 1:
        raise ValueError("No valid finite points available.")
        
    c_valid = c_arr[valid_mask]
    r_valid = r_arr[valid_mask]
    
    # Sort ascending and average duplicates
    sort_idx = np.argsort(c_valid)
    c_sorted = c_valid[sort_idx]
    r_sorted = r_valid[sort_idx]
    
    unique_c, inv_idx = np.unique(c_sorted, return_inverse=True)
    if len(unique_c) < len(c_sorted):
        unique_r = np.zeros_like(unique_c)
        for i in range(len(unique_c)):
            unique_r[i] = np.mean(r_sorted[inv_idx == i])
        c_sorted, r_sorted = unique_c, unique_r
        
    # Filter points within or bracketing [min_coverage, max_coverage]
    # Build curve strictly across [min_coverage, max_coverage]
    c_pts = []
    r_pts = []
    
    # Left boundary extension/interpolation
    if c_sorted[0] > min_coverage:
        c_pts.append(float(min_coverage))
        r_pts.append(float(r_sorted[0]))
    elif c_sorted[0] < min_coverage:
        # Interpolate risk at min_coverage
        r_at_min = float(np.interp(min_coverage, c_sorted, r_sorted))
        c_pts.append(float(min_coverage))
        r_pts.append(r_at_min)
        
    for c, r in zip(c_sorted, r_sorted):
        if min_coverage <= c <= max_coverage:
            if not c_pts or c > c_pts[-1]:
                c_pts.append(float(c))
                r_pts.append(float(r))
                
    # Right boundary extension/interpolation
    if c_pts[-1] < max_coverage:
        if c_sorted[-1] < max_coverage:
            c_pts.append(float(max_coverage))
            r_pts.append(float(r_sorted[-1]))
        else:
            r_at_max = float(np.interp(max_coverage, c_sorted, r_sorted))
            c_pts.append(float(max_coverage))
            r_pts.append(r_at_max)
            
    return _trapezoid_integration(np.array(r_pts), np.array(c_pts))


def compute_full_aurc(
    coverages: np.ndarray,
    selective_risks: np.ndarray,
) -> float:
    """
    Compute Full-Range Area Under the Risk-Coverage Curve (Full-AURC) over [0.0, 1.0].
    
    Formula:
        Full_AURC = int_0^1 Selective_Risk(c) dc
        
    Integration Convention for Empty Accepted Set:
    - At c = 0.0 (empty accepted set), selective risk is empirically undefined (0/0).
    - Standard selective classification convention (Geifman & El-Yaniv):
      The risk at the lowest observed non-empty coverage c_min is extended horizontally down to c = 0.0:
          r(c) = r(c_min) for c in [0, c_min].
      This evaluates int_0^{c_min} r(c) dc = r(c_min) * c_min.
    - If c_max < 1.0, risk is extended horizontally up to c = 1.0:
          r(c) = r(c_max) for c in [c_max, 1.0].
    - Evaluates the trapezoidal integral over the full [0.0, 1.0] domain (interval width = 1.0).
    
    Parameters:
        coverages: 1D array of coverage values in [0.0, 1.0].
        selective_risks: 1D array of selective risks in [0.0, 1.0].
        
    Returns:
        float: Full-range AURC in [0.0, 1.0].
    """
    return compute_partial_aurc(
        coverages=coverages,
        selective_risks=selective_risks,
        min_coverage=0.0,
        max_coverage=1.0,
    )


def compute_aurc(
    coverages: np.ndarray,
    selective_risks: np.ndarray,
    min_coverage: float = 0.0,
) -> float:
    """
    Compute Area Under the Risk-Coverage Curve (AURC).
    
    Important Note on Terminology and Integration Domain:
    - If min_coverage == 0.0: This evaluates the raw trapezoid integral over the observed points
      c >= 0.0. To compute canonical full-range AURC with the standard horizontal left-extension
      to c=0.0, use compute_full_aurc().
    - If min_coverage > 0.0 (e.g. 0.5): This evaluates partial AURC over the observed points
      c >= min_coverage. For exact bounded partial AURC over [min_coverage, 1.0], use
      compute_partial_aurc().
    
    Parameters:
        coverages: 1D array of coverage values in [0.0, 1.0].
        selective_risks: 1D array of selective risks in [0.0, 1.0].
        min_coverage: Lower integration bound (default: 0.0).
        
    Returns:
        aurc: Float representing raw trapezoidal area under the risk-coverage curve.
    """
    c_arr = np.asarray(coverages, dtype=np.float64)
    r_arr = np.asarray(selective_risks, dtype=np.float64)
    
    if len(c_arr) != len(r_arr):
        raise ValueError(f"Array length mismatch: coverages={len(c_arr)}, risks={len(r_arr)}.")
    if len(c_arr) < 2:
        raise ValueError("At least 2 points are required to compute AURC.")
        
    # Filter valid finite points above min_coverage
    valid_mask = np.isfinite(c_arr) & np.isfinite(r_arr) & (c_arr >= min_coverage)
    if np.sum(valid_mask) < 2:
        raise ValueError(f"Fewer than 2 valid points available above min_coverage={min_coverage}.")
        
    c_valid = c_arr[valid_mask]
    r_valid = r_arr[valid_mask]
    
    # Sort strictly by coverage ascending
    sort_idx = np.argsort(c_valid)
    c_sorted = c_valid[sort_idx]
    r_sorted = r_valid[sort_idx]
    
    # If duplicate coverages exist due to discrete evaluation, average the risks
    unique_c, inv_idx = np.unique(c_sorted, return_inverse=True)
    if len(unique_c) < len(c_sorted):
        unique_r = np.zeros_like(unique_c)
        for i in range(len(unique_c)):
            unique_r[i] = np.mean(r_sorted[inv_idx == i])
        c_sorted, r_sorted = unique_c, unique_r
        
    return _trapezoid_integration(r_sorted, c_sorted)


def compute_random_rejection_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coverages: List[float],
    n_repeats: int = 50,
    seed: Optional[int] = 42,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute a statistically defensible Random Rejection baseline.
    
    Statistical Properties:
    - In expectation, uniform random rejection without replacement has a selective accuracy
      strictly equal to full-coverage baseline accuracy:
          E[Acc_random(c)] = Acc_full
          E[Risk_random(c)] = Risk_full = 1.0 - Acc_full
      for all coverage levels c in (0.0, 1.0].
    - Random acceptance NEVER inspects labels when choosing accepted samples. Labels are only
      used post-hoc to evaluate accuracy on the randomly accepted subset.
    - Monte Carlo sampling across `n_repeats` estimates empirical sampling variance.
    
    Parameters:
        y_true: Ground truth class labels, shape (N,).
        y_pred: Predicted class labels, shape (N,).
        coverages: List of target coverage levels (e.g. [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]).
        n_repeats: Number of Monte Carlo random rejection iterations per coverage level.
        seed: Random seed for reproducible sampling.
        
    Returns:
        dict containing:
            'target_coverages': Array of requested coverages
            'actual_coverages': Array of actual sample ratios (M / N)
            'expected_accuracy': Analytical expected accuracy (constant scalar)
            'expected_risk': Analytical expected risk (constant scalar)
            'empirical_accuracy_mean': Mean accuracy across Monte Carlo repeats, shape (len(coverages),)
            'empirical_accuracy_std': Standard deviation of accuracy across repeats, shape (len(coverages),)
            'empirical_risk_mean': Mean selective risk across repeats, shape (len(coverages),)
            'empirical_risk_std': Standard deviation of selective risk across repeats, shape (len(coverages),)
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    n_total = len(y_t)
    
    if len(y_p) != n_total or n_total == 0:
        raise ValueError(f"Invalid input arrays: y_true={len(y_t)}, y_pred={len(y_p)}.")
        
    full_acc = float(np.mean(y_p == y_t))
    full_risk = float(1.0 - full_acc)
    
    rng = np.random.RandomState(seed)
    
    target_covs = np.array(sorted(coverages, reverse=True), dtype=np.float64)
    actual_covs = []
    mean_accs = []
    std_accs = []
    mean_risks = []
    std_risks = []
    
    for cov in target_covs:
        if cov <= 0.0 or cov > 1.0:
            raise ValueError(f"Coverage must be in (0.0, 1.0], got {cov}.")
            
        m_samples = max(1, int(np.round(cov * n_total)))
        actual_covs.append(m_samples / n_total)
        
        if m_samples == n_total:
            # Full coverage is deterministic
            mean_accs.append(full_acc)
            std_accs.append(0.0)
            mean_risks.append(full_risk)
            std_risks.append(0.0)
            continue
            
        rep_accs = []
        for _ in range(n_repeats):
            # Uniform random sampling WITHOUT inspecting labels
            accepted_indices = rng.choice(n_total, size=m_samples, replace=False)
            acc_rep = np.mean(y_p[accepted_indices] == y_t[accepted_indices])
            rep_accs.append(acc_rep)
            
        rep_accs = np.array(rep_accs, dtype=np.float64)
        mean_accs.append(float(np.mean(rep_accs)))
        std_accs.append(float(np.std(rep_accs, ddof=1)) if n_repeats > 1 else 0.0)
        mean_risks.append(float(1.0 - np.mean(rep_accs)))
        std_risks.append(float(np.std(1.0 - rep_accs, ddof=1)) if n_repeats > 1 else 0.0)
        
    return {
        "target_coverages": target_covs,
        "actual_coverages": np.array(actual_covs, dtype=np.float64),
        "expected_accuracy": full_acc,
        "expected_risk": full_risk,
        "empirical_accuracy_mean": np.array(mean_accs, dtype=np.float64),
        "empirical_accuracy_std": np.array(std_accs, dtype=np.float64),
        "empirical_risk_mean": np.array(mean_risks, dtype=np.float64),
        "empirical_risk_std": np.array(std_risks, dtype=np.float64),
    }


def determine_threshold_from_training_budget(
    uncertainties_train: np.ndarray,
    target_coverage: float,
) -> float:
    """
    Determine an operational uncertainty threshold strictly from training/OOF uncertainty distribution.
    
    Guarantees:
    - Held-out test subjects are NEVER observed when selecting this threshold.
    - Selects the empirical target_coverage quantile of training uncertainty.
    
    Parameters:
        uncertainties_train: 1D array of uncertainty scores from training or inner OOF data.
        target_coverage: Float in (0.0, 1.0].
        
    Returns:
        float: Threshold tau such that approximately target_coverage fraction of training
               samples satisfy u <= tau.
    """
    if target_coverage <= 0.0 or target_coverage > 1.0:
        raise ValueError(f"Target coverage must be in (0.0, 1.0], got {target_coverage}.")
    u_tr = np.asarray(uncertainties_train, dtype=np.float64)
    if len(u_tr) == 0:
        raise ValueError("Cannot determine threshold on empty training uncertainties.")
        
    if target_coverage >= 1.0:
        # Explicit 100% coverage rule: no-abstention reference case.
        # Returning np.inf guarantees all eligible test predictions are accepted regardless
        # of any test-set distribution shift or extreme test uncertainties.
        return float(np.inf)
    return float(np.percentile(u_tr, target_coverage * 100.0, method="linear"))
