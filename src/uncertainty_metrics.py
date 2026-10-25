"""
Phase D: Uncertainty Metrics, Calibration Evaluation, and Error-Detection Scoring.

Metrics:
- Multiclass Brier score
- Multiclass Log Loss (NLL) with safe clipping
- Top-label Expected Calibration Error (ECE) and Classwise ECE
- Uncertainty Scores: u_max (1 - max_prob), predictive entropy, normalized entropy
- Error-Detection Performance: AUROC and AUPRC for predicting incorrect classifications
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score


def compute_uncertainty_scores(probs: np.ndarray, epsilon: float = 1e-15) -> Dict[str, np.ndarray]:
    """
    Compute uncertainty metrics from multiclass probability distributions.
    
    Parameters:
        probs: Array of shape (N, C) containing calibrated probabilities summing to 1.0.
        epsilon: Small constant for safe logarithm.
        
    Returns:
        dict with:
            'u_max': 1.0 - max(p), range [0, 1 - 1/C]. Higher = more uncertain.
            'entropy': -sum(p * log(p)), range [0, log(C)].
            'norm_entropy': entropy / log(C), range [0, 1].
    """
    n_samples, n_classes = probs.shape
    
    # 1. Maximum Probability Uncertainty
    max_p = np.max(probs, axis=1)
    u_max = 1.0 - max_p
    
    # 2. Predictive Entropy (safe handling of 0*log(0) -> 0)
    p_clipped = np.clip(probs, epsilon, 1.0)
    entropy = -np.sum(probs * np.log(p_clipped), axis=1)
    
    # 3. Normalized Entropy in [0, 1]
    max_entropy = np.log(n_classes) if n_classes > 1 else 1.0
    norm_entropy = entropy / max_entropy
    
    return {
        "u_max": u_max,
        "entropy": entropy,
        "norm_entropy": norm_entropy,
    }


def compute_expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    """
    Compute top-label ECE (confidence-based) and classwise ECE with equal-width binning.
    
    Parameters:
        y_true: Ground truth labels (N,).
        probs: Calibrated probabilities (N, C).
        n_bins: Number of equal-width bins in [0, 1].
        
    Returns:
        (top_label_ece, classwise_ece, bin_details)
    """
    n_samples, n_classes = probs.shape
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    
    # Top-label ECE
    top_ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Include upper boundary in last bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_count = np.sum(in_bin)
        bin_counts.append(bin_count)
        
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            top_ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            
    # Classwise ECE
    classwise_ece = 0.0
    for c in range(n_classes):
        class_y_true = (y_true == c).astype(float)
        class_probs = probs[:, c]
        class_ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            if i == n_bins - 1:
                in_bin = (class_probs >= bin_lower) & (class_probs <= bin_upper)
            else:
                in_bin = (class_probs >= bin_lower) & (class_probs < bin_upper)
            count = np.sum(in_bin)
            if count > 0:
                acc = np.mean(class_y_true[in_bin])
                conf = np.mean(class_probs[in_bin])
                class_ece += (count / n_samples) * np.abs(acc - conf)
        classwise_ece += class_ece / n_classes
        
    bin_details = {
        "bin_accs": np.array(bin_accs),
        "bin_confs": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
        "bin_centers": (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0,
    }
    
    return float(top_ece), float(classwise_ece), bin_details


def compute_multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Compute multiclass Brier score:
        Brier = 1/N * sum_{i=1}^N sum_{c=1}^C (p_ic - y_ic)^2
    where y_ic is 1 if y_i == c else 0.
    """
    n_samples, n_classes = probs.shape
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_probability_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute full probability calibration quality metrics.
    """
    n_samples, n_classes = probs.shape
    p_safe = np.clip(probs, 1e-15, 1.0 - 1e-15)
    p_safe /= np.sum(p_safe, axis=1, keepdims=True)
    
    # 1. Brier score
    brier = compute_multiclass_brier_score(y_true, p_safe)
    
    # 2. Multiclass Log Loss (NLL)
    try:
        nll = float(log_loss(y_true, p_safe, labels=list(range(n_classes))))
    except Exception:
        one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
        one_hot[np.arange(n_samples), y_true] = 1.0
        nll = float(-np.mean(np.sum(one_hot * np.log(p_safe), axis=1)))
        
    # 3. Expected Calibration Error
    top_ece, class_ece, _ = compute_expected_calibration_error(y_true, p_safe, n_bins=n_bins)
    
    return {
        "brier_score": brier,
        "log_loss": nll,
        "top_label_ece": top_ece,
        "classwise_ece": class_ece,
    }


def compute_error_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty_score: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate whether an uncertainty score reliably ranks incorrect predictions above correct ones.
    
    Target:
        error_indicator = (y_pred != y_true)  # 1 = error, 0 = correct
        
    Returns:
        dict with 'auroc_error', 'auprc_error' (Average Precision).
        If only one class is present in error_indicator (e.g. 100% correct or 100% incorrect),
        returns np.nan with safe handling.
    """
    is_error = (y_pred != y_true).astype(int)
    n_errors = np.sum(is_error)
    n_samples = len(y_true)
    
    # Degenerate edge case: all correct or all wrong
    if n_errors == 0 or n_errors == n_samples:
        return {
            "error_rate": float(n_errors / n_samples),
            "auroc_error": np.nan,
            "auprc_error": np.nan,
        }
        
    try:
        auroc = float(roc_auc_score(is_error, uncertainty_score))
    except ValueError:
        auroc = np.nan
        
    try:
        auprc = float(average_precision_score(is_error, uncertainty_score))
    except ValueError:
        auprc = np.nan
        
    return {
        "error_rate": float(n_errors / n_samples),
        "auroc_error": auroc,
        "auprc_error": auprc,
    }
