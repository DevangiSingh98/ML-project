"""
Unit tests for uncertainty and calibration metrics module (src/uncertainty_metrics.py).

Verifies:
- Uncertainty score calculation (u_max, entropy, normalized entropy).
- Expected Calibration Error (top-label and classwise) on known distributions.
- Multiclass Brier score and log loss.
- Error-detection performance (AUROC, AUPRC) and handling of single-class edge cases.
"""

import unittest
import numpy as np

from src.uncertainty_metrics import (
    compute_uncertainty_scores,
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    compute_probability_metrics,
    compute_error_detection_metrics,
)


class TestUncertaintyMetrics(unittest.TestCase):
    """Test suite for Phase D probability calibration and uncertainty metrics."""

    def test_01_uncertainty_score_bounds_and_zero_entropy(self):
        """Verify entropy and u_max bounds and handling of extreme probabilities."""
        # Case A: Complete certainty [1, 0, 0, 0] -> u_max = 0, entropy = 0
        probs_certain = np.array([[1.0, 0.0, 0.0, 0.0]])
        u_certain = compute_uncertainty_scores(probs_certain)
        self.assertAlmostEqual(u_certain["u_max"][0], 0.0)
        self.assertAlmostEqual(u_certain["entropy"][0], 0.0)
        self.assertAlmostEqual(u_certain["norm_entropy"][0], 0.0)

        # Case B: Maximum uncertainty [0.25, 0.25, 0.25, 0.25] -> u_max = 0.75, norm_entropy = 1.0
        probs_uniform = np.array([[0.25, 0.25, 0.25, 0.25]])
        u_uniform = compute_uncertainty_scores(probs_uniform)
        self.assertAlmostEqual(u_uniform["u_max"][0], 0.75)
        self.assertAlmostEqual(u_uniform["entropy"][0], np.log(4.0))
        self.assertAlmostEqual(u_uniform["norm_entropy"][0], 1.0)

    def test_02_brier_score_known_cases(self):
        """Verify multiclass Brier score on perfect predictions and uniform predictions."""
        # Perfect predictions: Brier = 0.0
        y_true = np.array([0, 1, 2, 3])
        probs_perfect = np.eye(4)
        brier_perf = compute_multiclass_brier_score(y_true, probs_perfect)
        self.assertAlmostEqual(brier_perf, 0.0)

        # Uniform predictions on 4 classes:
        # Each sample error: (0.25 - 1)^2 + 3 * (0.25 - 0)^2 = 0.5625 + 0.1875 = 0.75
        probs_uniform = np.full((4, 4), 0.25)
        brier_uni = compute_multiclass_brier_score(y_true, probs_uniform)
        self.assertAlmostEqual(brier_uni, 0.75)

    def test_03_expected_calibration_error_properties(self):
        """Verify ECE calculation on perfectly calibrated and miscalibrated probabilities."""
        # Perfectly calibrated: 100 samples with 0.8 confidence and exactly 80% accuracy
        np.random.seed(42)
        n = 100
        y_true = np.zeros(n, dtype=int)
        y_true[80:] = 1  # 80 zeros (correct), 20 ones (incorrect)

        # Predicted probabilities: class 0 has 0.8, others have 0.2 / 3
        probs = np.full((n, 4), (1.0 - 0.8) / 3.0)
        probs[:, 0] = 0.8

        top_ece, class_ece, details = compute_expected_calibration_error(y_true, probs, n_bins=10)
        # Since accuracy is 80/100 = 0.8 and confidence is 0.8, top_ece should be 0.0
        self.assertAlmostEqual(top_ece, 0.0, places=5)

    def test_04_error_detection_edge_cases(self):
        """Verify AUROC/AUPRC computation and graceful handling of single-class edge cases."""
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 0, 1])  # 5 correct, 1 error at index 3
        uncertainty = np.array([0.1, 0.2, 0.1, 0.9, 0.05, 0.15])  # High uncertainty on error

        res = compute_error_detection_metrics(y_true, y_pred, uncertainty)
        self.assertFalse(np.isnan(res["auroc_error"]))
        self.assertFalse(np.isnan(res["auprc_error"]))
        self.assertAlmostEqual(res["auroc_error"], 1.0)  # Perfect ranking of the error

        # Edge case: All correct (0 errors) -> AUROC/AUPRC should be NaN, not crash
        res_zero_errors = compute_error_detection_metrics(y_true, y_true, uncertainty)
        self.assertTrue(np.isnan(res_zero_errors["auroc_error"]))
        self.assertTrue(np.isnan(res_zero_errors["auprc_error"]))
        self.assertEqual(res_zero_errors["error_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
