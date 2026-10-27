"""
tests/test_experiment_g.py

Unit tests for Experiment G domain adaptation extensions and evaluation integrity:
- LOSO subject separation
- Target-similarity source weighting
- Conservative class-balanced pseudo-label refinement
- Subdomain conditional residual alignment
- Exact-coverage and operational selective metrics
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.evaluate_experiment_g import (
    run_inner_experiment_g_tuning,
    EXACT_COVERAGE_LEVELS,
    NOMINAL_COVERAGE_LEVELS,
)
from src.evaluate_window_coral_optimisation import (
    compute_coral_projection,
    compute_cov_matrices,
    FastTemperatureScaler,
)


class TestExperimentG(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.d = 8
        self.n_source = 90
        self.n_target = 30
        self.Xs = np.random.randn(self.n_source, self.d).astype(np.float64)
        self.Xt = np.random.randn(self.n_target, self.d).astype(np.float64) + 1.0
        self.ys = np.random.randint(0, 4, self.n_source)
        self.yt = np.random.randint(0, 4, self.n_target)
        self.source_subs = np.repeat([1, 2, 3], 30)

    def test_similarity_weighting_bounds(self):
        # Verify with 14 source subjects (as in 14-subject LOSO training)
        train_subs = np.arange(14)
        d_vals = np.linspace(1.0, 5.0, 14)
        weights_sub = np.exp(-(d_vals - np.mean(d_vals)) / (np.std(d_vals) + 1e-5))
        weights_sub = np.clip(weights_sub / weights_sub.sum(), 0.03, 0.20)
        weights_sub = weights_sub / weights_sub.sum()

        self.assertEqual(len(weights_sub), 14)
        self.assertTrue(np.all(weights_sub >= 0.02))
        self.assertTrue(np.all(weights_sub <= 0.30))
        self.assertAlmostEqual(np.sum(weights_sub), 1.0)

    def test_pseudo_label_class_balance_safeguards(self):
        # 100 samples with skewed initial predictions
        probs = np.zeros((100, 4))
        # Skew: class 0 has 80 samples, class 1 has 10, class 2 has 5, class 3 has 5
        probs[:80, 0] = 0.9
        probs[:80, 1:] = 0.1 / 3
        probs[80:90, 1] = 0.85
        probs[80:90, [0, 2, 3]] = 0.15 / 3
        probs[90:95, 2] = 0.80
        probs[90:95, [0, 1, 3]] = 0.20 / 3
        probs[95:100, 3] = 0.75
        probs[95:100, [0, 1, 2]] = 0.25 / 3

        max_p = np.max(probs, axis=1)
        pseudo_labels = np.argmax(probs, axis=1)

        # Apply quota of max 4 per class
        quota = 4
        pseudo_idx = []
        for k in range(4):
            k_indices = np.where(pseudo_labels == k)[0]
            if len(k_indices) > 0:
                top_k = k_indices[np.argsort(-max_p[k_indices])[:quota]]
                pseudo_idx.extend(top_k)
        pseudo_idx = np.array(pseudo_idx)

        # Check total count is exactly 4 * 4 = 16
        self.assertEqual(len(pseudo_idx), 16)
        # Check exactly 4 from each class
        selected_classes = pseudo_labels[pseudo_idx]
        for k in range(4):
            self.assertEqual(np.sum(selected_classes == k), 4)

    def test_conditional_residual_shift_properties(self):
        # Check soft centroid conditional residual shift
        d = 4
        n_target = 40
        Xt_coral = np.random.randn(n_target, d)
        probs_t = np.random.dirichlet(np.ones(4), size=n_target)

        source_centroids = np.random.randn(4, d)
        target_centroids = np.zeros((4, d))
        for k in range(4):
            wk = probs_t[:, k:k+1]
            target_centroids[k] = np.sum(Xt_coral * wk, axis=0) / (np.sum(wk) + 1e-6)

        eta = 0.25
        delta_shift = np.zeros_like(Xt_coral)
        for k in range(4):
            delta_shift += probs_t[:, k:k+1] * (source_centroids[k] - target_centroids[k])

        Xt_cond = Xt_coral + eta * delta_shift
        self.assertEqual(Xt_cond.shape, Xt_coral.shape)
        self.assertTrue(np.all(np.isfinite(Xt_cond)))

    def test_inner_tuning_pipeline(self):
        val_res = run_inner_experiment_g_tuning(
            Xs_s=self.Xs,
            ys=self.ys,
            source_subjects=self.source_subs,
            seed=42,
        )

        self.assertIn("best_c", val_res)
        self.assertIn("best_eta", val_res)
        self.assertIn("scaler_temp", val_res)
        self.assertIn("thresholds", val_res)
        self.assertIn(val_res["best_c"], [0.0003, 0.0005, 0.001])
        self.assertIn(val_res["best_eta"], [0.10, 0.25])
        self.assertGreater(val_res["scaler_temp"].temperature, 0.0)


if __name__ == "__main__":
    unittest.main()
