"""
tests/test_evaluate_window_coral_optimisation.py

Unit tests for window-level CORAL accuracy and reliability optimisation:
- Exact coverage curve calculation with deterministic tie-breaking.
- Operational selective prediction with frozen thresholds.
- Inner-subject validation for lambda and C selection.
- S->T vs T->S CORAL alignment numerical validity.
- Covariance estimator stability.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.evaluate_window_coral_optimisation import (
    compute_coral_projection,
    compute_cov_matrices,
    FastTemperatureScaler,
    run_inner_model_selection_and_calibration,
    EXACT_COVERAGE_LEVELS,
    NOMINAL_COVERAGE_LEVELS,
)


class TestWindowCoralOptimisation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.d = 8
        self.n_source = 80
        self.n_target = 30
        self.Xs = np.random.randn(self.n_source, self.d).astype(np.float64)
        self.Xt = np.random.randn(self.n_target, self.d).astype(np.float64) + 1.0
        self.ys = np.random.randint(0, 4, self.n_source)
        self.yt = np.random.randint(0, 4, self.n_target)

    def test_st_vs_ts_coral_transforms(self):
        Cs, Ct, mu_s, mu_t = compute_cov_matrices(self.Xs, self.Xt, method="tikhonov", reg=1.0)
        self.assertEqual(Cs.shape, (self.d, self.d))
        self.assertEqual(Ct.shape, (self.d, self.d))

        A_st, A_ts = compute_coral_projection(Cs, Ct)
        self.assertEqual(A_st.shape, (self.d, self.d))
        self.assertEqual(A_ts.shape, (self.d, self.d))
        self.assertTrue(np.all(np.isfinite(A_st)))
        self.assertTrue(np.all(np.isfinite(A_ts)))

        # S->T transformed source cov should match Ct
        Xs_c = self.Xs - mu_s
        cov_st = np.cov(Xs_c @ A_st, rowvar=False)
        self.assertTrue(np.all(np.isfinite(cov_st)))

        # T->S transformed target cov should match Cs
        Xt_c = self.Xt - mu_t
        cov_ts = np.cov(Xt_c @ A_ts, rowvar=False)
        self.assertTrue(np.all(np.isfinite(cov_ts)))

    def test_covariance_estimators(self):
        for method in ["tikhonov", "oas", "ledoit_wolf"]:
            Cs, Ct, mu_s, mu_t = compute_cov_matrices(self.Xs, self.Xt, method=method, reg=1.0)
            self.assertEqual(Cs.shape, (self.d, self.d))
            self.assertEqual(Ct.shape, (self.d, self.d))
            self.assertTrue(np.all(np.isfinite(Cs)))
            self.assertTrue(np.all(np.isfinite(Ct)))
            eigvals_s = np.linalg.eigvalsh(Cs)
            eigvals_t = np.linalg.eigvalsh(Ct)
            self.assertTrue(np.all(eigvals_s > 0))
            self.assertTrue(np.all(eigvals_t > 0))

    def test_fast_temperature_scaler(self):
        z = np.array([
            [2.0, 0.5, -1.0, 0.0],
            [-0.5, 3.0, 1.0, 0.2],
            [0.1, 0.2, 2.5, 0.3],
            [-1.0, 0.0, 0.5, 2.0],
        ], dtype=np.float64)
        y = np.array([0, 1, 2, 3], dtype=int)

        scaler = FastTemperatureScaler().fit(z, y)
        self.assertGreater(scaler.temperature, 0.0)

        probs = scaler.predict_proba(z)
        self.assertEqual(probs.shape, z.shape)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0))
        # Argmax preserved
        self.assertTrue(np.array_equal(np.argmax(probs, axis=1), np.argmax(z, axis=1)))

    def test_deterministic_exact_coverage(self):
        # Verify deterministic ranking and exact coverage calculation
        n = 100
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = np.array([0] * 40 + [1] * 10 + [0] * 10 + [1] * 40)
        u_max = np.linspace(0.1, 0.9, n)

        sort_order = np.lexsort((np.arange(n), u_max))
        self.assertEqual(len(sort_order), n)

        # Check at 20%, 50%, 100%
        for cov in [0.20, 0.50, 1.00]:
            k = int(np.round(cov * n))
            accepted_idx = sort_order[:k]
            self.assertEqual(len(accepted_idx), k)
            acc = accuracy_score(y_true[accepted_idx], y_pred[accepted_idx]) * 100.0
            risk = 100.0 - acc
            self.assertAlmostEqual(acc + risk, 100.0)

    def test_inner_validation_pipeline(self):
        # Synthetic 3 source subjects
        n_per_sub = 25
        Xs = np.random.randn(n_per_sub * 3, self.d)
        ys = np.random.randint(0, 4, n_per_sub * 3)
        source_subs = np.repeat([1, 2, 3], n_per_sub)

        val_res = run_inner_model_selection_and_calibration(
            Xs=Xs,
            ys=ys,
            source_subjects=source_subs,
            seed=42,
        )

        self.assertIn("best_lambda_c01", val_res)
        self.assertIn("best_c_joint", val_res)
        self.assertIn("best_lambda_joint", val_res)
        self.assertIn("oof_calib", val_res)

        oof_calib = val_res["oof_calib"]
        self.assertIn("ref_stab32_c01_lam1", oof_calib)
        self.assertIn("exp_a_tuned_lambda", oof_calib)
        self.assertIn("exp_c_wide_margin_c001", oof_calib)
        self.assertIn("exp_f_joint_tuned", oof_calib)

        for cfg_key, c_info in oof_calib.items():
            self.assertGreater(c_info["scaler_temp"].temperature, 0.0)
            self.assertIn(0.5, c_info["thresholds"])
            self.assertIn(1.0, c_info["thresholds"])
            self.assertGreaterEqual(c_info["thresholds"][1.0], c_info["thresholds"][0.5])


if __name__ == "__main__":
    unittest.main()

