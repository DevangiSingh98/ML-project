"""
tests/test_evaluate_window_coral.py

Unit tests for window-level CORAL domain adaptation, covariance shrinkage,
temperature scaling calibration, and uncertainty-aware unreliability prediction.
"""

import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.evaluate_window_coral import (
    compute_coral_projection,
    compute_covariance_estimates,
    compute_uncertainty_metrics_dict,
    TemperatureScaler,
    run_inner_validation_for_all_methods,
)


class TestEvaluateWindowCORAL(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_source = 100
        self.n_target = 30
        self.d = 10
        self.Xs = np.random.randn(self.n_source, self.d).astype(np.float64)
        self.Xt = np.random.randn(self.n_target, self.d).astype(np.float64) + 1.5

    def test_compute_coral_projection_properties(self):
        Cs = np.cov(self.Xs, rowvar=False) + 1.0 * np.eye(self.d)
        Ct = np.cov(self.Xt, rowvar=False) + 1.0 * np.eye(self.d)

        A_st, A_ts = compute_coral_projection(Cs, Ct)

        self.assertEqual(A_st.shape, (self.d, self.d))
        self.assertEqual(A_ts.shape, (self.d, self.d))
        self.assertTrue(np.all(np.isfinite(A_st)))
        self.assertTrue(np.all(np.isfinite(A_ts)))

        # Alignment check: Cov((Xt - mu_t) @ A_ts) should approximately equal Cs
        mu_t = self.Xt.mean(axis=0)
        Xt_c = self.Xt - mu_t
        aligned_cov = np.cov(Xt_c @ A_ts, rowvar=False)
        self.assertTrue(np.all(np.isfinite(aligned_cov)))

    def test_covariance_estimates_methods(self):
        for method in ["tikhonov", "ledoit_wolf", "oas"]:
            Cs, Ct, mu_s, mu_t, rho_s, rho_t = compute_covariance_estimates(
                self.Xs, self.Xt, method=method, reg=1.0
            )
            self.assertEqual(Cs.shape, (self.d, self.d))
            self.assertEqual(Ct.shape, (self.d, self.d))
            self.assertTrue(np.all(np.isfinite(Cs)))
            self.assertTrue(np.all(np.isfinite(Ct)))
            self.assertTrue(np.allclose(Cs, Cs.T, atol=1e-5))
            self.assertTrue(np.allclose(Ct, Ct.T, atol=1e-5))
            # Eigenvalues must be positive
            eigvals_s = np.linalg.eigvalsh(Cs)
            eigvals_t = np.linalg.eigvalsh(Ct)
            self.assertTrue(np.all(eigvals_s > 0))
            self.assertTrue(np.all(eigvals_t > 0))

    def test_temperature_scaler(self):
        z = np.array([
            [2.0, 0.5, -1.0, 0.0],
            [-0.5, 3.0, 1.0, 0.2],
            [0.1, 0.2, 2.5, 0.3],
            [-1.0, 0.0, 0.5, 2.0],
        ], dtype=np.float64)
        y = np.array([0, 1, 2, 3], dtype=int)

        scaler = TemperatureScaler().fit(z, y)
        self.assertGreater(scaler.temperature, 0.0)

        probs = scaler.predict_proba(z)
        self.assertEqual(probs.shape, z.shape)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0))

        # Argmax must be strictly preserved
        self.assertTrue(np.array_equal(np.argmax(probs, axis=1), np.argmax(z, axis=1)))

    def test_uncertainty_metrics_dict(self):
        probs = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.25, 0.25, 0.25, 0.25],
        ])
        z = np.array([
            [3.0, 1.0, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0],
        ])
        u_dict = compute_uncertainty_metrics_dict(probs, z)
        self.assertIn("u_max", u_dict)
        self.assertIn("entropy", u_dict)
        self.assertIn("margin", u_dict)
        self.assertIn("prob_margin", u_dict)

        # First sample has high confidence, second has maximum uncertainty
        self.assertLess(u_dict["u_max"][0], u_dict["u_max"][1])
        self.assertLess(u_dict["entropy"][0], u_dict["entropy"][1])
        self.assertGreater(u_dict["margin"][0], u_dict["margin"][1])

    def test_inner_validation_for_all_methods(self):
        # Synthetic 3 subjects
        n_sub = 20
        d = 6
        Xs = np.random.randn(n_sub * 3, d)
        ys = np.random.randint(0, 4, n_sub * 3)
        source_subs = np.repeat([1, 2, 3], n_sub)

        results = run_inner_validation_for_all_methods(
            Xs=Xs,
            ys=ys,
            source_subjects=source_subs,
            methods=["baseline", "tikhonov", "oas"],
            C=0.01,
            n_splits=3,
        )
        self.assertEqual(set(results.keys()), {"baseline", "tikhonov", "oas"})
        for m, (scaler_temp, thresholds, val_acc) in results.items():
            self.assertGreater(scaler_temp.temperature, 0.0)
            self.assertIn(1.0, thresholds)
            self.assertIn(0.8, thresholds)
            self.assertIn(0.5, thresholds)
            self.assertGreaterEqual(thresholds[1.0], thresholds[0.8])
            self.assertGreaterEqual(thresholds[0.8], thresholds[0.5])


if __name__ == "__main__":
    unittest.main()
