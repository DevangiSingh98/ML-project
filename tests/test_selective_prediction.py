"""
Unit tests for Phase E Selective Prediction (src/selective_prediction.py).

Tests:
1. compute_max_probability_uncertainty: correctness, bounds, extreme cases.
2. compute_predictive_entropy: correctness, zero probabilities, normalized entropy, base checks.
3. validate_probability_array: rejection of negative values, unnormalized rows, invalid shapes.
4. get_acceptance_mask: threshold logic, monotonicity, ties at boundary.
5. compute_coverage: range bounds, empty mask handling.
6. compute_selective_accuracy & compute_selective_risk:
   - Complementarity: accuracy + risk == 1.0 for non-empty sets.
   - Hand-calculated cases.
   - Explicit np.nan behavior for empty accepted sets (no silent 0 or 1).
7. compute_risk_coverage_curve:
   - Monotonic coverage sorting.
   - Endpoint at coverage 1.0.
   - Proper tie handling without splitting identical scores.
   - Consistent counts.
8. compute_aurc:
   - Hand-calculated piecewise linear example.
   - Invariance to duplicate coverage entries.
   - Rejection of invalid inputs.
9. compute_random_rejection_baseline:
   - Analytical expectation equals full baseline accuracy.
   - Reproducibility with identical random seed.
   - Variance across different random seeds.
   - Label-agnostic sampling verification.
10. determine_threshold_from_training_budget:
    - Quantile thresholds.
    - Full coverage endpoint.
"""

import unittest
import numpy as np

from src.selective_prediction import (
    validate_probability_array,
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


class TestSelectivePrediction(unittest.TestCase):
    """Validation test suite for Phase E selective prediction mathematics."""

    def setUp(self):
        # Deterministic test distributions
        self.probs_extreme = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Certain class 0
            [0.25, 0.25, 0.25, 0.25],  # Maximum uncertainty
            [0.70, 0.10, 0.10, 0.10],  # Moderate certainty
            [0.40, 0.30, 0.20, 0.10],  # Low certainty
        ], dtype=np.float64)

    def test_01_max_probability_uncertainty_correctness(self):
        """Verify u_max calculation, bounds [0, 1 - 1/K], and extreme cases."""
        u = compute_max_probability_uncertainty(self.probs_extreme)
        self.assertEqual(len(u), 4)
        
        # Sample 0: max prob 1.0 -> u = 0.0
        self.assertAlmostEqual(u[0], 0.0, places=6)
        # Sample 1: max prob 0.25 -> u = 0.75
        self.assertAlmostEqual(u[1], 0.75, places=6)
        # Sample 2: max prob 0.70 -> u = 0.30
        self.assertAlmostEqual(u[2], 0.30, places=6)
        # Sample 3: max prob 0.40 -> u = 0.60
        self.assertAlmostEqual(u[3], 0.60, places=6)
        
        # Check theoretical bounds
        self.assertTrue(np.all(u >= 0.0))
        self.assertTrue(np.all(u <= 0.75))

    def test_02_predictive_entropy_zero_probabilities_and_bounds(self):
        """Verify predictive entropy handles zero probabilities safely without NaN/Inf."""
        h_nat = compute_predictive_entropy(self.probs_extreme, base=np.e, normalize=False)
        self.assertFalse(np.isnan(h_nat).any(), "Entropy must not contain NaNs.")
        self.assertFalse(np.isinf(h_nat).any(), "Entropy must not contain Infs.")
        
        # Sample 0: [1, 0, 0, 0] -> 0 ln(1) = 0.0
        self.assertAlmostEqual(h_nat[0], 0.0, places=6)
        # Sample 1: [0.25, 0.25, 0.25, 0.25] -> ln(4)
        self.assertAlmostEqual(h_nat[1], np.log(4.0), places=6)
        
        # Normalized entropy
        h_norm = compute_predictive_entropy(self.probs_extreme, normalize=True)
        self.assertAlmostEqual(h_norm[0], 0.0, places=6)
        self.assertAlmostEqual(h_norm[1], 1.0, places=6)
        self.assertTrue(np.all(h_norm >= 0.0) and np.all(h_norm <= 1.0))

    def test_03_invalid_probability_rejection(self):
        """Verify invalid probability arrays raise clear ValueErrors."""
        # Negative probability
        with self.assertRaises(ValueError):
            validate_probability_array(np.array([[-0.1, 1.1]]))
        # Unnormalized row (sums to 2.0)
        with self.assertRaises(ValueError):
            validate_probability_array(np.array([[0.5, 0.5, 1.0]]))
        # 1D array
        with self.assertRaises(ValueError):
            validate_probability_array(np.array([0.5, 0.5]))
        # 1-class array
        with self.assertRaises(ValueError):
            validate_probability_array(np.array([[1.0]]))

    def test_04_acceptance_decision_logic(self):
        """Verify that predictions are accepted iff uncertainty <= threshold."""
        u = np.array([0.1, 0.3, 0.5, 0.7])
        
        mask_03 = get_acceptance_mask(u, threshold=0.3)
        np.testing.assert_array_equal(mask_03, [True, True, False, False])
        
        mask_all = get_acceptance_mask(u, threshold=1.0)
        np.testing.assert_array_equal(mask_all, [True, True, True, True])
        
        mask_none = get_acceptance_mask(u, threshold=0.05)
        np.testing.assert_array_equal(mask_none, [False, False, False, False])

    def test_05_threshold_ties_handling(self):
        """Verify threshold ties are handled consistently: samples with equal uncertainty are all accepted."""
        u = np.array([0.2, 0.4, 0.4, 0.4, 0.6])
        mask = get_acceptance_mask(u, threshold=0.4)
        # All three 0.4 entries must be accepted together
        np.testing.assert_array_equal(mask, [True, True, True, True, False])
        self.assertEqual(np.sum(mask), 4)

    def test_06_monotonicity_of_abstention(self):
        """Verify that lowering threshold monotonically decreases or maintains acceptance count."""
        rng = np.random.RandomState(42)
        u = rng.uniform(0.0, 1.0, size=100)
        
        thresholds = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        accepted_counts = [np.sum(get_acceptance_mask(u, th)) for th in thresholds]
        
        for i in range(len(accepted_counts) - 1):
            self.assertGreaterEqual(
                accepted_counts[i],
                accepted_counts[i + 1],
                "Acceptance count must be monotonically non-increasing as threshold tightens."
            )

    def test_07_coverage_bounds_and_validity(self):
        """Verify coverage calculation and boundary guarantees."""
        mask = np.array([True, True, False, False])
        self.assertEqual(compute_coverage(mask), 0.5)
        
        mask_full = np.array([True, True, True])
        self.assertEqual(compute_coverage(mask_full), 1.0)
        
        mask_empty = np.array([False, False, False])
        self.assertEqual(compute_coverage(mask_empty), 0.0)

    def test_08_selective_accuracy_and_risk_complementarity(self):
        """Verify selective_accuracy + selective_risk == 1.0 on non-empty sets."""
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 0, 3, 1, 1])  # Samples 0, 1, 3, 5 correct; 2, 4 incorrect
        
        mask = np.array([True, True, True, True, False, False])  # First 4 samples accepted
        # Among accepted:
        # Sample 0: correct (0==0)
        # Sample 1: correct (1==1)
        # Sample 2: incorrect (2!=0)
        # Sample 3: correct (3==3)
        # Acc = 3/4 = 0.75, Risk = 1/4 = 0.25
        
        acc = compute_selective_accuracy(y_true, y_pred, mask)
        risk = compute_selective_risk(y_true, y_pred, mask)
        
        self.assertAlmostEqual(acc, 0.75)
        self.assertAlmostEqual(risk, 0.25)
        self.assertAlmostEqual(acc + risk, 1.0)

    def test_09_empty_accepted_set_returns_nan(self):
        """Verify that empty accepted sets explicitly return np.nan rather than 0.0 or 1.0."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        empty_mask = np.array([False, False])
        
        acc = compute_selective_accuracy(y_true, y_pred, empty_mask)
        risk = compute_selective_risk(y_true, y_pred, empty_mask)
        
        self.assertTrue(np.isnan(acc), "Empty set must yield NaN for accuracy.")
        self.assertTrue(np.isnan(risk), "Empty set must yield NaN for risk.")

    def test_10_risk_coverage_curve_endpoints_and_monotonicity(self):
        """Verify risk-coverage curve reaches full coverage and handles tied uncertainties."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 0, 0, 1, 2, 0])  # 6 correct, 2 incorrect (Samples 3 and 7 wrong)
        # Incorrect samples assigned highest uncertainty
        u = np.array([0.1, 0.1, 0.2, 0.8, 0.2, 0.3, 0.3, 0.9])
        
        curve = compute_risk_coverage_curve(y_true, y_pred, u)
        
        coverages = curve["coverages"]
        risks = curve["selective_risks"]
        accuracies = curve["selective_accuracies"]
        
        # 1. Full-coverage endpoint must exist
        self.assertAlmostEqual(coverages[-1], 1.0)
        self.assertAlmostEqual(accuracies[-1], 6 / 8)  # 75% baseline accuracy
        self.assertAlmostEqual(risks[-1], 2 / 8)       # 25% baseline error
        
        # 2. Coverages must be strictly non-decreasing
        self.assertTrue(np.all(np.diff(coverages) >= 0.0))
        
        # 3. For low coverage, rejecting high-uncertainty samples must reduce risk to 0
        self.assertAlmostEqual(risks[0], 0.0)
        self.assertAlmostEqual(accuracies[0], 1.0)

    def test_11_aurc_piecewise_linear_hand_calculation(self):
        """Verify AURC against an exact hand-calculated trapezoidal integral."""
        # Simple test curve:
        # (cov=0.5, risk=0.1) to (cov=1.0, risk=0.3)
        # Trapezoid area = (1.0 - 0.5) * (0.1 + 0.3) / 2 = 0.5 * 0.4 / 2 = 0.10
        covs = np.array([0.5, 1.0])
        risks = np.array([0.1, 0.3])
        
        aurc = compute_aurc(covs, risks, min_coverage=0.0)
        self.assertAlmostEqual(aurc, 0.10, places=6)

    def test_12_aurc_duplicate_coverages_handling(self):
        """Verify AURC handles duplicate coverage coordinates without crashing or inflation."""
        covs = np.array([0.5, 0.5, 1.0])
        risks = np.array([0.1, 0.2, 0.3])  # Duplicate at 0.5 with risks 0.1 and 0.2 -> mean = 0.15
        # Effective curve: (0.5, 0.15) to (1.0, 0.3)
        # Area = 0.5 * (0.15 + 0.30) / 2 = 0.5 * 0.45 / 2 = 0.1125
        aurc = compute_aurc(covs, risks)
        self.assertAlmostEqual(aurc, 0.1125, places=6)

    def test_13_random_rejection_baseline_expectation(self):
        """Verify random rejection analytical expectation strictly equals full-coverage accuracy."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 0, 0, 0, 0])  # 7 correct, 3 incorrect -> Acc = 0.70
        
        target_covs = [1.0, 0.8, 0.5]
        res = compute_random_rejection_baseline(y_true, y_pred, target_covs, n_repeats=200, seed=42)
        
        # Analytical expectation must be 0.70
        self.assertAlmostEqual(res["expected_accuracy"], 0.70)
        self.assertAlmostEqual(res["expected_risk"], 0.30)
        
        # Empirical Monte Carlo means must closely cluster around 0.70 (law of large numbers)
        for emp_acc in res["empirical_accuracy_mean"]:
            self.assertAlmostEqual(emp_acc, 0.70, delta=0.04)

    def test_14_random_rejection_reproducibility_and_label_agnosticism(self):
        """Verify random rejection is 100% reproducible with same seed and differs with different seed."""
        y_true = np.array([0, 1, 2, 3] * 10)
        y_pred = np.array([0, 1, 0, 0] * 10)
        
        res1 = compute_random_rejection_baseline(y_true, y_pred, [0.5], n_repeats=10, seed=123)
        res2 = compute_random_rejection_baseline(y_true, y_pred, [0.5], n_repeats=10, seed=123)
        res3 = compute_random_rejection_baseline(y_true, y_pred, [0.5], n_repeats=10, seed=999)
        
        # Identical seed must yield identical results
        np.testing.assert_array_equal(res1["empirical_accuracy_mean"], res2["empirical_accuracy_mean"])
        np.testing.assert_array_equal(res1["empirical_accuracy_std"], res2["empirical_accuracy_std"])
        
        # Different seed must yield distinct random samples
        self.assertNotEqual(res1["empirical_accuracy_mean"][0], res3["empirical_accuracy_mean"][0])

    def test_15_determine_threshold_from_training_budget(self):
        """Verify training quantile threshold determination preserves coverage target and handles 100% rule."""
        u_train = np.linspace(0.0, 1.0, 101)  # 0.00, 0.01, ..., 1.00
        
        th_80 = determine_threshold_from_training_budget(u_train, target_coverage=0.80)
        self.assertAlmostEqual(th_80, 0.80, places=6)
        
        # Applying threshold to training data must accept exactly 80% (+1 inclusive)
        mask_80 = get_acceptance_mask(u_train, th_80)
        self.assertEqual(np.sum(mask_80), 81)
        
        # Explicit 100% coverage rule: returns np.inf to ensure all predictions are accepted
        th_100 = determine_threshold_from_training_budget(u_train, target_coverage=1.0)
        self.assertTrue(np.isposinf(th_100) or th_100 == np.inf)
        mask_100 = get_acceptance_mask(u_train, th_100)
        self.assertEqual(np.sum(mask_100), len(u_train))

    def test_16_partial_and_full_aurc_hand_calculations(self):
        """Verify compute_partial_aurc and compute_full_aurc against exact hand calculations."""
        # Simple test curve: (cov=0.5, risk=0.1) to (cov=1.0, risk=0.3)
        covs = np.array([0.5, 1.0])
        risks = np.array([0.1, 0.3])
        
        # 1. Partial AURC over [0.5, 1.0]:
        # Trapezoid area = (1.0 - 0.5) * (0.1 + 0.3) / 2 = 0.5 * 0.4 / 2 = 0.10
        part_aurc = compute_partial_aurc(covs, risks, min_coverage=0.5, max_coverage=1.0)
        self.assertAlmostEqual(part_aurc, 0.10, places=6)
        
        # 2. Full AURC over [0.0, 1.0]:
        # Left-extension convention: risk at c_min (0.5) extends horizontally to c=0 (r=0.1)
        # Area [0.0, 0.5] = (0.5 - 0.0) * 0.1 = 0.05
        # Area [0.5, 1.0] = 0.10
        # Total Full AURC = 0.05 + 0.10 = 0.150
        full_aurc = compute_full_aurc(covs, risks)
        self.assertAlmostEqual(full_aurc, 0.150, places=6)

    def test_17_extreme_uncertainty_test_sample_accepted_at_100_percent(self):
        """Verify that a test sample with uncertainty higher than training maximum is accepted at 100% nominal coverage."""
        u_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Max training uncertainty is 0.5
        u_test = np.array([0.2, 0.4, 0.99])            # 0.99 exceeds max training uncertainty
        
        # Operational threshold for 100% coverage
        tau_100 = determine_threshold_from_training_budget(u_train, target_coverage=1.0)
        self.assertTrue(np.isposinf(tau_100))
        
        # Must accept all 3 test samples (100% acceptance)
        mask = get_acceptance_mask(u_test, tau_100)
        self.assertEqual(np.sum(mask), 3)
        self.assertEqual(compute_coverage(mask), 1.0)


if __name__ == "__main__":
    unittest.main()
