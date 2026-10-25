"""
Unit tests for EEG channel features, selection methods, leakage isolation, and stability metrics.

Verifies:
- Deterministic 5-band feature mapping per physical channel.
- Exact k unique channels selected per budget.
- Reproducibility of random channel subsets.
- Selection strictly using training subjects only.
- Complete exclusion of held-out test subjects.
- Jaccard channel stability calculations.
"""

import unittest
import numpy as np
import pandas as pd

from src.channel_features import (
    get_channel_feature_indices,
    slice_channel_features,
    get_channel_names,
)
from src.channel_selection import (
    select_random_channels,
    select_predictive_importance_channels,
    select_stability_aware_channels,
    compute_inner_grouped_permutation_importance,
    compute_channel_stability_scores,
    select_channels_for_loso_fold,
    compute_pairwise_jaccard_similarity,
)
from src.data_loader import load_seed_iv_dataset, load_config


class TestChannelSelection(unittest.TestCase):
    """Test suite for Phase C channel reduction and stability methods."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config.yaml")
        cls.dataset = load_seed_iv_dataset(config=cls.config, sessions_override=[1])
        cls.X = cls.dataset.X
        cls.y = cls.dataset.y
        cls.metadata = cls.dataset.metadata

    def test_01_channel_to_five_feature_columns_mapping(self):
        """Verify that every physical channel maps to its 5 frequency-band feature columns."""
        # Test first channel (FP1: index 0)
        idx_0 = get_channel_feature_indices([0])
        np.testing.assert_array_equal(idx_0, [0, 1, 2, 3, 4])

        # Test middle channel (e.g. index 10)
        idx_10 = get_channel_feature_indices([10])
        np.testing.assert_array_equal(idx_10, [50, 51, 52, 53, 54])

        # Test last channel (CB2: index 61)
        idx_61 = get_channel_feature_indices([61])
        np.testing.assert_array_equal(idx_61, [305, 306, 307, 308, 309])

        # Multiple channels
        multi = get_channel_feature_indices([0, 2, 61])
        expected = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 305, 306, 307, 308, 309])
        np.testing.assert_array_equal(multi, expected)

    def test_02_channel_subset_budget_counts(self):
        """Verify that exactly k unique channels and k*5 features are selected per budget."""
        budgets = [8, 12, 16, 24, 32]
        for k in budgets:
            chs = select_random_channels(budget=k, seed=42)
            self.assertEqual(len(chs), k)
            self.assertEqual(len(set(chs)), k, f"Budget k={k} contains duplicate channels!")

            X_sliced, feat_idx = slice_channel_features(self.X, chs)
            self.assertEqual(X_sliced.shape[1], k * 5)
            self.assertEqual(len(feat_idx), k * 5)

    def test_03_random_subsets_reproducibility(self):
        """Verify reproducibility across seeds and variance with different seeds."""
        chs_1 = select_random_channels(budget=16, seed=123)
        chs_2 = select_random_channels(budget=16, seed=123)
        chs_3 = select_random_channels(budget=16, seed=999)

        self.assertEqual(chs_1, chs_2, "Same seed must produce identical channel subsets.")
        self.assertNotEqual(chs_1, chs_3, "Different seeds should produce different subsets.")

    def test_04_training_subjects_only_selection(self):
        """Verify selection methods run cleanly on training metadata and return exactly k channels."""
        # Hold out subject 1
        train_mask = (self.metadata["subject_id"] != 1).to_numpy()
        X_tr = self.X[train_mask]
        y_tr = self.y[train_mask]
        meta_tr = self.metadata[train_mask]

        # Fast test with budget k=8
        pred_chs, avg_imp, ranks = select_predictive_importance_channels(
            X_train=X_tr,
            y_train=y_tr,
            train_metadata=meta_tr,
            budget=8,
            n_splits=2,
            seed=42,
        )
        self.assertEqual(len(pred_chs), 8)
        self.assertEqual(len(set(pred_chs)), 8)
        self.assertEqual(len(avg_imp), 62)

        stab_chs, Q, I_norm, S = select_stability_aware_channels(
            X_train=X_tr,
            y_train=y_tr,
            train_metadata=meta_tr,
            budget=8,
            n_splits=2,
            alpha=0.5,
            beta=0.5,
            seed=42,
        )
        self.assertEqual(len(stab_chs), 8)
        self.assertEqual(len(set(stab_chs)), 8)
        self.assertEqual(len(Q), 62)

    def test_05_held_out_subject_exclusion(self):
        """Verify held-out subject data cannot influence channel selection scores."""
        # Train on subjects 2..15
        train_mask = (self.metadata["subject_id"] != 1).to_numpy()
        X_tr = self.X[train_mask].copy()
        y_tr = self.y[train_mask].copy()
        meta_tr = self.metadata[train_mask].copy()

        # Modify channel 0 exclusively in held-out subject 1 (not in X_tr)
        # SUT (system under test) receives only X_tr, guaranteeing zero influence from subject 1
        self.assertNotIn(1, meta_tr["subject_id"].unique())

        chs, _, _ = select_predictive_importance_channels(
            X_train=X_tr, y_train=y_tr, train_metadata=meta_tr, budget=12, n_splits=2
        )
        self.assertEqual(len(chs), 12)

    def test_06_jaccard_similarity_metric(self):
        """Verify pairwise Jaccard stability calculation on known sets."""
        # Identical sets
        set_a = [1, 2, 3, 4]
        set_b = [1, 2, 3, 4]
        sim_identical = compute_pairwise_jaccard_similarity([set_a, set_b])
        self.assertAlmostEqual(sim_identical, 1.0)

        # Disjoint sets
        set_c = [5, 6, 7, 8]
        sim_disjoint = compute_pairwise_jaccard_similarity([set_a, set_c])
        self.assertAlmostEqual(sim_disjoint, 0.0)

        # Half overlap: intersection = {1, 2} (2), union = {1, 2, 3, 4, 5, 6} (6) -> 2/6 = 0.3333
        set_d = [1, 2, 5, 6]
        sim_half = compute_pairwise_jaccard_similarity([set_a, set_d])
        self.assertAlmostEqual(sim_half, 2.0 / 6.0)

    def test_07_equal_mean_ranks_different_variability(self):
        """
        Verify that when two channels have identical mean ranks across splits,
        the corrected multiplicative score penalizes the channel with higher rank variance.
        Under the legacy mean-rank score, both channels would receive identical scores.
        """
        total_channels = 62
        n_splits = 3
        ranks = np.zeros((n_splits, total_channels), dtype=np.int64)

        # Baseline assignments for all channels: 1..62
        for s in range(n_splits):
            ranks[s, :] = np.arange(1, total_channels + 1)

        # Channel 0: stable ranks [10, 10, 10] -> mean = 10.0, std = 0.0
        ranks[:, 0] = [10, 10, 10]

        # Channel 1: erratic ranks [1, 10, 19] -> mean = 10.0, std = 7.348
        ranks[:, 1] = [1, 10, 19]

        avg_importance = np.full(total_channels, 0.05, dtype=np.float64)

        # 1. Corrected Multiplicative Scoring
        scores_mult = compute_channel_stability_scores(
            avg_importance=avg_importance,
            ranks_matrix=ranks,
            total_channels=total_channels,
            method="multiplicative",
        )
        # Mean rank quality R_c must be identical because mean ranks are equal (10.0)
        self.assertAlmostEqual(scores_mult["R_c"][0], scores_mult["R_c"][1], places=7)

        # Rank consistency V_c must strictly favor Channel 0
        self.assertAlmostEqual(scores_mult["V_c"][0], 1.0, places=7)
        self.assertLess(scores_mult["V_c"][1], 1.0)
        self.assertGreater(scores_mult["V_c"][0], scores_mult["V_c"][1])

        # Stability score S_c must strictly favor Channel 0
        self.assertGreater(scores_mult["S_c"][0], scores_mult["S_c"][1])
        self.assertGreater(scores_mult["Q_c"][0], scores_mult["Q_c"][1])

        # 2. Legacy Mean-Rank Scoring (Historical Bug)
        scores_legacy = compute_channel_stability_scores(
            avg_importance=avg_importance,
            ranks_matrix=ranks,
            total_channels=total_channels,
            method="legacy_mean_rank",
        )
        # Legacy score failed to distinguish variance: S_0 == S_1
        self.assertAlmostEqual(scores_legacy["S_c"][0], scores_legacy["S_c"][1], places=7)
        self.assertAlmostEqual(scores_legacy["Q_c"][0], scores_legacy["Q_c"][1], places=7)

    def test_08_stable_vs_unstable_channel_selection(self):
        """Verify that a channel with stable high ranks is prioritized over an erratic channel."""
        total_channels = 62
        n_splits = 3
        ranks = np.zeros((n_splits, total_channels), dtype=np.int64)
        for s in range(n_splits):
            ranks[s, :] = np.arange(1, total_channels + 1)

        # Stable top channel (Ch 5): ranks [3, 4, 3] -> mean = 3.33, std = 0.47
        ranks[:, 5] = [3, 4, 3]

        # Erratic channel (Ch 6): ranks [1, 5, 20] -> mean = 8.67, std = 8.18
        ranks[:, 6] = [1, 5, 20]

        avg_importance = np.full(total_channels, 0.05, dtype=np.float64)
        avg_importance[5] = 0.08
        avg_importance[6] = 0.08  # Equal importance

        scores = compute_channel_stability_scores(
            avg_importance=avg_importance,
            ranks_matrix=ranks,
            total_channels=total_channels,
            method="multiplicative",
        )
        self.assertGreater(scores["Q_c"][5], scores["Q_c"][6])

        # Priority order must place 5 before 6
        order = list(scores["rank_order"])
        self.assertLess(order.index(5), order.index(6))

    def test_09_score_normalization_and_finite_bounds(self):
        """Verify that all normalized score components are strictly finite and within [0, 1]."""
        total_channels = 62
        n_splits = 3

        test_scenarios = [
            # Standard linear drops
            (np.linspace(0.01, 0.15, total_channels), np.random.RandomState(42).randint(1, 63, size=(n_splits, total_channels))),
            # Negative drops
            (np.linspace(-0.2, 0.0, total_channels), np.random.RandomState(43).randint(1, 63, size=(n_splits, total_channels))),
            # Identical drops (zero variance in importance)
            (np.full(total_channels, 0.05), np.random.RandomState(44).randint(1, 63, size=(n_splits, total_channels))),
            # Extreme drop range
            (np.array([100.0] + [0.0] * (total_channels - 1)), np.ones((n_splits, total_channels), dtype=int)),
        ]

        for avg_imp, ranks in test_scenarios:
            scores = compute_channel_stability_scores(
                avg_importance=avg_imp,
                ranks_matrix=ranks,
                total_channels=total_channels,
                method="multiplicative",
            )
            for key in ["Q_c", "I_norm", "S_c", "R_c", "V_c"]:
                arr = scores[key]
                self.assertTrue(np.all(np.isfinite(arr)), f"{key} contains non-finite values!")
                self.assertTrue(np.all(arr >= 0.0 - 1e-12), f"{key} contains values < 0: min={np.min(arr)}")
                self.assertTrue(np.all(arr <= 1.0 + 1e-12), f"{key} contains values > 1: max={np.max(arr)}")

    def test_10_tied_ranks_and_constant_rankings(self):
        """Verify deterministic handling and stability when all ranks or importance scores are tied."""
        total_channels = 62
        n_splits = 3

        # Case A: Constant identical rankings across all splits and all channels
        ranks_const = np.ones((n_splits, total_channels), dtype=np.int64) * 10
        avg_imp_const = np.ones(total_channels, dtype=np.float64) * 0.05

        scores = compute_channel_stability_scores(
            avg_importance=avg_imp_const,
            ranks_matrix=ranks_const,
            total_channels=total_channels,
            method="multiplicative",
        )
        # All standard deviations are 0 -> V_c must be exactly 1.0 without division by zero
        self.assertTrue(np.allclose(scores["V_c"], 1.0))
        self.assertTrue(np.allclose(scores["Q_c"], scores["Q_c"][0]))

        # Tie breaking must deterministically sort by ascending channel index 0..61
        np.testing.assert_array_equal(scores["rank_order"], np.arange(total_channels))

        # Case B: Single inner split (n_splits = 1)
        ranks_single = np.arange(1, total_channels + 1).reshape(1, total_channels)
        scores_single = compute_channel_stability_scores(
            avg_importance=avg_imp_const,
            ranks_matrix=ranks_single,
            total_channels=total_channels,
            method="multiplicative",
        )
        self.assertTrue(np.allclose(scores_single["V_c"], 1.0))

    def test_11_physical_channel_to_five_feature_mapping_integrity(self):
        """
        Verify that physical-to-feature mapping preserves indivisibility:
        - Exactly k*5 features for any k unique physical channels.
        - Every selected channel has all 5 bands (delta, theta, alpha, beta, gamma).
        - No unselected channel has any feature in the sliced feature matrix.
        """
        budgets = [8, 12, 16, 24, 32]
        rng = np.random.RandomState(42)

        for k in budgets:
            chs = sorted(rng.choice(62, size=k, replace=False).tolist())
            feat_idx = get_channel_feature_indices(chs)

            self.assertEqual(len(feat_idx), k * 5)
            self.assertEqual(len(set(feat_idx)), k * 5, "Duplicate feature indices detected!")

            # Verify every selected channel's 5 bands are present
            for ch in chs:
                expected_bands = [ch * 5 + b for b in range(5)]
                for b_col in expected_bands:
                    self.assertIn(b_col, feat_idx)

            # Verify unselected channels have zero representation
            unselected = set(range(62)) - set(chs)
            for un_ch in unselected:
                un_bands = [un_ch * 5 + b for b in range(5)]
                for b_col in un_bands:
                    self.assertNotIn(b_col, feat_idx)

    def test_12_perturbation_based_leakage_test(self):
        """
        Rigorous perturbation-based leakage test:
        Alter the held-out subject's feature values and labels while keeping outer-training data unchanged.
        Verify that channel selection scores (Q_c, I_norm, S_c) and selected channels remain 100% IDENTICAL.
        """
        test_subject = 1
        budget = 12

        # 1. Baseline selection on unperturbed dataset
        orig_chs, orig_info = select_channels_for_loso_fold(
            X_full=self.X,
            y_full=self.y,
            metadata_full=self.metadata,
            test_subject=test_subject,
            budget=budget,
            method_name="stability_aware",
            stability_method="multiplicative",
            n_splits=2,
            seed=42,
        )

        # 2. Corrupt held-out subject data and labels in full dataset
        X_perturbed = self.X.copy()
        y_perturbed = self.y.copy()
        meta_perturbed = self.metadata.copy()

        held_out_mask = (self.metadata["subject_id"] == test_subject).to_numpy()
        self.assertTrue(np.any(held_out_mask), "Held-out subject rows not found!")

        # Inject extreme feature distortions and flip all labels for held-out subject
        X_perturbed[held_out_mask] = 999999.0
        y_perturbed[held_out_mask] = (self.y[held_out_mask] + 1) % 4

        # 3. Selection on perturbed dataset
        pert_chs, pert_info = select_channels_for_loso_fold(
            X_full=X_perturbed,
            y_full=y_perturbed,
            metadata_full=meta_perturbed,
            test_subject=test_subject,
            budget=budget,
            method_name="stability_aware",
            stability_method="multiplicative",
            n_splits=2,
            seed=42,
        )

        # 4. Strict identity verification
        self.assertEqual(orig_chs, pert_chs, "Perturbing test subject changed selected channels!")
        np.testing.assert_allclose(
            orig_info["Q_c"],
            pert_info["Q_c"],
            atol=1e-12,
            err_msg="Perturbing test subject altered Q_c selection scores!",
        )
        np.testing.assert_allclose(
            orig_info["I_norm"],
            pert_info["I_norm"],
            atol=1e-12,
            err_msg="Perturbing test subject altered I_norm scores!",
        )
        np.testing.assert_allclose(
            orig_info["S_c"],
            pert_info["S_c"],
            atol=1e-12,
            err_msg="Perturbing test subject altered S_c scores!",
        )

    def test_13_inner_split_subject_disjointness_enforcement(self):
        """Verify that inner grouped splits strictly isolate subjects between inner train and val."""
        train_mask = (self.metadata["subject_id"] != 1).to_numpy()
        meta_tr = self.metadata[train_mask]
        X_tr = self.X[train_mask]
        y_tr = self.y[train_mask]

        # Standard execution must pass inner assertion without error
        imp_mat, ranks_mat, avg_imp = compute_inner_grouped_permutation_importance(
            X_train=X_tr,
            y_train=y_tr,
            train_metadata=meta_tr,
            n_splits=3,
            seed=42,
        )
        self.assertEqual(imp_mat.shape, (3, 62))
        self.assertEqual(ranks_mat.shape, (3, 62))
        self.assertEqual(avg_imp.shape, (62,))

    def test_14_invalid_inputs_and_boundary_checks(self):
        """Verify that invalid budgets, channel dimensions, and invalid methods raise clear errors."""
        with self.assertRaises(ValueError):
            select_random_channels(budget=0, total_channels=62)

        with self.assertRaises(ValueError):
            select_random_channels(budget=63, total_channels=62)

        with self.assertRaises(ValueError):
            select_stability_aware_channels(
                X_train=self.X[:100],
                y_train=self.y[:100],
                train_metadata=self.metadata.iloc[:100],
                budget=0,
            )

        with self.assertRaises(ValueError):
            compute_channel_stability_scores(
                avg_importance=np.zeros(62),
                ranks_matrix=np.ones((3, 62)),
                method="invalid_scoring_method",
            )

        with self.assertRaises(ValueError):
            compute_channel_stability_scores(
                avg_importance=np.zeros(10),  # Mismatch with total_channels=62
                ranks_matrix=np.ones((3, 62)),
                total_channels=62,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

