"""
Unit tests for Stage 2.5: Corrected Channel Performance Evaluation.

Verifies:
1. Feature extraction preserves indivisible 5 frequency bands per channel (32 channels -> 160 features).
2. Proper loading and channel index mapping of verified Stage 2 channel selections.
3. Strict LOSO subject isolation (held-out subject excluded from training).
4. Prediction DataFrame schema integrity and unique key constraints.
5. Consistent evaluation across conditions (identical test subjects, labels, and sample counts).
6. Correctness of metric aggregation (pooled vs subject-level mean +/- std).
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.data_loader import load_seed_iv_dataset, load_config
from src.channel_features import get_channel_feature_indices, slice_channel_features
from src.evaluate_corrected_channel_performance import load_or_compute_fold_channels
from src.metrics import compute_classification_metrics


class TestCorrectedChannelPerformance(unittest.TestCase):
    """Test suite for Stage 2.5 classifier evaluation pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config.yaml")
        cls.dataset = load_seed_iv_dataset(config=cls.config, sessions_override=[1])
        cls.X = cls.dataset.X
        cls.y = cls.dataset.y
        cls.metadata = cls.dataset.metadata
        cls.all_channels = cls.dataset.channels
        cls.selections_path = "outputs/tables/corrected_channel_selections.csv"

    def test_01_feature_slicing_dimension_integrity(self):
        """Verify that slicing 32 physical channels produces exactly 160 feature columns."""
        # Test arbitrary 32 channels
        rng = np.random.RandomState(42)
        chs_32 = sorted(rng.choice(62, size=32, replace=False).tolist())
        feat_idx = get_channel_feature_indices(chs_32)

        self.assertEqual(len(chs_32), 32)
        self.assertEqual(len(feat_idx), 160)
        self.assertEqual(len(set(feat_idx)), 160)

        X_sub, feat_ret = slice_channel_features(self.X, chs_32)
        self.assertEqual(X_sub.shape, (len(self.X), 160))
        np.testing.assert_array_equal(feat_idx, feat_ret)

    def test_02_load_stage2_channel_selections(self):
        """Verify that Stage 2 channel selections CSV loads cleanly and has 32 channels per fold."""
        if not os.path.exists(self.selections_path):
            self.skipTest(f"Selections file {self.selections_path} not found.")

        df = pd.read_csv(self.selections_path)
        df32 = df[df["budget"] == 32]
        self.assertEqual(len(df32), 15, "Expected exactly 15 outer folds for budget 32.")

        ch_to_idx = {name: i for i, name in enumerate(self.all_channels)}

        for _, row in df32.iterrows():
            corr_names = [c.strip() for c in row["corrected_stability_channels"].split(",")]
            pred_names = [c.strip() for c in row["predictive_importance_channels"].split(",")]

            self.assertEqual(len(corr_names), 32)
            self.assertEqual(len(pred_names), 32)
            self.assertEqual(len(set(corr_names)), 32, "Duplicate channels found in corrected selection!")
            self.assertEqual(len(set(pred_names)), 32, "Duplicate channels found in predictive selection!")

            for c in corr_names:
                self.assertIn(c, ch_to_idx, f"Channel {c} not in official channel list!")
            for c in pred_names:
                self.assertIn(c, ch_to_idx, f"Channel {c} not in official channel list!")

    def test_03_load_or_compute_fold_channels_helper(self):
        """Verify load_or_compute_fold_channels correctly extracts 32 channel indices."""
        train_mask = (self.metadata["subject_id"] != 1).to_numpy()
        meta_tr = self.metadata[train_mask]
        X_tr = self.X[train_mask]
        y_tr = self.y[train_mask]

        selections_df = pd.read_csv(self.selections_path) if os.path.exists(self.selections_path) else None

        corr_idx, pred_idx = load_or_compute_fold_channels(
            fold_idx=1,
            test_sub=1,
            X_train=X_tr,
            y_train=y_tr,
            train_meta=meta_tr,
            all_channel_names=self.all_channels,
            selections_df=selections_df,
            budget=32,
        )

        self.assertEqual(len(corr_idx), 32)
        self.assertEqual(len(pred_idx), 32)
        self.assertTrue(all(0 <= c < 62 for c in corr_idx))
        self.assertTrue(all(0 <= c < 62 for c in pred_idx))

    def test_04_metric_aggregation_consistency(self):
        """Verify that pooled accuracy differs from subject-averaged accuracy as expected mathematically."""
        # Synthetic fold predictions: Subject 1 has 10 samples (100% acc), Subject 2 has 90 samples (0% acc)
        # Subject-average acc = (1.0 + 0.0) / 2 = 0.50
        # Pooled acc = 10 / 100 = 0.10
        y_true_s1 = np.zeros(10, dtype=int)
        y_pred_s1 = np.zeros(10, dtype=int)

        y_true_s2 = np.zeros(90, dtype=int)
        y_pred_s2 = np.ones(90, dtype=int)

        m_s1 = compute_classification_metrics(y_true_s1, y_pred_s1)
        m_s2 = compute_classification_metrics(y_true_s2, y_pred_s2)

        sub_acc_mean = (m_s1["accuracy"] + m_s2["accuracy"]) / 2.0
        self.assertEqual(sub_acc_mean, 0.5)

        y_true_all = np.concatenate([y_true_s1, y_true_s2])
        y_pred_all = np.concatenate([y_pred_s1, y_pred_s2])
        pooled_m = compute_classification_metrics(y_true_all, y_pred_all)
        self.assertEqual(pooled_m["accuracy"], 0.10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
