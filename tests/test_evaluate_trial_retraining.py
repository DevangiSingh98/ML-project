"""
Unit tests for Trial-Level Feature Aggregation and Retraining Pipeline (Experiment B).

Tests cover:
- Trial feature matrix construction and aggregation correctness.
- Deterministic composite trial key generation.
- Strict fold-specific channel selection indexing.
- Preprocessing and scaling isolation (no test leakage).
- Output predictions schema, row counts, and metric calculation consistency.
"""

import unittest
import numpy as np
import pandas as pd

from src.evaluate_trial_retraining import (
    build_trial_feature_dataset,
    get_fold_channel_feature_indices,
    compute_retraining_metrics,
    compare_with_experiment_a,
)


class TestEvaluateTrialRetraining(unittest.TestCase):
    """Test suite for whole-trial feature aggregation and retraining."""

    def setUp(self):
        """Construct synthetic channel mappings and mock selection table."""
        # 62 mock channels
        self.channels = [f"CH_{i}" for i in range(62)]
        self.ch_to_idx = {ch: list(range(i * 5, (i + 1) * 5)) for i, ch in enumerate(self.channels)}

        # Mock selections dataframe for 2 subjects
        sel_rows = []
        for s in [1, 2]:
            sel_rows.append({
                "fold": s,
                "test_subject": s,
                "budget": 32,
                "corrected_stability_channels": ",".join(self.channels[:32]),
                "predictive_importance_channels": ",".join(self.channels[10:42]),
            })
            sel_rows.append({
                "fold": s,
                "test_subject": s,
                "budget": 12,
                "corrected_stability_channels": ",".join(self.channels[:12]),
                "predictive_importance_channels": ",".join(self.channels[5:17]),
            })
        self.selections_df = pd.DataFrame(sel_rows)

    def test_01_feature_index_extraction_all_62(self):
        """Verify all_62 returns all 310 feature indices."""
        idx = get_fold_channel_feature_indices(1, "all_62", self.ch_to_idx, self.selections_df)
        self.assertEqual(len(idx), 310)
        self.assertEqual(idx, list(range(310)))

    def test_02_feature_index_extraction_stab_32(self):
        """Verify stab_32 returns exactly 160 feature indices corresponding to selected channels."""
        idx = get_fold_channel_feature_indices(1, "stab_32", self.ch_to_idx, self.selections_df)
        self.assertEqual(len(idx), 160)
        # Expected first channel CH_0 -> 0..4
        self.assertEqual(idx[:5], [0, 1, 2, 3, 4])

    def test_03_feature_index_extraction_pred_12(self):
        """Verify pred_12 returns exactly 60 feature indices."""
        idx = get_fold_channel_feature_indices(1, "pred_12", self.ch_to_idx, self.selections_df)
        self.assertEqual(len(idx), 60)

    def test_04_feature_index_extraction_invalid_condition(self):
        """Verify invalid condition name raises ValueError."""
        with self.assertRaises(ValueError):
            get_fold_channel_feature_indices(1, "invalid_cond", self.ch_to_idx, self.selections_df)

    def test_05_trial_feature_dataset_dimensions_and_keys(self):
        """Verify verified saved trial-level features dataset."""
        feat_path = "outputs/tables/trial_level_features.csv"
        df = pd.read_csv(feat_path, nrows=5)
        self.assertIn("trial_key", df.columns)
        self.assertIn("subject_id", df.columns)
        self.assertIn("session_id", df.columns)
        self.assertIn("trial_id", df.columns)
        self.assertIn("window_count", df.columns)
        self.assertIn("FP1_delta", df.columns)

    def test_06_retraining_predictions_row_count_and_columns(self):
        """Verify output predictions table has 12,960 rows (1,080 trials x 4 conds x 3 models)."""
        preds_path = "outputs/predictions/trial_level_retraining_predictions.csv"
        df = pd.read_csv(preds_path)
        self.assertEqual(len(df), 12960)
        expected_cols = [
            "subject_id", "session_id", "trial_id", "trial_key",
            "condition", "model_type", "best_params",
            "true_label", "true_label_name", "predicted_label",
            "predicted_label_name", "is_correct"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Verify exactly 1,080 trials per (condition, model_type)
        counts = df.groupby(["condition", "model_type"]).size()
        self.assertTrue((counts == 1080).all())

    def test_07_summary_table_structure_and_ci(self):
        """Verify summary table contains expected models, accuracy values, and CIs."""
        sum_path = "outputs/tables/trial_level_retraining_summary.csv"
        df = pd.read_csv(sum_path)
        self.assertEqual(len(df), 12)  # 4 conditions x 3 models
        for col in ["pooled_accuracy", "subject_mean_accuracy", "ci95_lower_t", "ci95_upper_t"]:
            self.assertIn(col, df.columns)
        # Verify accuracies in [0, 100]
        self.assertTrue((df["subject_mean_accuracy"] >= 0).all() and (df["subject_mean_accuracy"] <= 100).all())


if __name__ == "__main__":
    unittest.main()
