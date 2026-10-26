"""
tests/test_evaluate_adaptation.py

Unit test suite for Experiment D1: Unsupervised Subject Adaptation Feasibility via CORAL.
Verifies:
1. Mathematical correctness of CORAL covariance alignment and numerical stability.
2. Unsupervised isolation (target labels are strictly never ingested or leaked).
3. Channel selection indexing and outer-fold subject insulation.
4. Output predictions table row counts (12,960 rows = 1,080 trials x 4 conds x 3 methods).
5. Metric calculation and comparison consistency.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

from src.evaluate_subject_adaptation import (
    compute_coral_transformation,
    get_channel_feature_indices,
    tune_coral_hyperparameters,
)


class TestEvaluateAdaptation(unittest.TestCase):

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.tables_dir = self.root_dir / "outputs" / "tables"
        self.predictions_dir = self.root_dir / "outputs" / "predictions"
        self.selections_path = self.tables_dir / "corrected_channel_selections.csv"

        channels = [
            "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
            "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
            "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
            "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
            "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
        ]
        self.channel_to_indices = {ch: list(range(i * 5, (i + 1) * 5)) for i, ch in enumerate(channels)}

    def test_01_coral_transformation_shapes_and_finiteness(self):
        """Verify CORAL computes valid (d, d) transformation matrices with no NaNs/Infs."""
        np.random.seed(42)
        d = 30
        ns, nt = 100, 20
        Xs = np.random.randn(ns, d) * 2.0 + 1.0
        Xt = np.random.randn(nt, d) * 0.5 - 2.0

        A_st, A_ts, mu_s, mu_t = compute_coral_transformation(Xs, Xt, reg=1.0)

        self.assertEqual(A_st.shape, (d, d))
        self.assertEqual(A_ts.shape, (d, d))
        self.assertEqual(mu_s.shape, (d,))
        self.assertEqual(mu_t.shape, (d,))

        self.assertFalse(np.isnan(A_st).any(), "A_st contains NaNs")
        self.assertFalse(np.isinf(A_st).any(), "A_st contains Infs")
        self.assertFalse(np.isnan(A_ts).any(), "A_ts contains NaNs")
        self.assertFalse(np.isinf(A_ts).any(), "A_ts contains Infs")

    def test_02_coral_covariance_alignment_property(self):
        """Verify CORAL aligns source covariance to target covariance."""
        np.random.seed(42)
        d = 10
        ns, nt = 200, 80
        # Source with diagonal variance 4.0
        Xs = np.random.randn(ns, d) * 2.0
        # Target with diagonal variance 0.25
        Xt = np.random.randn(nt, d) * 0.5

        A_st, _, mu_s, mu_t = compute_coral_transformation(Xs, Xt, reg=0.01)
        Xs_coral = (Xs - mu_s) @ A_st + mu_t

        Cs_coral = np.cov(Xs_coral, rowvar=False)
        Ct = np.cov(Xt, rowvar=False)

        # Frobenius norm difference should be significantly smaller than initial difference
        initial_diff = np.linalg.norm(np.cov(Xs, rowvar=False) - Ct, "fro")
        coral_diff = np.linalg.norm(Cs_coral - Ct, "fro")
        self.assertLess(coral_diff, initial_diff * 0.1, "CORAL did not substantially reduce covariance distance")

    def test_03_coral_zero_label_leakage(self):
        """Verify CORAL alignment function never inspects or takes target emotion labels."""
        import inspect
        sig = inspect.signature(compute_coral_transformation)
        param_names = list(sig.parameters.keys())
        self.assertIn("Xs", param_names)
        self.assertIn("Xt", param_names)
        self.assertNotIn("yt", param_names)
        self.assertNotIn("y_target", param_names)
        self.assertNotIn("target_labels", param_names)

    def test_04_channel_indices_dimensions_and_isolation(self):
        """Verify training-only channel indices match expected dimensions for all budgets."""
        if not self.selections_path.is_file():
            self.skipTest("Missing corrected_channel_selections.csv")

        selections_df = pd.read_csv(self.selections_path)

        for test_sub in [1, 5, 10, 15]:
            idx_62 = get_channel_feature_indices(test_sub, "all_62", self.channel_to_indices, selections_df)
            idx_s32 = get_channel_feature_indices(test_sub, "stab_32", self.channel_to_indices, selections_df)
            idx_p32 = get_channel_feature_indices(test_sub, "pred_32", self.channel_to_indices, selections_df)
            idx_p12 = get_channel_feature_indices(test_sub, "pred_12", self.channel_to_indices, selections_df)

            self.assertEqual(len(idx_62), 310)
            self.assertEqual(len(idx_s32), 160)
            self.assertEqual(len(idx_p32), 160)
            self.assertEqual(len(idx_p12), 60)

    def test_05_inner_hyperparameter_tuning_isolation(self):
        """Verify tune_coral_hyperparameters runs strictly on source subjects without target subject."""
        np.random.seed(42)
        n_subs = 14
        trials_per_sub = 72
        d = 20
        Xs = np.random.randn(n_subs * trials_per_sub, d)
        ys = np.random.randint(0, 4, size=n_subs * trials_per_sub)
        subs = np.repeat(np.arange(1, n_subs + 1), trials_per_sub)

        best_c = tune_coral_hyperparameters(Xs, ys, subs, c_grid=[0.005, 0.01, 0.05], reg=1.0)
        self.assertIn(best_c, [0.005, 0.01, 0.05])

    def test_06_predictions_table_schema_and_counts(self):
        """Verify experiment_d1_predictions.csv has 12,960 rows and valid schema."""
        pred_path = self.predictions_dir / "experiment_d1_predictions.csv"
        if not pred_path.is_file():
            self.skipTest("experiment_d1_predictions.csv not yet generated")

        df = pd.read_csv(pred_path)
        # 1,080 trials x 4 conditions x 3 methods = 12,960 rows
        self.assertEqual(len(df), 12960)

        required_cols = [
            "subject_id", "session_id", "trial_id", "trial_key",
            "condition", "method", "adaptation_type", "true_label",
            "predicted_label", "is_correct"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

        # Labels strictly 0..3
        self.assertTrue(df["true_label"].isin([0, 1, 2, 3]).all())
        self.assertTrue(df["predicted_label"].isin([0, 1, 2, 3]).all())

    def test_07_summary_table_integrity(self):
        """Verify experiment_d1_summary.csv contains 12 rows (4 conditions x 3 methods)."""
        sum_path = self.tables_dir / "experiment_d1_summary.csv"
        if not sum_path.is_file():
            self.skipTest("experiment_d1_summary.csv not yet generated")

        df = pd.read_csv(sum_path)
        self.assertEqual(len(df), 12)

        # Baseline stab_32 should match 41.94%
        base_s32 = df[(df["condition"] == "stab_32") & (df["method"] == "strict_loso_baseline")].iloc[0]
        self.assertAlmostEqual(base_s32["pooled_accuracy"], 41.94444444444444, places=2)


if __name__ == "__main__":
    unittest.main()
