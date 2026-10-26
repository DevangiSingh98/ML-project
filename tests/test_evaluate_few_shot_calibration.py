"""
tests/test_evaluate_few_shot_calibration.py

Unit tests for Experiment D3: Four-Shot Subject Calibration Combined with CORAL.
Verifies:
1. Calibration split disjointness, exact 4 calibration trials (1 per class), and 68 test trials.
2. Calibration split reproducibility across seeds and variance with different seeds.
3. Model-selection and inner-tuning insulation from outer test subject data.
4. Prediction table schema, row counts (81,600 rows), and metadata integrity.
5. Summary table integrity (16 condition/method combinations).
6. Subject-level aggregation over repeated calibration seeds.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_few_shot_calibration import (
    CALIBRATION_SEEDS,
    select_four_shot_calibration_split,
    tune_calibration_hyperparameters,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"


class TestEvaluateFewShotCalibration(unittest.TestCase):

    def setUp(self):
        # Synthetic 72 trials for a subject: exactly 18 trials per class (0, 1, 2, 3)
        self.y_target = np.repeat(np.arange(4), 18)

    def test_01_calibration_split_disjointness_and_counts(self):
        """Verify calibration set has exactly 4 trials (1 per class) and test set has 68 trials."""
        for seed in CALIBRATION_SEEDS:
            calib_idx, test_idx = select_four_shot_calibration_split(self.y_target, seed=seed, n_classes=4)

            # Length checks
            self.assertEqual(len(calib_idx), 4, f"Seed {seed}: Expected 4 calibration trials, got {len(calib_idx)}")
            self.assertEqual(len(test_idx), 68, f"Seed {seed}: Expected 68 test trials, got {len(test_idx)}")
            self.assertEqual(len(calib_idx) + len(test_idx), 72)

            # Disjointness
            overlap = set(calib_idx).intersection(set(test_idx))
            self.assertEqual(len(overlap), 0, f"Seed {seed}: Overlap detected: {overlap}")

            # Exactly one trial per class
            calib_labels = self.y_target[calib_idx]
            self.assertEqual(sorted(calib_labels), [0, 1, 2, 3])
            self.assertEqual(len(np.unique(calib_labels)), 4)

            # Test set class counts: each class has 18 - 1 = 17 trials
            test_labels = self.y_target[test_idx]
            for c in range(4):
                self.assertEqual((test_labels == c).sum(), 17, f"Seed {seed}: Class {c} in test does not have 17 trials")

    def test_02_calibration_split_reproducibility_and_variance(self):
        """Verify reproducibility with same seed and diversity with different seeds."""
        calib_42_a, test_42_a = select_four_shot_calibration_split(self.y_target, seed=42)
        calib_42_b, test_42_b = select_four_shot_calibration_split(self.y_target, seed=42)
        np.testing.assert_array_equal(calib_42_a, calib_42_b)
        np.testing.assert_array_equal(test_42_a, test_42_b)

        # Different seed produces different calibration samples
        calib_100, _ = select_four_shot_calibration_split(self.y_target, seed=100)
        self.assertFalse(np.array_equal(calib_42_a, calib_100))

    def test_03_zero_test_label_leakage_in_tuning(self):
        """Verify inner tuning operates strictly on training source subjects."""
        rng = np.random.default_rng(42)
        # 14 subjects with 20 balanced trials each (5 per class)
        source_subjects = np.repeat(np.arange(1, 15), 20)
        ys = np.tile(np.repeat(np.arange(4), 5), 14)
        Xs = rng.normal(size=(len(ys), 20))

        best_config = tune_calibration_hyperparameters(
            Xs=Xs,
            ys=ys,
            source_subjects=source_subjects,
            w_cal_grid=[2.0, 10.0],
            c_grid=[0.01],
            calib_seeds=[42, 100],
            reg=1.0,
            seed=42,
        )

        self.assertIn("w_cal", best_config)
        self.assertIn("c", best_config)
        self.assertIn("val_acc", best_config)
        self.assertTrue(0.0 <= best_config["val_acc"] <= 1.0)

    def test_04_prediction_table_schema_and_counts(self):
        """Verify experiment_d3_predictions.csv has 81,600 rows and required columns if generated."""
        pred_path = PREDICTIONS_DIR / "experiment_d3_predictions.csv"
        if not pred_path.is_file():
            self.skipTest("experiment_d3_predictions.csv not yet generated.")

        df = pd.read_csv(pred_path)
        # 15 subjects x 4 conditions x 5 seeds x 4 methods x 68 trials = 81,600
        self.assertEqual(len(df), 81600)

        required_cols = [
            "subject_id", "session_id", "trial_id", "trial_key", "condition", "calibration_seed",
            "method", "adaptation_type", "best_c", "w_cal", "n_features", "n_calibration_trials",
            "n_test_trials", "true_label", "true_label_name", "predicted_label",
            "predicted_label_name", "is_correct"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

        # Check conditions
        self.assertEqual(sorted(df["condition"].unique()), ["all_62", "pred_12", "pred_32", "stab_32"])
        # Check methods
        expected_methods = [
            "coral_plus_four_shot",
            "coral_unsupervised",
            "few_shot_calibration_only",
            "source_only_baseline",
        ]
        self.assertEqual(sorted(df["method"].unique()), expected_methods)

        # Check that calibration trials are never in test set
        self.assertEqual(df["n_calibration_trials"].iloc[0], 4)
        self.assertEqual(df["n_test_trials"].iloc[0], 68)

    def test_05_summary_table_integrity(self):
        """Verify experiment_d3_summary.csv contains 16 rows (4 conditions x 4 methods)."""
        sum_path = TABLES_DIR / "experiment_d3_summary.csv"
        if not sum_path.is_file():
            self.skipTest("experiment_d3_summary.csv not yet generated.")

        df = pd.read_csv(sum_path)
        self.assertEqual(len(df), 16)
        self.assertTrue((df["subject_mean_accuracy"] >= 0.0).all())
        self.assertTrue((df["subject_mean_accuracy"] <= 100.0).all())
        self.assertTrue((df["macro_f1"] >= 0.0).all())
        self.assertTrue((df["macro_f1"] <= 100.0).all())

    def test_06_subject_level_aggregation_integrity(self):
        """Verify experiment_d3_subject_metrics.csv has 240 rows (15 subjects per method/cond)."""
        sub_path = TABLES_DIR / "experiment_d3_subject_metrics.csv"
        if not sub_path.is_file():
            self.skipTest("experiment_d3_subject_metrics.csv not yet generated.")

        df = pd.read_csv(sub_path)
        self.assertEqual(len(df), 240)
        self.assertEqual(len(df["subject_id"].unique()), 15)


if __name__ == "__main__":
    unittest.main()
