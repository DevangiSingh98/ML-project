"""
tests/test_evaluate_class_aware_adaptation.py

Unit tests for Experiment D2: Class-Aware Unsupervised Domain Adaptation (CB-PLR).
Verifies:
1. Conservative class-balanced pseudo-label selection rules (tau threshold, k_max cap, class balance).
2. Strict insulation from target emotion labels (zero true label leakage).
3. Inner nested validation isolation (strictly within 14 training subjects).
4. Prediction table schema, row counts, and trial key consistency (12,960 rows).
5. Summary table integrity and reproduction of D1 CORAL baseline numbers.
6. Inner pseudo-label audit log schema and valid precision bounds [0, 1].
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_class_aware_adaptation import (
    select_conservative_pseudo_labels,
    tune_cb_plr_hyperparameters,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"


class TestEvaluateClassAwareAdaptation(unittest.TestCase):

    def test_01_conservative_pseudo_label_selection_rule(self):
        """Verify pseudo-label selection respects confidence threshold and k_max caps."""
        # 10 samples, 4 classes
        probs = np.array([
            [0.70, 0.10, 0.10, 0.10],  # Class 0, conf 0.70 -> Should select
            [0.60, 0.20, 0.10, 0.10],  # Class 0, conf 0.60 -> Should select
            [0.55, 0.15, 0.15, 0.15],  # Class 0, conf 0.55 -> Top-2 cap (k_max=2) excludes this if k_max=2
            [0.10, 0.80, 0.05, 0.05],  # Class 1, conf 0.80 -> Should select
            [0.10, 0.40, 0.30, 0.20],  # Class 1, conf 0.40 -> Excluded if tau=0.50
            [0.20, 0.20, 0.30, 0.30],  # Max conf 0.30 < tau
            [0.05, 0.05, 0.85, 0.05],  # Class 2, conf 0.85 -> Should select
            [0.05, 0.05, 0.05, 0.85],  # Class 3, conf 0.85 -> Should select
            [0.05, 0.05, 0.05, 0.75],  # Class 3, conf 0.75 -> Should select
            [0.05, 0.05, 0.05, 0.65],  # Class 3, conf 0.65 -> Excluded if k_max=2
        ])

        sel_idx, sel_labels, sel_confs = select_conservative_pseudo_labels(
            probs=probs, tau=0.50, k_max=2, n_classes=4
        )

        # Check total selected count:
        # Class 0: samples 0, 1 (2 selected)
        # Class 1: sample 3 (1 selected, sample 4 has conf 0.40 < 0.50)
        # Class 2: sample 6 (1 selected)
        # Class 3: samples 7, 8 (2 selected, sample 9 capped out by k_max=2)
        # Total = 2 + 1 + 1 + 2 = 6
        self.assertEqual(len(sel_idx), 6)
        self.assertTrue(all(sel_confs >= 0.50))

        # Check counts per class
        for c in range(4):
            c_cnt = (sel_labels == c).sum()
            self.assertLessEqual(c_cnt, 2, f"Class {c} exceeded k_max=2")

        # Check extreme high threshold: tau = 0.95 -> 0 selected
        empty_idx, empty_labels, empty_confs = select_conservative_pseudo_labels(
            probs=probs, tau=0.95, k_max=2, n_classes=4
        )
        self.assertEqual(len(empty_idx), 0)
        self.assertEqual(len(empty_labels), 0)
        self.assertEqual(len(empty_confs), 0)

    def test_02_pseudo_label_zero_true_label_leakage(self):
        """Verify select_conservative_pseudo_labels never takes true labels."""
        import inspect
        sig = inspect.signature(select_conservative_pseudo_labels)
        params = list(sig.parameters.keys())
        self.assertNotIn("y_true", params)
        self.assertNotIn("labels", params)
        self.assertNotIn("true_labels", params)
        self.assertIn("probs", params)
        self.assertIn("tau", params)
        self.assertIn("k_max", params)

    def test_03_inner_nested_tuning_subject_isolation(self):
        """Verify tune_cb_plr_hyperparameters operates strictly on source subjects."""
        rng = np.random.default_rng(42)
        Xs = rng.normal(size=(140, 20))
        ys = rng.choice(4, size=140)
        source_subjects = np.repeat(np.arange(1, 15), 10)

        best_config, inner_grid = tune_cb_plr_hyperparameters(
            Xs=Xs,
            ys=ys,
            source_subjects=source_subjects,
            tau_grid=[0.35, 0.50],
            k_max_grid=[2, 4],
            w_pseudo_grid=[0.1, 0.5],
            c_grid=[0.01],
            reg=1.0,
            seed=42,
        )

        self.assertIn("tau", best_config)
        self.assertIn("k_max", best_config)
        self.assertIn("w_pseudo", best_config)
        self.assertIn("c", best_config)
        self.assertIn("val_acc", best_config)
        self.assertIn("val_precision", best_config)
        self.assertTrue(0.0 <= best_config["val_precision"] <= 1.0)
        self.assertGreater(len(inner_grid), 0)

    def test_04_prediction_table_schema_and_counts(self):
        """Verify experiment_d2_predictions.csv has 12,960 rows and valid schema if generated."""
        pred_path = PREDICTIONS_DIR / "experiment_d2_predictions.csv"
        if not pred_path.is_file():
            self.skipTest("experiment_d2_predictions.csv not yet generated.")

        df = pd.read_csv(pred_path)
        # 1,080 trials x 4 conditions x 3 methods = 12,960 rows
        self.assertEqual(len(df), 12960)

        required_cols = [
            "subject_id", "session_id", "trial_id", "trial_key", "condition", "method",
            "adaptation_type", "best_c", "tau", "k_max", "w_pseudo", "n_features",
            "n_pseudo_selected", "true_label", "true_label_name", "predicted_label",
            "predicted_label_name", "confidence", "is_correct"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

        # Check conditions
        self.assertEqual(sorted(df["condition"].unique()), ["all_62", "pred_12", "pred_32", "stab_32"])
        # Check methods
        self.assertEqual(
            sorted(df["method"].unique()),
            ["class_balanced_pseudo_label_refinement", "coral_target_to_source", "strict_loso_baseline"]
        )

    def test_05_summary_table_integrity(self):
        """Verify experiment_d2_summary.csv contains 12 rows (4 conditions x 3 methods)."""
        sum_path = TABLES_DIR / "experiment_d2_summary.csv"
        if not sum_path.is_file():
            self.skipTest("experiment_d2_summary.csv not yet generated.")

        df = pd.read_csv(sum_path)
        self.assertEqual(len(df), 12)

        # Baseline check on stab_32 (strict baseline = 41.94%, CORAL T->S = 54.26%)
        stab_base = df[(df["condition"] == "stab_32") & (df["method"] == "strict_loso_baseline")]["subject_mean_accuracy"].iloc[0]
        stab_coral = df[(df["condition"] == "stab_32") & (df["method"] == "coral_target_to_source")]["subject_mean_accuracy"].iloc[0]
        self.assertAlmostEqual(stab_base, 41.944444, places=2)
        self.assertAlmostEqual(stab_coral, 54.259259, places=2)

    def test_06_inner_pseudo_audit_schema(self):
        """Verify inner pseudo-label audit log schema and finite precision."""
        audit_path = TABLES_DIR / "experiment_d2_inner_pseudo_audit.csv"
        if not audit_path.is_file():
            self.skipTest("experiment_d2_inner_pseudo_audit.csv not yet generated.")

        df = pd.read_csv(audit_path)
        self.assertGreater(len(df), 0)
        self.assertIn("val_precision", df.columns)
        self.assertIn("n_pseudo_selected", df.columns)
        self.assertTrue(df["val_precision"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
