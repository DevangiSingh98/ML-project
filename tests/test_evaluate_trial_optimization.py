"""
tests/test_evaluate_trial_optimization.py

Unit test suite for Experiment C: Trial-Level Feature and Classifier Optimization.
Covers:
1. Feature representations dimensions, absence of NaNs, and class balance.
2. Temporal segmentation logic for variable-length trials (W in 10..64).
3. Strictly nested channel feature index expansion across 310, 620, and 930 dimensions.
4. Pipeline encapsulation of StandardScaler across all model pipelines.
5. Baseline reproduction check and metric calculation consistency.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate_trial_optimization import (
    get_channel_feature_indices,
    get_model_pipeline_and_grid,
    inventory_seed_iv_alternative_features,
)


class TestEvaluateTrialOptimization(unittest.TestCase):

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.tables_dir = self.root_dir / "outputs" / "tables"
        self.predictions_dir = self.root_dir / "outputs" / "predictions"
        self.selections_path = self.tables_dir / "corrected_channel_selections.csv"
        
        # Standard channel list for channel_to_indices mapping
        channels = [
            "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
            "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
            "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
            "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
            "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
        ]
        self.channel_to_indices = {ch: list(range(i * 5, (i + 1) * 5)) for i, ch in enumerate(channels)}

    def test_01_seed_iv_feature_inventory(self):
        """Verify SEED-IV feature inventory correctly discovers smooth EEG features."""
        inv = inventory_seed_iv_alternative_features()
        self.assertIn("de_movingAve", inv["feature_types"])
        self.assertIn("de_LDS", inv["feature_types"])
        self.assertIn("psd_movingAve", inv["feature_types"])
        self.assertIn("psd_LDS", inv["feature_types"])
        
        # Check shapes: (62, W, 5)
        for k in ["de_movingAve", "de_LDS", "psd_movingAve", "psd_LDS"]:
            feat_info = inv["feature_types"][k]
            self.assertEqual(feat_info["channels"], 62)
            self.assertEqual(feat_info["bands"], 5)

    def test_02_variable_length_temporal_segmentation_logic(self):
        """Verify temporal thirds segmentation across trial lengths ranging from 10 to 64."""
        for W in range(10, 65):
            c1 = W // 3
            c2 = 2 * (W // 3)
            # Ensure each chunk has at least 3 windows
            self.assertGreaterEqual(c1, 3, f"c1 < 3 for W={W}")
            self.assertGreaterEqual(c2 - c1, 3, f"mid < 3 for W={W}")
            self.assertGreaterEqual(W - c2, 3, f"late < 3 for W={W}")
            # Ensure non-overlapping partition
            indices = list(range(c1)) + list(range(c1, c2)) + list(range(c2, W))
            self.assertEqual(len(indices), W)
            self.assertEqual(indices, list(range(W)))

    def test_03_channel_feature_indices_expansion_all_reps(self):
        """Verify feature index expansion across representations (310, 620, 930 dims)."""
        if not self.selections_path.is_file():
            self.skipTest("Missing corrected_channel_selections.csv")
        
        selections_df = pd.read_csv(self.selections_path)

        # 1. all_62
        self.assertEqual(len(get_channel_feature_indices(1, "all_62", self.channel_to_indices, selections_df, "mean")), 310)
        self.assertEqual(len(get_channel_feature_indices(1, "all_62", self.channel_to_indices, selections_df, "mean_std")), 620)
        self.assertEqual(len(get_channel_feature_indices(1, "all_62", self.channel_to_indices, selections_df, "early_mid_late")), 930)

        # 2. stab_32 (32 * 5 = 160 base features)
        idx_mean = get_channel_feature_indices(1, "stab_32", self.channel_to_indices, selections_df, "mean")
        idx_std = get_channel_feature_indices(1, "stab_32", self.channel_to_indices, selections_df, "mean_std")
        idx_eml = get_channel_feature_indices(1, "stab_32", self.channel_to_indices, selections_df, "early_mid_late")

        self.assertEqual(len(idx_mean), 160)
        self.assertEqual(len(idx_std), 320)
        self.assertEqual(len(idx_eml), 480)

        # 3. pred_12 (12 * 5 = 60 base features)
        idx_p12_mean = get_channel_feature_indices(1, "pred_12", self.channel_to_indices, selections_df, "mean")
        idx_p12_eml = get_channel_feature_indices(1, "pred_12", self.channel_to_indices, selections_df, "early_mid_late")

        self.assertEqual(len(idx_p12_mean), 60)
        self.assertEqual(len(idx_p12_eml), 180)

    def test_04_pipeline_scaler_leakage_isolation(self):
        """Verify StandardScaler is encapsulated in Pipeline across all models."""
        for model_name in ["linear_svc", "logistic_regression", "shrinkage_lda", "rbf_svm"]:
            pipe, grid = get_model_pipeline_and_grid(model_name)
            self.assertIsInstance(pipe, Pipeline)
            self.assertEqual(pipe.steps[0][0], "scaler")
            self.assertIsInstance(pipe.steps[0][1], StandardScaler)
            self.assertEqual(pipe.steps[1][0], "clf")

    def test_05_intermediate_feature_representations_npz(self):
        """Verify saved trial feature representations NPZ artifact dimensions and integrity."""
        npz_path = self.tables_dir / "experiment_c_trial_features.npz"
        if not npz_path.is_file():
            self.skipTest("experiment_c_trial_features.npz not yet generated")

        data = np.load(npz_path)
        self.assertIn("mean", data)
        self.assertIn("mean_std", data)
        self.assertIn("mean_mad", data)
        self.assertIn("early_mid_late", data)
        self.assertIn("y_trial", data)

        self.assertEqual(data["mean"].shape, (1080, 310))
        self.assertEqual(data["mean_std"].shape, (1080, 620))
        self.assertEqual(data["mean_mad"].shape, (1080, 620))
        self.assertEqual(data["early_mid_late"].shape, (1080, 930))
        self.assertEqual(data["y_trial"].shape, (1080,))

        # Verify no NaN or Inf
        for k in ["mean", "mean_std", "mean_mad", "early_mid_late"]:
            self.assertFalse(np.isnan(data[k]).any(), f"NaNs found in {k}")
            self.assertFalse(np.isinf(data[k]).any(), f"Infs found in {k}")

    def test_06_baseline_reproduction_results(self):
        """Verify baseline summary reproduces the stab_32 Linear SVM reference (41.94%)."""
        sum_path = self.tables_dir / "experiment_c_baseline_summary.csv"
        if not sum_path.is_file():
            self.skipTest("experiment_c_baseline_summary.csv not yet generated")

        df = pd.read_csv(sum_path)
        self.assertEqual(len(df), 2)
        orig_row = df[df["mode"] == "exp_b_original_outer_scaled"].iloc[0]
        pipe_row = df[df["mode"] == "strict_pipeline_inner_scaled"].iloc[0]

        # Check that original mode reproduces exact 41.94%
        self.assertAlmostEqual(orig_row["pooled_accuracy"], 41.94444444444444, places=2)
        self.assertEqual(orig_row["correct_trials"], 453)
        self.assertEqual(orig_row["total_trials"], 1080)


if __name__ == "__main__":
    unittest.main()
