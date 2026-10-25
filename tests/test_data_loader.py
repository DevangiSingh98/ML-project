"""
Tests for SEED-IV Data Loader (src/data_loader.py).

Uses Python's standard `unittest` framework (no external dependencies required)
so tests execute reliably in environments where `pytest` is not yet installed.
Compatible with `pytest` execution once authorized and installed.
"""

import os
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.data_loader import (
    load_config,
    load_channel_order,
    generate_feature_mappings,
    load_seed_iv_dataset,
    generate_dataset_summary,
    OFFICIAL_LABELS,
    LABEL_NAMES,
    SUBJECT_GENDERS,
    DatasetContainer,
)


class TestSEEDIVDataLoader(unittest.TestCase):
    """Validation test suite for SEED-IV data loading and formatting."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config.yaml")
        cls.data_dir = Path(cls.config["dataset"]["data_dir"])

    def test_01_config_structure(self):
        """Verify configuration contains all locked project parameters."""
        self.assertEqual(self.config["dataset"]["name"], "SEED-IV")
        self.assertEqual(self.config["dataset"]["feature_type"], "de_LDS")
        self.assertEqual(self.config["dataset"]["sessions"], [1, 2, 3])
        self.assertEqual(self.config["dataset"]["total_channels"], 62)
        self.assertEqual(self.config["dataset"]["num_frequency_bands"], 5)
        self.assertEqual(self.config["dataset"]["total_features"], 310)
        self.assertEqual(self.config["evaluation"]["strategy"], "LOSO")
        self.assertEqual(self.config["channel_reduction"]["budgets"], [8, 12, 16, 24, 32, 62])

    def test_02_channel_order_loading(self):
        """Verify official 62 channels are correctly loaded and ordered."""
        channels = load_channel_order(self.data_dir)
        self.assertEqual(len(channels), 62)
        self.assertEqual(channels[0], "FP1")
        self.assertEqual(channels[1], "FPZ")
        self.assertEqual(channels[2], "FP2")
        self.assertEqual(channels[-1], "CB2")
        self.assertEqual(len(set(channels)), 62, "All 62 channel names must be unique.")

    def test_03_feature_mappings(self):
        """Verify deterministic 62-channel x 5-band feature name and index mapping."""
        channels = load_channel_order(self.data_dir)
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
        feat_names, ch_to_idx = generate_feature_mappings(channels, bands)

        self.assertEqual(len(feat_names), 310)
        self.assertEqual(len(ch_to_idx), 62)

        # Check first channel (FP1)
        self.assertEqual(feat_names[0], "FP1_delta")
        self.assertEqual(feat_names[4], "FP1_gamma")
        self.assertEqual(ch_to_idx["FP1"], [0, 1, 2, 3, 4])

        # Check last channel (CB2)
        self.assertEqual(feat_names[305], "CB2_delta")
        self.assertEqual(feat_names[309], "CB2_gamma")
        self.assertEqual(ch_to_idx["CB2"], [305, 306, 307, 308, 309])

        # Slicing test: selecting 8 channels yields 40 features
        k_channels = channels[:8]
        k_indices = [idx for ch in k_channels for idx in ch_to_idx[ch]]
        self.assertEqual(len(k_indices), 40)

    def test_04_official_labels_integrity(self):
        """Verify official labels for all 3 sessions have 24 balanced trials."""
        for sess_id in [1, 2, 3]:
            labels = OFFICIAL_LABELS[sess_id]
            self.assertEqual(len(labels), 24, f"Session {sess_id} must have 24 trial labels.")
            counts = {c: labels.count(c) for c in range(4)}
            for c in range(4):
                self.assertEqual(counts[c], 6, f"Session {sess_id} must have 6 trials for class {c}.")

    def test_05_single_session_loading(self):
        """Test loading a single session (Session 1) to verify array validity without loading all 3."""
        container = load_seed_iv_dataset(
            config=self.config,
            sessions_override=[1],
        )
        self.assertIsInstance(container, DatasetContainer)
        self.assertEqual(container.X.ndim, 2)
        self.assertEqual(container.X.shape[1], 310)
        self.assertEqual(container.X.shape[0], 12765)  # 15 subjects * 851 windows
        self.assertEqual(len(container.y), 12765)
        self.assertFalse(np.isnan(container.X).any(), "Feature matrix contains NaN.")
        self.assertFalse(np.isinf(container.X).any(), "Feature matrix contains Inf.")

    def test_06_metadata_integrity(self):
        """Verify metadata rows align with expected bounds and labels."""
        container = load_seed_iv_dataset(
            config=self.config,
            sessions_override=[1],
        )
        meta = container.metadata
        self.assertEqual(len(meta), 12765)
        self.assertEqual(set(meta["subject_id"].unique()), set(range(1, 16)))
        self.assertEqual(list(meta["session_id"].unique()), [1])
        self.assertEqual(set(meta["trial_id"].unique()), set(range(1, 25)))
        self.assertEqual(set(meta["label"].unique()), {0, 1, 2, 3})
        # Check correspondence between y and metadata label column
        np.testing.assert_array_equal(container.y, meta["label"].to_numpy())

    def test_07_summary_table_generation(self):
        """Verify summary CSV generation and inspect structure."""
        summary_path = Path("outputs/tables/dataset_summary.csv")
        self.assertTrue(summary_path.is_file(), "outputs/tables/dataset_summary.csv must exist.")

        df = pd.read_csv(summary_path)
        self.assertIn("category", df.columns)
        self.assertIn("total_windows", df.columns)

        # Check total row
        total_row = df[df["category"] == "total"]
        self.assertEqual(len(total_row), 1)
        self.assertEqual(int(total_row["total_windows"].values[0]), 37575)
        self.assertEqual(int(total_row["sessions"].values[0]), 3)
        self.assertEqual(int(total_row["trials"].values[0]), 1080)

        # Check 15 subject rows
        sub_rows = df[df["category"] == "subject"]
        self.assertEqual(len(sub_rows), 15)

        # Check 3 session rows
        sess_rows = df[df["category"] == "session"]
        self.assertEqual(len(sess_rows), 3)

        # Check 4 class rows
        class_rows = df[df["category"] == "class"]
        self.assertEqual(len(class_rows), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
