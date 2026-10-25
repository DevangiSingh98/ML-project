"""
Unit tests for LOSO split generation, leakage safeguards, and metrics computation.

Verifies:
- Complete subject disjointness for all 15 outer LOSO folds.
- Exactly one subject held out per fold, and every subject held out exactly once.
- Strict fold sample counts (35,070 train, 2,505 test).
- Inner GroupKFold subject disjointness.
- Determinism and reproducibility of split generation.
- Classification metrics accuracy and edge-case handling on known examples.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.data_loader import load_seed_iv_dataset, load_config
from src.subject_split import get_loso_splits, get_inner_group_splits
from src.metrics import compute_classification_metrics, compute_confusion_matrix_df


class TestSubjectSplitAndMetrics(unittest.TestCase):
    """Test suite for subject splitting, leakage checks, and metrics."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config.yaml")
        # Load dataset metadata for split testing
        cls.dataset = load_seed_iv_dataset(config=cls.config)
        cls.metadata = cls.dataset.metadata

    def test_01_loso_15_folds_generated(self):
        """Verify that exactly 15 outer folds are generated."""
        splits = list(get_loso_splits(self.metadata))
        self.assertEqual(len(splits), 15)

    def test_02_loso_subject_disjointness_and_counts(self):
        """Verify strict subject disjointness and window counts in every fold."""
        seen_test_subjects = []

        for fold_idx, test_sub, train_idx, test_idx in get_loso_splits(self.metadata):
            seen_test_subjects.append(test_sub)

            # Check disjointness
            train_subs = set(self.metadata.iloc[train_idx]["subject_id"].unique())
            test_subs = set(self.metadata.iloc[test_idx]["subject_id"].unique())

            self.assertTrue(
                train_subs.isdisjoint(test_subs),
                f"Fold {fold_idx} failed: Train subjects {train_subs} overlap with test {test_subs}"
            )
            self.assertEqual(test_subs, {test_sub})
            self.assertEqual(len(train_subs), 14)

            # Check window counts (pooled 3 sessions)
            self.assertEqual(len(test_idx), 2505, f"Fold {fold_idx} test count mismatch")
            self.assertEqual(len(train_idx), 35070, f"Fold {fold_idx} train count mismatch")
            self.assertEqual(len(train_idx) + len(test_idx), 37575)

        # Check every subject 1..15 held out exactly once
        self.assertEqual(sorted(seen_test_subjects), list(range(1, 16)))

    def test_03_split_reproducibility(self):
        """Verify that generating LOSO splits repeatedly yields identical index arrays."""
        splits_1 = list(get_loso_splits(self.metadata))
        splits_2 = list(get_loso_splits(self.metadata))

        for (f1, sub1, tr1, te1), (f2, sub2, tr2, te2) in zip(splits_1, splits_2):
            self.assertEqual(f1, f2)
            self.assertEqual(sub1, sub2)
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_04_inner_group_splits(self):
        """Verify inner GroupKFold splits strictly isolate subjects."""
        # Test on outer training fold 1 (14 subjects)
        _, _, train_idx, _ = next(get_loso_splits(self.metadata))
        train_meta = self.metadata.iloc[train_idx].reset_index(drop=True)

        inner_splits = list(get_inner_group_splits(train_meta, n_splits=5))
        self.assertEqual(len(inner_splits), 5)

        for inner_idx, inner_tr, inner_val in inner_splits:
            inner_tr_subs = set(train_meta.iloc[inner_tr]["subject_id"].unique())
            inner_val_subs = set(train_meta.iloc[inner_val]["subject_id"].unique())
            self.assertTrue(
                inner_tr_subs.isdisjoint(inner_val_subs),
                f"Inner fold {inner_idx} leakage detected!"
            )

    def test_05_metrics_calculation_known_example(self):
        """Verify metrics computation against a known hand-calculated test case."""
        # 8 samples, 4 classes (2 samples per class)
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        # Preds:
        # Class 0: [0, 1] -> 1 correct, 1 wrong (predicted as 1)
        # Class 1: [1, 1] -> 2 correct
        # Class 2: [2, 2] -> 2 correct
        # Class 3: [3, 0] -> 1 correct, 1 wrong (predicted as 0)
        # Correct: 6 / 8 = 0.75 accuracy
        y_pred = np.array([0, 1, 1, 1, 2, 2, 3, 0])

        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        # Per class recall: class 0 = 0.5, class 1 = 1.0, class 2 = 1.0, class 3 = 0.5
        # Balanced accuracy = (0.5 + 1.0 + 1.0 + 0.5) / 4 = 0.75
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.75)

        self.assertAlmostEqual(metrics["recall_neutral"], 0.5)
        self.assertAlmostEqual(metrics["recall_sad"], 1.0)
        self.assertAlmostEqual(metrics["recall_fear"], 1.0)
        self.assertAlmostEqual(metrics["recall_happy"], 0.5)

        # Confusion matrix checks
        cm_df = compute_confusion_matrix_df(y_true, y_pred)
        self.assertEqual(cm_df.shape, (4, 4))
        self.assertEqual(cm_df.loc["true_neutral", "pred_neutral"], 1)
        self.assertEqual(cm_df.loc["true_neutral", "pred_sad"], 1)
        self.assertEqual(cm_df.loc["true_sad", "pred_sad"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
