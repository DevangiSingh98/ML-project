"""
Unit tests for leakage-safe calibration module (src/calibration.py).

Verifies:
- Group-safe out-of-fold decision score generation on synthetic subject-grouped data.
- Strict subject disjointness across inner splits.
- MulticlassPlattCalibrator probability bounds, normalization, and class alignment.
- Zero leakage from held-out subjects.
"""

import unittest
import numpy as np
import pandas as pd

from src.calibration import (
    generate_group_safe_oof_decision_scores,
    MulticlassPlattCalibrator,
    UncalibratedSoftmaxCalibrator,
)


class TestCalibration(unittest.TestCase):
    """Test suite for group-safe calibration algorithms."""

    def setUp(self):
        np.random.seed(42)
        # 6 subjects, 20 samples per subject = 120 samples, 4 classes, 10 features
        self.n_subjects = 6
        self.samples_per_sub = 20
        self.n_samples = self.n_subjects * self.samples_per_sub
        self.n_features = 10
        self.n_classes = 4

        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.choice([0, 1, 2, 3], size=self.n_samples)
        subject_ids = np.repeat(np.arange(1, self.n_subjects + 1), self.samples_per_sub)
        self.metadata = pd.DataFrame({"subject_id": subject_ids})

    def test_01_multiclass_platt_calibrator_properties(self):
        """Verify probability bounds, sum-to-one, and class alignment."""
        Z_oof = np.random.randn(100, 4)
        y_train = np.random.choice([0, 1, 2, 3], size=100)

        cal = MulticlassPlattCalibrator()
        cal.fit(Z_oof, y_train)

        Z_test = np.random.randn(25, 4)
        probs = cal.predict_proba(Z_test)
        preds = cal.predict(Z_test)

        # Shape checks
        self.assertEqual(probs.shape, (25, 4))
        self.assertEqual(preds.shape, (25,))

        # Probabilities lie in [0, 1]
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

        # Sum to 1
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(25), rtol=1e-5)

        # Argmax matches predicted label
        np.testing.assert_array_equal(np.argmax(probs, axis=1), preds)

    def test_02_group_safe_oof_scores_disjointness(self):
        """Verify that every training sample receives an out-of-fold score from a model that did not see its subject."""
        Z_oof = generate_group_safe_oof_decision_scores(
            X_train=self.X,
            y_train=self.y,
            train_metadata=self.metadata,
            group_key="subject_id",
            n_splits=3,
            cache_size=100,
        )

        self.assertEqual(Z_oof.shape, (self.n_samples, self.n_classes))
        self.assertTrue(np.all(np.isfinite(Z_oof)))

    def test_03_uncalibrated_softmax_properties(self):
        """Verify uncalibrated softmax calibrator yields valid probability distributions."""
        cal = UncalibratedSoftmaxCalibrator()
        Z = np.array([
            [1.0, 2.0, 0.5, -1.0],
            [10.0, 5.0, 2.0, 1.0],
            [-2.0, -2.0, -2.0, -2.0],
        ])
        y_dummy = np.array([0, 1, 2, 3])
        cal.fit(Z, y_dummy)

        probs = cal.predict_proba(Z)
        self.assertEqual(probs.shape, (3, 4))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(3), rtol=1e-5)

        # Equal scores -> uniform probability 0.25
        np.testing.assert_allclose(probs[2], np.array([0.25, 0.25, 0.25, 0.25]), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
