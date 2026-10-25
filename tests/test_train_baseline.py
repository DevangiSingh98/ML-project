"""
Tests for Phase B Baseline training components, pipeline isolation, and prediction alignment.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocessing import create_baseline_pipeline
from src.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix_df,
)


class TestBaselineComponents(unittest.TestCase):
    """Test suite for baseline model construction, scaling isolation, and outputs."""

    def test_01_pipeline_structure(self):
        """Verify baseline pipeline has StandardScaler and SVC(probability=False)."""
        pipe = create_baseline_pipeline(C=1.0, gamma="scale")
        self.assertIsInstance(pipe, Pipeline)
        self.assertIn("scaler", pipe.named_steps)
        self.assertIn("svc", pipe.named_steps)
        self.assertIsInstance(pipe.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(pipe.named_steps["svc"], SVC)
        self.assertFalse(pipe.named_steps["svc"].probability, "Phase B must NOT use probability=True.")

    def test_02_scaling_isolation_check(self):
        """Verify that StandardScaler fits exclusively on training data and does not leak test statistics."""
        # Simulated train and test data with clearly different distributions
        X_train = np.random.normal(loc=10.0, scale=2.0, size=(100, 5))
        y_train = np.random.randint(0, 4, size=100)
        X_test = np.random.normal(loc=100.0, scale=20.0, size=(20, 5))

        pipe = create_baseline_pipeline()
        pipe.fit(X_train, y_train)

        # Scaler mean must match X_train mean, not X_test
        np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, X_train.mean(axis=0), rtol=1e-5)
        # Scaler must transform X_test without altering its stored mean/variance
        transformed_test = pipe.named_steps["scaler"].transform(X_test)
        self.assertEqual(transformed_test.shape, (20, 5))
        np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, X_train.mean(axis=0), rtol=1e-5)

    def test_03_prediction_dataframe_alignment_logic(self):
        """Verify that prediction records preserve full metadata alignment."""
        dummy_meta = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "session_id": [1, 1, 1],
            "trial_id": [1, 1, 1],
            "window_id": [0, 1, 2],
            "label": [0, 1, 2],
            "label_name": ["neutral", "sad", "fear"],
        })
        y_pred = np.array([0, 1, 3])
        dummy_meta["predicted_label"] = y_pred
        dummy_meta["is_correct"] = (dummy_meta["label"] == dummy_meta["predicted_label"]).astype(int)

        self.assertEqual(list(dummy_meta["is_correct"]), [1, 1, 0])
        self.assertEqual(len(dummy_meta), 3)

    def test_04_confusion_matrix_sum_matches_predictions(self):
        """Verify confusion matrix sum equals total prediction count."""
        y_true = np.random.randint(0, 4, size=200)
        y_pred = np.random.randint(0, 4, size=200)

        cm_df = compute_confusion_matrix_df(y_true, y_pred)
        self.assertEqual(cm_df.values.sum(), 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
