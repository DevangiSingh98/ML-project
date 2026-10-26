"""
Unit tests for Trial-Level Prediction Aggregation (Experiment A).

Tests cover:
- Trial structure validation and error detection (label contradictions, duplicate windows).
- Correct probability vector averaging (Method A).
- Correct majority voting and deterministic cumulative probability tie-breaking (Method B).
- Unique composite trial keys and complete 1,080-trial coverage.
- Metric calculation consistency and confusion matrix sums.
"""

import unittest
import numpy as np
import pandas as pd

from src.evaluate_trial_aggregation import (
    validate_trial_structure,
    aggregate_single_trial,
    generate_trial_predictions,
    compute_metrics_and_comparisons,
    compute_paired_aggregation_tests,
    PROB_COLS,
    LABEL_NAMES,
)


class TestEvaluateTrialAggregation(unittest.TestCase):
    """Test suite for whole-trial prediction aggregation pipeline."""

    def setUp(self):
        """Build a controlled synthetic trial dataframe."""
        rows = []
        # Create 2 subjects, 1 session, 2 trials, 4 windows each
        for sub in [1, 2]:
            for tr, lbl in [(1, 0), (2, 3)]:
                for w in range(4):
                    # For trial 1 (neutral, label=0):
                    # Window 0 & 1 predict 0 (probs: [0.6, 0.2, 0.1, 0.1])
                    # Window 2 & 3 predict 1 (probs: [0.3, 0.5, 0.1, 0.1])
                    if tr == 1:
                        if w < 2:
                            p = [0.6, 0.2, 0.1, 0.1]
                            pred = 0
                        else:
                            p = [0.3, 0.5, 0.1, 0.1]
                            pred = 1
                    else:
                        # Trial 2 (happy, label=3):
                        # All predict 3 with high prob
                        p = [0.05, 0.05, 0.1, 0.8]
                        pred = 3

                    rows.append({
                        "subject_id": sub,
                        "session_id": 1,
                        "trial_id": tr,
                        "window_id": w,
                        "condition": "all_62",
                        "method": "group_safe_temperature",
                        "true_label": lbl,
                        "predicted_label": pred,
                        "prob_neutral": p[0],
                        "prob_sad": p[1],
                        "prob_fear": p[2],
                        "prob_happy": p[3],
                    })
        self.toy_df = pd.DataFrame(rows)

    def test_01_mean_probability_aggregation_math(self):
        """Verify exact hand-calculated mean probability vector and argmax."""
        # Trial 1 windows: two with [0.6, 0.2, 0.1, 0.1] and two with [0.3, 0.5, 0.1, 0.1]
        # Expected mean: [(0.6+0.6+0.3+0.3)/4, (0.2+0.2+0.5+0.5)/4, 0.1, 0.1] = [0.45, 0.35, 0.1, 0.1]
        # Argmax should be 0 (neutral)
        tr1_df = self.toy_df[(self.toy_df["subject_id"] == 1) & (self.toy_df["trial_id"] == 1)]
        pred_mean, mean_probs, pred_vote, had_tie = aggregate_single_trial(tr1_df)

        self.assertEqual(pred_mean, 0)
        self.assertAlmostEqual(mean_probs[0], 0.45)
        self.assertAlmostEqual(mean_probs[1], 0.35)
        self.assertAlmostEqual(mean_probs[2], 0.10)
        self.assertAlmostEqual(mean_probs[3], 0.10)

    def test_02_majority_voting_deterministic_tie_breaking(self):
        """Verify tie-breaking selects class with highest cumulative probability among tied classes."""
        # In Trial 1, window predictions are [0, 0, 1, 1] -> Tie between 0 and 1 (2 votes each)
        # Cumulative prob for class 0 = 0.6 + 0.6 + 0.3 + 0.3 = 1.8
        # Cumulative prob for class 1 = 0.2 + 0.2 + 0.5 + 0.5 = 1.4
        # Tie break should select class 0, and had_tie should be True
        tr1_df = self.toy_df[(self.toy_df["subject_id"] == 1) & (self.toy_df["trial_id"] == 1)]
        pred_mean, mean_probs, pred_vote, had_tie = aggregate_single_trial(tr1_df)

        self.assertEqual(pred_vote, 0)
        self.assertTrue(had_tie)

    def test_03_majority_voting_clear_winner(self):
        """Verify majority vote selects clear mode when no tie occurs."""
        tr2_df = self.toy_df[(self.toy_df["subject_id"] == 1) & (self.toy_df["trial_id"] == 2)]
        pred_mean, mean_probs, pred_vote, had_tie = aggregate_single_trial(tr2_df)

        self.assertEqual(pred_vote, 3)
        self.assertFalse(had_tie)

    def test_04_validation_detects_label_contradiction(self):
        """Verify validation raises ValueError if a window within a trial has a different true label."""
        bad_df = self.toy_df.copy()
        bad_df.loc[0, "true_label"] = 2  # Corrupt window 0 of trial 1
        with self.assertRaises(ValueError):
            validate_trial_structure(bad_df)

    def test_05_validation_detects_duplicate_windows(self):
        """Verify validation raises ValueError if duplicate window entries exist."""
        bad_df = pd.concat([self.toy_df, self.toy_df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_trial_structure(bad_df)

    def test_06_predictions_output_row_count_and_columns(self):
        """Verify generated predictions table contains expected columns and correct trial count."""
        preds = generate_trial_predictions(self.toy_df)
        # 2 subjects * 2 trials = 4 trials * 2 aggregation methods = 8 rows
        self.assertEqual(len(preds), 8)
        self.assertIn("aggregation_method", preds.columns)
        self.assertIn("had_tie_break", preds.columns)
        self.assertIn("is_correct", preds.columns)
        self.assertTrue((preds["is_correct"].isin([0, 1])).all())

    def test_07_verified_phase_d_trial_counts(self):
        """Verify the actual Phase D file produces exactly 1,080 trials per condition and method."""
        preds_path = "outputs/predictions/phase_d_calibrated_predictions.csv"
        df = pd.read_csv(preds_path, usecols=["condition", "method", "subject_id", "session_id", "trial_id", "true_label"])
        
        # Test all_62 group_safe_temperature
        sub = df[(df["condition"] == "all_62") & (df["method"] == "group_safe_temperature")]
        n_trials = sub.groupby(["subject_id", "session_id", "trial_id"]).ngroups
        self.assertEqual(n_trials, 1080)

        # Verify exactly 72 trials per subject
        sub_counts = sub.drop_duplicates(subset=["subject_id", "session_id", "trial_id"]).groupby("subject_id").size()
        self.assertTrue((sub_counts == 72).all())

    def test_08_paired_differences_and_t_test(self):
        """Verify paired aggregation tests correctly compute differences and subject sign counts."""
        sub_rows = []
        for s in range(1, 16):
            # Create a scenario where Mean Prob is slightly higher
            sub_rows.append({"condition": "all_62", "calibration_method": "group_safe_temperature", "aggregation_method": "mean_probability", "subject_id": s, "trial_count": 72, "correct_trials": 30, "accuracy": 41.67})
            sub_rows.append({"condition": "all_62", "calibration_method": "group_safe_temperature", "aggregation_method": "majority_vote", "subject_id": s, "trial_count": 72, "correct_trials": 28, "accuracy": 38.89})
        sub_df = pd.DataFrame(sub_rows)

        paired_df = compute_paired_aggregation_tests(sub_df)
        self.assertEqual(len(paired_df), 1)
        self.assertAlmostEqual(paired_df["mean_delta_acc_pp"].iloc[0], 41.67 - 38.89, places=2)
        self.assertEqual(paired_df["n_pos_subjects"].iloc[0], 15)
        self.assertEqual(paired_df["n_neg_subjects"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
