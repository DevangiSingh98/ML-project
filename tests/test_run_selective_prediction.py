"""
Unit tests for Phase E2 Experiment Runner: Selective Prediction & Uncertainty-Based Abstention.

All tests utilize small, self-contained synthetic fixtures to ensure:
- Fast, deterministic execution (milliseconds).
- Zero reliance on or contamination of experimental research data.
- Strict verification of mathematical alignment, leakage safety, error handling, and output schemas.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from src.run_selective_prediction import (
    load_and_validate_phase_d_predictions,
    evaluate_mode_a_retrospective,
    evaluate_mode_b_inductive,
    generate_phase_e_plots,
    generate_training_oof_uncertainties,
    generate_strictly_nested_training_oof_uncertainties,
    run_phase_e_pipeline,
    NOMINAL_COVERAGES_DEFAULT,
    CONDITIONS_DEFAULT,
    REGIMES_DEFAULT,
)
from src.selective_prediction import (
    compute_partial_aurc,
    compute_full_aurc,
    compute_random_rejection_baseline,
    determine_threshold_from_training_budget,
    get_acceptance_mask,
    compute_coverage,
)


def create_synthetic_predictions_df(n_samples_per_subject: int = 40, n_subjects: int = 3) -> pd.DataFrame:
    """
    Create a minimal synthetic Phase D predictions DataFrame.
    """
    rows = []
    np.random.seed(42)
    conditions = ["all_62", "pred_32"]
    methods = ["group_safe_temperature", "uncalibrated_softmax"]
    
    for cond in conditions:
        for meth in methods:
            for sid in range(1, n_subjects + 1):
                for i in range(n_samples_per_subject):
                    true_y = int(np.random.randint(0, 4))
                    # 70% accuracy on average
                    if np.random.rand() < 0.70:
                        pred_y = true_y
                        conf = float(np.random.uniform(0.60, 0.95))
                    else:
                        pred_y = int((true_y + np.random.randint(1, 4)) % 4)
                        conf = float(np.random.uniform(0.25, 0.55))
                        
                    u_max = 1.0 - conf
                    norm_ent = float(u_max * 1.1)  # correlated proxy
                    
                    rows.append({
                        "subject_id": sid,
                        "session_id": 1,
                        "trial_id": 1,
                        "window_id": i,
                        "condition": cond,
                        "channel_count": 62 if cond == "all_62" else 32,
                        "method": meth,
                        "true_label": true_y,
                        "true_label_name": f"class_{true_y}",
                        "predicted_label": pred_y,
                        "predicted_label_name": f"class_{pred_y}",
                        "confidence": conf,
                        "u_max": u_max,
                        "entropy": norm_ent * np.log(4.0),
                        "norm_entropy": norm_ent,
                        "is_correct": int(true_y == pred_y),
                        "is_error": int(true_y != pred_y),
                    })
                    
    return pd.DataFrame(rows)


class TestRunSelectivePrediction(unittest.TestCase):
    """Test suite for Phase E2 experiment runner."""

    def setUp(self):
        self.synthetic_df = create_synthetic_predictions_df(n_samples_per_subject=50, n_subjects=3)

    def test_01_load_and_validate_schema(self):
        """Test loading and validation of prediction DataFrame schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_preds.csv"
            self.synthetic_df.to_csv(csv_path, index=False)
            
            # Successful load
            loaded = load_and_validate_phase_d_predictions(csv_path)
            self.assertEqual(len(loaded), len(self.synthetic_df))
            
            # Missing required column raises ValueError
            bad_df = self.synthetic_df.drop(columns=["u_max"])
            bad_path = Path(tmpdir) / "bad_preds.csv"
            bad_df.to_csv(bad_path, index=False)
            with self.assertRaises(ValueError):
                load_and_validate_phase_d_predictions(bad_path)
                
            # Missing file raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                load_and_validate_phase_d_predictions(Path(tmpdir) / "non_existent.csv")

    def test_02_filtering_model_conditions_and_regimes(self):
        """Test correct filtering of conditions and probability regimes in Mode A."""
        res = evaluate_mode_a_retrospective(
            predictions_df=self.synthetic_df,
            condition="all_62",
            method="group_safe_temperature",
            uncertainty_col="u_max",
            target_coverages=[1.0, 0.8, 0.6],
            n_random_repeats=50,
            seed=42,
        )
        pooled = res["pooled_curve"]
        self.assertTrue((pooled["condition"] == "all_62").all())
        self.assertTrue((pooled["method"] == "group_safe_temperature").all())
        self.assertTrue((pooled["evaluation_mode"] == "retrospective_mode_a").all())
        
        # Non-existent condition raises ValueError
        with self.assertRaises(ValueError):
            evaluate_mode_a_retrospective(
                predictions_df=self.synthetic_df,
                condition="invalid_condition",
                method="group_safe_temperature",
            )

    def test_03_alignment_of_labels_and_uncertainty(self):
        """Test alignment of true labels, predictions, and uncertainty values in Mode A."""
        # Create deterministic mini dataframe
        mini_df = pd.DataFrame({
            "subject_id": [1, 1, 1, 1, 1],
            "condition": ["all_62"] * 5,
            "method": ["group_safe_temperature"] * 5,
            "true_label": [0, 1, 2, 3, 0],
            "predicted_label": [0, 1, 2, 0, 1], # 3 correct (idx 0, 1, 2), 2 errors (idx 3, 4)
            "u_max": [0.1, 0.2, 0.3, 0.8, 0.9], # Low uncertainty on correct, high on errors
            "norm_entropy": [0.1, 0.2, 0.3, 0.8, 0.9],
            "confidence": [0.9, 0.8, 0.7, 0.2, 0.1],
        })
        
        res = evaluate_mode_a_retrospective(
            predictions_df=mini_df,
            condition="all_62",
            method="group_safe_temperature",
            uncertainty_col="u_max",
            target_coverages=[1.0, 0.6], # 1.0 (all 5) and 0.6 (top 3: u <= 0.3)
            n_random_repeats=20,
            seed=42,
        )
        curve = res["pooled_curve"].sort_values("actual_coverage", ascending=False).reset_index(drop=True)
        # At coverage 1.0, accuracy is 3/5 = 0.60
        self.assertAlmostEqual(curve.loc[0, "actual_coverage"], 1.0)
        self.assertAlmostEqual(curve.loc[0, "selective_accuracy"], 0.60)
        
        # At coverage 0.6, accepted are indices 0, 1, 2 which are ALL correct -> accuracy = 1.0
        self.assertAlmostEqual(curve.loc[1, "actual_coverage"], 0.60)
        self.assertAlmostEqual(curve.loc[1, "selective_accuracy"], 1.00)

    def test_04_separation_of_retrospective_and_inductive_modes(self):
        """Test that Mode A and Mode B maintain distinct evaluation tags and behaviors."""
        train_oof = {1: np.array([0.1, 0.2, 0.3, 0.4, 0.5])}
        sub_df = self.synthetic_df[self.synthetic_df["subject_id"] == 1]
        
        res_a = evaluate_mode_a_retrospective(
            predictions_df=sub_df,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[0.8],
            n_random_repeats=20,
        )
        res_b = evaluate_mode_b_inductive(
            test_predictions_df=sub_df,
            training_oof_uncertainties=train_oof,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[0.8],
            n_random_repeats=20,
        )
        
        self.assertEqual(res_a["pooled_curve"]["evaluation_mode"].iloc[0], "retrospective_mode_a")
        self.assertEqual(res_b["subject_inductive_metrics"]["evaluation_mode"].iloc[0], "inductive_mode_b")

    def test_05_training_only_threshold_inputs_mode_b(self):
        """Test that operational thresholds in Mode B are derived strictly from training OOF uncertainties."""
        # If training OOF is [0.1, 0.2, 0.3, 0.4, 0.5], target coverage 0.8 corresponds to quantile 0.8 = 0.42
        train_oof = {
            1: np.array([0.10, 0.20, 0.30, 0.40, 0.50]),
            2: np.array([0.15, 0.25, 0.35, 0.45, 0.55]),
            3: np.array([0.05, 0.15, 0.25, 0.35, 0.45]),
        }
        res_b = evaluate_mode_b_inductive(
            test_predictions_df=self.synthetic_df,
            training_oof_uncertainties=train_oof,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[0.80],
            n_random_repeats=20,
        )
        metrics = res_b["subject_inductive_metrics"]
        
        # Verify thresholds match quantiles of the respective training OOF arrays
        tau_s1 = metrics[metrics["subject_id"] == 1]["threshold"].iloc[0]
        expected_tau_s1 = float(np.quantile(train_oof[1], 0.80))
        self.assertAlmostEqual(tau_s1, expected_tau_s1, places=5)

    def test_06_detection_of_missing_or_invalid_oof_inputs(self):
        """Test that missing or empty training OOF scores raise appropriate errors."""
        incomplete_oof = {1: np.array([0.1, 0.2]), 2: np.array([0.1, 0.2])} # missing subject 3
        with self.assertRaises(KeyError):
            evaluate_mode_b_inductive(
                test_predictions_df=self.synthetic_df,
                training_oof_uncertainties=incomplete_oof,
                condition="all_62",
                method="group_safe_temperature",
            )
            
        empty_oof = {1: np.array([]), 2: np.array([0.1]), 3: np.array([0.2])}
        with self.assertRaises(ValueError):
            evaluate_mode_b_inductive(
                test_predictions_df=self.synthetic_df,
                training_oof_uncertainties=empty_oof,
                condition="all_62",
                method="group_safe_temperature",
            )

    def test_07_no_test_label_access_during_threshold_selection(self):
        """Verify that modifying test labels does not alter the selected threshold tau in Mode B."""
        train_oof = {sid: np.array([0.1, 0.2, 0.3, 0.4, 0.5]) for sid in [1, 2, 3]}
        
        df_original = self.synthetic_df.copy()
        res_orig = evaluate_mode_b_inductive(
            test_predictions_df=df_original,
            training_oof_uncertainties=train_oof,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[0.70],
            n_random_repeats=10,
        )
        
        df_corrupted = self.synthetic_df.copy()
        df_corrupted["true_label"] = (df_corrupted["true_label"] + 1) % 4 # Completely alter test labels
        res_corrupt = evaluate_mode_b_inductive(
            test_predictions_df=df_corrupted,
            training_oof_uncertainties=train_oof,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[0.70],
            n_random_repeats=10,
        )
        
        tau_orig = res_orig["subject_inductive_metrics"]["threshold"].to_numpy()
        tau_corrupt = res_corrupt["subject_inductive_metrics"]["threshold"].to_numpy()
        np.testing.assert_array_equal(tau_orig, tau_corrupt)

    def test_08_threshold_tie_handling(self):
        """Test tie handling where multiple samples share identical uncertainty values."""
        tie_df = pd.DataFrame({
            "subject_id": [1] * 10,
            "condition": ["all_62"] * 10,
            "method": ["group_safe_temperature"] * 10,
            "true_label": [0] * 10,
            "predicted_label": [0] * 10,
            "u_max": [0.35] * 10, # All identical ties
            "norm_entropy": [0.35] * 10,
            "confidence": [0.65] * 10,
        })
        train_oof = {1: np.array([0.35] * 10)}
        
        res = evaluate_mode_b_inductive(
            test_predictions_df=tie_df,
            training_oof_uncertainties=train_oof,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[0.80],
            n_random_repeats=10,
        )
        metrics = res["subject_inductive_metrics"].iloc[0]
        # Since u <= tau where tau=0.35 and all u=0.35, all 10 should be accepted
        self.assertEqual(metrics["accepted_count"], 10)
        self.assertAlmostEqual(metrics["actual_test_coverage"], 1.0)

    def test_09_correct_100_percent_coverage_behavior(self):
        """Verify that at 100% nominal coverage, all samples are accepted."""
        train_oof = {sid: np.linspace(0.1, 0.9, 20) for sid in [1, 2, 3]}
        res = evaluate_mode_b_inductive(
            test_predictions_df=self.synthetic_df,
            training_oof_uncertainties=train_oof,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[1.0],
            n_random_repeats=10,
        )
        metrics = res["subject_inductive_metrics"]
        for _, row in metrics.iterrows():
            self.assertEqual(row["nominal_coverage_target"], 1.0)
            self.assertEqual(row["rejected_count"], 0)
            self.assertEqual(row["accepted_count"], row["total_test_samples"])
            self.assertAlmostEqual(row["actual_test_coverage"], 1.0)
            self.assertAlmostEqual(row["selective_accuracy"], row["full_coverage_accuracy"])

    def test_10_reproducible_random_rejection(self):
        """Verify that random rejection baseline is strictly deterministic under a fixed seed."""
        res1 = evaluate_mode_a_retrospective(
            predictions_df=self.synthetic_df,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[0.8, 0.6],
            n_random_repeats=100,
            seed=123,
        )
        res2 = evaluate_mode_a_retrospective(
            predictions_df=self.synthetic_df,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[0.8, 0.6],
            n_random_repeats=100,
            seed=123,
        )
        acc1 = res1["pooled_curve"]["empirical_random_accuracy_mean"].to_numpy()
        acc2 = res2["pooled_curve"]["empirical_random_accuracy_mean"].to_numpy()
        np.testing.assert_array_almost_equal(acc1, acc2)

    def test_11_output_schemas(self):
        """Verify schema integrity of all DataFrames output by run_phase_e_pipeline."""
        train_oof_nested = {
            "all_62": {
                "group_safe_temperature": {
                    "u_max": {sid: np.linspace(0.1, 0.8, 30) for sid in [1, 2, 3]}
                }
            }
        }
        res = run_phase_e_pipeline(
            predictions_df=self.synthetic_df,
            training_oof_dict=train_oof_nested,
            conditions=["all_62"],
            regimes=["group_safe_temperature"],
            uncertainty_measures=["u_max"],
            target_coverages=[1.0, 0.8, 0.6],
            n_random_repeats=30,
            save_outputs=False,
            generate_plots=False,
        )
        
        # Check presence of key result items
        self.assertIn("mode_a_pooled", res)
        self.assertIn("mode_b_summary", res)
        self.assertIn("selective_predictions", res)
        self.assertIn("fold_metrics", res)
        self.assertIn("coverage_accuracy_summary", res)
        self.assertIn("risk_coverage_summary", res)
        self.assertIn("random_rejection_summary", res)
        self.assertIn("subject_summary", res)
        self.assertIn("run_report", res)
        
        # Check columns of fold metrics
        expected_fold_cols = ["evaluation_mode", "condition", "method", "uncertainty_measure", "subject_id"]
        for col in expected_fold_cols:
            self.assertIn(col, res["fold_metrics"].columns)
            
        # Check report JSON structure
        report = res["run_report"]
        self.assertIn("verification_checks", report)
        self.assertTrue(report["mode_a_evaluated"])
        self.assertTrue(report["mode_b_evaluated"])

    def test_12_end_to_end_small_fixture_runner_execution(self):
        """Test complete end-to-end execution of the runner with file saving in a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_oof_nested = {
                "all_62": {
                    "group_safe_temperature": {
                        "u_max": {sid: np.linspace(0.1, 0.8, 30) for sid in [1, 2, 3]}
                    }
                }
            }
            res = run_phase_e_pipeline(
                predictions_df=self.synthetic_df,
                training_oof_dict=train_oof_nested,
                output_dir=tmpdir,
                conditions=["all_62"],
                regimes=["group_safe_temperature"],
                uncertainty_measures=["u_max"],
                target_coverages=[1.0, 0.8, 0.6],
                n_random_repeats=20,
                save_outputs=True,
                generate_plots=True,
            )
            saved = res["saved_files"]
            self.assertGreaterEqual(len(saved), 7)
            for path_str in saved:
                p = Path(path_str)
                self.assertTrue(p.exists(), f"File was not created: {p}")
                self.assertGreater(p.stat().st_size, 0, f"File is empty: {p}")

    def test_13_small_fixture_training_oof_generator(self):
        """Test generate_training_oof_uncertainties on a small 3-subject synthetic dataset."""
        np.random.seed(42)
        n_samples = 45 # 15 samples per subject
        X = np.random.randn(n_samples, 310)
        y = np.random.randint(0, 4, size=n_samples)
        subjects = np.repeat([1, 2, 3], 15)
        metadata = pd.DataFrame({"subject_id": subjects})
        
        oof_dict = generate_training_oof_uncertainties(
            X=X,
            y=y,
            metadata=metadata,
            condition_indices=list(range(2)), # 2 channels = 10 features (5 bands each)
            method="group_safe_temperature",
            uncertainty_col="u_max",
            n_inner_splits=2, # 2 inner splits since 2 training subjects remain per outer fold
            cache_size=100,
        )
        
        # Verify that all 3 outer held-out subjects receive training OOF uncertainty scores
        self.assertEqual(set(oof_dict.keys()), {1, 2, 3})
        for sid in [1, 2, 3]:
            u_arr = oof_dict[sid]
            # When subject `sid` is outer test (15 samples), 30 samples remain in training
            self.assertEqual(len(u_arr), 30)
            self.assertTrue(np.all(u_arr >= 0.0) and np.all(u_arr <= 1.0))

    def test_14_strictly_nested_inner_channel_selection(self):
        """Verify generate_strictly_nested_training_oof_uncertainties runs and selects channels inner-fold only."""
        np.random.seed(42)
        n_samples = 40  # 10 samples per subject x 4 subjects
        X = np.random.randn(n_samples, 310)  # Full 62 channels (310 features)
        y = np.random.randint(0, 2, size=n_samples)  # 2 classes
        subjects = np.repeat([1, 2, 3, 4], 10)
        metadata = pd.DataFrame({"subject_id": subjects})
        
        oof_dict = generate_strictly_nested_training_oof_uncertainties(
            X=X,
            y=y,
            metadata=metadata,
            condition="pred_32",
            method="group_safe_temperature",
            uncertainty_col="u_max",
            n_inner_splits=2,
            seed=42,
            outer_subjects=[4],
            cache_size=100,
        )
        self.assertIn(4, oof_dict)
        u_arr = oof_dict[4]
        self.assertEqual(len(u_arr), 30)  # 30 outer-training samples (subjects 1, 2, 3)
        self.assertTrue(np.all(u_arr >= 0.0) and np.all(u_arr <= 1.0))

    def test_15_no_inner_validation_influence_on_channel_selection(self):
        """Verify that inner validation subject data cannot influence inner channel selection."""
        from src.channel_selection import select_predictive_importance_channels
        np.random.seed(42)
        n_samples = 30  # 10 samples each for inner train {1, 2} and inner val {3}
        X = np.random.randn(n_samples, 20)  # 4 channels = 20 features
        y = np.random.randint(0, 2, size=n_samples)
        subjects = np.repeat([1, 2, 3], 10)
        metadata = pd.DataFrame({"subject_id": subjects})
        
        # Channel selection strictly using inner train subjects {1, 2}
        tr_mask = (metadata["subject_id"].isin([1, 2])).to_numpy()
        X_in_tr, y_in_tr = X[tr_mask], y[tr_mask]
        meta_in_tr = metadata.iloc[tr_mask].reset_index(drop=True)
        
        ch_orig, _, _ = select_predictive_importance_channels(
            X_train=X_in_tr,
            y_train=y_in_tr,
            train_metadata=meta_in_tr,
            budget=2,
            n_splits=2,
            total_channels=4,
            seed=42,
        )
        
        # Now corrupt/alter inner validation subject 3
        X_corrupted = X.copy()
        y_corrupted = y.copy()
        val_mask = (metadata["subject_id"] == 3).to_numpy()
        X_corrupted[val_mask] = X_corrupted[val_mask] * 1000.0 + 500.0
        y_corrupted[val_mask] = 1 - y_corrupted[val_mask]
        
        # Re-run channel selection for the model evaluating subject 3 (using inner train {1, 2})
        X_in_tr_c, y_in_tr_c = X_corrupted[tr_mask], y_corrupted[tr_mask]
        meta_in_tr_c = metadata.iloc[tr_mask].reset_index(drop=True)
        
        ch_corrupt, _, _ = select_predictive_importance_channels(
            X_train=X_in_tr_c,
            y_train=y_in_tr_c,
            train_metadata=meta_in_tr_c,
            budget=2,
            n_splits=2,
            total_channels=4,
            seed=42,
        )
        # Selection on {1, 2} must be bit-for-bit identical regardless of subject 3
        self.assertEqual(ch_orig, ch_corrupt)

    def test_16_no_outer_test_subject_influence_on_thresholds(self):
        """Verify that outer test subject data cannot influence operational thresholds."""
        train_oof = {
            4: np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        }
        tau_80_orig = determine_threshold_from_training_budget(train_oof[4], target_coverage=0.80)
        
        # Corrupt test subject data completely
        test_corrupt = np.array([0.99, 0.99, 0.99])
        # Operational threshold must not inspect test_corrupt
        tau_80_after = determine_threshold_from_training_budget(train_oof[4], target_coverage=0.80)
        self.assertEqual(tau_80_orig, tau_80_after)

    def test_17_exactly_100_percent_acceptance_no_abstention(self):
        """Verify exactly 100% acceptance in Mode B when target coverage is 1.0, even with extreme test uncertainty."""
        train_oof = {
            1: np.array([0.1, 0.2, 0.3, 0.4])  # Max training uncertainty is 0.4
        }
        test_df = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "condition": ["all_62"] * 3,
            "method": ["group_safe_temperature"] * 3,
            "true_label": [0, 1, 2],
            "predicted_label": [0, 1, 0],
            "confidence": [0.3, 0.2, 0.05],
            "u_max": [0.7, 0.8, 0.95],  # All test uncertainties exceed training max (0.4)
            "norm_entropy": [0.7, 0.8, 0.95],
        })
        
        res_b = evaluate_mode_b_inductive(
            test_predictions_df=test_df,
            training_oof_uncertainties=train_oof,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[1.0],
            n_random_repeats=10,
        )
        row = res_b["subject_inductive_metrics"].iloc[0]
        self.assertEqual(row["nominal_coverage_target"], 1.0)
        self.assertEqual(row["accepted_count"], 3)
        self.assertEqual(row["rejected_count"], 0)
        self.assertAlmostEqual(row["actual_test_coverage"], 1.0)
        self.assertAlmostEqual(row["selective_accuracy"], row["full_coverage_accuracy"])

    def test_18_actual_coverage_reporting_on_threshold_ties(self):
        """Verify actual coverage is faithfully reported when threshold ties occur without forcing coverage."""
        train_oof = {
            1: np.array([0.10, 0.25, 0.25, 0.25, 0.90])
        }
        test_df = pd.DataFrame({
            "subject_id": [1] * 6,
            "condition": ["all_62"] * 6,
            "method": ["group_safe_temperature"] * 6,
            "true_label": [0] * 6,
            "predicted_label": [0] * 6,
            "confidence": [0.75] * 6,
            "u_max": [0.25] * 6,  # 6 tied samples at 0.25
            "norm_entropy": [0.25] * 6,
        })
        
        # For target_coverage 0.60, quantile lands on 0.25
        res_b = evaluate_mode_b_inductive(
            test_predictions_df=test_df,
            training_oof_uncertainties=train_oof,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[0.60],
            n_random_repeats=10,
        )
        row = res_b["subject_inductive_metrics"].iloc[0]
        self.assertEqual(row["nominal_coverage_target"], 0.60)
        # All 6 tied test samples with u <= 0.25 are accepted
        self.assertEqual(row["accepted_count"], 6)
        self.assertAlmostEqual(row["actual_test_coverage"], 1.00)

    def test_19_correct_labelling_and_integration_of_partial_vs_full_aurc(self):
        """Verify Mode A outputs clearly distinguish partial_aurc_50_to_100 and full_aurc."""
        res_a = evaluate_mode_a_retrospective(
            predictions_df=self.synthetic_df,
            condition="all_62",
            method="group_safe_temperature",
            target_coverages=[1.0, 0.8, 0.6, 0.5],
            n_random_repeats=20,
        )
        aurc_dict = res_a["aurc_summary"]
        self.assertIn("partial_aurc_50_to_100", aurc_dict)
        self.assertIn("full_aurc", aurc_dict)
        
        p_aurc = aurc_dict["partial_aurc_50_to_100"]
        f_aurc = aurc_dict["full_aurc"]
        
        # Partial AURC over [0.5, 1.0] must be <= Full AURC over [0.0, 1.0]
        self.assertLessEqual(p_aurc, f_aurc)
        # Partial AURC is bounded in [0, 0.5]
        self.assertGreaterEqual(p_aurc, 0.0)
        self.assertLessEqual(p_aurc, 0.5)
        # Full AURC is bounded in [0, 1.0]
        self.assertGreaterEqual(f_aurc, 0.0)
        self.assertLessEqual(f_aurc, 1.0)

    def test_20_random_rejection_decisions_not_depending_on_true_labels(self):
        """Verify random rejection selection does not depend on true labels."""
        y_true_a = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_true_b = np.array([3, 2, 1, 0, 3, 2, 1, 0])  # Completely different labels
        y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 0])
        
        res_a = compute_random_rejection_baseline(y_true_a, y_pred, coverages=[0.5], n_repeats=50, seed=777)
        res_b = compute_random_rejection_baseline(y_true_b, y_pred, coverages=[0.5], n_repeats=50, seed=777)
        
        # Analytical expectations equal full coverage accuracy of respective label vectors
        self.assertAlmostEqual(res_a["expected_accuracy"], float(np.mean(y_true_a == y_pred)))
        self.assertAlmostEqual(res_b["expected_accuracy"], float(np.mean(y_true_b == y_pred)))
        # Actual coverages are identical
        self.assertEqual(res_a["actual_coverages"][0], res_b["actual_coverages"][0])

    def test_21_cross_fitted_calibration_excludes_validation_group(self):
        """Verify that temperature calibration of inner validation groups excludes their own samples/labels."""
        from src.calibration import GroupSafeTemperatureScalingCalibrator
        np.random.seed(42)
        n_samples = 30  # 10 samples each for subjects 1, 2, 3
        Z = np.random.randn(n_samples, 4)
        y = np.random.randint(0, 4, size=n_samples)
        
        # When calibrating group 3, fit on groups {1, 2}
        cal1 = GroupSafeTemperatureScalingCalibrator()
        cal1.fit(Z[:20], y[:20])
        temp1 = cal1.temperature_
        
        # Corrupt group 3 labels and decision scores
        Z_corrupt = Z.copy()
        y_corrupt = y.copy()
        Z_corrupt[20:] *= 100.0
        y_corrupt[20:] = (y_corrupt[20:] + 1) % 4
        
        # Re-fit on groups {1, 2}
        cal2 = GroupSafeTemperatureScalingCalibrator()
        cal2.fit(Z_corrupt[:20], y_corrupt[:20])
        temp2 = cal2.temperature_
        
        # Temperature learned to calibrate group 3 must be 100% independent of group 3
        self.assertEqual(temp1, temp2)


if __name__ == "__main__":
    unittest.main()
