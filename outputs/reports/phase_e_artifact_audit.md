# Phase E0: Selective Prediction & Abstention — Existing Artifact Audit

**Project Title:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 37,575 feature windows, 62 channels, 5 frequency bands = 310 features, 4 emotion classes)  
**Evaluation Protocol:** 15-Fold Outer Leave-One-Subject-Out (LOSO) Cross-Validation  
**Auditor:** Antigravity AI (Independent Verification Agent)  
**Date of Audit:** 2026-10-25  
**Workspace:** `C:\Users\singh\OneDrive\Desktop\ML Project`  
**Deliverable File:** `outputs/reports/phase_e_artifact_audit.md`  

---

## 1. Executive Summary

This audit assesses the feasibility, data availability, methodological safety, and architectural requirements for implementing **Phase E (Selective Prediction & Uncertainty-Based Abstention)**.

### Principal Findings:
1. **Phases A–D are Fully Verified and Intact:** Ingestion, 15-fold outer LOSO partitioning, RBF-SVM baseline, channel selection (predictive, stability-aware, and random), and leakage-safe probability calibration are complete, mathematically verified, and validated by 41 passing unit tests.
2. **Phase E Does Not Currently Exist:** There is zero source code, zero saved prediction outputs, and zero summary tables for selective prediction or abstention in the workspace.
3. **Rich Test-Level Uncertainty Artifacts Are Saved:** [`outputs/predictions/phase_d_calibrated_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/phase_d_calibrated_predictions.csv) contains **450,900 complete prediction rows** with exact probabilities, confidences, $u_{\max}$, predictive entropy, normalized entropy, true labels, and error indicators across all 15 outer test folds.
4. **Training-Only Inner Out-of-Fold (OOF) Scores Were Not Persisted to Disk:** In Phase D, inner out-of-fold decision scores ($Z_{\text{oof}}$) were generated in runtime memory to fit temperature $T$ and Platt parameters, but were not saved to disk.
5. **Phase E Can Be Executed Rapidly Without Retraining the Main Models:** 
   - **Test-level Risk-Coverage Analysis & AURC** can be computed **immediately from existing artifacts in seconds**.
   - **Inductive Leakage-Safe Operational Thresholding** requires generating training OOF uncertainty scores on the 14 training subjects, which takes **~75 seconds of CPU compute across all 15 folds** without modifying or retraining final outer models.

---

## 2. Answers to Specific Audit Questions

---

### Question 1: What is the current implementation status of Phases A–D?

| Phase | Description | Implementation Status | Supporting Evidence & Artifacts | Test Suite Status |
| :--- | :--- | :---: | :--- | :---: |
| **Phase A** | Dataset Ingestion, LDS Feature Extraction, & Metadata Formatting | **Complete** | [`src/data_loader.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/data_loader.py)<br>[`outputs/tables/dataset_summary.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/dataset_summary.csv) | 7/7 PASS (`test_data_loader.py`) |
| **Phase B** | Outer LOSO Partitioning & Full-Channel RBF-SVM Baseline | **Complete** | [`src/subject_split.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/subject_split.py), [`src/train_baseline.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/train_baseline.py)<br>[`outputs/predictions/baseline_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/baseline_predictions.csv) | 9/9 PASS (`test_subject_split.py`, `test_train_baseline.py`) |
| **Phase C** | Channel Reduction & Selection Stability Evaluation | **Complete & Audited** | [`src/channel_selection.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/channel_selection.py), [`src/evaluate_corrected_channel_performance.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/evaluate_corrected_channel_performance.py)<br>[`outputs/tables/channel_reduction_comparison.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/channel_reduction_comparison.csv) | 18/18 PASS (`test_channel_selection.py`, `test_corrected_channel_performance.py`) |
| **Phase D** | Probability Calibration & Uncertainty Estimation | **Complete & Audited** | [`src/calibration.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/calibration.py), [`src/run_calibration.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py)<br>[`outputs/predictions/phase_d_calibrated_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/phase_d_calibrated_predictions.csv) | 7/7 PASS (`test_calibration.py`, `test_uncertainty_metrics.py`) |
| **Phase E** | Selective Prediction & Uncertainty-Based Abstention | **NOT IMPLEMENTED** | *Zero source files, zero outputs, zero tables* | *None* |

---

### Question 2: What columns and information are actually present in the saved Phase D predictions?

The primary prediction artifact is [`outputs/predictions/phase_d_calibrated_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/phase_d_calibrated_predictions.csv) (103.8 MB, 450,900 rows).

#### Evaluated Matrix Dimensions:
- 15 outer LOSO folds $\times$ 2,505 test windows per fold = **37,575 test samples**.
- 4 Channel Conditions: `all_62`, `pred_32`, `stab_32`, `pred_12`.
- 3 Probability Regimes: `uncalibrated_softmax`, `group_safe_temperature`, `group_safe_platt`.
- Total Rows: $37,575 \times 4 \times 3 = \mathbf{450,900}$ rows.

#### Verified Column Schema:
| Column Name | Data Type | Meaning & Verified Content |
| :--- | :---: | :--- |
| `subject_id` | `int64` | Held-out outer test subject ID ($1 \dots 15$) |
| `session_id` | `int64` | Experimental session ($1, 2, 3$) |
| `trial_id` | `int64` | Trial clip ID ($1 \dots 24$) |
| `window_id` | `int64` | Consecutive 4-second sliding window within trial |
| `label` | `int64` | True integer emotion label ($0, 1, 2, 3$) |
| `label_name` | `object` | True emotion string (`neutral`, `sad`, `fear`, `happy`) |
| `condition` | `object` | Channel subset evaluated (`all_62`, `pred_32`, `stab_32`, `pred_12`) |
| `channel_count` | `int64` | Physical channel count ($62, 32, 32, 12$) |
| `method` | `object` | Probability regime (`uncalibrated_softmax`, `group_safe_temperature`, `group_safe_platt`) |
| `temperature` | `float64` | Fitted temperature $T$ ($3.76 - 4.05$ for temperature scaling; NaN for others) |
| `true_label` | `int64` | Duplicated true integer label |
| `true_label_name` | `object` | Duplicated true emotion string |
| `predicted_label` | `int64` | Model predicted class ($\arg\max_c p_c$) |
| `predicted_label_name` | `object` | Predicted emotion name string |
| `prob_neutral` | `float64` | Calibrated probability for class 0 ($p_0 \in [0, 1]$) |
| `prob_sad` | `float64` | Calibrated probability for class 1 ($p_1 \in [0, 1]$) |
| `prob_fear` | `float64` | Calibrated probability for class 2 ($p_2 \in [0, 1]$) |
| `prob_happy` | `float64` | Calibrated probability for class 3 ($p_3 \in [0, 1]$) |
| `confidence` | `float64` | Top-label confidence: $\hat{p} = \max_k p_k \in [0.25, 1.0]$ |
| `u_max` | `float64` | Maximum probability uncertainty: $u_{\max} = 1 - \max_k p_k \in [0, 0.75]$ |
| `entropy` | `float64` | Shannon predictive entropy: $H(p) = -\sum p_k \ln p_k \in [0, 1.3863]$ |
| `norm_entropy` | `float64` | Normalized entropy: $H(p) / \ln 4 \in [0, 1.0]$ |
| `is_correct` | `int64` | 1 if `predicted_label == true_label`, else 0 |
| `is_error` | `int64` | 1 if `predicted_label != true_label`, else 0 |

#### What is NOT Present:
- **Raw Decision Scores ($z_0 \dots z_3$):** Not stored directly (though algebraically recoverable for temperature scaling via $z_k \propto T \ln p_k$).
- **Training Samples / Inner Out-of-Fold Scores ($Z_{\text{oof}}$):** Zero training samples exist in the CSV. Every single row represents a held-out test subject.

---

### Question 3: Are training-only out-of-fold predictions available for selecting abstention thresholds?

**NO.**  
An exhaustive audit of the workspace confirms that **inner out-of-fold predictions were never saved to disk**.

In [`src/run_calibration.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py) (lines 120–135), the function `generate_group_safe_oof_decision_scores()` computed $Z_{\text{oof}}$ as a transient in-memory NumPy array across the 14 training subjects (`GroupKFold(n_splits=3)`). That array was immediately consumed by `calibrator.fit(Z_oof, y_train)` to learn the temperature parameter $T$, and was then garbage-collected. Only the outer held-out test predictions were written to `phase_d_calibrated_predictions.csv`.

---

### Question 4: Can thresholds be selected without using the held-out subject's labels or uncertainty distribution?

**YES, but the distinction between Inductive and Empirical evaluation must be strictly maintained:**

1. **Test-Set Peeking (Methodologically Invalid):**  
   If an algorithm computes the 80th percentile of uncertainty on the *held-out test subject* to achieve 80% coverage on that subject, it is using transductive information from the test set. In a real-world BCI application, single samples arrive sequentially from an unseen subject, making test-set quantiles unavailable.
2. **Inductive Leakage-Safe Threshold Determination (Methodologically Sound):**  
   To set an operational threshold $\tau_\gamma$ targeting coverage $\gamma$ (e.g., $\gamma = 0.80$), $\tau_\gamma$ must be chosen as the empirical $\gamma$-quantile of uncertainty on the **training data** (specifically, out-of-fold uncertainty on the 14 training subjects). When this fixed threshold $\tau_\gamma$ is applied to the unseen test subject, the resulting test coverage will fluctuate naturally around $\gamma$ (e.g., $78\%$ or $83\%$). Reporting actual achieved test coverage alongside selective accuracy is the gold standard of selective prediction.
3. **Empirical Risk-Coverage Curve Evaluation (Valid Evaluation Tool):**  
   Sorting test samples by uncertainty to sweep coverage from 100% down to 50% is valid for computing Area Under the Risk-Coverage Curve (AURC), provided it is explicitly designated as an *empirical risk-coverage evaluation* rather than an inductive decision rule.

---

### Question 5: Can Phase E be implemented using existing artifacts, or is additional model fitting necessary?

**Phase E can be implemented largely using existing artifacts, with only a small, fast auxiliary computation if inductive training thresholds are desired:**

#### Path A: Evaluation from Existing Artifacts (100% Ready, 0 Retraining, ~5 seconds compute)
- Using the 450,900 saved predictions in `phase_d_calibrated_predictions.csv`, we can immediately compute:
  1. Empirical Risk-Coverage curves ($r(\gamma)$ vs. $\gamma \in [0.5, 1.0]$).
  2. Selective accuracy on retained samples: $\text{Acc}(\gamma) = \frac{\sum_{i \in \text{Accepted}} \mathbf{1}(\hat{y}_i = y_i)}{|\text{Accepted}|}$.
  3. Area Under the Risk-Coverage Curve (AURC) and Excess-AURC.
  4. Comparison against the theoretical Random Abstention baseline ($r_{\text{random}}(\gamma) \equiv r_{\text{base}}$).
  5. Both pooled and per-subject risk-coverage trajectories.

#### Path B: Inductive Thresholding on Training OOF Scores (Requires Lightweight OOF Generation, ~75s compute)
- If inductive operational thresholds are required, we do **not** need to retrain the outer models or redo Phase D.
- We simply run `generate_group_safe_oof_decision_scores()` across the 14 training subjects for the 15 outer folds. On modern multi-core CPU, this takes **~75 seconds total** across all 15 folds.

---

### Question 6: Which prediction rule is used for each existing result?

| Project Stage | Output Artifact | Prediction Rule Function | Mathematical Definition | Concordance with Discrete Voting |
| :--- | :--- | :--- | :--- | :---: |
| **Phase B Baseline** | [`baseline_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/baseline_predictions.csv) | `pipeline.predict(X_test)` | Discrete One-vs-One (OvO) Majority Voting | 100.0% (Definition) |
| **Phase C Channels** | [`channel_reduction_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/channel_reduction_predictions.csv) | `pipeline.predict(X_test)` | Discrete One-vs-One (OvO) Majority Voting | 100.0% (Definition) |
| **Stage 2.5 Channels**| [`corrected_channel_performance_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/corrected_channel_performance_predictions.csv) | Both `predict` and `margin_predict` saved | Rule 1: `SVC.predict`<br>Rule 2: $\arg\max_c z_c$ | 95.16% on all-62<br>96.84% on corr-stab-32 |
| **Phase D Calibration** | [`phase_d_calibrated_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/phase_d_calibrated_predictions.csv) | `calibrator.predict(Z_test)` | Continuous Margin Argmax: $\arg\max_c p_c \equiv \arg\max_c z_c$ | 95.16% on all-62<br>(1,817 low-margin tie breaks) |

*Decision for Phase E:* Phase E must evaluate selective prediction on **Continuous Margin Argmax ($\arg\max_c p_c$)**, because the uncertainty score ($u_{\max} = 1 - \max p_c$) is derived from the exact same calibrated probability distribution.

---

### Question 7: Are the existing predictions and metrics internally consistent?

**YES. Verified 100% mathematically consistent:**
1. **Probability Bounds:** For all 450,900 rows, $p_k \ge 0$, $\sum_{k=0}^3 p_k = 1.0 \pm 10^{-6}$.
2. **Confidence-Uncertainty Link:** $\text{confidence} = \max_k p_k$, and $u_{\max} = 1.0 - \text{confidence}$ holds strictly across all rows.
3. **Entropy Bounds:** Entropy $H(p) \in [0, \ln 4 \approx 1.3863]$, and $H_{\text{norm}} \in [0, 1.0]$.
4. **Prediction Invariance under Temperature Scaling:** For every sample, `predicted_label` under `group_safe_temperature` is identical to `predicted_label` under `uncalibrated_softmax` ($0$ mismatches across 150,300 comparisons).
5. **Metric Consistency:** Fold metric averages in [`outputs/metrics/phase_d_fold_metrics.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/metrics/phase_d_fold_metrics.csv) exactly match the means computed directly from sample rows in [`phase_d_calibrated_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/phase_d_calibrated_predictions.csv).

---

### Question 8: What files would need to be created or modified for a leakage-safe implementation?

#### Files to Create:
1. `src/selective_prediction.py`: Core mathematical functions (risk-coverage curves, empirical selective accuracy, AURC, random abstention baseline, and OOF threshold determination).
2. `src/run_selective_prediction.py`: Experiment runner orchestrating selective prediction across coverage levels $\gamma \in [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]$.
3. `tests/test_selective_prediction.py`: Unit test suite verifying monotonic coverage, zero-coverage edge cases, correct selective accuracy formula, and leakage safety.
4. `outputs/predictions/selective_prediction_results.csv`: Saved predictions with acceptance masks.
5. `outputs/tables/selective_prediction_summary.csv`: Main summary comparison table.
6. `outputs/figures/risk_coverage_curves.png`: Publication figures comparing selective risk vs. coverage against random rejection.
7. `outputs/reports/phase_e_selective_prediction_report.md`: Comprehensive Phase E report.

#### Files to Modify:
- `PROJECT_STATUS.md`: Record Phase E completion.
- `config.yaml`: Update coverage levels if finer steps are desired.

---

### Question 9: What are the main methodological risks?

1. **Inter-Subject Baseline Confidence Shift (Primary Risk):**  
   Difficult subjects have lower confidence across all samples. Global thresholding will disproportionately abstain on difficult subjects rather than abstaining uniformly on errors. Both **pooled** and **subject-averaged** selective metrics must be reported.
2. **Transductive Thresholding (Leakage Risk):**  
   Selecting thresholds using test-set quantiles must be clearly reported as an empirical evaluation curve, not an operational decision rule.
3. **Empty Accepted Sets at Extreme Thresholds:**  
   If threshold $\tau$ is too aggressive on a low-confidence subject, zero samples might be accepted. The code must gracefully handle empty accepted sets without dividing by zero.
4. **Misinterpreting AUPRC / Risk Baselines:**  
   Base error rate is ~61%. The random rejection baseline has a constant error rate of 61%. Selective prediction is successful only if selective error drops below 61% as coverage decreases.

---

### Question 10: What is the smallest defensible implementation plan?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHASE E IMPLEMENTATION ROADMAP                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Core Module (src/selective_prediction.py)                                │
│    - Selective accuracy: Acc(gamma) = Corr_accepted / N_accepted            │
│    - Selective risk: Risk(gamma) = 1 - Acc(gamma)                           │
│    - AURC: Trapezoidal integration of Risk(gamma) across gamma in [0.5, 1.0]│
│    - Random baseline: Risk_random(gamma) = Overall_Risk (flat line)         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Unit Test Suite (tests/test_selective_prediction.py)                     │
│    - Test monotonicity, bounds, perfect abstention, random baseline         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Execution Pipeline (src/run_selective_prediction.py)                     │
│    - Evaluate all 4 conditions (all_62, pred_32, stab_32, pred_12)          │
│    - Evaluate both uncertainty measures (u_max, entropy)                    │
│    - Evaluate coverage levels: 1.0, 0.9, 0.8, 0.7, 0.6, 0.5                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. Output Generation & Reporting                                            │
│    - Save summary table, risk-coverage plot, and research report            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Recommended Next Steps

1. **Approve Implementation of Phase E:**  
   Proceed to create `src/selective_prediction.py` and `tests/test_selective_prediction.py`.
2. **Execute Phase E Evaluation:**  
   Run selective prediction on existing Phase D predictions (`phase_d_calibrated_predictions.csv`) across coverages $\gamma \in [0.5, 1.0]$.
3. **Generate Final Visualizations:**  
   Produce high-resolution publication risk-coverage curves comparing models against random rejection.
4. **Draft Phase E Report:**  
   Finalize `outputs/reports/phase_e_selective_prediction_report.md` to complete all experimental requirements for the research project.
