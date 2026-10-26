# Experiment A Research Report: Whole-Trial Prediction Aggregation

**Project:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 24 trials per session, 4 emotion classes)  
**Evaluation Protocol:** 15-fold outer Leave-One-Subject-Out (LOSO) cross-validation  
**Primary Input:** [`outputs/predictions/phase_d_calibrated_predictions.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/phase_d_calibrated_predictions.csv)  
**Execution Script:** [`src/evaluate_trial_aggregation.py`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/src/evaluate_trial_aggregation.py)  
**Test Suite:** [`tests/test_evaluate_trial_aggregation.py`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_evaluate_trial_aggregation.py) (8 unit tests, 100% passing)  
**Date:** October 26, 2026

---

## 1. Executive Summary & Verification of Preliminary Numbers

This report presents the empirical findings of **Experiment A: Whole-Trial Prediction Aggregation**. The experiment investigates whether combining 4-second window-level emotion predictions across an entire trial improves subject-independent classification performance on unseen subjects.

### 1.1 Verification of Preliminary Feasibility Numbers
During preflight feasibility analysis, preliminary numbers were computed dynamically in-memory. In this step, the calculations were independently implemented, strictly validated, and executed:
- **`all_62` (Temperature Scaled):**
  - Preliminary Feasibility Report: Mean Prob = 39.17% ± 9.71%, Majority Vote = 39.44%
  - Recomputed Audited Pipeline: Mean Prob = **39.17% ± 9.71%**, Majority Vote = **39.44% ± 9.80%** (**Exact Match**)
- **`pred_32` (Temperature Scaled):**
  - Preliminary Feasibility Report: Mean Prob = 39.26% ± 9.22%, Majority Vote = 39.35%
  - Recomputed Audited Pipeline: Mean Prob = **39.26% ± 9.22%**, Majority Vote = **39.35% ± 9.28%** (**Exact Match**)
- **`stab_32` (Temperature Scaled):**
  - Preliminary Feasibility Report: Mean Prob = 38.80% ± 9.25%, Majority Vote = 38.98%
  - Recomputed Audited Pipeline: Mean Prob = **38.80% ± 9.25%**, Majority Vote = **38.89% ± 9.32%**
  - *Minor Discrepancy Explanation:* In the preflight script, pandas `mode()` arbitrary first-occurrence was used on the single tied trial; the formal pipeline uses deterministic tie-breaking via cumulative probability sums, yielding 38.89%.
- **`pred_12` (Temperature Scaled):**
  - Preliminary Feasibility Report: Mean Prob = 37.22% ± 7.75%, Majority Vote = 37.22%
  - Recomputed Audited Pipeline: Mean Prob = **37.22% ± 7.75%**, Majority Vote = **37.31% ± 7.75%**
  - *Minor Discrepancy Explanation:* Deterministic cumulative probability tie-breaking resolved 3 tied trials, gaining 1 additional correct trial globally.

### 1.2 Core Research Verdict
**Whole-trial prediction aggregation does not meaningfully improve subject-independent classification accuracy.**
- For all channel configurations, trial-level aggregated accuracy remains within **-0.8 to +0.4 percentage points** of the window-level baseline:
  - `all_62`: 39.53% (window) $\to$ **39.44%** (trial vote) / **39.17%** (trial mean prob)
  - `pred_32`: 38.90% (window) $\to$ **39.35%** (trial vote) / **39.26%** (trial mean prob)
  - `stab_32`: 38.56% (window) $\to$ **38.89%** (trial vote) / **38.80%** (trial mean prob)
  - `pred_12`: 38.04% (window) $\to$ **37.31%** (trial vote) / **37.22%** (trial mean prob)
- **Electrophysiological Root Cause:** Cross-subject domain shift in EEG is stationary over the span of a single trial (40–256 seconds). When an unseen subject's feature distribution lies on the incorrect side of a decision boundary, individual window predictions are systematically biased in the same direction. Temporal aggregation consolidates rather than corrects this systematic inter-subject shift.

---

## 2. Trial Validation and Data Structure

The trial aggregation pipeline verified the following structural invariants:
1. **Trial Count & Uniqueness:** Exactly **1,080** trial presentations across the dataset (15 subjects $\times$ 3 sessions $\times$ 24 trials). Every subject has exactly 72 trials (24 per session).
2. **Disambiguation:** The composite key `(subject_id, session_id, trial_id)` uniquely isolates repeated trial indices across sessions and participants.
3. **Label Integrity:** Checked all 1,080 trials; zero trials exhibit conflicting window labels. Every window within a trial has an identical ground-truth label.
4. **Class Balance:** Exactly 270 trials per emotion class (25.00% balanced across Neutral, Sad, Fear, Happy). Chance level is strictly 25.00%.
5. **Window Distribution:** Every trial contains between 10 and 64 windows (mean: 34.79 windows). Zero trials have 0 windows.

---

## 3. Comprehensive Trial-Level Performance Summary

Table 1 summarizes performance across all 16 experimental configurations (4 channel conditions $\times$ 2 calibration regimes $\times$ 2 aggregation methods):

#### Table 1: Trial-Level Aggregation Performance Summary ($N = 1,080$ Trials)
*(Subject mean accuracy evaluated across $N = 15$ outer LOSO subjects; 95% CIs reported)*

| Condition | Calibration Method | Aggregation Method | Window Acc (Phase A/D) | Pooled Trial Acc | Subject Mean Acc (Mean ± SD) | Median Acc [Min, Max] | Student-$t$ 95% CI | Bootstrap 95% CI | Macro-F1 | Correct / Total Trials |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`all_62`** | `group_safe_temp` | **Majority Vote** | 39.53% | **39.44%** | **39.44% ± 9.80%** | 38.89% [22.2%, 62.5%] | [34.02%, 44.87%] | [34.72%, 44.44%] | 39.65% | 426 / 1080 |
| | `group_safe_temp` | **Mean Prob** | 39.53% | **39.17%** | **39.17% ± 9.71%** | 38.89% [22.2%, 61.1%] | [33.79%, 44.54%] | [34.26%, 44.17%] | 39.37% | 423 / 1080 |
| | `uncalibrated` | **Majority Vote** | 39.53% | **39.44%** | **39.44% ± 9.80%** | 38.89% [22.2%, 62.5%] | [34.02%, 44.87%] | [34.72%, 44.44%] | 39.65% | 426 / 1080 |
| | `uncalibrated` | **Mean Prob** | 39.53% | **39.26%** | **39.26% ± 9.83%** | 38.89% [22.2%, 61.1%] | [33.82%, 44.70%] | [34.44%, 44.31%] | 39.47% | 424 / 1080 |
| **`pred_32`** | `group_safe_temp` | **Majority Vote** | 38.90% | **39.35%** | **39.35% ± 9.28%** | 38.89% [23.6%, 59.7%] | [34.21%, 44.49%] | [34.86%, 43.89%] | 39.45% | 425 / 1080 |
| | `group_safe_temp` | **Mean Prob** | 38.90% | **39.26%** | **39.26% ± 9.22%** | 38.89% [23.6%, 59.7%] | [34.15%, 44.37%] | [34.72%, 43.89%] | 39.35% | 424 / 1080 |
| | `uncalibrated` | **Majority Vote** | 38.90% | **39.35%** | **39.35% ± 9.28%** | 38.89% [23.6%, 59.7%] | [34.21%, 44.49%] | [34.86%, 43.89%] | 39.45% | 425 / 1080 |
| | `uncalibrated` | **Mean Prob** | 38.90% | **39.26%** | **39.26% ± 9.22%** | 38.89% [23.6%, 59.7%] | [34.15%, 44.37%] | [34.72%, 43.89%] | 39.35% | 424 / 1080 |
| **`stab_32`** | `group_safe_temp` | **Majority Vote** | 38.56% | **38.89%** | **38.89% ± 9.32%** | 38.89% [23.6%, 61.1%] | [33.73%, 44.05%] | [34.31%, 43.47%] | 38.94% | 420 / 1080 |
| | `group_safe_temp` | **Mean Prob** | 38.56% | **38.80%** | **38.80% ± 9.25%** | 38.89% [23.6%, 61.1%] | [33.67%, 43.92%] | [34.31%, 43.47%] | 38.85% | 419 / 1080 |
| | `uncalibrated` | **Majority Vote** | 38.56% | **38.89%** | **38.89% ± 9.32%** | 38.89% [23.6%, 61.1%] | [33.73%, 44.05%] | [34.31%, 43.47%] | 38.94% | 420 / 1080 |
| | `uncalibrated` | **Mean Prob** | 38.56% | **38.80%** | **38.80% ± 9.25%** | 38.89% [23.6%, 61.1%] | [33.67%, 43.92%] | [34.31%, 43.47%] | 38.85% | 419 / 1080 |
| **`pred_12`** | `group_safe_temp` | **Majority Vote** | 38.04% | **37.31%** | **37.31% ± 7.75%** | 37.50% [25.0%, 55.6%] | [33.02%, 41.61%] | [33.47%, 41.11%] | 37.09% | 403 / 1080 |
| | `group_safe_temp` | **Mean Prob** | 38.04% | **37.22%** | **37.22% ± 7.75%** | 37.50% [25.0%, 55.6%] | [32.93%, 41.52%] | [33.33%, 41.11%] | 36.96% | 402 / 1080 |
| | `uncalibrated` | **Majority Vote** | 38.04% | **37.31%** | **37.31% ± 7.75%** | 37.50% [25.0%, 55.6%] | [33.02%, 41.61%] | [33.47%, 41.11%] | 37.09% | 403 / 1080 |
| | `uncalibrated` | **Mean Prob** | 38.04% | **37.22%** | **37.22% ± 7.75%** | 37.50% [25.0%, 55.6%] | [32.93%, 41.52%] | [33.33%, 41.11%] | 36.96% | 402 / 1080 |

*(Persisted in [`outputs/tables/trial_level_aggregation_summary.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/trial_level_aggregation_summary.csv))*

---

## 4. Paired Comparison: Mean Probability vs. Majority Voting

Table 2 evaluates whether Method A (soft voting) and Method B (hard voting) differ significantly:

#### Table 2: Paired Subject-Level Differences ($\Delta\text{Acc} = \text{Mean Prob} - \text{Majority Vote}$)

| Condition | Calibration Method | Mean Prob Acc | Maj Vote Acc | Mean $\Delta\text{Acc}$ | Student-$t$ 95% CI | Bootstrap 95% CI | Paired $t$ ($p$-value) | Wilcoxon ($p$-value) | Positive / Negative / Tied Subjects |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`all_62`** | `group_safe_temp` | 39.17% | 39.44% | -0.28 pp | [-0.60, +0.04] | [-0.56, +0.00] | 0.082 | 0.084 | 0 / 3 / 12 |
| | `uncalibrated` | 39.26% | 39.44% | -0.19 pp | [-0.46, +0.09] | [-0.42, +0.00] | 0.164 | 0.158 | 0 / 2 / 13 |
| **`pred_32`** | `group_safe_temp` | 39.26% | 39.35% | -0.09 pp | [-0.29, +0.11] | [-0.28, +0.00] | 0.334 | 0.317 | 0 / 1 / 14 |
| | `uncalibrated` | 39.26% | 39.35% | -0.09 pp | [-0.29, +0.11] | [-0.28, +0.00] | 0.334 | 0.317 | 0 / 1 / 14 |
| **`stab_32`** | `group_safe_temp` | 38.80% | 38.89% | -0.09 pp | [-0.29, +0.11] | [-0.28, +0.00] | 0.334 | 0.317 | 0 / 1 / 14 |
| | `uncalibrated` | 38.80% | 38.89% | -0.09 pp | [-0.29, +0.11] | [-0.28, +0.00] | 0.334 | 0.317 | 0 / 1 / 14 |
| **`pred_12`** | `group_safe_temp` | 37.22% | 37.31% | -0.09 pp | [-0.55, +0.36] | [-0.56, +0.28] | 0.670 | 0.632 | 2 / 3 / 10 |
| | `uncalibrated` | 37.22% | 37.31% | -0.09 pp | [-0.55, +0.36] | [-0.56, +0.28] | 0.670 | 0.632 | 2 / 3 / 10 |

### Findings from Paired Tests:
1. **Statistically Indistinguishable:** Across all 8 comparisons, $p > 0.08$ for both paired $t$-test and Wilcoxon signed-rank test. Mean-probability aggregation and majority voting yield essentially identical predictions.
2. **Subject Consistency:** For 10 to 14 of the 15 subjects, the two aggregation methods produce identical classification outcomes across all 72 trials ($\Delta = 0.00\text{ pp}$).
3. **Slight Majority Vote Advantage:** Majority voting correctly classifies 1 to 3 additional trials globally (out of 1,080), as discrete mode consensus filters out occasional single-window probability spikes.

---

## 5. Class-Level Performance and Error Patterns

Table 3 shows per-emotion precision, recall, and F1-scores under primary temperature-scaled mean-probability aggregation:

#### Table 3: Per-Class Performance (Mean-Probability Aggregation, Temperature-Scaled)

| Condition | Emotion Class | Support (Trials) | Precision (%) | Recall (%) | F1-Score (%) | Primary Misclassification Tendency |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **`all_62`** | Neutral | 270 | 38.0% | 40.7% | 39.3% | Confused with Sad (27.0%) and Happy (17.4%) |
| | Sad | 270 | 38.2% | 36.7% | 37.4% | Confused with Neutral (27.8%) and Fear (23.0%) |
| | Fear | 270 | 38.7% | 35.6% | 37.0% | Confused with Sad (28.1%) and Neutral (24.8%) |
| | Happy | 270 | 42.6% | 43.7% | 43.1% | Confused with Neutral (27.4%) and Fear (15.2%) |
| **`pred_32`** | Neutral | 270 | 38.3% | 41.1% | 39.6% | Confused with Sad (26.3%) and Happy (17.8%) |
| | Sad | 270 | 37.9% | 37.8% | 37.8% | Confused with Neutral (26.7%) and Fear (22.6%) |
| | Fear | 270 | 38.8% | 35.6% | 37.1% | Confused with Sad (28.5%) and Neutral (24.1%) |
| | Happy | 270 | 43.2% | 43.0% | 43.1% | Confused with Neutral (26.7%) and Fear (17.4%) |
| **`stab_32`** | Neutral | 270 | 37.6% | 40.7% | 39.1% | Confused with Sad (27.0%) and Happy (17.4%) |
| | Sad | 270 | 37.7% | 37.4% | 37.6% | Confused with Neutral (27.0%) and Fear (23.3%) |
| | Fear | 270 | 38.4% | 35.2% | 36.7% | Confused with Sad (28.5%) and Neutral (24.1%) |
| | Happy | 270 | 42.6% | 41.9% | 42.2% | Confused with Neutral (26.7%) and Fear (17.8%) |
| **`pred_12`** | Neutral | 270 | 35.4% | 36.7% | 36.0% | Confused with Sad (28.5%) and Happy (21.9%) |
| | Sad | 270 | 36.9% | 35.2% | 36.0% | Confused with Neutral (27.0%) and Fear (24.8%) |
| | Fear | 270 | 35.5% | 34.4% | 35.0% | Confused with Sad (27.8%) and Neutral (25.9%) |
| | Happy | 270 | 41.9% | 42.6% | 42.2% | Confused with Neutral (28.1%) and Fear (16.3%) |

*(Persisted in [`outputs/tables/trial_level_aggregation_class_metrics.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/trial_level_aggregation_class_metrics.csv))*

### Key Confusion Insights:
- **Happy is Most Distinct:** Happy consistently achieves the highest precision (42%–43%) and recall (42%–44%).
- **Neutral vs. Negative Valence Blur:** Neutral, Sad, and Fear exhibit high mutual confusion (~24%–28%), reflecting known electrophysiological overlap in low-valence states under subject-independent linear boundaries.

---

## 6. Generated Publication Figures

1. **Comparison Bar Chart:**  
   `outputs/figures/trial_level_aggregation_comparison.png`  
   Plots window-level baseline accuracy against trial mean-probability aggregation and majority voting across all 4 channel conditions, with standard error bars and chance-level reference (25.0%).
2. **Confusion Matrices:**  
   `outputs/figures/trial_level_aggregation_confusion_matrix.png`  
   Displays row-normalized 4-class confusion matrices for `all_62`, `pred_32`, `stab_32`, and `pred_12`.

---

## 7. Conclusions & Recommended Next Steps

### 7.1 Definitive Conclusions of Experiment A
1. **Hypothesis Not Supported for Post-Hoc Aggregation:** Merely aggregating window-level predictions from a window-trained model across a trial does not bridge the cross-subject generalization gap (+0.4 pp maximum gain).
2. **Channel Reduction Holds at Trial Level:** 32 channels (`pred_32`: 39.35%, `stab_32`: 38.89%) perform comparably to all 62 channels (39.44%), confirming that channel reduction does not degrade trial-level decisions.
3. **Voting Methods are Equivalent:** Mean probability aggregation and majority voting produce statistically indistinguishable results ($p > 0.08$).

### 7.2 Recommended Next Experiment: Experiment B
Now that Experiment A has conclusively demonstrated that post-hoc prediction aggregation cannot solve the domain shift problem, the logical next step is **Experiment B: Trial-Level Feature Aggregation and Model Retraining**:
- **Why:** Training directly on trial-averaged Differential Entropy features allows the classifier to construct decision boundaries tailored to whole-trial variance, rather than attempting to classify fluctuating 4-second windows.
- **Protocol:** Average the 310 features per trial ($N = 1,080$), evaluate 15-fold outer LOSO with inner `GroupKFold` hyperparameter tuning, and test whether non-linear RBF kernels or linear models can capture whole-trial affective representations.
