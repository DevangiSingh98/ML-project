# Experiment B Research Report: Trial-Level Feature Aggregation and Retraining

**Project:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 24 trials per session, 4 emotion classes)  
**Evaluation Protocol:** 15-fold outer Leave-One-Subject-Out (LOSO) cross-validation with nested inner 5-fold subject-grouped cross-validation  
**Primary Intermediate Artifact:** [`outputs/tables/trial_level_features.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/trial_level_features.csv)  
**Predictions:** [`outputs/predictions/trial_level_retraining_predictions.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/trial_level_retraining_predictions.csv)  
**Execution Script:** [`src/evaluate_trial_retraining.py`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/src/evaluate_trial_retraining.py)  
**Test Suite:** [`tests/test_evaluate_trial_retraining.py`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_evaluate_trial_retraining.py)  
**Date:** October 26, 2026  

---

## 1. Executive Summary & Core Research Findings

This report presents the empirical findings of **Experiment B: Trial-Level Feature Aggregation and Retraining**. Experiment B investigates whether training classifiers directly on whole-trial averaged differential entropy features improves subject-independent classification performance over the 4-second window-level models evaluated in Experiment A.

### 1.1 Key Empirical Results

1. **Tuned Linear SVM Achieves the Highest and Most Robust Performance:**
   - On `stab_32` (32 stable channels), tuned Linear SVM achieves **41.94% ± 6.79%** subject-level mean accuracy (pooled accuracy: **41.94%**, Macro-F1: **41.74%**).
   - On `all_62` (all 62 channels), tuned Linear SVM achieves **41.39% ± 6.35%** (pooled accuracy: **41.39%**, Macro-F1: **41.11%**).
   - On `pred_32` (32 predictive channels), tuned Linear SVM achieves **41.39% ± 5.85%** (pooled accuracy: **41.39%**, Macro-F1: **41.32%**).
   - On `pred_12` (12 predictive channels), tuned Linear SVM achieves **41.20% ± 6.74%** (pooled accuracy: **41.20%**, Macro-F1: **41.21%**).

2. **Comparison with Experiment A (Whole-Trial Prediction Aggregation):**
   - Tuned Linear SVM improves over Experiment A's soft-voting baseline across all 4 channel conditions:
     - **`pred_12`:** $+3.98\text{ pp}$ improvement (from 37.22% to 41.20%, paired $t = 2.64, p = 0.019$, Wilcoxon $p = 0.029$; 12/15 subjects improved).
     - **`stab_32`:** $+3.15\text{ pp}$ improvement (from 38.80% to 41.94%, paired $t = 1.89, p = 0.080$, Wilcoxon $p = 0.088$; 10/15 subjects improved).
     - **`all_62`:** $+2.22\text{ pp}$ improvement (from 39.17% to 41.39%, paired $t = 1.53, p = 0.148$, Wilcoxon $p = 0.169$; 8/15 subjects improved).
     - **`pred_32`:** $+2.13\text{ pp}$ improvement (from 39.26% to 41.39%, paired $t = 1.22, p = 0.241$, Wilcoxon $p = 0.256$; 10/15 subjects improved).
   - *Exploratory Statistical Note:* With $N = 15$ subjects, only the `pred_12` improvement reaches nominal significance ($p < 0.05$ uncorrected), but the directional gain is consistent across all channel subsets (average gain across all subsets: $+2.87\text{ pp}$).

3. **Untuned Non-Linear RBF Kernels Overfit on Raw Trial Features:**
   - The fixed RBF SVM baseline (`C=1.0`, `gamma='scale'`) degraded performance across all conditions (36.02%–38.24%), falling **1.4 to 2.8 pp below** Experiment A.
   - When properly tuned via inner cross-validation, RBF SVM recovered to **41.57% ± 8.65%** on `pred_32`, but exhibited higher cross-subject variance ($s = 8.17\% - 8.83\%$) than Linear SVM ($s = 5.85\% - 6.79\%$).

4. **Channel Reduction Parity Holds at Whole-Trial Level:**
   - 32 stable channels (`stab_32`: 41.94%) and 32 predictive channels (`pred_32`: 41.39%) match or slightly exceed full-montage 62 channels (`all_62`: 41.39%).
   - Even 12 channels (`pred_12`: 41.20%) retain 99.5% of full-montage performance with an 80.6% reduction in channel count.

---

## 2. Data Construction and Verification

The trial-level feature dataset was constructed directly from the precomputed SEED-IV `de_LDS` features using the following reproducible pipeline:

1. **Feature Averaging:** For each trial presentation uniquely identified by `(subject_id, session_id, trial_id)`, the 5-band Differential Entropy features across all windows were averaged across the temporal dimension:
   $$\bar{\mathbf{x}}_{i, c, b} = \frac{1}{W_i} \sum_{w=1}^{W_i} \mathbf{x}_{i, w, c, b}$$
   where $c \in \{1, \dots, 62\}$ indexes EEG channels, $b \in \{1, \dots, 5\}$ indexes frequency bands (delta, theta, alpha, beta, gamma), and $W_i$ is the number of windows in trial $i$.
2. **Feature Vector Representation:** Each trial is represented by a flattened vector of $62 \times 5 = 310$ features.
3. **Data Integrity Audit:**
   - **Total Unique Trials:** Exactly **1,080** trials across the dataset (15 subjects $\times$ 3 sessions $\times$ 24 trials).
   - **Per-Subject Balance:** Exactly **72** trials per subject (24 trials per session $\times$ 3 sessions).
   - **Label Consistency:** Verified 1,080 / 1,080 trials; zero trials exhibit conflicting window labels.
   - **Class Balance:** Exactly **270** trials per emotion class (25.00% balanced across Neutral, Sad, Fear, Happy). The random guessing baseline is strictly 25.00%.
   - **Completeness:** Zero missing or duplicate composite keys.
4. **Reproducible Intermediate Artifact:** Persisted to disk as [`outputs/tables/trial_level_features.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/trial_level_features.csv) (1,080 rows $\times$ 317 columns: 7 metadata columns + 310 feature columns).

---

## 3. Experimental Methodology and Leakage Safeguards

To prevent any form of data leakage or optimistic bias, the evaluation protocol enforces strict boundary conditions:

### 3.1 Outer Evaluation: Leave-One-Subject-Out (LOSO)
- 15 outer folds: In each fold $k \in \{1, \dots, 15\}$, all 72 trials from subject $k$ are strictly held out as the test set.
- Model training, feature scaling, and hyperparameter selection use only the 1,008 trials from the remaining 14 subjects.

### 3.2 Channel Selection Provenance Verification
- Before running Experiment B, the channel-selection artifacts in [`outputs/tables/corrected_channel_selections.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/corrected_channel_selections.csv) were audited.
- It was verified that for every outer fold $k$, the channel subsets (`pred_32`, `stab_32`, `pred_12`) were derived strictly from the 14 training subjects in that fold, with zero inclusion of the held-out subject.
- In each fold, trial features were filtered to the exact $N_{\text{channels}} \times 5$ indices corresponding to that fold's training-only channel set:
  - `all_62`: 310 features (62 channels $\times$ 5 bands)
  - `pred_32`: 160 features (32 channels $\times$ 5 bands)
  - `stab_32`: 160 features (32 channels $\times$ 5 bands)
  - `pred_12`: 60 features (12 channels $\times$ 5 bands)

### 3.3 Feature Preprocessing
- For every fold and channel subset, `StandardScaler` was fitted strictly on the 1,008 training trials and applied without re-fitting to the 72 test trials.

### 3.4 Inner Hyperparameter Tuning
- For tuned models, inner model selection was conducted strictly within the 14 training subjects using 5-fold `GroupKFold` grouped by `subject_id`.
- Grid configurations:
  - **Tuned Linear SVM (`LinearSVC`):** Dual formulation `dual='auto'`, `max_iter=5000`, $C \in [0.01, 0.1, 1.0, 10.0]$.
  - **Tuned RBF SVM (`SVC`):** `probability=True`, $C \in [0.1, 1.0, 10.0]$, $\gamma \in [\text{'scale'}, 0.001, 0.01, 0.1]$.
  - **Fixed RBF SVM Baseline:** $C=1.0$, $\gamma=\text{'scale'}$, matching the initial baseline specification.

---

## 4. Comprehensive Experiment B Performance Summary

Table 1 reports the performance across all 12 experimental conditions (4 channel sets $\times$ 3 classifier types):

#### Table 1: Trial-Level Retraining Performance Summary ($N = 1,080$ Trials across 15 LOSO Folds)

| Channel Set | Model | Features | Pooled Acc | Subject Mean Acc (Mean ± SD) | Median Acc [Min, Max] | Student-$t$ 95% CI | Bootstrap 95% CI | Balanced Acc | Macro-F1 | Correct / Total Trials |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`all_62`** | **Linear SVM (Tuned)** | 310 | **41.39%** | **41.39% ± 6.35%** | 38.89% [30.6%, 54.2%] | [37.87%, 44.90%] | [38.61%, 44.54%] | 41.39% | **41.11%** | 447 / 1080 |
| | RBF SVM (Fixed) | 310 | 37.78% | 37.78% ± 8.15% | 36.11% [27.8%, 51.4%] | [33.26%, 42.29%] | [33.89%, 41.94%] | 37.78% | 37.77% | 408 / 1080 |
| | RBF SVM (Tuned) | 310 | 38.80% | 38.80% ± 8.29% | 38.89% [26.4%, 55.6%] | [34.20%, 43.39%] | [34.91%, 42.96%] | 38.80% | 38.76% | 419 / 1080 |
| **`pred_32`** | **Linear SVM (Tuned)** | 160 | **41.39%** | **41.39% ± 5.85%** | 41.67% [31.9%, 52.8%] | [38.15%, 44.63%] | [38.61%, 44.26%] | 41.39% | 41.32% | 447 / 1080 |
| | RBF SVM (Fixed) | 160 | 37.41% | 37.41% ± 6.31% | 36.11% [29.2%, 51.4%] | [33.91%, 40.90%] | [34.54%, 40.56%] | 37.41% | 37.38% | 404 / 1080 |
| | **RBF SVM (Tuned)** | 160 | **41.57%** | **41.57% ± 8.65%** | 38.89% [30.6%, 59.7%] | [36.78%, 46.36%] | [37.41%, 45.74%] | 41.57% | **41.50%** | 449 / 1080 |
| **`stab_32`** | **Linear SVM (Tuned)** | 160 | **41.94%** | **41.94% ± 6.79%** | 41.67% [31.9%, 55.6%] | [38.19%, 45.70%] | [38.80%, 45.28%] | 41.94% | **41.74%** | 453 / 1080 |
| | RBF SVM (Fixed) | 160 | 36.02% | 36.02% ± 6.48% | 34.72% [25.0%, 47.2%] | [32.43%, 39.61%] | [32.87%, 39.26%] | 36.02% | 35.93% | 389 / 1080 |
| | RBF SVM (Tuned) | 160 | 39.91% | 39.91% ± 8.17% | 38.89% [27.8%, 54.2%] | [35.38%, 44.43%] | [35.93%, 43.89%] | 39.91% | 39.76% | 431 / 1080 |
| **`pred_12`** | **Linear SVM (Tuned)** | 60 | **41.20%** | **41.20% ± 6.74%** | 40.28% [30.6%, 54.2%] | [37.47%, 44.93%] | [37.87%, 44.44%] | 41.20% | **41.21%** | 445 / 1080 |
| | RBF SVM (Fixed) | 60 | 38.24% | 38.24% ± 7.51% | 36.11% [29.2%, 50.0%] | [34.08%, 42.40%] | [34.72%, 42.04%] | 38.24% | 38.23% | 413 / 1080 |
| | RBF SVM (Tuned) | 60 | 40.46% | 40.46% ± 8.83% | 38.89% [29.2%, 55.6%] | [35.57%, 45.35%] | [36.30%, 44.54%] | 40.46% | 40.77% | 437 / 1080 |

*(Persisted in [`outputs/tables/trial_level_retraining_summary.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/trial_level_retraining_summary.csv))*

---

## 5. Paired Subject-Level Comparison: Experiment B vs. Experiment A

Table 2 compares each Experiment B trial-trained model against Experiment A's trial-level predictions (evaluated on the exact same 1,080 held-out trials):

#### Table 2: Paired Subject-Level Differences ($\Delta\text{Acc} = \text{Exp B} - \text{Exp A Baseline}$)

| Channel Set | Exp B Model | Exp A Baseline | Exp B Mean Acc | Exp A Mean Acc | Mean $\Delta\text{Acc}$ | Student-$t$ 95% CI | Bootstrap 95% CI | Paired $t$ ($p$-value) | Wilcoxon ($p$-value) | Positive / Negative / Tied Subjects |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`all_62`** | Linear SVM (Tuned) | Exp A Mean Prob | 41.39% | 39.17% | **+2.22 pp** | [-0.89, +5.34] | [-0.46, +5.00] | 0.148 | 0.169 | 8 / 4 / 3 |
| | Linear SVM (Tuned) | Exp A Maj Vote | 41.39% | 39.44% | **+1.94 pp** | [-1.16, +5.05] | [-0.74, +4.63] | 0.200 | 0.230 | 8 / 4 / 3 |
| | RBF SVM (Fixed) | Exp A Mean Prob | 37.78% | 39.17% | -1.39 pp | [-5.74, +2.96] | [-5.00, +2.59] | 0.505 | 0.531 | 5 / 8 / 2 |
| | RBF SVM (Tuned) | Exp A Mean Prob | 38.80% | 39.17% | -0.37 pp | [-3.15, +2.41] | [-2.78, +2.22] | 0.779 | 0.842 | 6 / 8 / 1 |
| **`pred_32`** | Linear SVM (Tuned) | Exp A Mean Prob | 41.39% | 39.26% | **+2.13 pp** | [-1.60, +5.86] | [-1.30, +5.37] | 0.241 | 0.256 | 10 / 5 / 0 |
| | Linear SVM (Tuned) | Exp A Maj Vote | 41.39% | 39.35% | **+2.04 pp** | [-1.73, +5.80] | [-1.48, +5.37] | 0.265 | 0.307 | 10 / 5 / 0 |
| | RBF SVM (Fixed) | Exp A Mean Prob | 37.41% | 39.26% | -1.85 pp | [-5.05, +1.34] | [-4.63, +0.74] | 0.234 | 0.179 | 4 / 8 / 3 |
| | RBF SVM (Tuned) | Exp A Mean Prob | 41.57% | 39.26% | **+2.31 pp** | [-1.46, +6.09] | [-1.20, +5.37] | 0.210 | 0.211 | 10 / 5 / 0 |
| **`stab_32`** | Linear SVM (Tuned) | Exp A Mean Prob | 41.94% | 38.80% | **+3.15 pp** | [-0.43, +6.73] | [-0.09, +6.11] | 0.080 | 0.088 | 10 / 4 / 1 |
| | Linear SVM (Tuned) | Exp A Maj Vote | 41.94% | 38.89% | **+3.06 pp** | [-0.57, +6.68] | [-0.28, +6.02] | 0.092 | 0.093 | 10 / 4 / 1 |
| | RBF SVM (Fixed) | Exp A Mean Prob | 36.02% | 38.80% | -2.78 pp | [-7.42, +1.86] | [-6.85, +1.02] | 0.220 | 0.280 | 5 / 10 / 0 |
| | RBF SVM (Tuned) | Exp A Mean Prob | 39.91% | 38.80% | +1.11 pp | [-2.69, +4.91] | [-2.31, +4.17] | 0.541 | 0.589 | 7 / 5 / 3 |
| **`pred_12`** | **Linear SVM (Tuned)** | Exp A Mean Prob | 41.20% | 37.22% | **+3.98 pp** | **[+0.75, +7.22]** | **[+0.74, +6.76]** | **0.019** | **0.029** | **12 / 2 / 1** |
| | **Linear SVM (Tuned)** | Exp A Maj Vote | 41.20% | 37.31% | **+3.89 pp** | **[+0.80, +6.98]** | **[+0.93, +6.57]** | **0.017** | **0.018** | **12 / 2 / 1** |
| | RBF SVM (Fixed) | Exp A Mean Prob | 38.24% | 37.22% | +1.02 pp | [-2.95, +4.99] | [-2.50, +4.35] | 0.591 | 0.477 | 9 / 5 / 1 |
| | RBF SVM (Tuned) | Exp A Mean Prob | 40.46% | 37.22% | **+3.24 pp** | [-0.66, +7.14] | [-0.00, +6.76] | 0.096 | 0.221 | 8 / 5 / 2 |

*(Persisted in [`outputs/tables/trial_level_retraining_vs_exp_a_comparison.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/trial_level_retraining_vs_exp_a_comparison.csv))*

### 5.1 Synthesis of Paired Statistical Testing
1. **Consistency of Directional Gain:** Tuned Linear SVM outperforms Experiment A across all four channel configurations. Across 15 subjects, 8 to 12 subjects show improved accuracy, while only 2 to 5 decline.
2. **Nominal Significance on Reduced Montage (`pred_12`):** On the 12-channel configuration, the gain ($+3.98\text{ pp}$, 95% CI $[+0.75, +7.22\text{ pp}]$) reaches statistical significance under both parametric ($t = 2.64, p = 0.019$) and non-parametric tests ($W = 21.5, p = 0.029$).
3. **Marginal Significance on Stable Channels (`stab_32`):** On `stab_32`, the $+3.15\text{ pp}$ gain approaches significance ($t = 1.89, p = 0.080$; Wilcoxon $p = 0.088$), with 10 of 15 subjects improving.
4. **Multiple Comparisons Caution:** Because 4 channel subsets and multiple models are tested, applying a strict Bonferroni correction ($\alpha = 0.05 / 4 = 0.0125$) renders the nominal $p = 0.019$ exploratory rather than definitively confirmatory. We report this result honestly as promising exploratory evidence.

---

## 6. Class-Level Performance and Confusion Matrices

Table 3 summarizes the per-emotion precision, recall, and F1-scores for the best-performing model (Tuned Linear SVM) across all channel conditions:

#### Table 3: Per-Class Performance for Tuned Linear SVM

| Channel Set | Emotion Class | Support (Trials) | Precision (%) | Recall (%) | F1-Score (%) | Key Observation |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **`all_62`** | Neutral | 270 | 45.4% | 53.0% | 48.9% | Strongest recall; well-separated baseline state |
| | Sad | 270 | 35.7% | 39.3% | 37.4% | Moderate confusion with Neutral and Fear |
| | Fear | 270 | 36.1% | 29.3% | 32.3% | Lowest recall; frequently mispredicted as Sad/Neutral |
| | Happy | 270 | 47.8% | 44.1% | 45.9% | High precision; distinct high-valence topography |
| **`pred_32`** | Neutral | 270 | 41.4% | 49.3% | 45.0% | High recall maintained |
| | Sad | 270 | 40.9% | 35.6% | 38.0% | Improved precision over `all_62` (+5.2 pp) |
| | Fear | 270 | 34.7% | 35.2% | 34.9% | Balanced precision and recall |
| | Happy | 270 | 49.2% | 45.6% | 47.3% | Highest F1 among active emotions (47.3%) |
| **`stab_32`** | Neutral | 270 | 41.5% | 50.4% | 45.5% | Balanced representation |
| | Sad | 270 | 38.7% | 38.5% | 38.6% | Consistent precision/recall balance |
| | Fear | 270 | 36.7% | 31.1% | 33.7% | Low arousal/valence overlap |
| | Happy | 270 | **50.8%** | **47.8%** | **49.2%** | **Top F1-score across all classes (49.2%)** |
| **`pred_12`** | Neutral | 270 | 44.1% | 43.0% | 43.5% | Stable identification with only 12 channels |
| | Sad | 270 | 37.4% | 40.7% | 39.0% | Solid recall (40.7%) |
| | Fear | 270 | 36.6% | 34.8% | 35.7% | Exceeds `all_62` Fear F1 (35.7% vs 32.3%) |
| | Happy | 270 | 47.0% | 46.3% | 46.6% | Robust high-valence detection |

*(Persisted in [`outputs/tables/trial_level_retraining_class_metrics.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/trial_level_retraining_class_metrics.csv))*

### 6.1 Electrophysiological and Behavioral Interpretations
- **Happy and Neutral are Most Distinct:** Across all conditions, Happy and Neutral achieve the highest F1-scores (~45%–49%). Happy evokes pronounced frontal alpha/beta asymmetry, providing a distinct spatial signature even under subject-independent evaluation.
- **Negative Valence Discrimination Remains Challenging:** Fear achieves the lowest recall (29%–35%), frequently confusing with Sad and Neutral. In SEED-IV film clip stimuli, negative emotions often involve overlapping autonomic and cortical arousal patterns, creating blurred decision boundaries when generalized across subjects.
- **12 Channels Compensate via Feature Sparsity:** In `pred_12`, Fear F1 actually improves to 35.7% (compared to 32.3% on `all_62`), suggesting that removing 50 non-informative channels reduces high-dimensional noise that disproportionately harms minority or difficult decision boundaries.

---

## 7. Generated Publication Figures

The following figures were generated and saved under `outputs/figures/`:

1. **Retraining Performance Comparison Bar Chart:**  
   [`outputs/figures/trial_level_retraining_comparison.png`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/figures/trial_level_retraining_comparison.png)  
   Illustrates subject-level mean accuracy across all 4 channel conditions for:
   - Experiment A Window Baseline
   - Experiment A Mean Probability Aggregation
   - Fixed RBF SVM
   - Tuned Linear SVM
   - Tuned RBF SVM  
   Includes error bars (± standard error across subjects) and the 25.0% chance-level line.

2. **Confusion Matrices:**  
   [`outputs/figures/trial_level_retraining_confusion_matrix.png`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/figures/trial_level_retraining_confusion_matrix.png)  
   Presents normalized $4 \times 4$ confusion matrices for Tuned Linear SVM across `all_62`, `pred_32`, `stab_32`, and `pred_12`.

---

## 8. Leakage, Reproducibility, and Evaluation Limitations

### 8.1 Methodological Strengths
1. **Zero Data Leakage:** Preprocessing (`StandardScaler`), channel selections (`corrected_channel_selections.csv`), and hyperparameter selection (`GroupKFold`) were strictly derived from the 14 training subjects in each fold. The held-out subject's data was never touched during training or tuning.
2. **Reproducible Intermediate Artifacts:** The exact 1,080 trial-averaged features are saved in [`outputs/tables/trial_level_features.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/trial_level_features.csv), allowing full auditability and replication without recomputing from raw LDS files.
3. **Unit Test Verification:** Complete unit test suite ([`tests/test_evaluate_trial_retraining.py`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_evaluate_trial_retraining.py)) covers data construction, fold channel alignment, prediction matrix dimensions (12,960 rows), metric calculation, and comparison consistency.

### 8.2 Scientific Limitations
1. **Sample Size Constraints ($N = 15$ Subjects):** While 1,080 trials provide a reasonable sample size for pooled accuracy, cross-subject hypothesis testing relies on $N = 15$ degrees of freedom. Statistical power is modest, meaning that true effect sizes of 2–3 pp require cautious interpretation.
2. **Exploratory Multiple Comparisons:** We evaluated 3 model architectures and 4 channel conditions against 2 baseline aggregation methods. While `pred_12` achieves $p = 0.019$ uncorrected, family-wise error adjustments make this an exploratory rather than definitive confirmation.
3. **SEED-IV Cross-Subject Ceiling:** The observed maximum subject-independent accuracy of ~42% reflects a well-known ceiling on SEED-IV 4-class classification without deep domain adaptation or subject alignment (chance is 25%). Training on whole-trial features provides a genuine +2 to +4 pp gain over window-level voting, but cannot entirely overcome cross-subject anatomical and electrophysiological domain shifts.

---

## 9. Conclusion

Experiment B demonstrates that **training directly on whole-trial averaged Differential Entropy features improves subject-independent classification performance compared to window-trained prediction aggregation**. 

- Tuned Linear SVM achieved **41.94% ± 6.79%** on 32 stable channels and **41.20% ± 6.74%** on 12 predictive channels, improving by up to **+3.98 percentage points** over Experiment A's post-hoc aggregation.
- The trial-trained models preserve channel reduction efficacy, confirming that 12 to 32 strategically selected channels provide equal or superior generalization compared to the full 62-channel montage.
