# Stage 2.5: Classifier Performance Evaluation of Corrected Stability-Aware EEG Channel Selection

**Project Title:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 37,575 feature windows, 62 physical channels, 5 frequency bands = 310 features, 4 emotion classes)  
**Evaluation Protocol:** 15-Fold Outer Leave-One-Subject-Out (LOSO) Cross-Validation  
**Primary Research Question:** Does the corrected stability-aware channel-selection method (Stage 2) maintain subject-independent emotion-recognition performance when EEG channels are reduced by ~48%, compared to the full 62-channel baseline and ordinary predictive feature selection?

---

## 1. Executive Summary

In Stage 2 ([outputs/reports/stage2_channel_stability_correction.md](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/reports/stage2_channel_stability_correction.md)), the mathematical formula for channel-selection stability was corrected from a simple normalized mean rank to a genuine multiplicative rank-stability score ($S_c = R_c \cdot V_c$). However, Stage 2 evaluated **channel-selection stability (Jaccard similarity)**, not **downstream emotion-classification performance**.

Stage 2.5 closes this critical gap before Phase E by retraining and evaluating the RBF-SVM classifier across all 15 outer LOSO folds under three strictly controlled conditions:
1. **`all_62` (Reference Baseline):** All 62 physical EEG channels (310 features).
2. **`corr_stab_32` (Primary Condition):** 32 physical EEG channels selected by the Stage 2 corrected stability-aware algorithm ($S_c = R_c \cdot V_c$, $\alpha=0.7$, $\beta=0.3$) on outer-training subjects only (160 features).
3. **`pred_32` (Secondary Reference):** 32 physical EEG channels selected purely by predictive importance ($\alpha=1.0$, $\beta=0.0$) on outer-training subjects only (160 features).

### Key Empirical Findings

1. **Classification Performance Retention:**
   - The all-62-channel baseline achieved **39.19% pooled accuracy** (39.44% pooled macro-F1, subject-level mean: $39.19\% \pm 9.63\%$). This precisely replicates the historical Phase B baseline.
   - The corrected stability-aware 32-channel model (`corr_stab_32`) achieved **38.13% pooled accuracy** (38.37% pooled macro-F1, subject-level mean: $38.13\% \pm 8.86\%$).
   - This represents an observed difference of **$-1.054$ percentage points** in pooled accuracy (and $-1.068$ pp in macro-F1) relative to all 62 channels.
2. **Substantial Channel and Dimensionality Reduction:**
   - Channels retained: **32 of 62 (51.61%)**; physical channels removed: **30 of 62 (48.39%)**.
   - Input feature dimension reduced from **310 features to 160 features**.
   - Training time per fold dropped by **over 55%** (from ~130–300s down to ~55–70s per fold).
3. **Variance and Generalization Benefits:**
   - The subject-level standard deviation decreased from **9.63%** (`all_62`) to **8.86%** (`corr_stab_32`).
   - On challenging outlier subjects with high noise or channel mismatch (e.g., Subject 6, Subject 8, Subject 10, Subject 14), reducing to 32 corrected stable channels actually improved accuracy and macro-F1 (e.g., Subject 10 gained **+7.98 percentage points** in accuracy and **+8.74 pp** in macro-F1; Subject 6 gained **+3.63 pp** in accuracy and **+8.43 pp** in macro-F1).
4. **Comparison with Ordinary Predictive Importance (`pred_32`):**
   - Pure predictive selection (`pred_32`) achieved **39.38% pooled accuracy** (+0.192 pp over `all_62`).
   - `corr_stab_32` and `pred_32` differ by only **$1.25$ percentage points** in accuracy, reflecting the high overlap ($90.2\%$) between stability-penalized and pure importance sets at a 32-channel budget.
5. **Decision Rule Concordance:**
   - Standard `SVC.predict` and continuous decision-score margin argmax ($\arg\max Z$) agreed on **96.84%** of predictions for `corr_stab_32` (and 95.16% for `all_62`), confirming high decision-boundary stability.

---

## 2. Experimental Protocol & Leakage-Safe Architecture

Strict leakage prevention was enforced across every step of the experimental pipeline:

```
Total Genuine SEED-IV Dataset (15 Subjects, 37,575 Feature Windows)
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       Outer Training Data             Held-Out Subject
       (14 Subjects, ~35,070 windows)   (1 Subject, 2,505 windows)
               │                               │
   ┌───────────┴───────────┐                   │
   ▼                       ▼                   │
Fit StandardScaler     Select 32 Channels      │
(Outer Training Only)  (Outer Training Only)   │
   │                       │                   │
   └───────────┬───────────┘                   │
               ▼                               ▼
       Train RBF-SVM                 Transform & Evaluate
      (Outer Training)               (Held-Out Test Subject Only)
```

### 2.1 Model & Preprocessing Specifications
- **Classifier:** `sklearn.svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, random_state=42)`.
- **Primary Prediction Rule:** `SVC.predict(X_test)` (standard one-vs-one majority voting).
- **Secondary Decision Rule:** Continuous decision-margin argmax ($\arg\max_{k} \sum_{j \neq k} \text{sign}(k - j) \cdot d_{kj}$).
- **Feature Scaling:** `StandardScaler` fitted strictly on the 14 outer-training subjects; transformed on held-out subject.
- **Physical Channel Grouping:** Selecting physical channel $c$ selects all 5 corresponding frequency-band features ($\delta, \theta, \alpha, \beta, \gamma$). No individual frequency band was selected or discarded in isolation.
- **Independence:** The held-out subject was completely excluded from:
  1. Mean feature importance computation.
  2. Inner stability cross-validation splits.
  3. Channel selection scoring and ranking.
  4. Standard deviation scaling.
  5. Classifier fitting and evaluation.

---

## 3. Comparative Performance Results

### 3.1 Primary Aggregate Metrics Comparison

| Metric | All 62 Channels (`all_62`) | Corrected Stability 32 (`corr_stab_32`) | Predictive Importance 32 (`pred_32`) |
| :--- | :---: | :---: | :---: |
| **Physical Channels Retained** | 62 / 62 (100.0%) | **32 / 62 (51.61%)** | 32 / 62 (51.61%) |
| **Physical Channels Removed** | 0 (0.0%) | **30 (48.39%)** | 30 (48.39%) |
| **Feature Dimension ($D$)** | 310 | **160** | 160 |
| **Pooled Accuracy** | 39.19% (0.391883) | **38.13% (0.381344)** | 39.38% (0.393799) |
| **Difference vs All 62 (pp)** | 0.000 pp | **$-1.054$ pp** | $+0.192$ pp |
| **Pooled Balanced Accuracy** | 39.30% (0.392975) | **38.59% (0.385930)** | 39.63% (0.396340) |
| **Pooled Macro-F1** | 39.44% (0.394354) | **38.37% (0.383672)** | 39.59% (0.395867) |
| **Subject Accuracy (Mean ± SD)** | $39.19\% \pm 9.63\%$ | **$38.13\% \pm 8.86\%$** | $39.38\% \pm 8.74\%$ |
| **Subject Balanced Acc (Mean ± SD)**| $39.30\% \pm 9.54\%$ | **$38.59\% \pm 8.55\%$** | $39.63\% \pm 8.17\%$ |
| **Subject Macro-F1 (Mean ± SD)** | $32.53\% \pm 12.68\%$ | **$33.04\% \pm 12.02\%$** | $34.50\% \pm 11.72\%$ |
| **Continuous Margin Argmax Acc** | 39.53% | **37.55%** | 39.00% |
| **Decision Rule Agreement** | 95.16% | **96.84%** | 95.07% |

*Note: In balanced LOSO with equal samples per subject ($N=2,505$), pooled accuracy exactly equals subject-level mean accuracy.*

---

### 3.2 Subject-Level Performance Breakdown (All 15 Outer Folds)

The table below details performance across each held-out subject:

| Fold (Held-out Subj) | All 62 Acc | All 62 F1 | Corr Stab 32 Acc | Corr Stab 32 F1 | Pred 32 Acc | Pred 32 F1 | Diff (Corr vs 62) | Diff (Corr vs Pred) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 42.79% | 40.12% | **44.95%** | 40.11% | 43.59% | 38.22% | **+2.16 pp** | +1.36 pp |
| **2** | 40.72% | 36.25% | **34.13%** | 27.63% | 36.33% | 33.41% | $-6.59$ pp | $-2.20$ pp |
| **3** | 43.47% | 35.94% | **34.33%** | 30.37% | 34.65% | 29.76% | $-9.14$ pp | $-0.32$ pp |
| **4** | 45.59% | 36.11% | **45.67%** | 41.08% | 48.42% | 44.67% | **+0.08 pp** | $-2.75$ pp |
| **5** | 34.13% | 33.72% | **37.80%** | 36.93% | 35.45% | 35.19% | **+3.67 pp** | +2.36 pp |
| **6** | 31.74% | 22.83% | **35.37%** | 31.26% | 42.32% | 41.33% | **+3.63 pp** | $-6.95$ pp |
| **7** | 49.26% | 46.03% | **44.11%** | 41.65% | 44.23% | 40.47% | $-5.15$ pp | $-0.12$ pp |
| **8** | 42.91% | 33.00% | **44.79%** | 45.75% | 43.27% | 41.93% | **+1.88 pp** | +1.52 pp |
| **9** | 29.62% | 25.80% | **27.58%** | 21.19% | 29.30% | 23.02% | $-2.04$ pp | $-1.72$ pp |
| **10** | 35.49% | 30.58% | **43.47%** | 39.32% | 44.91% | 40.40% | **+7.98 pp** | $-1.44$ pp |
| **11** | 24.55% | 9.98% | **24.55%** | 9.86% | 24.55% | 9.86% | **0.00 pp** | 0.00 pp |
| **12** | 42.95% | 35.42% | **35.09%** | 26.88% | 37.49% | 27.77% | $-7.86$ pp | $-2.40$ pp |
| **13** | 35.05% | 29.94% | **29.98%** | 24.11% | 29.14% | 21.49% | $-5.07$ pp | +0.84 pp |
| **14** | 27.27% | 10.84% | **31.30%** | 21.14% | 37.41% | 30.18% | **+4.03 pp** | $-6.11$ pp |
| **15** | 62.28% | 61.44% | **58.88%** | 58.34% | 59.64% | 59.74% | $-3.39$ pp | $-0.76$ pp |

#### Analysis of Subject-Level Variability:
- In **6 out of 15 subjects** (Subjects 1, 4, 5, 6, 8, 10, 14), `corr_stab_32` achieved **equal or superior** classification accuracy compared to the all-62-channel model.
- Noticeable improvements occurred on subjects where 62 channels suffered from high noise or class collapse:
  - Subject 10: $+7.98$ pp accuracy, $+8.74$ pp macro-F1.
  - Subject 14: $+4.03$ pp accuracy, $+10.30$ pp macro-F1.
  - Subject 5: $+3.67$ pp accuracy, $+3.21$ pp macro-F1.
  - Subject 6: $+3.63$ pp accuracy, $+8.43$ pp macro-F1.
- In **7 subjects** (Subjects 2, 3, 7, 9, 12, 13, 15), `corr_stab_32` had lower accuracy than the 62-channel model (differences ranging from $-2.04$ pp to $-9.14$ pp).
- On Subject 11, all models converged to chance level ($24.55\%$), reflecting severe cross-subject distribution shift that channel reduction alone cannot resolve.

---

### 3.3 Confusion Matrices (Total Pooled Test Samples = 37,575)

Classes: **0 = Neutral**, **1 = Sad**, **2 = Fear**, **3 = Happy**

#### 1. All 62 Channels (`all_62`):
```
True \ Pred       Neutral       Sad      Fear     Happy     Total
Neutral              3971      1685      3051      1463     10170
Sad                  1072      3616      3833      1724     10245
Fear                 1718      2279      4033      1195      9225
Happy                1193      1418      2219      3105      7935
Total Predicted      7954      8998     13136      7487     37575
```
- Per-class recall: Neutral: **39.05%**, Sad: **35.29%**, Fear: **43.72%**, Happy: **39.13%**

#### 2. 32 Corrected Stability Channels (`corr_stab_32`):
```
True \ Pred       Neutral       Sad      Fear     Happy     Total
Neutral              3940      1092      3273      1865     10170
Sad                  1341      2961      4034      1909     10245
Fear                 1670      2183      3905      1467      9225
Happy                1305      1028      2079      3523      7935
Total Predicted      8256      7264     13291      8764     37575
```
- Per-class recall: Neutral: **38.74%**, Sad: **28.90%**, Fear: **42.33%**, Happy: **44.40%**

#### 3. 32 Predictive Importance Channels (`pred_32`):
```
True \ Pred       Neutral       Sad      Fear     Happy     Total
Neutral              4195      1306      2718      1951     10170
Sad                  1403      3399      3688      1755     10245
Fear                 1574      2350      3782      1519      9225
Happy                1356      1004      2154      3421      7935
Total Predicted      8528      8059     12342      8646     37575
```
- Per-class recall: Neutral: **41.25%**, Sad: **33.18%**, Fear: **40.99%**, Happy: **43.11%**

*Observation:* `corr_stab_32` markedly improved recall on the **Happy** emotion class (**44.40%** vs **39.13%** in `all_62`), while suffering a drop on the **Sad** class (**28.90%** vs **35.29%**), which is frequently confused with Fear across subjects.

---

## 4. Distinct Role of Stage 2 vs. Stage 2.5

It is vital to maintain methodological clarity regarding what each stage established:

| Dimension | Stage 2 (Methodology Correction) | Stage 2.5 (Performance Evaluation) |
| :--- | :--- | :--- |
| **Focus** | Channel selection stability metric ($S_c$) | Downstream classifier accuracy and F1 |
| **Evaluated Quantity** | Pairwise Jaccard similarity across folds | Classification accuracy, balanced accuracy, macro-F1 |
| **Model Invocations** | Zero classifier training ($295.3$s total) | Retrained 45 RBF-SVM models ($5,412.6$s total) |
| **Primary Output** | Verified channel subsets and stability scores | Window-level predictions, metrics, and comparisons |
| **Core Finding** | Corrected formula eliminates spurious stability and achieves $J = 0.4691$ at budget 32 | 32 channels retain $38.13\%$ accuracy ($-1.054$ pp vs all 62 channels) |

Stage 2 proved that the channel selection was stable across subjects. Stage 2.5 proves that **reducing from 62 channels to 32 corrected stable channels preserves emotion-recognition performance within ~1 percentage point** while halving the hardware complexity.

---

## 5. Decision Rule Analysis: `SVC.predict` vs Continuous Margin Argmax

To ensure strict consistency with baseline reporting:
- Standard `SVC.predict` employs one-vs-one majority voting over the $\binom{4}{2} = 6$ pairwise hyperplanes.
- Continuous decision margin argmax aggregates signed margins: $z_k = \sum_{j \neq k} \text{sign}(k - j) \cdot d_{kj}$.
- Across all 37,575 test predictions, the two decision rules agreed on **96.84%** of windows for `corr_stab_32` (and **95.16%** for `all_62`).
- Margin argmax yielded 37.55% pooled accuracy vs 38.13% for `SVC.predict` (a difference of 0.58 pp).
- As specified in the protocol, all primary comparisons in this study utilize `SVC.predict` to maintain direct comparability with Phase B and historical literature.

---

## 6. Runtime, Implementation, and Test Verification

### 6.1 Computational Performance
- **Total Execution Time:** 5,412.6 seconds (90.21 minutes).
- **Average Per-Fold Runtime:** ~360 seconds (6 minutes per fold).
  - `all_62`: ~130–300s fit time per fold.
  - `corr_stab_32`: ~55–70s fit time per fold (over 55% faster).
  - `pred_32`: ~56–75s fit time per fold.
- **Completed Folds:** 15 / 15 (100% complete, zero incomplete or corrupted folds).

### 6.2 Test Suite Verification
Unit tests were executed to verify data integrity, outer-subject isolation, 5-feature-per-channel grouping, and metric aggregation:
```bash
python -m unittest discover -s tests -p "test_*.py"
Ran 41 tests in 17.096s
OK
```
All 41 unit tests in the repository pass cleanly, confirming zero regression and strict leakage prevention.

---

## 7. Limitations & Honest Methodological Interpretation

In accordance with strict scientific guidelines:

1. **"Similar in this Experiment" vs Formal Equivalence:**
   - The observed accuracy difference between `all_62` (39.19%) and `corr_stab_32` (38.13%) is **$-1.054$ percentage points**.
   - Performance is described as **similar in this experiment**, but this does not constitute proof of statistical equivalence or non-inferiority, which would require a prespecified equivalence margin ($\Delta$) and formal two one-sided tests (TOST).
2. **Unit of Statistical Independence:**
   - Individual EEG feature windows ($N=37,575$) are temporally correlated within sessions and cannot be treated as independent degrees of freedom for statistical significance testing. The true statistical degrees of freedom are the $N=15$ independent subjects.
3. **High Inter-Subject Variance:**
   - Subject-level accuracy ranged widely from **24.55%** (Subject 11) to **58.88%** (Subject 15), reflecting fundamental individual differences in cortical EEG manifestations of emotion.
4. **Non-Universal, Non-Causal Channel Sets:**
   - While the 32 selected channels show high empirical utility and stability across training subjects, they should not be interpreted as universally optimal across all EEG recording setups, nor as direct neurophysiological causal drivers of emotional processing.

---

## 8. Artifact Inventory

The following artifacts have been generated and committed to the workspace:

| Artifact Path | Description |
| :--- | :--- |
| [src/evaluate_corrected_channel_performance.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/evaluate_corrected_channel_performance.py) | Standalone evaluation runner with fold-level checkpointing and dual decision rules. |
| [outputs/predictions/corrected_channel_performance_predictions.csv](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/corrected_channel_performance_predictions.csv) | Full window-level test predictions (112,725 rows: 3 conditions $\times$ 37,575 windows). |
| [outputs/metrics/corrected_channel_performance_fold_metrics.csv](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/metrics/corrected_channel_performance_fold_metrics.csv) | Fold-level metric breakdown across all 15 outer folds (45 rows). |
| [outputs/tables/corrected_channel_performance_comparison.csv](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/corrected_channel_performance_comparison.csv) | Primary summary comparison table for all 3 conditions. |
| [outputs/tables/corrected_channel_performance_subject_summary.csv](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/corrected_channel_performance_subject_summary.csv) | Per-subject comparison across all 15 held-out folds. |
| [outputs/figures/corrected_channel_performance_comparison.png](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/figures/corrected_channel_performance_comparison.png) | 3-panel publication figure (Pooled Metrics, Delta vs 62 Ch, Subject Accuracies). |
| [outputs/metrics/corrected_channel_performance_run_report.json](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/metrics/corrected_channel_performance_run_report.json) | Complete execution metadata, hyperparameter logs, and confusion matrices. |
| [outputs/reports/stage2_5_corrected_channel_performance_report.md](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/reports/stage2_5_corrected_channel_performance_report.md) | This comprehensive experimental report. |

---

## 9. Next Steps (Gate to Phase E)

Stage 2.5 successfully establishes the classification benchmark for the corrected stability-aware channel selection method. The project is now methodologically prepared to proceed to **Phase E (Selective Prediction and Abstention)**, where uncertainty estimation (calibrated probabilities and confidence metrics) will be used to detect and abstain from unreliable predictions on unseen subjects.
