# Discrepancy Analysis: Phase B vs. Phase D Baseline Accuracy

**Project:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 37,575 EEG windows, 62 channels, 310 features)  
**Evaluation Protocol:** 15-Fold Outer Leave-One-Subject-Out (LOSO) Cross-Validation  
**Auditor:** Senior ML Researcher & Code Auditor  
**Date:** 2026-10-21  
**Status:** **100% RESOLVED AND RECONCILED**  

---

## 1. Executive Summary

Phase B established the 62-channel all-data baseline with a reported pooled classification accuracy of **39.1883%** (14,725 correct out of 37,575 windows). In Phase D, the uncalibrated all-62 condition (`all_62` with `uncalibrated_softmax`) reported a pooled accuracy of **39.5263%** (14,852 correct out of 37,575 windows), representing a net gain of **127 correct predictions** (+0.3380%).

A rigorous, sample-by-sample forensic audit was conducted across all 37,575 predictions to determine whether this difference stemmed from data leakage, feature ordering, random seeds, hyperparameter shifts, or decision rules.

### Key Audit Finding
The models, features, train/test subject splits, scaling parameters, and SVM kernels used in Phase B and Phase D are **100% IDENTICAL**. The exact root cause is the decision rule applied during prediction:
1. **Phase B used LibSVM discrete one-vs-one majority voting** (`sklearn.svm.SVC.predict`).
2. **Phase D used continuous decision-margin argmax** ($\arg\max_c Z_c = \arg\max_c \text{decision\_function}(X)_c$), which is the mathematically required input representation for softmax normalization and temperature scaling.

On **95.16% of samples** (35,758 / 37,575), both decision rules produce **100% identical predictions**. The 1,817 mismatched samples (4.84%) represent low-confidence boundary instances where binary voting produced a tie or split vote, whereas the continuous decision margin broke the tie via distance to the separating hyperplanes.

---

## 2. Component-by-Component Equivalence Audit

To verify whether any unintended discrepancy existed between the Phase B baseline runner ([src/train_baseline.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/train_baseline.py)) and the Phase D calibration runner ([src/run_calibration.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py)), every component was cross-checked:

| Component | Phase B Baseline (`train_baseline.py`) | Phase D All-62 (`run_calibration.py`) | Audit Status |
| :--- | :--- | :--- | :--- |
| **Dataset Ingestion** | `load_seed_iv_dataset()` | `load_seed_iv_dataset()` | **IDENTICAL** |
| **Total Windows ($N$)** | 37,575 | 37,575 | **IDENTICAL** |
| **Features per Window** | 310 ($62 \text{ chs} \times 5 \text{ bands}$) | 310 ($62 \text{ chs} \times 5 \text{ bands}$) | **IDENTICAL** |
| **Feature Ordering** | `channel_idx * 5 + band_idx` | `channel_idx * 5 + band_idx` | **IDENTICAL** |
| **Class Label Encoding** | 0: Neutral, 1: Sad, 2: Fear, 3: Happy | 0: Neutral, 1: Sad, 2: Fear, 3: Happy | **IDENTICAL** (0 label mismatches) |
| **Outer LOSO Splits** | 15 folds, subject-disjoint | 15 folds, subject-disjoint | **IDENTICAL** (Same train/test indices) |
| **Preprocessing** | `StandardScaler` fitted on train only | `StandardScaler` fitted on train only | **IDENTICAL** |
| **Model Type** | RBF-SVM | RBF-SVM | **IDENTICAL** |
| **Hyperparameters** | $C=1.0$, $\gamma=\text{'scale'}$, kernel='rbf' | $C=1.0$, $\gamma=\text{'scale'}$, kernel='rbf' | **IDENTICAL** |
| **Decision Rule** | `pipeline.predict(X_test)` | $\arg\max_c Z_c$ on `pipeline.decision_function(X_test)` | **DIFFERENT (Root Cause)** |

---

## 3. Mathematical Forensic Analysis of the Decision Rules

In multiclass classification with $K=4$ classes, `scikit-learn`'s `SVC` constructs $\binom{K}{2} = \binom{4}{2} = 6$ binary one-vs-one (OvO) classifiers:
$$\{(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)\}$$

### Rule 1: Phase B Discrete One-vs-One Voting (`SVC.predict`)
For a sample $x$, each of the 6 binary classifiers casts a discrete vote:
$$V_c(x) = \sum_{j \neq c} \mathbf{1}(\text{sign}(f_{c, j}(x)) > 0)$$
The predicted class is:
$$\hat{y}_{\text{vote}} = \arg\max_{c \in \{0, 1, 2, 3\}} V_c(x)$$
If two or more classes tie with the same number of votes (e.g., two classes receive 2 votes each), LibSVM breaks the tie using its internal heuristic (often selecting the first tied class or comparing pairwise margin sums).

### Rule 2: Phase D Continuous Decision-Margin Argmax (`argmax(decision_function)`)
When preparing scores for probability calibration (Platt scaling, temperature scaling, or softmax), the model requires continuous real-valued confidence scores $Z \in \mathbb{R}^K$. `scikit-learn` computes this via an OvR transformation of the pairwise decision values:
$$z_c(x) = \sum_{j \neq c} f_{c, j}(x)$$
The predicted class under this regime is:
$$\hat{y}_{\text{margin}} = \arg\max_{c \in \{0, 1, 2, 3\}} z_c(x)$$

### Why the Rules Disagree on Boundary Samples
When a sample is well away from the decision boundary, the class that wins the majority of binary votes also has the highest aggregate margin:
$$V_{c^*}(x) = 3 \implies \hat{y}_{\text{vote}} = \hat{y}_{\text{margin}} = c^*$$
This occurs on **35,758 out of 37,575 samples (95.16%)**.

However, when a sample lies in an ambiguous region near the intersection of multiple hyperplanes, split voting occurs (e.g., $V_0=2, V_1=2, V_2=1, V_3=1$). Discrete voting treats every pairwise victory equally ($+1$), ignoring whether the sample cleared the margin by $+0.01$ or $+2.50$. In contrast, $z_c(x)$ integrates the continuous distances.

### Empirical Validation on the Mismatched Samples
Comparing the 1,817 mismatched samples:
- **Mean Confidence on Matched Samples (35,758 rows):** **0.6902**
- **Mean Confidence on Mismatched Samples (1,817 rows):** **0.3519** (near random guess $0.250$)
- **Phase B Accuracy on Mismatched Samples:** 22.56% (410 / 1,817)
- **Phase D Accuracy on Mismatched Samples:** 29.55% (537 / 1,817)
- **Net Difference:** $537 - 410 = \mathbf{+127}$ correct predictions ($14,852 - 14,725 = 127$).

Continuous margin argmax resolved boundary ties slightly more effectively than discrete vote-counting, producing a $+0.338\%$ increase in pooled accuracy.

---

## 4. Subject-by-Subject Breakdown

| Subject | Test Samples | Mismatched Samples | Mismatch % | Phase B Accuracy (`SVC.predict`) | Phase D Accuracy ($\arg\max Z$) | Difference |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2,505 | 37 | 1.48% | 42.79% | 42.79% | +0.00% |
| **2** | 2,505 | 42 | 1.68% | 40.72% | 40.96% | +0.24% |
| **3** | 2,505 | 822 | 32.81% | 43.47% | 40.88% | -2.59% |
| **4** | 2,505 | 153 | 6.11% | 45.59% | 51.70% | +6.11% |
| **5** | 2,505 | 203 | 8.10% | 34.13% | 33.93% | -0.20% |
| **6** | 2,505 | **0** | **0.00%** | **31.74%** | **31.74%** | **+0.00%** |
| **7** | 2,505 | 10 | 0.40% | 49.26% | 48.86% | -0.40% |
| **8** | 2,505 | 45 | 1.80% | 42.91% | 43.87% | +0.96% |
| **9** | 2,505 | 352 | 14.05% | 29.62% | 32.42% | +2.79% |
| **10** | 2,505 | 58 | 2.32% | 35.49% | 34.89% | -0.60% |
| **11** | 2,505 | **0** | **0.00%** | **24.55%** | **24.55%** | **+0.00%** |
| **12** | 2,505 | 48 | 1.92% | 42.95% | 42.08% | -0.88% |
| **13** | 2,505 | **0** | **0.00%** | **35.05%** | **35.05%** | **+0.00%** |
| **14** | 2,505 | **0** | **0.00%** | **27.27%** | **27.27%** | **+0.00%** |
| **15** | 2,505 | 47 | 1.88% | 62.28% | 61.92% | -0.36% |
| **Pooled**| **37,575** | **1,817** | **4.84%** | **39.1883%** | **39.5263%** | **+0.3380%** |

Notice that for 4 subjects (6, 11, 13, 14), there are **zero mismatches** (100.0% agreement between voting and continuous margin). Subjects with higher mismatch counts (Subjects 3, 4, 9) had higher proportions of ambiguous, low-margin feature windows.

---

## 5. Mismatch Behavior Across Reduced Channel Conditions

The exact same phenomenon was evaluated across the Phase C reduced-channel conditions:

| Condition | Selected Channels in Phase C vs Phase D | Phase C Accuracy (`SVC.predict`) | Phase D Accuracy ($\arg\max Z$) | Prediction Mismatches | Mismatch % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`all_62`** | 62 / 62 (100% match) | 39.19% | 39.53% | 1,817 | 4.84% |
| **`pred_32`** | 32 / 32 (100% match across all 15 folds) | 39.28% | 38.90% | 1,828 | 4.86% |
| **`stab_32`** | 32 / 32 (100% match across all 15 folds) | 38.58% | 38.56% | 1,707 | 4.54% |
| **`pred_12`** | 12 / 12 (100% match across all 15 folds) | 38.31% | 38.04% | 1,537 | 4.09% |

In all conditions, between **95.14% and 95.91% of predictions are identical**. The difference in pooled accuracy ranges from $-0.02\%$ (`stab_32`) to $+0.34\%$ (`all_62`), well within the subject standard deviation of $\pm 9.6\%$.

---

## 6. Methodological Decision for the Project & Paper

### Which Decision Rule is Correct?
Both rules are mathematically valid, but serve different evaluation purposes:
1. **For Uncalibrated Baseline SVMs (Phase B & C):** `SVC.predict` is the standard library default for discrete multiclass SVM predictions.
2. **For Calibrated Probabilities & Uncertainty Estimation (Phase D & E):** Continuous decision scores $Z \in \mathbb{R}^K$ are required to fit temperature scaling ($T$) and compute calibrated probabilities ($p_c = \text{softmax}(z_c / T)$). Because temperature scaling divides $Z$ by scalar $T > 0$, the argmax of the probabilities is algebraically forced to match $\arg\max_c Z_c$:
$$\arg\max_{c} \frac{\exp(z_c / T)}{\sum_j \exp(z_j / T)} \equiv \arg\max_c z_c$$
Using `SVC.predict` while reporting probabilities from $\text{softmax}(Z / T)$ would create an inconsistency where the predicted label would not match the maximum calibrated probability.

### Recommended Reporting in Research Manuscripts
In the research paper, report both figures transparently:
- *"Baseline RBF-SVM achieves 39.19% ± 9.63% under LibSVM majority voting, and 39.53% ± 9.77% under continuous margin argmax. When probabilities are calibrated via group-safe temperature scaling, the argmax prediction rule strictly preserves the 39.53% accuracy while reducing Expected Calibration Error from 27.85% down to 3.74%."*

---

## 7. Audit Conclusion

The discrepancy between Phase B (39.19%) and Phase D (39.53%) is **fully explained, mathematically justified, and verified at the sample level**. It does not reflect any defect, data leak, or code bug. Phase D baseline outputs are certified sound.
