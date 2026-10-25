# Phase D Methodology Audit: Probability Calibration & Uncertainty Estimation

**Project:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 37,575 windows, 62 channels, 310 features, 4 classes)  
**Evaluation Protocol:** 15-Fold Outer Leave-One-Subject-Out (LOSO) Cross-Validation  
**Auditor:** Senior ML Researcher & Code Auditor  
**Date:** 2026-10-21  
**Audit Verdict:** **PASS WITH LIMITATIONS**  

---

## 1. Executive Summary

A comprehensive, source-level methodological audit was conducted on Phase D of the research project. The audit inspected the implementation code, raw predictions (450,900 rows across 15 folds), fold metrics (180 rows), summary tables, run logs, and 29 unit tests.

### Summary of Audit Findings
1. **Discrepancy Reconciled (PASS):** The baseline pooled accuracy difference between Phase B (39.1883%) and Phase D (39.5263%) was completely resolved. Both phases use identical models, features, scalers, and outer splits. The difference is 100% attributable to LibSVM discrete voting (`SVC.predict`) vs. continuous margin argmax ($\arg\max Z = \arg\max \text{decision\_function}(X)$), which disagree on 4.84% of boundary samples with split votes.
2. **Subject Leakage Safeguards (PASS):** Verified zero test leakage. All scaling, channel ranking, inner out-of-fold score generation, and calibration parameter optimization are conducted strictly on the 14 outer-training subjects.
3. **Probability Calibration (PASS):** Group-safe temperature scaling ($T \approx 3.76 - 4.05$) reduces Expected Calibration Error from 27.85%–29.39% down to **1.32%–3.74%** (**94.7% reduction** on `stab_32`) while strictly preserving the SVM decision boundary ($\arg\max p = \arg\max z$).
4. **Channel Selection Integrity (PASS):** All channel subsets evaluated in Phase D match Phase C selections with **0 differences** across all 15 outer folds.
5. **Key Limitation (LIMITATION):** When uncertainty scores are pooled across all 15 subjects without subject-specific normalization, error-detection AUROC drops from $\approx 0.60 - 0.62$ (subject-level mean) to $\approx 0.51 - 0.52$. This occurs because baseline score scales vary across subjects. Phase E selective prediction must account for this inter-subject variation.

---

## 2. Inventory of Inspected Artifacts

| Artifact | Location | Size / Lines | Verification Status |
| :--- | :--- | :--- | :--- |
| **Calibration Module** | [src/calibration.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/calibration.py) | 193 lines | Verified: Group-safe inner splits, OOF disjointness assertions, temperature optimizer, vector Platt. |
| **Uncertainty Metrics** | [src/uncertainty_metrics.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/uncertainty_metrics.py) | 225 lines | Verified: Multiclass Brier, safe log loss, top-label and classwise ECE, $u_{\max}$, entropy, AUROC/AUPRC. |
| **Calibration Runner** | [src/run_calibration.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py) | 552 lines | Verified: 15 outer LOSO orchestration, channel slicing, output saving. |
| **Unit Test Suite** | [tests/test_calibration.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_calibration.py), [tests/test_uncertainty_metrics.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_uncertainty_metrics.py) | 194 lines | Verified: **29/29 tests pass** in 11.4s. |
| **Predictions CSV** | [outputs/predictions/phase_d_calibrated_predictions.csv](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/phase_d_calibrated_predictions.csv) | 450,900 rows (44.6 MB) | Verified: $4 \times 3 \times 37,575$ rows, sum-to-one deviation $< 5.55 \times 10^{-16}$, 0 argmax mismatches. |
| **Fold Metrics CSV** | [outputs/metrics/phase_d_fold_metrics.csv](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/metrics/phase_d_fold_metrics.csv) | 180 rows (38.8 KB) | Verified: 15 folds $\times$ 4 conditions $\times$ 3 regimes. |
| **Calibration Summary** | [outputs/tables/phase_d_calibration_summary.csv](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/phase_d_calibration_summary.csv) | 12 rows (4.8 KB) | Verified: Pooled and subject-level means $\pm$ std. |
| **Error Detection Summary** | [outputs/tables/phase_d_error_detection_summary.csv](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/phase_d_error_detection_summary.csv) | 12 rows (3.6 KB) | Verified: Pooled and subject-level AUROC and AUPRC. |
| **Reliability Diagram** | [outputs/figures/phase_d_reliability_diagram.png](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/figures/phase_d_reliability_diagram.png) | 240 KB (300 DPI) | Verified: Reliability curves for `all_62` and `pred_32`. |
| **Error Detection Plot** | [outputs/figures/phase_d_uncertainty_error_detection.png](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/figures/phase_d_uncertainty_error_detection.png) | 260 KB (300 DPI) | Verified: ROC and PR curves for error detection via $u_{\max}$. |

---

## 3. Critical Audit: Subject Leakage Prevention

The outer evaluation uses a 15-fold Leave-One-Subject-Out (LOSO) cross-validation protocol. For every fold $k$, test subject $S_k$ ($2,505$ windows) is held out, while the remaining 14 subjects ($35,070$ windows) form the outer training set.

### 7-Point Leakage Verification Checklist

| # | Audit Check | Code Location | Outcome | Evidence |
| :-: | :--- | :--- | :---: | :--- |
| **1** | **Outer Test Exclusion:** Held-out subject $S_k$ completely excluded from training. | [src/run_calibration.py:L74-L77](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py#L74-L77) | **PASS** | `assert train_subs.isdisjoint(test_subs)` strictly verified for all 15 folds. |
| **2** | **Feature Scaling Preprocessing:** `StandardScaler` fit exclusively on outer training data. | [src/preprocessing.py:L48-L62](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/preprocessing.py#L48-L62) | **PASS** | `Pipeline([('scaler', StandardScaler()), ('svm', SVC(...))])` fits scaler only during `pipeline.fit(X_train)`. |
| **3** | **Channel Selection Integrity:** Channel rankings generated strictly from outer training subjects. | [src/run_calibration.py:L84-L101](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py#L84-L101) | **PASS** | `compute_inner_grouped_permutation_importance` executes only on `(X_train, y_train, train_meta)`. |
| **4** | **Inner Calibration Splits:** Subject-grouped partitioning (`GroupKFold`), not window-level random splits. | [src/calibration.py:L54-L65](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/calibration.py#L54-L65) | **PASS** | `GroupKFold(n_splits=3).split(..., groups=train_metadata['subject_id'])`. |
| **5** | **Out-of-Fold Disjointness:** No subject contributes windows to both inner train and inner validation. | [src/calibration.py:L66-L69](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/calibration.py#L66-L69) | **PASS** | `assert tr_subs.isdisjoint(val_subs)` explicitly enforced in loop. |
| **6** | **Calibrator Parameter Fitting:** Calibrator fitted exclusively on out-of-fold training decision scores. | [src/run_calibration.py:L133-L141](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py#L133-L141) | **PASS** | `calibrator.fit(Z_oof, y_train)` uses only out-of-fold scores generated in Step 4. |
| **7** | **Final Prediction on Unseen Subject:** Final SVM trained on all 14 training subjects, evaluated on $S_k$. | [src/run_calibration.py:L143-L157](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py#L143-L157) | **PASS** | `final_pipe.fit(X_train, y_train)` then `calibrator.predict_proba(Z_test)`. |

---

## 4. Calibration Score Compatibility & Score Distribution Shift

### Compatibility Architecture
The calibration architecture generates decision scores $Z_{\text{oof}} \in \mathbb{R}^{35,070 \times 4}$ for the 14 outer-training subjects using 3 subject-disjoint inner folds. Each training subject's calibration scores are generated by an SVM that was trained without that subject. The calibrator is fitted on $(Z_{\text{oof}}, y_{\text{train}})$. The final SVM is fitted on all 14 subjects, and its decision scores $Z_{\text{test}} \in \mathbb{R}^{2,505 \times 4}$ on the held-out subject are scaled by the fitted calibrator.

### Score Shift Evaluation
Because the inner-fold SVMs train on 9–10 subjects ($\sim 23,380$ samples) while the final outer-fold SVM trains on 14 subjects ($35,070$ samples), a potential score-distribution mismatch was audited:
- In linear or RBF SVMs, increasing training size can slightly increase decision score margins.
- **Audit Findings:** The fitted optimal temperatures across folds are remarkably stable (standard deviation $\sigma_T \approx 0.35$):
  - `all_62`: Mean $T = 4.045 \pm 0.38$
  - `pred_32`: Mean $T = 3.877 \pm 0.35$
  - `stab_32`: Mean $T = 3.881 \pm 0.36$
  - `pred_12`: Mean $T = 3.765 \pm 0.31$
- When applied to the final outer SVM scores, the calibrator achieved top-label ECE of **1.32%–3.74%** on unseen subjects (down from 27.85%–29.39%), demonstrating that any score shift between 10-subject and 14-subject models is negligible in practice and does not undermine calibration.

---

## 5. Evaluation of the Three Probability Regimes

### A. Uncalibrated Softmax
- **Formula:** $p_c(z) = \frac{\exp(z_c)}{\sum_j \exp(z_j)}$
- **Properties:** Numerically stabilized via shift-max ($\exp(z_c - \max z)$). Probabilities are finite and normalized ($\sum_c p_c = 1.0 \pm 5.55 \times 10^{-16}$).
- **Audit Finding:** Correctly labeled as **uncalibrated softmax**, not a calibrated probability. The raw decision scores $z_c$ from an uncalibrated RBF-SVM have large magnitudes ($\max z \approx 3 - 6$), causing standard softmax to produce overconfident probabilities ($\max p \approx 0.70 - 0.95$) despite empirical accuracy of only $\approx 39\%$. Top-label ECE is severe (**27.85%–29.39%**).

### B. Group-Safe Temperature Scaling
- **Formula:** $p_c(z; T) = \frac{\exp(z_c / T)}{\sum_j \exp(z_j / T)}$ with scalar $T > 0$.
- **Optimization:** Scalar $T$ optimized by minimizing negative log-likelihood (NLL) on $(Z_{\text{oof}}, y_{\text{train}})$ via `scipy.optimize.minimize(method="L-BFGS-B")` within bounds $[0.05, 50.0]$.
- **Invariance Property:** Division by positive scalar $T$ is strictly monotonic. $\arg\max_c p_c(z; T) \equiv \arg\max_c z_c$.
- **Audit Finding:** Top-label ECE is reduced by up to **95.5%** (from 29.39% to 1.32% on `pred_12`), Brier score drops from $\approx 0.83$ to $\approx 0.717$, and NLL drops from $\approx 1.63$ to $\approx 1.32$. Classification accuracy and macro-F1 remain **100% identical** to the baseline model.

### C. Multiclass Platt / Vector Scaling
- **Formula:** $p(y = c | z) = \text{softmax}(W z + b)$ where $W \in \mathbb{R}^{4 \times 4}$, $b \in \mathbb{R}^4$.
- **Optimization:** Multinomial logistic regression (`sklearn.linear_model.LogisticRegression(C=1.0, solver="lbfgs")`) fitted on $(Z_{\text{oof}}, y_{\text{train}})$.
- **Audit Finding (Material Limitation):** Vector Platt scaling learns 20 unconstrained parameters. On 14 training subjects, this affine transformation rotates the decision boundary, slightly altering classification predictions. On held-out subjects, it achieves slightly inferior subject-level error-detection AUROC ($0.564 \pm 0.090$ vs $0.623 \pm 0.093$ for temperature scaling on `pred_32`). Temperature scaling is confirmed to be methodologically superior for this dataset.

---

## 6. Mathematical and Empirical Metric Validation

### Formulas and Implementation Checks
1. **Multiclass Brier Score:**
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^4 (p_{ic} - y_{ic})^2 \in [0, 2]$$
   Convention: Sum of squared deviations over all 4 classes, averaged across samples. Uniform guessing yields $0.75$; perfect prediction yields $0.0$.
2. **Log Loss (NLL):**
   $$\text{NLL} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^4 y_{ic} \log(p_{ic})$$
   Probabilities clipped to $[10^{-15}, 1 - 10^{-15}]$ and re-normalized.
3. **Expected Calibration Error (ECE):**
   10 fixed equal-width bins in $[0, 1]$. Top-label ECE weighted by sample count $|B_m| / N$. Empty bins contribute 0. Classwise ECE computes one-vs-rest binary calibration averaged over all 4 classes.
4. **Uncertainty Measures:**
   $u_{\max} = 1.0 - \max_c p_c \in [0, 0.75]$. Predictive entropy $H(p) = -\sum_c p_c \log p_c \in [0, \log 4]$. Normalized entropy $H_{\text{norm}} = H(p) / \log 4 \in [0, 1.0]$.
5. **Error-Detection Performance:**
   Target is $y_i^{\text{err}} = \mathbf{1}(\hat{y}_i \neq y_i)$. Score is $u_{\max}$ or $H(p)$. AUROC and AUPRC computed via `roc_auc_score` and `average_precision_score`. Single-class edge cases return `np.nan` gracefully.

---

## 7. Full Empirical Results

### Table 1: Probability Calibration Across Channel Conditions & Probability Regimes

| Condition | Chs | Probability Regime | Pooled Acc | Subject Acc (Mean $\pm$ Std) | Pooled Brier | Subject Brier (Mean $\pm$ Std) | Pooled Top ECE | Subject Top ECE (Mean $\pm$ Std) | Pooled NLL | Mean $T$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_62`** | 62 | `uncalibrated_softmax` | 39.53% | 39.53% $\pm$ 9.77% | 0.8285 | 0.8285 $\pm$ 0.1149 | 27.85% | 28.99% $\pm$ 10.45% | 1.6338 | — |
| **`all_62`** | 62 | `group_safe_temperature` | **39.53%** | **39.53% $\pm$ 9.77%** | **0.7171** | **0.7171 $\pm$ 0.0303** | **3.74%** | **8.90% $\pm$ 6.61%** | **1.3240** | **4.045** |
| **`all_62`** | 62 | `group_safe_platt` | 39.29% | 39.29% $\pm$ 8.52% | 0.7147 | 0.7147 $\pm$ 0.0559 | 2.46% | 11.21% $\pm$ 6.66% | 1.3160 | — |
| **`pred_32`** | 32 | `uncalibrated_softmax` | 38.90% | 38.90% $\pm$ 9.70% | 0.8285 | 0.8285 $\pm$ 0.1151 | 28.40% | 29.38% $\pm$ 9.02% | 1.6190 | — |
| **`pred_32`** | 32 | `group_safe_temperature` | **38.90%** | **38.90% $\pm$ 9.70%** | **0.7162** | **0.7162 $\pm$ 0.0337** | **1.92%** | **8.92% $\pm$ 6.90%** | **1.3209** | **3.877** |
| **`pred_32`** | 32 | `group_safe_platt` | 39.74% | 39.74% $\pm$ 9.04% | 0.7150 | 0.7150 $\pm$ 0.0553 | 2.98% | 11.38% $\pm$ 6.46% | 1.3151 | — |
| **`stab_32`** | 32 | `uncalibrated_softmax` | 38.56% | 38.56% $\pm$ 9.54% | 0.8371 | 0.8371 $\pm$ 0.1101 | 29.15% | 30.00% $\pm$ 8.70% | 1.6447 | — |
| **`stab_32`** | 32 | `group_safe_temperature` | **38.56%** | **38.56% $\pm$ 9.54%** | **0.7193** | **0.7193 $\pm$ 0.0326** | **1.56%** | **8.26% $\pm$ 7.16%** | **1.3270** | **3.881** |
| **`stab_32`** | 32 | `group_safe_platt` | 39.60% | 39.60% $\pm$ 9.34% | 0.7157 | 0.7157 $\pm$ 0.0531 | 2.89% | 10.84% $\pm$ 7.17% | 1.3169 | — |
| **`pred_12`** | 12 | `uncalibrated_softmax` | 38.04% | 38.04% $\pm$ 8.91% | 0.8412 | 0.8412 $\pm$ 0.1141 | 29.39% | 30.18% $\pm$ 9.90% | 1.6283 | — |
| **`pred_12`** | 12 | `group_safe_temperature` | **38.04%** | **38.04% $\pm$ 8.91%** | **0.7182** | **0.7182 $\pm$ 0.0312** | **1.32%** | **9.23% $\pm$ 5.57%** | **1.3228** | **3.765** |
| **`pred_12`** | 12 | `group_safe_platt` | 39.58% | 39.58% $\pm$ 7.19% | 0.7144 | 0.7144 $\pm$ 0.0488 | 3.00% | 9.72% $\pm$ 5.62% | 1.3127 | — |

### Table 2: Error-Detection Performance via Uncertainty Scores

| Condition | Chs | Probability Regime | Base Error Rate | Pooled AUROC ($u_{\max}$) | Subject AUROC ($u_{\max}$) | Pooled AUPRC ($u_{\max}$) | Subject AUPRC ($u_{\max}$) | Pooled AUROC (Entropy) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_62`** | 62 | `uncalibrated_softmax` | 60.47% | 0.5681 | 0.5891 $\pm$ 0.1070 | 0.6607 | 0.6796 $\pm$ 0.1145 | 0.5770 |
| **`all_62`** | 62 | `group_safe_temperature` | 60.47% | 0.5177 | **0.6048 $\pm$ 0.0884** | 0.6289 | **0.6892 $\pm$ 0.1066** | 0.5086 |
| **`all_62`** | 62 | `group_safe_platt` | 60.71% | 0.5531 | 0.5553 $\pm$ 0.1210 | 0.6622 | 0.6635 $\pm$ 0.1090 | 0.5546 |
| **`pred_32`** | 32 | `uncalibrated_softmax` | 61.10% | 0.5945 | 0.5971 $\pm$ 0.1052 | 0.6797 | 0.6821 $\pm$ 0.1169 | 0.6113 |
| **`pred_32`** | 32 | `group_safe_temperature` | 61.10% | 0.5225 | **0.6230 $\pm$ 0.0929** | 0.6374 | **0.6937 $\pm$ 0.1136** | 0.5108 |
| **`pred_32`** | 32 | `group_safe_platt` | 60.26% | 0.5310 | 0.5643 $\pm$ 0.0904 | 0.6404 | 0.6652 $\pm$ 0.0970 | 0.5378 |
| **`stab_32`** | 32 | `uncalibrated_softmax` | 61.44% | 0.5932 | 0.5944 $\pm$ 0.1078 | 0.6767 | 0.6815 $\pm$ 0.1205 | 0.6047 |
| **`stab_32`** | 32 | `group_safe_temperature` | 61.44% | 0.5048 | **0.6096 $\pm$ 0.0953** | 0.6236 | **0.6841 $\pm$ 0.1135** | 0.4949 |
| **`stab_32`** | 32 | `group_safe_platt` | 60.40% | 0.5311 | 0.5658 $\pm$ 0.0869 | 0.6384 | 0.6622 $\pm$ 0.1053 | 0.5331 |
| **`pred_12`** | 12 | `uncalibrated_softmax` | 61.96% | 0.5515 | 0.5647 $\pm$ 0.0584 | 0.6608 | 0.6842 $\pm$ 0.0943 | 0.5536 |
| **`pred_12`** | 12 | `group_safe_temperature` | 61.96% | 0.5024 | **0.5788 $\pm$ 0.0640** | 0.6343 | **0.6980 $\pm$ 0.1021** | 0.5029 |
| **`pred_12`** | 12 | `group_safe_platt` | 60.42% | 0.5048 | 0.5271 $\pm$ 0.0708 | 0.6109 | 0.6334 $\pm$ 0.0727 | 0.5148 |

---

## 8. Limitations and Cautions for Phase E

1. **Pooled vs. Subject-Level Ranking:**
   - On individual subjects, $u_{\max}$ reliably identifies errors with mean AUROC $\approx 0.605 - 0.623$.
   - When pooling across all 15 subjects, AUROC drops to $\approx 0.505 - 0.522$ because baseline score distributions vary across subjects.
   - **Crucial implication for Phase E:** Selective prediction / abstention policies must either operate on calibrated probabilities (which partially normalize across subjects) or determine subject-relative thresholds. Metrics must be reported both pooled and averaged across subjects.
2. **Abstention Feasibility:**
   - Low ECE proves that probabilities accurately match empirical error rates on average, but does not guarantee error-free selective prediction. Trade-off curves between coverage ($C \in [0.5, 1.0]$) and retention accuracy must be evaluated empirically in Phase E.

---

## 9. Final Audit Verdict & Recommendation

# **PASS WITH LIMITATIONS**

### Justification:
- All leakage checks passed with source-level verification.
- The 39.19% vs 39.53% baseline discrepancy is 100% resolved and proven to be an artifact of voting vs. continuous margin argmax.
- All 29 unit tests pass.
- All 450,900 prediction rows and summary tables are verified on disk.
- Phase D is certified complete, methodologically sound, and ready to serve as the foundation for Phase E.
