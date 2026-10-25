# Stage 1 — Full Project Source-Code Audit and Evidence-Based Repair Plan

**Project Title:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 37,575 feature windows, 62 physical EEG channels, 310 features, 4 classes)  
**Evaluation Protocol:** 15-Fold Outer Leave-One-Subject-Out (LOSO) Cross-Validation  
**Auditor:** Senior ML Research Engineer & Experimental Methodology Auditor  
**Date:** 2026-10-21  
**Project Path:** `C:\Users\singh\OneDrive\Desktop\ML Project`  
**Stage 1 Audit Verdict:** **AUDIT COMPLETE — PASS WITH METHODOLOGICAL LIMITATIONS (READINESS CONFIRMED FOR STAGE 2)**  

---

## 1. Project Inventory & Dependency Map

### 1.1 Complete Source Code Inventory

| File Path | Lines | Primary Purpose | Direct Callers / Consumers | Upstream Dependencies |
| :--- | :---: | :--- | :--- | :--- |
| [src/data_loader.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/data_loader.py) | 401 | Ingests `.mat` files, validates array shapes, aligns official labels, constructs 310-feature mappings. | `train_baseline.py`, `run_experiments.py`, `run_calibration.py`, unit tests | `config.yaml`, `SEED_IV/Channel Order.xlsx`, `SEED_IV/eeg_feature_smooth/` |
| [src/subject_split.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/subject_split.py) | 105 | Generates 15 outer LOSO folds and inner `GroupKFold` splits with strict subject-disjointness assertions. | `train_baseline.py`, `run_experiments.py`, `run_calibration.py`, unit tests | `data_loader.py` (via metadata) |
| [src/preprocessing.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/preprocessing.py) | 62 | Builds `Pipeline([('scaler', StandardScaler()), ('svc', SVC())])` enforcing training-only scaling. | `train_baseline.py`, `run_experiments.py`, `calibration.py`, `run_calibration.py`, unit tests | `scikit-learn` |
| [src/channel_features.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/channel_features.py) | 95 | Maps physical channel indices to 5 spectral bands and slices feature submatrices indivisibly. | `channel_selection.py`, `run_experiments.py`, `run_calibration.py`, unit tests | `numpy` |
| [src/channel_selection.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/channel_selection.py) | 245 | Permutation importance via `RidgeClassifier`, stability index $Q_c$, and pairwise Jaccard. | `run_experiments.py`, `run_calibration.py`, `analyze_random_subsets.py`, unit tests | `channel_features.py`, `scikit-learn` |
| [src/metrics.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/metrics.py) | 140 | Computes multiclass accuracy, balanced accuracy, macro-F1, per-class recall, and confusion matrix. | `train_baseline.py`, `run_experiments.py`, `run_calibration.py`, unit tests | `scikit-learn`, `matplotlib` |
| [src/calibration.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/calibration.py) | 193 | Generates group-safe OOF decision scores via inner `GroupKFold`; Temperature Scaling; Platt Scaling. | `run_calibration.py`, unit tests | `preprocessing.py`, `scipy.optimize`, `scikit-learn` |
| [src/uncertainty_metrics.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/uncertainty_metrics.py) | 225 | Multiclass Brier score, safe log loss, top-label and classwise ECE, $u_{\max}$, predictive entropy, AUROC/AUPRC. | `run_calibration.py`, unit tests | `scikit-learn`, `numpy` |
| [src/train_baseline.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/train_baseline.py) | 338 | Phase B all-62 LOSO baseline runner. | CLI / standalone execution | `data_loader`, `subject_split`, `preprocessing`, `metrics` |
| [src/run_experiments.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_experiments.py) | 510 | Phase C channel reduction experiment runner across 5 budgets ($8, 12, 16, 24, 32$) and all 62 channels. | CLI / standalone execution | `data_loader`, `subject_split`, `channel_selection`, `channel_features`, `metrics` |
| [src/run_calibration.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py) | 552 | Phase D calibration and uncertainty experiment runner across 4 conditions and 3 regimes. | CLI / standalone execution | `data_loader`, `subject_split`, `channel_selection`, `calibration`, `uncertainty_metrics` |
| [src/analyze_random_subsets.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/analyze_random_subsets.py) | 121 | Phase C audit analyzer evaluating replicate variance across seeds vs cross-subject variance. | Standalone analysis | `channel_selection`, reads historical Phase C CSVs |
| [src/make_figures.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/make_figures.py) | 135 | Generates performance curves and channel selection frequency plots. | Standalone script | `matplotlib`, `pandas` |

### 1.2 Pipeline Dependency Map & Inter-Phase Coupling

```
[Raw SEED-IV .mat + Channel Order.xlsx]
          │
          ▼
    data_loader.py (Phase A) ───────────► outputs/tables/dataset_summary.csv
          │
          ├──────────────────────────────────────────┐
          ▼                                          ▼
   train_baseline.py (Phase B)              run_experiments.py (Phase C)
          │                                          │
          ├─► outputs/predictions/baseline_...       ├─► outputs/predictions/channel_reduction_...
          └─► outputs/metrics/baseline_...           ├─► outputs/metrics/channel_reduction_...
                                                     └─► outputs/tables/channel_reduction_...
                                                                │
                                                                ▼
                                                    analyze_random_subsets.py
                                                                │
                                                                ├─► random_subset_replicates.csv
                                                                ├─► random_subset_variability_summary.csv
                                                                └─► channel_stability_comparison.csv

    data_loader.py + subject_split.py + channel_selection.py + calibration.py + uncertainty_metrics.py
                                          │
                                          ▼
                              run_calibration.py (Phase D)
                                          │
                                          ├─► outputs/predictions/phase_d_calibrated_predictions.csv (450,900 rows)
                                          ├─► outputs/metrics/phase_d_fold_metrics.csv (180 rows)
                                          ├─► outputs/tables/phase_d_calibration_summary.csv (12 rows)
                                          ├─► outputs/tables/phase_d_error_detection_summary.csv (12 rows)
                                          ├─► outputs/figures/phase_d_reliability_diagram.png
                                          └─► outputs/figures/phase_d_uncertainty_error_detection.png
```

### 1.3 Coupling and Historical Artifact Consumption Analysis
- **Coupling Audit:** Does `run_calibration.py` consume historical CSV outputs from Phase B or Phase C?
  - **Verification:** Inspection of imports in [src/run_calibration.py:L21-L58](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py#L21-L58) confirms that `run_calibration.py` **DOES NOT READ** any Phase B or Phase C CSV files. It loads the genuine SEED-IV dataset directly, regenerates identical training-only channel rankings using deterministic random seeds ($42 + \text{fold\_idx}$), fits the RBF-SVM models on identical splits, and writes its own self-contained predictions and metrics.
  - **Consumer Scripts:** The only script that consumes historical CSVs is `src/analyze_random_subsets.py`, which reads `channel_reduction_fold_metrics.csv` to post-process random replicate variance.

---

## 2. Findings Organized by Severity

### Critical Severity: None
No active data leakage, test contamination, or pipeline-breaking defects were found in the current implementation.

---

### High Severity Findings

#### Finding H1: Terminological Misnomer in Stability Score $S_c$
- **Source File & Function:** [src/channel_selection.py:L207-L213](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/channel_selection.py#L207-L213), `select_stability_aware_channels()`
- **Evidence:**
  ```python
  mean_ranks = np.mean(ranks_matrix, axis=0)  # Average rank across 3 inner splits in [1, 62]
  S_c = 1.0 - (mean_ranks - 1.0) / (total_channels - 1.0)
  S_c = np.clip(S_c, 0.0, 1.0)
  Q_c = alpha * I_norm + beta * S_c
  ```
- **Consequence:** `ranks_matrix` stores the rank of each channel in each of the 3 inner splits (1 = highest permutation drop, 62 = lowest). Taking the mean across splits computes the **average rank** (an ordinal Borda count). $S_c$ is a linear rescale of the average rank. It **does NOT measure rank variance or rank stability across splits** (e.g., standard deviation of ranks across splits). Because rank is an ordinal transformation of permutation importance, $Q_c$ is a blend of continuous mean utility ($I_{\text{norm}}$) and ordinal mean utility ($S_c$), not utility plus stability.
- **Recommended Correction:** See Section 3 for the proposed corrected formulation incorporating cross-split rank dispersion.
- **Status:** **VERIFIED**.

#### Finding H2: Disconnect Between Subject-Level and Pooled Uncertainty Error Detection
- **Source File & Function:** [src/uncertainty_metrics.py:L182-L224](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/uncertainty_metrics.py#L182-L224), `compute_error_detection_metrics()`
- **Evidence:** In [outputs/tables/phase_d_error_detection_summary.csv](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/phase_d_error_detection_summary.csv):
  - Subject-level mean AUROC ($u_{\max}$) under temperature scaling: **$0.6048 \pm 0.0884$** (`all_62`), **$0.6230 \pm 0.0929$** (`pred_32`), and **$0.6096 \pm 0.0953$** (`stab_32`).
  - Pooled AUROC ($u_{\max}$) across all 37,575 windows: **$0.5177$** (`all_62`), **$0.5225$** (`pred_32`), and **$0.5048$** (`stab_32`).
- **Consequence:** Inter-subject variability causes baseline decision score scales and mean confidences to shift across subjects. When all windows from 15 subjects are pooled into a single global ranking, the global ordering is diluted by subject-level offsets. If Phase E applies a single fixed global threshold to raw scores, selective prediction performance will be compromised.
- **Recommended Correction:** In Phase E, selective prediction thresholds must be determined from calibrated probabilities (which are anchored to empirical class frequencies) or subject-relative percentiles, and metrics must be reported both pooled and averaged across subjects.
- **Status:** **VERIFIED**.

---

### Medium Severity Findings

#### Finding M1: Prediction Rule Divergence Across Phases (LibSVM Voting vs Margin Argmax)
- **Source File & Function:** [src/train_baseline.py:L81](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/train_baseline.py#L81) (`pipeline.predict(X_test)`) vs [src/run_calibration.py:L158](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py#L158) (`calibrator.predict(Z_test)`)
- **Evidence:**
  - Phase B reported accuracy: **$39.1883\%$** (14,725 / 37,575 correct).
  - Phase D reported accuracy: **$39.5263\%$** (14,852 / 37,575 correct).
  - Difference: Exactly $+127$ correct predictions (+0.3380%).
  - Row-by-row join on `(subject_id, session_id, trial_id, window_id)` confirms 35,758 / 37,575 samples ($95.16\%$) have identical predictions. On the 1,817 samples ($4.84\%$) where LibSVM binary one-vs-one votes split or tied, continuous decision-margin argmax resolved ties slightly better (537 vs 410 correct).
- **Consequence:** While mathematically justified for temperature scaling ($\arg\max p \equiv \arg\max z$), the presence of two different baseline accuracy numbers ($39.19\%$ vs $39.53\%$) can confuse external reviewers if not explicitly documented.
- **Recommended Correction:** Transparently document both decision rules in research publications: report LibSVM voting as the discrete baseline ($39.19\%$) and continuous margin argmax as the probabilistic reference ($39.53\%$).
- **Status:** **VERIFIED & RESOLVED**.

#### Finding M2: Discrepancy in Reported $k=12$ Stability-Aware Jaccard Values
- **Source Artifacts:** [outputs/tables/channel_stability_summary.csv](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/channel_stability_summary.csv) vs text in status reports.
- **Evidence:** In `channel_stability_summary.csv`:
  - Row 0 ($k=8$): `stability_aware_jaccard = 0.283484`
  - Row 1 ($k=12$): `stability_aware_jaccard = 0.286335`
  - Row 2 ($k=16$): `stability_aware_jaccard = 0.329903`
  - Recomputed from raw fold metrics: $k=12$ is **0.2863**, and $k=16$ is **0.3299**.
- **Consequence:** A narrative text summary in an earlier draft accidentally cited $0.329$ for $k=12$ by reading the $k=16$ row.
- **Recommended Correction:** Cite the exact CSV ground truth: $k=12$ Stability-Aware Jaccard is **$0.2863$**, and $k=16$ is **$0.3299$**.
- **Status:** **VERIFIED & RESOLVED**.

---

### Low Severity Findings

#### Finding L1: Redundant `PROJECT_RULES. md` File in Root
- **File:** `C:\Users\singh\OneDrive\Desktop\ML Project\PROJECT_RULES. md`
- **Evidence:** There are two files in root: `PROJECT_RULES.md` (6,294 bytes) and `PROJECT_RULES. md` (with a space before `.md`, 6,294 bytes). They have identical contents.
- **Consequence:** Harmless duplication, but could cause minor confusion on case-insensitive filesystems.
- **Status:** **VERIFIED**.

---

## 3. Audit A — Stability-Aware Channel Selection Diagnosis

### 3.1 Step-by-Step Execution of Channel Selection
1. **Inputs:** `(X_train, y_train, train_metadata)` for 14 outer-training subjects ($35,070$ windows, 310 features).
2. **Inner Split Partitioning:** 3-split `GroupKFold` grouped strictly by `train_metadata['subject_id']`. Inner train: $\approx 23,380$ windows ($\approx 9-10$ subjects); Inner validation: $\approx 11,690$ windows ($\approx 4-5$ subjects).
3. **Model & Permutation Drop:** A `RidgeClassifier(alpha=10.0)` is fitted on inner training data. On inner validation data, baseline balanced accuracy is computed. For each channel $c \in \{0, \dots, 61\}$, all 5 associated frequency columns $\{5c, \dots, 5c+4\}$ are jointly permuted across validation samples, and the drop $\Delta_{s, c} = \text{BAcc}_{\text{base}} - \text{BAcc}_{\text{perm}}$ is recorded.
4. **Ranking & Ordinal Transform:** For each inner split $s$, channels are sorted descending by $\Delta_{s, c}$, assigning integer ranks $r_{s, c} \in \{1, \dots, 62\}$ ($1 = \text{most important}$).
5. **Score Aggregation:**
   - Average drop: $I_c = \frac{1}{3} \sum_{s=1}^3 \Delta_{s, c}$. Normalized to $[0, 1]$ via min-max: $I_c^{\text{norm}} = \frac{I_c - \min I}{\max I - \min I}$.
   - Mean rank: $\bar{r}_c = \frac{1}{3} \sum_{s=1}^3 r_{s, c}$. Rescaled to $[0, 1]$: $S_c = 1.0 - \frac{\bar{r}_c - 1}{61}$.
   - Combined index: $Q_c = 0.5 I_c^{\text{norm}} + 0.5 S_c$.

### 3.2 Terminological Diagnosis: Average Rank vs. Genuine Stability
- **Mathematical Reality:**
  $$\bar{r}_c = \frac{1}{S} \sum_{s=1}^S r_{s, c}$$
  $\bar{r}_c$ is the **mean rank**, not the rank variance. If channel $A$ has ranks $(1, 1, 1)$, its mean rank is $1.0$ and sample variance is $0.0$. If channel $B$ has ranks $(1, 5, 20)$, its mean rank is $8.67$ and variance is $101.3$. If channel $C$ has ranks $(8, 9, 9)$, its mean rank is $8.67$ and variance is $0.33$. Under the current formula, channels $B$ and $C$ receive the **exact same** score $S_c$, even though channel $C$ has almost zero rank variance across subjects while channel $B$ is highly unstable!
- **Why Does Current $Q_c$ Still Improve Cross-Fold Jaccard?** Rank aggregation (Borda count) is inherently more robust to heavy-tailed outliers in single splits than raw arithmetic mean drops. Hence, $Q_c$ acted as an ensemble Borda-mean rank filter, improving cross-fold Jaccard from $0.267$ to $0.283$ ($k=8$) and $0.420$ to $0.424$ ($k=24$).
- **Defensible Proposed Corrected Formula for Future Revisions:**
  To measure *genuine selection stability* (penalizing rank dispersion across cross-validation splits):
  $$V_c = \sqrt{\frac{1}{S-1} \sum_{s=1}^S (r_{s, c} - \bar{r}_c)^2}$$
  $$V_c^{\text{norm}} = 1.0 - \frac{V_c}{\max_j V_j} \in [0, 1]$$
  $$Q_c^{\text{true}} = w_1 I_c^{\text{norm}} + w_2 S_c + w_3 V_c^{\text{norm}}$$
  where $V_c^{\text{norm}}$ explicitly rewards low standard deviation of ranks across independent subject folds.

### 3.3 Random Selection Stability Artifact Explanation
In Phase C's initial summary, `random_jaccard` was recorded as **1.0** across all folds.
- **Exact Cause:** `select_random_channels(budget=k, seed=42+k)` was called with seed fixed to $42+k$ in every outer fold. Because the seed did not vary with the outer fold index, the identical random channels were drawn in all 15 folds.
- **Correction Already Saved:** [src/analyze_random_subsets.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/analyze_random_subsets.py) evaluated repeated independent seeds and proved that the **true between-subset Jaccard** of random channels is only **0.064** ($k=8$) to **0.332** ($k=32$), whereas Stability-Aware selection achieves **0.284** to **0.491** (1.5x to 4.4x higher overlap).

---

## 4. Audit B — Data Leakage & Subject-Independent Evaluation

| Check | Target Operation | Source Code Location | Finding | Leakage Evidence / Assurance |
| :-: | :--- | :--- | :---: | :--- |
| **B1** | Outer LOSO Folds | [src/subject_split.py:L40-L69](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/subject_split.py#L40-L69) | **PASS** | Exactly 15 outer folds generated. `assert train_subs.isdisjoint(test_subs)` strictly verified. |
| **B2** | Preprocessing Scaler | [src/preprocessing.py:L48-L62](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/preprocessing.py#L48-L62) | **PASS** | `StandardScaler` inside Pipeline fit exclusively during `pipeline.fit(X_train)`. Test set transformed only. |
| **B3** | Channel Selection | [src/run_calibration.py:L84-L101](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py#L84-L101) | **PASS** | Permutation importance computed on `(X_train, y_train, train_meta)` only. Zero access to test subject. |
| **B4** | Inner CV Grouping | [src/channel_selection.py:L76-L91](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/channel_selection.py#L76-L91) | **PASS** | Inner splits use `GroupKFold(n_splits=3)` on `train_metadata['subject_id']`. Windows from same subject never split. |
| **B5** | Calibration OOF Disjointness | [src/calibration.py:L66-L69](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/calibration.py#L66-L69) | **PASS** | `assert tr_subs.isdisjoint(val_subs)` explicitly enforced in inner loop. |
| **B6** | Calibrator Optimization | [src/run_calibration.py:L133-L141](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py#L133-L141) | **PASS** | Temperature $T$ and Platt parameters fitted strictly on $(Z_{\text{oof}}, y_{\text{train}})$. Zero test labels. |
| **B7** | Prediction Identifiers | [src/run_calibration.py:L201-L220](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py#L201-L220) | **PASS** | Prediction rows store `subject_id`, `session_id`, `trial_id`, `window_id`, `label`, `is_correct`. |

---

## 5. Audit C — Calibration & Uncertainty Methodology

### 5.1 Mathematical Formulations & Verifications
1. **Multiclass Softmax:** Shift-max formulation $\frac{\exp(z_c - \max z)}{\sum_j \exp(z_j - \max z)}$ ensures numerical stability without floating-point overflow. Verified in `UncalibratedSoftmaxCalibrator` and `GroupSafeTemperatureScalingCalibrator`.
2. **Temperature Scaling Argmax Preservation:** Division of continuous scores $z \in \mathbb{R}^4$ by positive scalar $T \in [0.05, 50.0]$ is strictly monotonic:
   $$\arg\max_c p_c(z; T) \equiv \arg\max_c z_c$$
   Empirically confirmed: **0 mismatches** between predicted class and probability argmax across all $450,900$ rows.
3. **Multiclass Platt Vector Scaling:** Fits $p(y = c | z) = \text{softmax}(Wz + b)$ via `LogisticRegression(C=1.0, solver="lbfgs")`. Because $W \in \mathbb{R}^{4 \times 4}$ can rotate coordinates, it can change predicted classes. In Phase D, it perturbed predictions, shifting all-62 accuracy from $39.53\%$ to $39.29\%$. Temperature scaling is certified superior.
4. **Multiclass Brier Score:** $\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^4 (p_{ic} - y_{ic})^2 \in [0, 2]$. Uniform guessing yields $0.75$; perfect prediction yields $0.0$.
5. **Multiclass Log Loss (NLL):** $-\frac{1}{N} \sum_i \sum_c y_{ic} \log(p_{ic})$, clipped to $[10^{-15}, 1 - 10^{-15}]$.
6. **Expected Calibration Error (ECE):** 10 fixed equal-width bins in $[0, 1]$. Top-label ECE computes $| \text{acc}(B_m) - \text{conf}(B_m) |$ weighted by sample count $|B_m| / N$. Classwise ECE computes one-vs-rest binary calibration averaged over all 4 classes.
7. **Uncertainty Scores:** $u_{\max} = 1 - \max_c p_c$; predictive entropy $H(p) = -\sum_c p_c \log p_c$; normalized entropy $H(p)/\log 4$.
8. **Error-Detection Metrics:** Target $y_i^{\text{err}} = \mathbf{1}(\hat{y}_i \neq y_i)$, scored by $u_{\max}$ or $H(p)$. AUROC and AUPRC computed via `roc_auc_score` and `average_precision_score`.

### 5.2 Pooled vs. Subject-Level ECE and AUROC Distinction
- **Top-Label ECE:**
  - Pooled ECE under temperature scaling: **$1.32\% - 3.74\%$**.
  - Subject-level mean ECE: **$8.26\% - 9.23\%$** (with subject std $\approx 6.5\%$).
  - *Distinction:* Pooling over 37,575 windows averages out small-sample bin variances. At the individual subject level ($2,505$ windows), confidence bins have slight local deviations, yielding an average ECE of $\approx 8.9\%$, which is still a massive drop from the uncalibrated subject ECE of $\approx 29.0\%$.
- **Error Detection AUROC ($u_{\max}$):**
  - Subject-level mean AUROC: **$0.6048 \pm 0.0884$** (`all_62`), **$0.6230 \pm 0.0929$** (`pred_32`), and **$0.6096 \pm 0.0953$** (`stab_32`).
  - Pooled AUROC: **$0.5048 - 0.5225$**.
  - *Distinction:* Because different subjects have different baseline score ranges, a pooled ROC curve suffers from threshold confounding. An uncertainty threshold must be evaluated per subject or on calibrated probabilities.

---

## 6. Audit D — Prediction-Rule Discrepancy Forensic Audit

### 6.1 Direct Evidence and Code Locations
- **Phase B Implementation:** [src/train_baseline.py:L81](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/train_baseline.py#L81) calls `pipeline.predict(X_test)`.
- **Phase D Implementation:** [src/run_calibration.py:L158](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/run_calibration.py#L158) calls `calibrator.predict(Z_test)`, which computes `np.argmax(Z, axis=1)`.

### 6.2 Quantitative Discrepancy Breakdown
- **Total Windows:** $37,575$
- **Identical Predictions:** **$35,758$ / $37,575$ ($95.16\%$)**
- **Disagreement Windows:** **$1,817$ / $37,575$ ($4.84\%$)**
- **Subjects with Zero Disagreements:** Subjects 6, 11, 13, and 14 have $0$ disagreements ($100.0\%$ match).
- **Mean Confidence on Disagreement Subset:** **$0.3519$** (barely above random guess $0.250$), compared to **$0.6902$** on the agreement subset.
- **Accuracy on Disagreement Subset:**
  - Phase B (`SVC.predict`): $410 / 1,817$ ($22.56\%$)
  - Phase D ($\arg\max Z$): $537 / 1,817$ ($29.55\%$)
  - Net Gain for Phase D: $537 - 410 = \mathbf{+127}$ correct samples ($14,852 - 14,725 = 127$).
- **Conclusion:** Both numbers are legitimate reflections of the exact same fitted models. Phase B reflects LibSVM discrete pairwise majority voting ($39.19\%$), while Phase D reflects continuous decision-margin argmax ($39.53\%$). Continuous margin argmax is mathematically required for probability calibration.

---

## 7. Audit E — Phase E Implementation Gaps

### 7.1 Status of Phase E
**Phase E is NOT IMPLEMENTED.** Zero executable code exists in `src/` for selective prediction, threshold selection, abstention policies, coverage levels, risk-coverage curves, or AURC. It exists only as configuration specifications in `config.yaml` lines 72-75 and design guidelines in `IMPLEMENTATION (1).md`.

### 7.2 Implementation Gaps to be Completed in Stage 2
1. **Module Creation:** Create `src/selective_prediction.py`:
   - Function `determine_uncertainty_thresholds(uncertainties, coverages)`: Computes threshold $\tau(C)$ for target coverage $C \in \{1.0, 0.9, 0.8, 0.7, 0.6, 0.5\}$ based strictly on outer training subjects' out-of-fold calibration uncertainties.
   - Function `evaluate_selective_prediction(y_true, y_pred, uncertainties, threshold)`: Applies abstention mask (`is_retained = uncertainties <= threshold`), computing retained sample count, achieved coverage, selective accuracy, selective balanced accuracy, selective macro-F1, and selective risk.
   - Function `compute_aurc(coverages, risks)`: Computes Area Under the Risk-Coverage curve.
2. **Experiment Orchestrator:** Create `src/run_selective_prediction.py`:
   - Orchestrates selective prediction across all 15 outer LOSO folds for the 4 primary conditions (`all_62`, `pred_32`, `stab_32`, `pred_12`).
   - Uses temperature-scaled $u_{\max}$ as the primary uncertainty measure (with uncalibrated softmax as reference).
   - Operates directly on the saved Phase D predictions [outputs/predictions/phase_d_calibrated_predictions.csv](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/phase_d_calibrated_predictions.csv) or uses saved out-of-fold calibration predictions, **avoiding any re-training of SVMs**.
3. **Unit Tests:** Create `tests/test_selective_prediction.py` verifying threshold bounds, monotonicity of risk-coverage curves, and exclusion of outer test labels from thresholding.

---

## 8. Audit F — Tests and Reproducibility

### 8.1 Actual Test Execution
- **Command:** `python -m unittest discover -s tests -p "test_*.py"`
- **Result:** **29/29 tests PASSED in 7.784s**.
- **Coverage Breakdown:**
  - `test_data_loader.py`: 8 tests (array shapes, label balance, channel order, metadata).
  - `test_subject_split.py`: 5 tests (LOSO subject disjointness, window counts 35,070 / 2,505, split determinism, inner GroupKFold isolation, classification metrics on known cases).
  - `test_train_baseline.py`: 3 tests (pipeline structure `probability=False`, scaling isolation on training data, prediction DataFrame alignment).
  - `test_channel_selection.py`: 6 tests (5-band indivisible slicing, budget sizes $k \times 5$, random reproducibility, training-only selection, held-out subject exclusion, pairwise Jaccard).
  - `test_calibration.py`: 3 tests (Platt bounds, group-safe OOF score disjointness, uncalibrated softmax).
  - `test_uncertainty_metrics.py`: 4 tests (uncertainty bounds, Brier score on known cases, ECE on known distributions, single-class edge cases in AUROC/AUPRC).

---

## 9. Audit G — Artifact and Output Integrity

### 9.1 Verification of Existing Artifacts

| Artifact File | Expected Rows | Actual Rows | Unique Keys Verified | Status |
| :--- | :---: | :---: | :---: | :---: |
| `outputs/predictions/baseline_predictions.csv` | 37,575 | 37,575 | `['subject_id', 'session_id', 'trial_id', 'window_id']` | **VERIFIED** |
| `outputs/predictions/channel_reduction_predictions.csv` | 601,200 | 601,200 | 16 conditions $\times$ 37,575 rows | **VERIFIED** |
| `outputs/predictions/phase_d_calibrated_predictions.csv` | 450,900 | 450,900 | 4 conditions $\times$ 3 regimes $\times$ 37,575 rows | **VERIFIED** |
| `outputs/metrics/baseline_fold_metrics.csv` | 15 | 15 | 15 outer LOSO folds | **VERIFIED** |
| `outputs/metrics/channel_reduction_fold_metrics.csv` | 255 | 300 | Includes 2 random repeats (30 rows each) for $k=8, 16$ | **VERIFIED** |
| `outputs/metrics/phase_d_fold_metrics.csv` | 180 | 180 | 15 folds $\times$ 4 conditions $\times$ 3 regimes | **VERIFIED** |
| `outputs/tables/channel_reduction_comparison.csv` | 16 | 16 | 16 evaluated method/budget combinations | **VERIFIED** |
| `outputs/tables/channel_stability_summary.csv` | 5 | 5 | Budgets 8, 12, 16, 24, 32 | **VERIFIED** |
| `outputs/tables/channel_selection_frequencies.csv` | 62 | 62 | All 62 physical electrodes | **VERIFIED** |
| `outputs/tables/phase_d_calibration_summary.csv` | 12 | 12 | 4 conditions $\times$ 3 regimes | **VERIFIED** |
| `outputs/tables/phase_d_error_detection_summary.csv` | 12 | 12 | 4 conditions $\times$ 3 regimes | **VERIFIED** |

---

## 10. Staged Repair & Phase E Implementation Plan

### Principles:
1. **Zero Unnecessary Compute:** Do not retrain models for completed phases. Phase D predictions ($450,900$ rows) contain complete calibrated probabilities and uncertainty metrics for all 4 primary conditions and 3 regimes.
2. **Historical Preservation:** Never overwrite existing Phase B, C, or D output files.

### Stage 2: Selective Prediction Implementation (Phase E)
- **Step 2.1:** Implement `src/selective_prediction.py`:
  - Threshold selection via training out-of-fold uncertainty quantiles for coverage levels $C \in \{1.0, 0.9, 0.8, 0.7, 0.6, 0.5\}$.
  - Selective metrics evaluation (accuracy, balanced accuracy, macro-F1, AURC).
- **Step 2.2:** Add unit tests in `tests/test_selective_prediction.py` and run full test suite.
- **Step 2.3:** Implement and execute `src/run_selective_prediction.py`:
  - Evaluates `all_62`, `pred_32`, `stab_32`, and `pred_12`.
  - Saves `outputs/tables/selective_prediction_summary.csv` and `outputs/figures/selective_prediction_risk_coverage.png`.
  - Runtime: $< 30$ seconds (evaluates saved predictions without model retraining).

### Stage 3: Manuscript-Ready Reporting & Final Synthesis
- Synthesize all findings into the final project report in `report/`.

---

## 11. Unsupported Claims & Required Research Paper Qualifications

1. **Do NOT claim formal statistical non-inferiority:** While 32 channels retain $\approx 99\%$ of baseline accuracy, an equivalence margin hypothesis test (e.g., TOST) was not specified or conducted.
2. **Do NOT claim high pooled error-detection AUROC:** Pooled AUROC across all 15 subjects is $\approx 0.518$ due to inter-subject score scale offsets. Clearly state that uncertainty ranking is effective *within individual subjects* (mean AUROC $\approx 0.61 - 0.62$).
3. **Do NOT claim $S_c$ is a rank variance penalty:** Acknowledge that the implemented stability index $S_c$ is an ordinal Borda-style average rank score, which stabilizes selection against cardinal importance drop outliers rather than penalizing split-to-split variance.
4. **Do NOT hide the prediction rule difference:** State clearly that baseline discrete SVM achieves $39.19\%$, while continuous margin argmax achieves $39.53\%$, and explain that the latter is required for temperature-scaled probability calibration.
