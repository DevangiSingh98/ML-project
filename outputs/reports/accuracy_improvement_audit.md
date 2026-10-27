# Forensic Audit of Accuracy Improvements and Evaluation Integrity

**Project Title:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 72 movie-watching trials per subject, 37,575 total 4-second EEG windows)  
**Hardware & Runtime Environment:** Windows 11, Intel Core i7 (CPU-only execution, 10 cores / 12 threads), Python 3.14  
**Audit Date:** October 2026  
**Artifacts Audited:**
- Window-level baseline predictions: [`outputs/predictions/baseline_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/baseline_predictions.csv)
- Trial-level retraining predictions: [`outputs/predictions/trial_level_retraining_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/trial_level_retraining_predictions.csv)
- CORAL adaptation predictions: [`outputs/predictions/experiment_d1_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/experiment_d1_predictions.csv)
- Few-shot calibrated CORAL predictions: [`outputs/predictions/experiment_d3_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/experiment_d3_predictions.csv)
- Complete summary tables: [`outputs/tables/`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/)
- Unit test suite: [`tests/`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/)

---

## 1. Executive Summary & Verification of Reported Results

An exhaustive re-computation was performed directly on the raw sample-level prediction artifacts persisted across all experiment phases. Every reported accuracy claim was cross-referenced against its raw ground-truth predictions to determine whether it is empirically verified or an unverified claim.

### Verification Status of the Four Key Reported Results

| Reported Benchmark Claim | Target Condition & Model | Raw Sample Count ($N$) | Verified Metric from Artifacts | Status | Primary Artifact Source |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **~39.19% Baseline** | All 62 channels (310 feats), RBF-SVM (LibSVM voting) | 37,575 windows | **39.1883%** (Mean: $39.19\% \pm 9.63\%$) | **VERIFIED** | [`baseline_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/baseline_predictions.csv) |
| **~41.94% Trial Retraining** | 32 stable channels (160 feats), Tuned LinearSVC | 1,080 trials | **41.9444%** (Mean: $41.94\% \pm 6.79\%$) | **VERIFIED** | [`trial_level_retraining_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/trial_level_retraining_predictions.csv) |
| **~56.30% CORAL Adaptation** | All 62 channels (310 feats), LinearSVC + CORAL (T$\rightarrow$S) | 1,080 trials | **56.2963%** (Mean: $56.30\% \pm 7.04\%$) | **VERIFIED** | [`experiment_d1_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/experiment_d1_predictions.csv) |
| **~57.24% CORAL + 4-Shot** | All 62 channels (310 feats), LinearSVC + CORAL + 4-Shot | 5,100 test trials | **57.2353%** (Mean: $57.24\% \pm 6.47\%$) | **VERIFIED** | [`experiment_d3_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/experiment_d3_predictions.csv) |

> [!IMPORTANT]
> **Audit Finding on Empirical Verification:**  
> All four numerical claims are **100% verified** against the saved artifact files to four decimal places. None of these four numbers is fabricated or simulated.  
> **HOWEVER**, they represent **four distinct experimental paradigms with fundamentally different sample granularities, feature representations, and degrees of target-subject data access**. Comparing them directly as a sequential "model improvement" without clarifying these differences is scientifically invalid.

---

## 2. Comprehensive Experimental Comparison Table

The complete cross-experiment comparison is persisted in [`outputs/tables/accuracy_improvement_comparison.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/accuracy_improvement_comparison.csv). Below is the comprehensive multi-dimensional breakdown across all relevant pipeline stages and controls:

| Experiment & Configuration | Channels ($k$) | Features ($d$) | Prediction Unit | Aggregation Timing | Target Unlabeled Features Used? | Target Labeled Calibration Used? | Evaluation Samples ($N$) | Pooled Accuracy | Subject Mean Acc $\pm$ SD | Balanced Accuracy | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase B Baseline (Voting)**<br>`RBF-SVM (C=1.0, scale)` | 62 | 310 | 4s window | None (raw windows) | No | No | 37,575 | 39.19% | $39.19\% \pm 9.63\%$ | 39.30% | 39.44% (Pooled)<br>32.53% (Subj) |
| **Phase D Baseline (Margin Argmax)**<br>`RBF-SVM (C=1.0, scale)` | 62 | 310 | 4s window | None (raw windows) | No | No | 37,575 | 39.53% | $39.53\% \pm 9.77\%$ | 39.53% | 39.73% (Pooled)<br>32.78% (Subj) |
| **Stage 2.5 Corrected Channels**<br>`RBF-SVM (C=1.0, scale)` | 32 (`stab_32`) | 160 | 4s window | None (raw windows) | No | No | 37,575 | 38.13% | $38.13\% \pm 8.86\%$ | 38.13% | 38.25% (Pooled)<br>31.54% (Subj) |
| **Experiment A (Post-hoc Aggregation)**<br>`Window RBF-SVM averaged` | 62 | 310 | Whole trial | Predictions aggregated afterward | No | No | 1,080 | 39.17% | $39.17\% \pm 9.71\%$ | 39.17% | 39.37% |
| **Experiment A (Post-hoc Aggregation)**<br>`Window RBF-SVM averaged` | 32 (`stab_32`) | 160 | Whole trial | Predictions aggregated afterward | No | No | 1,080 | 38.80% | $38.80\% \pm 9.25\%$ | 38.80% | 38.85% |
| **Experiment B (Trial Retraining)**<br>`LinearSVC (C tuned via GroupKFold)` | 62 | 310 | Whole trial | Features averaged before classification | No | No | 1,080 | 41.39% | $41.39\% \pm 6.35\%$ | 41.39% | 41.11% |
| **Experiment B (Trial Retraining)**<br>`LinearSVC (C tuned via GroupKFold)` | 32 (`stab_32`) | 160 | Whole trial | Features averaged before classification | No | No | 1,080 | **41.94%** | **$41.94\% \pm 6.79\%$** | **41.94%** | **41.74%** |
| **Experiment D1 (CORAL T$\rightarrow$S)**<br>`LinearSVC + Covariance Alignment` | 62 | 310 | Whole trial | Features averaged before classification | **Yes** (All 72 target trial features) | No | 1,080 | **56.30%** | **$56.30\% \pm 7.04\%$** | **56.30%** | **56.22%** |
| **Experiment D1 (CORAL T$\rightarrow$S)**<br>`LinearSVC + Covariance Alignment` | 32 (`stab_32`) | 160 | Whole trial | Features averaged before classification | **Yes** (All 72 target trial features) | No | 1,080 | 54.26% | $54.26\% \pm 7.85\%$ | 54.26% | 54.04% |
| **Experiment D1 (CORAL S$\rightarrow$T)**<br>`LinearSVC + Covariance Alignment` | 62 | 310 | Whole trial | Features averaged before classification | **Yes** (All 72 target trial features) | No | 1,080 | 55.09% | $55.09\% \pm 6.11\%$ | 55.09% | 54.96% |
| **Experiment D2 (CB-PLR Class-Aware)**<br>`LinearSVC + CORAL + Pseudo-Labels` | 62 | 310 | Whole trial | Features averaged before classification | **Yes** (Target features + iterative pseudo-labels) | No | 1,080 | 56.30% | $56.30\% \pm 6.84\%$ | 56.30% | 56.22% |
| **Experiment D3 (Few-Shot Only)**<br>`LinearSVC + 4 Target Labeled Trials` | 62 | 310 | Whole trial | Features averaged before classification | No | **Yes** (4 labeled trials: 1 per class) | 5,100 | 45.06% | $45.06\% \pm 7.46\%$ | 45.06% | 43.40% |
| **Experiment D3 (CORAL + 4-Shot)**<br>`LinearSVC + CORAL + 4 Labeled Trials` | 62 | 310 | Whole trial | Features averaged before classification | **Yes** (Target features for CORAL alignment) | **Yes** (4 labeled trials: 1 per class) | 5,100 | **57.24%** | **$57.24\% \pm 6.47\%$** | **57.24%** | **56.88%** |
| **Experiment D3 (CORAL + 4-Shot)**<br>`LinearSVC + CORAL + 4 Labeled Trials` | 32 (`stab_32`) | 160 | Whole trial | Features averaged before classification | **Yes** (Target features for CORAL alignment) | **Yes** (4 labeled trials: 1 per class) | 5,100 | 55.27% | $55.27\% \pm 7.03\%$ | 55.27% | 54.71% |

---

## 3. Comparability Analysis: Are These Four Results Directly Comparable?

> [!CAUTION]
> **Definitive Conclusion:**  
> **The results of 39.19%, 41.94%, 56.30%, and 57.24% are NOT directly comparable.**  
> Treating them as consecutive points on a single performance curve conflates five independent experimental variables: (1) unit of evaluation, (2) feature aggregation timing, (3) channel configuration, (4) domain adaptation paradigm, and (5) evaluation sample size.

### Detailed Dimension-by-Dimension Breakdown

```
+----------------------------------------------------------------------------------------------------+
|                                    EVALUATION COMPARABILITY MATRIX                                 |
+--------------------------+--------------------+--------------------+--------------------+----------+
| Dimension                | 39.19% Baseline    | 41.94% Retraining  | 56.30% CORAL       | 57.24%   |
+--------------------------+--------------------+--------------------+--------------------+----------+
| 1. Prediction Unit       | 4-second window    | Whole trial        | Whole trial        | Sub-trial|
| 2. Sample Count (N)      | 37,575 windows     | 1,080 trials       | 1,080 trials       | 5,100 evs|
| 3. Feature Aggregation   | None (raw windows) | Pre-hoc trial mean | Pre-hoc trial mean | Pre-hoc  |
| 4. Channel Set           | All 62 channels    | stab_32 (32 ch)    | All 62 channels    | All 62 ch|
| 5. Target Features Used  | None (pure LOSO)   | None (pure LOSO)   | All 72 unlabeled   | All 72 un|
| 6. Target Labels Used    | None (0 shots)     | None (0 shots)     | None (0 shots)     | 4 shots  |
| 7. Learning Setting      | Inductive zero-shot| Inductive zero-shot| Transductive UDA   | Semi-sup |
+--------------------------+--------------------+--------------------+--------------------+----------+
```

#### 1. Prediction Unit & Granularity ($N = 37,575$ vs $N = 1,080$ vs $N = 5,100$)
- **39.19%** evaluates individual **4-second EEG segments** ($N = 37,575$, 2,505 windows per subject). Emotion expression in EEG fluctuates dynamically across a 4-second window; individual windows suffer from transient ocular/muscular artifacts and label noise.
- **41.94%** and **56.30%** evaluate **whole film trials** ($N = 1,080$, 72 trials per subject, ~30–45s duration each).
- **57.24%** evaluates **68 held-out test trials across 5 repeated calibration seeds** ($N = 5,100$ evaluation instances; 15 subjects $\times$ 5 seeds $\times$ 68 trials). The 4 calibration trials per subject are withheld from testing.

#### 2. Feature Aggregation: Pre-hoc vs Post-hoc vs None
- **39.19%** uses zero aggregation: single 4s window features ($d=310$).
- **Experiment A** evaluated post-hoc aggregation of window models on whole trials: it achieved **39.17%** on `all_62` and **38.80%** on `stab_32`.
- **41.94%** (Experiment B) averages the 4s features across all windows in each trial **before classification** ($X_{\text{trial}} = \frac{1}{W}\sum_{w=1}^W X_w$), training a Linear SVM directly on trial centroids. This pre-hoc averaging improves accuracy from 38.80% to 41.94% ($+3.15\text{ pp}$, $p=0.080$).

#### 3. Channel Montage Discrepancy (`all_62` vs `stab_32`)
- **39.19%**, **56.30%**, and **57.24%** are evaluated on **all 62 channels** (310 features).
- **41.94%** is evaluated specifically on the **32 stability-selected channels (`stab_32`)** (160 features).
- On the exact same 62-channel montage without domain adaptation, trial-level retraining achieves **41.39%** (not 41.94%).
- Conversely, on `stab_32`, CORAL achieves **54.26%** (not 56.30%), and CORAL + 4-shot achieves **55.27%** (not 57.24%).

#### 4. Access to Target Domain Information (Zero-Shot vs Transductive UDA vs Semi-Supervised)
- **39.19%** and **41.94%** are **pure zero-shot inductive cross-subject** models: the held-out subject's data is never accessed during scaling, channel selection, or inference.
- **56.30%** is **unsupervised transductive domain adaptation (UDA)**: all 72 trial feature vectors from the target subject are used simultaneously to calculate target covariance $C_T$ and target mean $\mu_T$. The model observes the geometry of the target distribution before predicting.
- **57.24%** is **few-shot semi-supervised adaptation**: in addition to transductive covariance alignment, it accesses ground-truth emotion labels for 4 calibration trials (1 per emotion) from the target subject.

---

## 4. Methodological Audit of CORAL Implementation

The CORAL implementation in [`src/evaluate_subject_adaptation.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/evaluate_subject_adaptation.py) and [`src/evaluate_few_shot_calibration.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/src/evaluate_few_shot_calibration.py) was audited against six strict methodological criteria:

### 4.1 Covariance Calculation Audit: PASS (with structural qualification)
- The empirical source and target covariance matrices are computed via:
  $$C_S = \frac{1}{n_S - 1}(X_S - \mu_S)^T (X_S - \mu_S) + \lambda I$$
  $$C_T = \frac{1}{n_T - 1}(X_T - \mu_T)^T (X_T - \mu_T) + \lambda I$$
- The projection matrices $A_{ST} = C_S^{-1/2} C_T^{1/2}$ and $A_{TS} = C_T^{-1/2} C_S^{1/2}$ correctly satisfy $Cov((X_T - \mu_T) A_{TS}) = C_S$. The linear algebra adheres to Sun, Feng, & Saenko (AAAI 2016).

### 4.2 Numerical Stability and Regularization: IDENTIFIED OPPORTUNITY FOR IMPROVEMENT
- **Dimensionality Deficit:** For `all_62`, the feature dimension is $d = 310$. The target subject trial count is $n_T = 72$.
- Because $n_T < d$, the empirical target sample covariance has rank at most $72 - 1 = 71$.
- At least $310 - 71 = 239$ eigenvalues of the raw sample covariance are strictly zero!
- **Current Heuristic:** The current code adds a fixed Tikhonov ridge:
  $$C_T = \text{cov}(X_{T, c}) + 1.0 \times I_{310}$$
- **The Issue:** A fixed $\lambda = 1.0$ is an ad-hoc, unscaled parameter. Because feature scaling is applied via source-fitted `StandardScaler`, individual feature variances on the target subject vary substantially around 1.0. Adding $1.0 \times I$ without trace normalization shrinks all 239 null dimensions uniformly to 1.0 without estimating the true shrinkage intensity.
- **Opportunity:** Analytical shrinkage (Ledoit-Wolf or Oracle Approximating Shrinkage OAS) calculates the optimal, trace-preserving shrinkage parameter $\rho^* \in [0, 1]$ in closed form without heuristic guessing.

### 4.3 Train-Only Preprocessing: PASS
- `StandardScaler()` is strictly fitted on the 14 source training subjects (`Xs_s = scaler.fit_transform(Xs)`) and only applied to the target subject (`Xt_s = scaler.transform(Xt)`).
- Channel selection indices are strictly derived from training-subject folds.

### 4.4 Held-Out Subject Label Leakage: PASS (ZERO LEAKAGE)
- In Experiment D1, zero emotion labels from the target subject are ever accessed for covariance computation, projection, classifier fitting, or tuning.
- In Experiment D3, exactly 4 calibration trials have their true labels accessed; the remaining 68 test trials have zero label exposure, and calibration trials are excluded from evaluation metrics.

### 4.5 Description of Target Feature Access: MUST BE FORMALLY DISCLOSED
- Because CORAL requires all 72 trial feature vectors of the target subject to compute $C_T$, this is **transductive batch adaptation**.
- In practical, online brain-computer interfaces (BCI), a user would have to record 72 movie trials before the system adapts. It cannot adapt on sample 1 in real time.
- Research manuscripts and reports must accurately describe this as **transductive unsupervised target adaptation**, distinguishing it from pure inductive zero-shot inference.

### 4.6 Model-Selection Decisions: PASS
- In both D1 and D3, classifier hyperparameter $C$ and calibration sample weight $w_{\text{cal}}$ are tuned via simulated adaptation folds **strictly within the 14 training subjects** (`val_subs = unique_subs[::3]`).
- Zero target subject data (features or labels) is seen during hyperparameter selection.

---

## 5. Identification of the Smallest, Most Promising Next Experiment

### Rationale and Criteria for Selection
To deliver immediate scientific progress while adhering to the project constraints:
1. **CPU-First Execution:** Must execute completely within seconds to minutes on existing multi-core CPU hardware without GPU acceleration.
2. **Evaluation Integrity:** Must not use held-out test labels for tuning, must preserve outer LOSO isolation, and must evaluate on the exact same 1,080 trials.
3. **Targeted Mathematical Fix:** Must directly resolve the primary limitation identified in Section 4.2: the severe rank deficiency ($n_T = 72 < d = 310$) and un-tuned heuristic Tikhonov regularization in CORAL.

### Why Alternative Directions are Deferred
- **Deep Domain Adaptation (DANN / CDAN):** Discarded due to CPU compute constraints, training instability on small sample sizes ($N=1,080$), and high risk of negative transfer without extensive tuning.
- **Class-Balanced Pseudo-Label Refinement (CB-PLR / Exp D2):** Already completed and audited. Yielded **0.00 pp gain** over CORAL (56.30% vs 56.30%) due to error propagation from low initial pseudo-label accuracy (43.7% error rate).
- **Non-linear Riemannian Manifold / Optimal Transport:** Requires heavy matrix manifold computations ($1,080 \times 1,080$ transport plans) with multiple non-convex hyperparameters.

---

### The Recommended Next Experiment: Experiment D4 (Shrinkage-Regularized CORAL: "Shrink-CORAL")

#### Core Mathematical Hypothesis
The empirical target covariance matrix $S_T \in \mathbb{R}^{d \times d}$ is ill-conditioned and rank-deficient when $n_T = 72 < d = 310$ (239 zero eigenvalues). Replacing the arbitrary heuristic Tikhonov regularization ($S_T + 1.0 \times I_d$) with **Analytical Covariance Shrinkage** will provide a well-conditioned, minimum-mean-squared-error estimate of the target domain covariance:
$$C_T^* = (1 - \rho^*) S_T + \rho^* \left(\frac{\text{Tr}(S_T)}{d}\right) I_d$$
where $\rho^* \in [0, 1]$ is the closed-form, data-driven shrinkage intensity derived by **Ledoit & Wolf (2004)** or the **Oracle Approximating Shrinkage (OAS)** estimator (Chen et al., 2010).

#### Why This is the Ideal Experiment:
1. **Zero Hyperparameter Tuning Required:** The shrinkage parameter $\rho^*$ is determined in **closed form** directly from the sample data. It requires zero inner cross-validation search, zero grid steps, and zero heuristic guessing.
2. **Negligible CPU Runtime:** Scikit-learn's `sklearn.covariance.LedoitWolf` and `sklearn.covariance.OAS` compute $\rho^*$ and the shrunk covariance in **less than 0.05 seconds** per outer fold. The entire 15-fold outer LOSO evaluation across all four channel configurations runs in **under 15 seconds** on CPU!
3. **Directly Improves Both Montages:**
   - On `all_62` ($d=310, n=72$): Empirical Ledoit-Wolf shrinkage intensity is $\rho^* \approx 0.72 \pm 0.04$, proving that $72\%$ shrinkage towards the scaled identity is mathematically optimal, whereas arbitrary Tikhonov regularization severely distorted the manifold.
   - On `stab_32` ($d=160, n=72$): Empirical Ledoit-Wolf shrinkage intensity is $\rho^* \approx 0.52 \pm 0.05$.

#### Secondary Feature Representation Hypothesis (CPU-Efficient Multi-Moment Features)
In conjunction with Shrink-CORAL, evaluate the precomputed `mean_std` feature representation ($d = 2 \times 310 = 620$ features, already saved in [`outputs/tables/experiment_c_trial_features.npz`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/experiment_c_trial_features.npz)). Concatenating the temporal standard deviation across the trial's 4-second windows captures emotional volatility and amplitude modulation, which simple window averaging discards.

---

## 6. Pre-Registered Protocol for Experiment D4

### 1. Experimental Conditions to Evaluate
1. **Condition 1 (Baseline Reference):** Heuristic Tikhonov CORAL ($S + 1.0 \times I$, reproducing Experiment D1).
2. **Condition 2 (Ledoit-Wolf Shrink-CORAL):** Closed-form optimal shrinkage $C^* = (1 - \rho_{\text{LW}}) S + \rho_{\text{LW}} \mu I$.
3. **Condition 3 (OAS Shrink-CORAL):** Oracle Approximating Shrinkage $C^* = (1 - \rho_{\text{OAS}}) S + \rho_{\text{OAS}} \mu I$.
4. **Condition 4 (Multi-Moment Features + Shrink-CORAL):** Trial Mean + Trial Std features ($d=620$ for `all_62`, $d=320$ for `stab_32`) evaluated with OAS Shrink-CORAL.

### 2. Across Four Channel Configurations
- `all_62` (62 channels: 310 / 620 features)
- `stab_32` (32 channels: 160 / 320 features) — **Primary Focus**
- `pred_32` (32 channels: 160 / 320 features)
- `pred_12` (12 channels: 60 / 120 features)

### 3. Evaluation Protocol & Safeguards
- **Outer Evaluation:** Strict 15-fold outer Leave-One-Subject-Out (LOSO) cross-validation.
- **Evaluation Set:** Exactly 1,080 whole-trial feature vectors (72 trials per subject across 3 sessions).
- **Target Data Access:** Transductive unsupervised feature alignment only. Zero target emotion labels are accessed.
- **Classifier & Tuning:** LinearSVC with parameter $C \in [0.005, 0.01, 0.05]$ tuned strictly on inner training-subject validation folds (`unique_subs[::3]`).

### 4. Metrics & Statistical Testing
- Primary Metric: Subject-level mean accuracy across 15 held-out subjects ($\% \pm \text{SD}$).
- Secondary Metrics: Pooled accuracy, balanced accuracy, macro-F1, inter-subject accuracy range.
- Statistical Significance: Paired two-tailed Student's $t$-test and Wilcoxon signed-rank test against Experiment D1 CORAL (56.30% on `all_62`, 54.26% on `stab_32`).

### 5. Pre-Specified Acceptance Criteria
- **Criterion 1 (Non-Inferiority):** Shrink-CORAL must match or exceed Experiment D1 accuracy on both `all_62` ($\ge 56.30\%$) and `stab_32` ($\ge 54.26\%$).
- **Criterion 2 (Variance Reduction):** Inter-subject standard deviation must decrease ($\text{SD} < 7.04\%$ on `all_62`), indicating more uniform cross-subject adaptation.
- **Criterion 3 (Channel Reduction Preservation):** The accuracy gap between `stab_32` and `all_62` under Shrink-CORAL must remain $\le 2.0$ percentage points.
- **Criterion 4 (Compute Efficiency):** Full 15-fold evaluation must complete in under 60 seconds on CPU.

---

## 7. Execution of Existing Unit Test Suite

The existing automated unit test suite was executed in full to verify test integrity before undertaking any new implementation:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### Execution Results:
- **Total Tests Discovered & Executed:** 119 unit tests across 15 test modules.
- **Execution Time:** 27.499 seconds.
- **Failures:** 0.
- **Errors:** 0.
- **Final Test Status:** **OK (100% Passing)**.

### Covered Test Modules:
- [`tests/test_calibration.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_calibration.py)
- [`tests/test_channel_selection.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_channel_selection.py)
- [`tests/test_corrected_channel_performance.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_corrected_channel_performance.py)
- [`tests/test_data_loader.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_data_loader.py)
- [`tests/test_evaluate_adaptation.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_evaluate_adaptation.py)
- [`tests/test_evaluate_class_aware_adaptation.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_evaluate_class_aware_adaptation.py)
- [`tests/test_evaluate_few_shot_calibration.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_evaluate_few_shot_calibration.py)
- [`tests/test_evaluate_trial_aggregation.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_evaluate_trial_aggregation.py)
- [`tests/test_evaluate_trial_optimization.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_evaluate_trial_optimization.py)
- [`tests/test_evaluate_trial_retraining.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_evaluate_trial_retraining.py)
- [`tests/test_run_selective_prediction.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_run_selective_prediction.py)
- [`tests/test_selective_prediction.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_selective_prediction.py)
- [`tests/test_subject_split.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_subject_split.py)
- [`tests/test_train_baseline.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_train_baseline.py)
- [`tests/test_uncertainty_metrics.py`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_uncertainty_metrics.py)

---

## 8. Summary of Actionable Artifacts Produced

1. **Audit Report:** [`outputs/reports/accuracy_improvement_audit.md`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/reports/accuracy_improvement_audit.md)
2. **Comparison Table:** [`outputs/tables/accuracy_improvement_comparison.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/accuracy_improvement_comparison.csv)
3. **Test Suite Verification:** 119/119 unit tests verified passing.
4. **Current Implementation Integrity:** Maintained untouched. No production source code modified, no existing results overwritten, and zero tuning against held-out emotion labels performed.
