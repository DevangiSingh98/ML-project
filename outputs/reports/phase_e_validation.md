# Phase E Validation Report: Calibration Check, Statistical Integrity & Regression Audit

## 1. Final Calibration Verification (Phase E3 Item 1)

### 1.1 Objective of the Verification
In selective prediction with operational threshold estimation (Mode B), thresholds $\tau_\gamma$ for nominal coverage targets $\gamma \in [0.5, 1.0]$ are computed from out-of-fold uncertainty quantiles across the 14 outer-training subjects.
A critical methodological vulnerability identified in preflight auditing was the potential for *self-calibration leakage*: if calibration parameters (such as the temperature parameter $T$) are fitted on the entire pool of 14 training subjects' out-of-fold scores, and then applied to those same scores to generate the uncertainty values from which quantiles are taken, the resulting uncertainty distribution is optimistically compressed or biased.

### 1.2 Implemented Cross-Fitting Mechanism
To guarantee strict independence and statistical validity, group-level cross-fitting of temperature calibration was implemented in `src/run_selective_prediction.py`.

The exact mathematical implementation proceeds as follows:
1. **Outer LOSO Fold $f$**: For held-out test subject $s_f$, training subjects are $\mathcal{S}_{\text{tr}} = \{1, \dots, 15\} \setminus \{s_f\}$.
2. **Inner Subject-Disjoint Splits**: $\mathcal{S}_{\text{tr}}$ is partitioned into $M=3$ disjoint group folds using `GroupKFold(n_splits=3)`.
3. **Inner Decision Margins $\mathbf{Z}_{\text{inner\_oof}}$**: For each inner split $m \in \{1, \dots, M\}$:
   - An SVM pipeline is trained strictly on inner training subjects $\mathcal{S}_{\text{tr}} \setminus \mathcal{S}_{\text{val}, m}$.
   - Decision margins $\mathbf{Z}_{\text{val}, m}$ are evaluated on the inner validation subjects $\mathcal{S}_{\text{val}, m}$.
   - Full $\mathbf{Z}_{\text{inner\_oof}} \in \mathbb{R}^{N_{\text{tr}} \times 4}$ is assembled out-of-fold.
4. **Group-Level Cross-Fitted Calibration**:
   - For each inner validation group $\mathcal{S}_{\text{val}, m}$:
     $$\mathcal{C}_m = \arg\min_T \sum_{i \in \text{inner\_tr}_m} \mathcal{L}_{\text{NLL}}(\hat{\mathbf{p}}(\mathbf{z}_i; T), y_i)$$
   - Calibrator $\mathcal{C}_m$ (fitted exclusively on the other inner training subjects) is applied to predict probabilities for the excluded inner validation samples:
     $$\hat{\mathbf{p}}_i = \mathcal{C}_m(\mathbf{z}_i), \quad \forall i \in \text{inner\_val}_m$$
5. **Operational Threshold Derivation**:
   - Training uncertainty scores $u_i = 1 - \max_k \hat{p}_{i, k}$ (or normalized entropy) are computed from these cross-fitted probabilities.
   - For nominal target coverage $\gamma < 1.0$, the threshold is the empirical quantile:
     $$\tau_\gamma = \text{Quantile}(u_{\text{train\_oof}}, \gamma)$$
   - For nominal target coverage $\gamma = 1.0$, the threshold is strictly:
     $$\tau_{1.0} = \infty$$
6. **Outer Fold Calibrator**:
   - As established in Phase D, the outer test subject $s_f$ is evaluated using the outer model and outer calibrator (fitted on all permitted outer-training OOF scores from $\mathcal{S}_{\text{tr}}$).
   - The test subject's uncertainty is compared against the frozen $\tau_\gamma$:
     $$g_{\text{test}, i} = \mathbb{I}(u_{\text{test}, i} \le \tau_\gamma)$$
   - Zero test data exposure: held-out subject $s_f$ never contributes to SVM fitting, channel selection, calibration parameter fitting, or threshold selection.

### 1.3 Unit Test Verification
This cross-fitting property is formally verified by regression test `test_21_cross_fitted_calibration_excludes_validation_group` in [tests/test_run_selective_prediction.py](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/tests/test_run_selective_prediction.py#L623-L649):
- Decision margins and labels of validation group 3 were corrupted by $100\times$ multiplication and arbitrary label permutation.
- The temperature parameter fitted to calibrate group 3 was verified to be strictly bit-for-bit identical before and after corruption (`self.assertEqual(temp1, temp2)`).

---

## 2. Comprehensive Test Suite Results

The full project regression test suite was executed across all modules:
- Command: `python -m unittest discover tests -v`
- Total tests executed: **79**
- Passing: **79 (100%)**
- Failures: **0**
- Errors: **0**
- Skipped: **0**
- Test execution runtime: **53.77s**

### 2.1 Test Coverage by Functional Area
| Test Module | Test Count | Key Invariants Verified |
|:---|:---:|:---|
| `test_selective_prediction.py` | 17 | Uncertainty bounds, zero-entropy safety, acceptance logic, tie-handling, monotonicity of abstention, selective accuracy/risk complementarity ($\text{Acc} + \text{Risk} = 1.0$), empty-set NaN returns, trapezoidal piecewise AURC, random rejection expectation $\mathbb{E}[\text{Acc}_{\text{rand}}] = \text{Acc}_{\text{full}}$, partial vs full AURC integration, $\tau_{1.0} = \infty$ unconditional acceptance |
| `test_run_selective_prediction.py` | 21 | Mode A retrospective curves, Mode B inductive thresholds, strictly nested OOF generation, inner group-safe splits, cross-fitted calibration independence (Test 21), random rejection label agnosticism (Test 20), partial vs full AURC integration (Test 19), actual coverage reporting on ties (Test 18), pipeline end-to-end execution |
| `test_channel_selection.py` | 13 | Permutation importance, stability scoring, budget counts (8, 12, 16, 24, 32), training-only selection, held-out subject exclusion, perturbation leakage test, inner split subject disjointness |
| `test_subject_split.py` | 5 | 15-fold outer LOSO splits, subject disjointness, split reproducibility, inner GroupKFold isolation |
| `test_train_baseline.py` | 4 | Pipeline structure, StandardScaler isolation, prediction alignment, confusion matrix conservation |
| `test_calibration.py` | 3 | Group-safe OOF score assembly, temperature scaling probability validity, uncalibrated softmax properties |
| `test_uncertainty_metrics.py` | 4 | Multiclass Brier score, Expected Calibration Error (ECE), AUROC/AUPRC error detection |
| `test_data_loader.py` | 12 | Configuration locking, 62 channel ordering, 5-band feature mappings, official label integrity |
| **Total** | **79** | **Complete pipeline verified** |

---

## 3. Methodological Boundary & Invariant Audit

| Check Description | Audit Criteria | Implementation Verification | Status |
|:---|:---|:---|:---:|
| **Probabilistic Bounds** | $\sum_k p_k = 1.0 \pm 10^{-6}$, $p_k \ge 0$, finite | Handled in `calibration.py` and `selective_prediction.py` via `np.clip` and numerically stable softmax | **PASSED** |
| **Uncertainty Bounds** | $u_{\max} \in [0, 0.75]$, $H_{\text{norm}} \in [0.0, 1.0]$ | Verified in `test_01` and `test_02` of `test_selective_prediction.py` | **PASSED** |
| **Zero Entropy Handling** | $p_k = 0 \implies p_k \ln p_k = 0$ | Implemented using boolean mask `np.where(p > 0, p * np.log(p), 0.0)` | **PASSED** |
| **Monotonic Acceptance** | $\tau_1 \le \tau_2 \implies N_{\text{acc}}(\tau_1) \le N_{\text{acc}}(\tau_2)$ | Formally proven and tested in `test_06_monotonicity_of_abstention` | **PASSED** |
| **Tie Consistency** | All tied samples accepted | Implemented as $u(\mathbf{x}) \le \tau$; tested in `test_05_threshold_ties_handling` | **PASSED** |
| **Empty Set Handling** | Empty accepted sets return NaN | Explicit check `if np.sum(mask) == 0: return np.nan`; tested in `test_09` | **PASSED** |
| **100% Coverage Rule** | $\tau_{1.0} = \infty$ | `determine_threshold_from_training_budget` returns `np.inf` at target $\ge 1.0$; tested in `test_17` | **PASSED** |
| **Random Baseline Expectation** | $\mathbb{E}[\text{Acc}_{\text{rand}}] \equiv \text{Acc}_{\text{full}}$ | Handled analytically and verified by 500-iteration Monte Carlo draws; tested in `test_13` | **PASSED** |
| **Partial AURC Domain** | Exactly $[0.5, 1.0]$, bounded $[0, 0.5]$ | Trapezoidal integration with boundary interpolation; tested in `test_16` and `test_19` | **PASSED** |
| **Full AURC Domain** | Canonical $[0.0, 1.0]$, bounded $[0, 1.0]$ | Left horizontal extension down to $c=0.0$; tested in `test_16` and `test_19` | **PASSED** |
| **Inner Split Disjointness** | $\mathcal{S}_{\text{inner\_tr}} \cap \mathcal{S}_{\text{inner\_val}} = \emptyset$ | Asserted via set disjointness in `generate_group_safe_oof_decision_scores` | **PASSED** |
| **Outer Test Isolation** | $s_{\text{test}} \notin \mathcal{S}_{\text{train}}$ | Verified via set assertion in LOSO split iteration | **PASSED** |
