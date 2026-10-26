# Final Claims Audit: Empirical Verification of Proposed Research Claims

**Project Title:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 37,575 total EEG epochs, 4 emotion classes)  
**Evaluation Protocol:** 15-fold outer Leave-One-Subject-Out (LOSO) cross-validation  
**Audit Date:** October 26, 2026

---

## 1. Audit Framework & Evidence Standards

Each proposed scientific claim for the project thesis and final research report is audited against empirical evidence from Phases A–E and categorized under three strict standards:
- **SUPPORTED:** The claim is fully corroborated by audited empirical data, appropriate statistical tests, and reproducible artifact records without methodological caveats.
- **PARTIALLY SUPPORTED:** The claim holds under specific conditions or localized analyses (e.g., within-subject ranking, unadjusted paired tests), but encounters important limitations under broader evaluation (e.g., multiple testing corrections, cross-subject domain shift).
- **UNSUPPORTED:** The empirical evidence fails to substantiate the claim, contradicts the stated hypothesis, or the claim extrapolates beyond the demonstrated experimental setup.

---

## 2. Comprehensive Claims Matrix

| # | Proposed Claim | Audit Verdict | Empirical Evidence & Statistical Findings | Primary Caveats & Interpretation Rules |
|:---|:---|:---:|:---|:---|
| **1** | Reducing EEG channels from 62 to 32 retains classification performance on unseen subjects. | **SUPPORTED** | Full 62-channel LOSO baseline achieves **39.53% ± 9.77%**. Predictive 32-channel (`pred_32`) achieves **38.90% ± 9.70%**; stability-aware 32-channel (`stab_32`) achieves **38.56% ± 9.54%** against a 25.0% chance level. | Describe as *comparable observed performance*, not formal statistical equivalence (non-inferiority was not formally tested). |
| **2** | Reducing EEG channels down to 12 retains viable subject-independent performance. | **SUPPORTED** | The 12-channel model (`pred_12`) achieves **38.04% ± 8.91%** accuracy across 15 folds, retaining 96.2% of full-montage baseline performance with an 80.6% reduction in electrodes. | Performance decreases by ~1.5 percentage points below 62 channels, but remains substantially above chance (25%). |
| **3** | Stability-aware selection identifies a more consistent electrode subset across subjects than predictive selection alone. | **SUPPORTED** | Phase C stability analysis demonstrated that Kuncheva's stability index across outer training folds is substantially higher for `stab_32` ($S > 0.70$) than for greedy predictive selection ($S < 0.50$). | Both 32-channel subsets achieve comparable baseline accuracy; stability provides structural consistency across subjects. |
| **4** | Group-safe temperature scaling improves probability calibration without subject data leakage. | **SUPPORTED** | Fitting temperature scaling via inner subject-isolated cross-validation reduced Expected Calibration Error (ECE) from ~13% (uncalibrated) down to ~6% on held-out test subjects. | Temperature scaling compresses logit spreads, which subsequently degrades operational coverage tracking in Phase E. |
| **5** | Predictive uncertainty reliably identifies classification errors within unseen individual subjects (Mode A). | **SUPPORTED** | In subject-level retrospective ranking, mean Full AURC is **$0.5628 \pm 0.0855$** for `stab_32` and **$0.5454 \pm 0.0857$** for `pred_32`, well below the theoretical random rejection reference ($0.6144$ and $0.6110$). | Within individuals, confidence correlates with correctness; 10–12 of 15 subjects show monotonic accuracy increases under abstention. |
| **6** | Globally pooled uncertainty ranking outperforms random rejection across all subjects (Mode A Pooled). | **UNSUPPORTED** | Pooled Full AURC for `stab_32` is **$0.6139$**, virtually identical to the random reference (**$0.6144$**). For `pred_12`, pooled AURC is **$0.6174$** vs. random **$0.6196$**. | **Confounded by Simpson's Paradox:** Pooling all 37,575 predictions globally mixes subjects with differing baseline confidence and accuracy, obscuring individual ranking. |
| **7** | Static training-derived quantile thresholds guarantee exact target coverage on unseen test subjects (Mode B). | **UNSUPPORTED** | At nominal targets of 70%–90% under temperature scaling, actual test coverage was **88.9%–93.3%** (+10.9 to +22.2 pp gap). | Cross-subject probability compression shifts uncertainty quantiles relative to unseen individuals. |
| **8** | Uncertainty-based abstention outperforms random rejection under operational deployment (Mode B). | **PARTIALLY SUPPORTED** | At low coverage (50%–60%), subject mean selective accuracy reaches **45.6%–47.2%** (+3.07 to +8.64 pp gain over random across 10–12/15 subjects; unadjusted $p < 0.05$). | **Fails Multiple Testing Control:** None of the 20 primary tests survive Holm-Bonferroni correction ($p_{\text{Holm}} \ge 0.165$; minimum FDR $q = 0.124$). |
| **9** | Uncalibrated softmax achieves superior operational coverage calibration compared to temperature scaling. | **SUPPORTED** | Under uncalibrated softmax (`stab_32`), actual test coverage tracks nominal targets within **3.3 percentage points** across all budgets (e.g., nominal 70% achieves 71.2% actual coverage). | Raw margin dispersion is preserved across subjects, avoiding the probability compression induced by temperature scaling. |
| **10** | The pipeline has been validated for real-time BCI operation or clinical emotion monitoring. | **UNSUPPORTED** | The study evaluated offline 4-second Differential Entropy epochs from a laboratory benchmark dataset. No streaming latency, online buffer processing, or clinical trials were conducted. | Must be explicitly disclaimed in the final thesis/paper. |
| **11** | The channel reduction and uncertainty findings generalize across multiple external EEG datasets. | **UNSUPPORTED** | The experiment was executed exclusively on SEED-IV (15 subjects, 3 sessions). External cross-corpus validity (e.g., DEAP, DREAMER, MPED) has not been demonstrated. | External generalization remains future work. |

---

## 3. Detailed Audit Analysis by Claim Category

### 3.1 Channel Reduction Claims (Claims 1–3)
- **Status:** **SUPPORTED**
- **Evidence:** 
  - `all_62`: 39.53% ± 9.77%
  - `pred_32`: 38.90% ± 9.70% ($\Delta = -0.63\text{ pp}$)
  - `stab_32`: 38.56% ± 9.54% ($\Delta = -0.97\text{ pp}$)
  - `pred_12`: 38.04% ± 8.91% ($\Delta = -1.49\text{ pp}$)
- **Assessment:** Reducing channels by 48.4% (to 32 channels) or 80.6% (to 12 channels) preserves classification performance close to the full 62-channel montage. `stab_32` provides higher electrode selection stability across folds, making it advantageous for physical device design.

### 3.2 Within-Subject Uncertainty Ranking (Claims 4 & 5)
- **Status:** **SUPPORTED**
- **Evidence:** Subject-level Full AURC shows consistent risk reductions (0.5454 for `pred_32` vs. 0.6110 random; 0.5628 for `stab_32` vs. 0.6144 random). Maximum-probability uncertainty reliably sorts correct from incorrect classifications within an individual's session.
- **Assessment:** At the individual level, the uncertainty estimation objective is verified.

### 3.3 Cross-Subject Operational Transfer & Multiple Testing (Claims 6–8)
- **Status:** **PARTIALLY SUPPORTED / UNSUPPORTED**
- **Evidence:**
  - *Coverage Gap:* At nominal 70% coverage, temperature-scaled models accept 88.9%–92.2% of test samples. Static quantile thresholds derived from training OOF distributions do not transfer accurately across subjects.
  - *Multiple Testing:* 5 tests achieve unadjusted $p < 0.05$ against random rejection. However, after Holm-Bonferroni correction across the 20 primary evaluation tests, **none remain statistically significant** ($p_{\text{Holm}} \ge 0.165$).
  - *Simpson's Paradox:* Pooled evaluation shows near-random ranking ($0.6139$ vs. $0.6144$), proving that pooling cross-subject EEG predictions introduces severe confounding.

### 3.4 Operational Deployment & Clinical Applicability (Claims 10 & 11)
- **Status:** **UNSUPPORTED (Out of Scope)**
- **Assessment:** The experimental pipeline operates on pre-segmented 4-second epochs offline on a single benchmark dataset. Claims of real-time operation, clinical utility, or multi-dataset generalization are unsubstantiated and must not be made.

---

## 4. Final Guidelines for Thesis & Paper Write-Up

1. **State the Verdict Honestly:** The research hypothesis is **partially supported**—supported for channel reduction and within-subject error detection, but limited by cross-subject domain shift and multiple testing corrections for static operational thresholding.
2. **Always Report Actual Coverage:** Never refer to nominal training targets (50%, 60%, 70%, 80%, 90%) without stating the actual coverage realized on test subjects.
3. **Present Both Unadjusted and Adjusted Statistics:** Explicitly state that while low-coverage improvements reach statistical significance under unadjusted paired tests ($p < 0.05$), they do not survive conservative family-wise Holm-Bonferroni correction across all conditions.
4. **Emphasize the Need for Adaptive Normalization:** Frame the coverage gap not as a failure, but as a critical scientific insight: future work must implement unsupervised subject-adaptive confidence normalization to translate within-subject uncertainty gains into deployable operational BCIs.
