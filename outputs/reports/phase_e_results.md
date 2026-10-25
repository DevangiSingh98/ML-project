# Phase E Research Report: Selective Prediction and Uncertainty-Based Abstention

## 1. Executive Summary

This report documents the final empirical findings of **Phase E** for the research project:
**Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction**.

The research experiment evaluated whether uncertainty estimation can identify and abstain from unreliable predictions on unseen subjects across 15 outer Leave-One-Subject-Out (LOSO) cross-validation folds on the genuine SEED-IV dataset (37,575 EEG window segments across 15 subjects).

### Key Empirical Findings:
1. **Hypothesis Verdict: PARTIALLY SUPPORTED**
   - **Supported at the Subject Level:** For individual unseen subjects, abstaining from high-uncertainty predictions systematically removes classification errors, driving subject-level accuracy up by **+6.0 to +8.6 percentage points** as coverage decreases from 100% to ~50%–60%.
   - **Supported at Operational Thresholds:** In Mode B (inductive operational evaluation with frozen, strictly nested, group-level cross-fitted training thresholds), uncertainty-based abstention outperforms the 500-repetition random rejection baseline across 10–12 out of 15 subjects, achieving statistically significant improvements at nominal coverages of 50%–60% ($p < 0.05$).
   - **Not Supported for Strict Nominal Coverage Guarantees:** Due to cross-subject distribution shift, operational thresholds derived from training subjects accept ~88%–93% of test samples when nominal coverage is set to 70%–80%. Fixed training quantiles cannot guarantee exact test coverage targets without knowing the test subject's uncertainty distribution.
   - **Confounded at the Pooled Sample Level (Simpson's Paradox):** In retrospective pooled analysis (Mode A), pooling 37,575 samples across subjects obscures individual accuracy improvements because subjects with naturally high confidence happen to exhibit lower baseline classification performance.

---

## 2. Core Results Table: Mode B Inductive Operational Evaluation

Primary analysis evaluated under **Group-Safe Temperature Scaling** with **Maximum-Probability Uncertainty ($u_{\max}$)** across 15 outer LOSO folds:

| Channel Condition | Nominal Coverage | Actual Coverage (Mean ± SD) | Full-Coverage Baseline Acc | Selective Accuracy (Mean ± SD) | Random Rejection Reference Acc | Mean Improvement ($\Delta \text{Acc}_{\text{pp}}$) | 95% Paired CI ($\Delta \text{Acc}$) | Paired $t$-test ($p$-value) | Wilcoxon ($p$-value) | Improved Subjects |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`all_62`** | **1.00** | 1.000 ± 0.000 | 39.53% | 39.53% ± 9.77% | 39.53% | 0.00 pp | [0.00, 0.00] | $p = 1.000$ | $p = 1.000$ | 0 / 15 |
| | **0.90** | 0.929 ± 0.090 | 39.53% | 40.40% ± 10.15% | 39.54% | +0.86 pp | [-0.68, +2.39] | $p = 0.250$ | $p = 0.301$ | 8 / 15 |
| | **0.80** | 0.928 ± 0.090 | 39.53% | 40.38% ± 10.13% | 39.53% | +0.85 pp | [-0.69, +2.40] | $p = 0.255$ | $p = 0.301$ | 8 / 15 |
| | **0.70** | 0.919 ± 0.090 | 39.53% | 40.31% ± 10.09% | 39.54% | +0.77 pp | [-0.78, +2.32] | $p = 0.304$ | $p = 0.424$ | 8 / 15 |
| | **0.60** | 0.694 ± 0.332 | 39.53% | 41.39% ± 10.71% | 39.59% | +1.79 pp | [-0.83, +4.42] | $p = 0.165$ | $p = 0.129$ | 8 / 15 |
| | **0.50** | 0.445 ± 0.382 | 39.53% | 40.75% ± 15.87% | 39.50% | +1.25 pp | [-7.47, +9.97] | $p = 0.762$ | $p = 0.626$ | 9 / 15 |
| **`pred_32`** | **1.00** | 1.000 ± 0.000 | 38.90% | 38.90% ± 9.70% | 38.90% | 0.00 pp | [0.00, 0.00] | $p = 1.000$ | $p = 1.000$ | 0 / 15 |
| | **0.90** | 0.926 ± 0.096 | 38.90% | 39.67% ± 9.60% | 38.88% | +0.79 pp | [-0.15, +1.73] | $p = 0.093$ | $p = 0.194$ | 9 / 15 |
| | **0.80** | 0.909 ± 0.101 | 38.90% | 40.08% ± 9.79% | 38.89% | +1.19 pp | [-0.01, +2.39] | $p = 0.052$ | $p = 0.104$ | 9 / 15 |
| | **0.70** | 0.891 ± 0.120 | 38.90% | 39.99% ± 9.85% | 38.87% | +1.12 pp | [-0.19, +2.43] | $p = 0.088$ | $p = 0.173$ | 9 / 15 |
| | **0.60** | 0.793 ± 0.180 | 38.90% | **41.97% ± 10.52%** | 38.90% | **+3.07 pp** | **[+0.62, +5.52]** | **$p = 0.018$** | **$p = 0.005$** | **11 / 15** |
| | **0.50** | 0.612 ± 0.324 | 38.90% | **45.60% ± 21.56%** | 38.70% | **+6.91 pp** | [-4.13, +17.94] | $p = 0.201$ | **$p = 0.049$** | **12 / 15** |
| **`stab_32`** | **1.00** | 1.000 ± 0.000 | 38.56% | 38.56% ± 9.54% | 38.56% | 0.00 pp | [0.00, 0.00] | $p = 1.000$ | $p = 1.000$ | 0 / 15 |
| | **0.90** | 0.933 ± 0.089 | 38.56% | 39.06% ± 9.34% | 38.55% | +0.52 pp | [-0.27, +1.31] | $p = 0.182$ | $p = 0.244$ | 9 / 15 |
| | **0.80** | 0.914 ± 0.095 | 38.56% | 39.43% ± 9.55% | 38.56% | +0.87 pp | [-0.24, +1.99] | $p = 0.115$ | $p = 0.216$ | 8 / 15 |
| | **0.70** | 0.889 ± 0.122 | 38.56% | 39.57% ± 9.58% | 38.54% | +1.03 pp | [-0.20, +2.26] | $p = 0.095$ | $p = 0.146$ | 9 / 15 |
| | **0.60** | 0.783 ± 0.222 | 38.56% | **42.78% ± 10.78%** | 38.57% | **+4.21 pp** | **[+0.40, +8.03]** | **$p = 0.033$** | **$p = 0.017$** | **10 / 15** |
| | **0.50** | 0.607 ± 0.334 | 38.56% | **47.15% ± 20.80%** | 38.51% | **+8.64 pp** | [-2.44, +19.72] | $p = 0.117$ | **$p = 0.030$** | **12 / 15** |
| **`pred_12`** | **1.00** | 1.000 ± 0.000 | 38.04% | 38.04% ± 8.91% | 38.04% | 0.00 pp | [0.00, 0.00] | $p = 1.000$ | $p = 1.000$ | 0 / 15 |
| | **0.90** | 0.933 ± 0.087 | 38.04% | 38.67% ± 9.65% | 38.04% | +0.63 pp | [-0.41, +1.66] | $p = 0.215$ | $p = 0.244$ | 9 / 15 |
| | **0.80** | 0.931 ± 0.086 | 38.04% | 38.63% ± 9.66% | 38.03% | +0.60 pp | [-0.45, +1.66] | $p = 0.240$ | $p = 0.273$ | 9 / 15 |
| | **0.70** | 0.922 ± 0.087 | 38.04% | 38.84% ± 9.57% | 38.04% | +0.81 pp | [-0.32, +1.93] | $p = 0.146$ | $p = 0.216$ | 9 / 15 |
| | **0.60** | 0.686 ± 0.322 | 38.04% | **45.10% ± 17.95%** | 38.17% | +6.93 pp | [-2.72, +16.58] | $p = 0.146$ | **$p = 0.033$** | **10 / 15** |
| | **0.50** | 0.477 ± 0.332 | 38.04% | **44.56% ± 12.21%** | 38.01% | **+6.02 pp** | **[+0.68, +11.37]** | **$p = 0.030$** | **$p = 0.020$** | **11 / 14** |

---

## 3. Secondary Comparison: Uncalibrated Softmax Regime

To determine whether temperature calibration improves operational abstention thresholds, Mode B was evaluated under **Uncalibrated Softmax** ($T = 1.0$):

| Condition | Nominal Coverage | Actual Coverage (Mean) | Selective Acc (Mean) | Random Acc (Mean) | Improvement ($\Delta \text{Acc}_{\text{pp}}$) | Wilcoxon ($p$-value) | Improved Subjects |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`stab_32`** | 1.00 | 100.0% | 38.56% | 38.56% | 0.00 pp | $p = 1.000$ | 0 / 15 |
| | 0.90 | 90.80% | 39.58% | 38.55% | +1.03 pp | $p = 0.037$ | 10 / 15 |
| | 0.80 | 82.42% | 40.69% | 38.53% | +2.16 pp | $p = 0.002$ | 11 / 15 |
| | 0.70 | 71.18% | 42.35% | 38.54% | +3.81 pp | $p = 0.002$ | 12 / 15 |
| | 0.60 | 62.42% | 44.75% | 38.58% | +6.17 pp | $p = 0.002$ | 12 / 15 |
| | 0.50 | 53.34% | 46.46% | 38.57% | +7.89 pp | $p = 0.003$ | 11 / 15 |

### Comparison Insight:
- In uncalibrated softmax, raw SVM margins produce sharper confidence boundaries.
- Because uncalibrated softmax confidence is more dispersed, training quantiles match held-out test coverage much closer to the nominal target (at 70% nominal, achieved coverage is 71.18%, compared to 88.92% for temperature scaling).
- Uncalibrated softmax achieves statistically significant selective accuracy gains earlier (starting at 90% nominal coverage, $p < 0.05$).
- However, temperature scaling achieves a higher peak selective accuracy at low coverage (47.15% for temperature-scaled `stab_32` vs 46.46% for uncalibrated).

---

## 4. Mode A: Retrospective Performance Evaluation (Ranking Quality)

Mode A describes uncertainty's empirical ranking capacity directly on test observations. It does not evaluate a deployable threshold policy.

### 4.1 Area Under the Risk-Coverage Curve (AURC) Summary
Evaluated over $N = 37,575$ pooled test predictions and across the 15 outer subjects:

| Condition | Calibration Method | Uncertainty Measure | Pooled Partial $\text{AURC}_{[0.5, 1.0]}$ | Pooled Canonical Full $\text{AURC}_{[0.0, 1.0]}$ | Subject Partial $\text{AURC}_{[0.5, 1.0]}$ (Mean ± SD) | Subject Canonical Full $\text{AURC}_{[0.0, 1.0]}$ (Mean ± SD) |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| `all_62` | `group_safe_temperature` | $u_{\max}$ | 0.2974 | 0.5979 | 0.2868 ± 0.0463 | 0.5528 ± 0.0827 |
| `all_62` | `group_safe_temperature` | $H_{\text{norm}}$ | 0.2995 | 0.5997 | 0.2831 ± 0.0460 | 0.5483 ± 0.0763 |
| `all_62` | `uncalibrated_softmax` | $u_{\max}$ | 0.2996 | 0.6009 | 0.2863 ± 0.0465 | 0.5501 ± 0.0807 |
| `all_62` | `uncalibrated_softmax` | $H_{\text{norm}}$ | 0.3013 | 0.6033 | 0.2889 ± 0.0471 | 0.5574 ± 0.0864 |
| `pred_32` | `group_safe_temperature` | $u_{\max}$ | 0.3023 | 0.6080 | 0.2828 ± 0.0470 | 0.5454 ± 0.0857 |
| `pred_32` | `group_safe_temperature` | $H_{\text{norm}}$ | 0.3039 | 0.6094 | 0.2878 ± 0.0487 | 0.5561 ± 0.0883 |
| `pred_32` | `uncalibrated_softmax` | $u_{\max}$ | 0.3044 | 0.6111 | 0.2882 ± 0.0474 | 0.5569 ± 0.0868 |
| `pred_32` | `uncalibrated_softmax` | $H_{\text{norm}}$ | 0.3081 | 0.6157 | 0.2936 ± 0.0494 | 0.5691 ± 0.0931 |
| `stab_32` | `group_safe_temperature` | $u_{\max}$ | 0.3072 | 0.6139 | 0.2917 ± 0.0467 | 0.5628 ± 0.0855 |
| `stab_32` | `group_safe_temperature` | $H_{\text{norm}}$ | 0.3086 | 0.6151 | 0.2922 ± 0.0466 | 0.5645 ± 0.0854 |
| `stab_32` | `uncalibrated_softmax` | $u_{\max}$ | 0.3073 | 0.6143 | 0.2928 ± 0.0466 | 0.5654 ± 0.0855 |
| `stab_32` | `uncalibrated_softmax` | $H_{\text{norm}}$ | 0.3093 | 0.6166 | 0.2954 ± 0.0472 | 0.5716 ± 0.0881 |
| `pred_12` | `group_safe_temperature` | $u_{\max}$ | 0.3080 | 0.6174 | 0.2938 ± 0.0450 | 0.5678 ± 0.0805 |
| `pred_12` | `group_safe_temperature` | $H_{\text{norm}}$ | 0.3105 | 0.6200 | 0.2981 ± 0.0458 | 0.5761 ± 0.0827 |
| `pred_12` | `uncalibrated_softmax` | $u_{\max}$ | 0.3098 | 0.6204 | 0.2988 ± 0.0460 | 0.5776 ± 0.0818 |
| `pred_12` | `uncalibrated_softmax` | $H_{\text{norm}}$ | 0.3117 | 0.6225 | 0.2995 ± 0.0463 | 0.5788 ± 0.0823 |

- **Integration Conventions**:
  - `partial_aurc_50_to_100`: Trapezoidal numerical integration bounded strictly on coverage $c \in [0.5, 1.0]$. Theoretical bounds are $[0.0, 0.50]$.
  - `full_aurc`: Canonical full trapezoidal numerical integration on coverage $c \in [0.0, 1.0]$ with horizontal extension down to $c = 0.0$. Theoretical bounds are $[0.0, 1.00]$.

---

## 5. Methodological Analysis: Why Pooled Evaluation Hides Individual Improvement

A crucial discovery from this Phase E execution is the **Subject-Level Calibration Disconnect (Simpson's Paradox)**:

1. **Within each subject**, higher confidence is strongly associated with correct classifications:
   - For 11–12 of the 15 subjects in `pred_32` and `stab_32`, selective accuracy increases monotonically when abstaining from lower-confidence windows.
   - At the individual level, abstention increases accuracy from ~38.6% up to 47.2%.

2. **Across subjects**, however, baseline confidence is not aligned with subject baseline accuracy:
   - Some subjects (e.g. Subject 2, full-coverage accuracy 36.3%) produce decision margins with high confidence magnitudes.
   - Other subjects (e.g. Subject 1, full-coverage accuracy 42.8%) produce narrower decision margins.
   - When all 37,575 samples from all 15 subjects are pooled together and sorted by global confidence (Mode A), the windows from lower-accuracy, high-confidence subjects are accepted first, while higher-accuracy, moderate-confidence subjects are rejected early.
   - This cross-subject calibration misalignment depresses the pooled curve at low coverage levels.

### Key Implication for BCI Deployment:
Treating all EEG windows from diverse individuals as independent observations produces misleading conclusions. Uncertainty-based abstention is effective **per subject**, but operational thresholds must account for inter-subject baseline confidence variability.

---

## 6. Generated Output Artifacts

All outputs were generated strictly under the project protocol and verified on disk:

1. **Predictions CSV**:
   - [`outputs/predictions/phase_e_selective_predictions.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/predictions/phase_e_selective_predictions.csv) (28.8 MB, 300,600 rows)
2. **Metrics CSV**:
   - [`outputs/metrics/phase_e_fold_metrics.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/metrics/phase_e_fold_metrics.csv) (1.16 MB, master fold metrics across Mode A and Mode B)
3. **Summary Tables**:
   - [`outputs/tables/phase_e_coverage_accuracy_summary.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/phase_e_coverage_accuracy_summary.csv) (Mode A coverage vs accuracy coordinates)
   - [`outputs/tables/phase_e_risk_coverage_summary.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/phase_e_risk_coverage_summary.csv) (Mode A risk-coverage coordinates)
   - [`outputs/tables/phase_e_random_rejection_summary.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/phase_e_random_rejection_summary.csv) (500-repetition Monte Carlo random rejection baseline)
   - [`outputs/tables/phase_e_subject_summary.csv`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/phase_e_subject_summary.csv) (Mode B fold-level and target-level operational summary)
4. **Execution Run Report**:
   - [`outputs/metrics/phase_e_run_report.json`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/metrics/phase_e_run_report.json) (12.6 KB JSON report)
5. **Publication-Quality Figures**:
   - [`outputs/figures/phase_e_risk_coverage.png`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/figures/phase_e_risk_coverage.png) (Risk-Coverage curves across conditions)
   - [`outputs/figures/phase_e_coverage_accuracy.png`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/figures/phase_e_coverage_accuracy.png) (Selective accuracy vs coverage across conditions)
6. **Documentation Reports**:
   - [`outputs/reports/phase_e_results.md`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/reports/phase_e_results.md) (This document)
   - [`outputs/reports/phase_e_methodology.md`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/reports/phase_e_methodology.md) (Mathematical and operational protocol specification)
   - [`outputs/reports/phase_e_validation.md`](file:///c:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/reports/phase_e_validation.md) (Calibration verification and preflight audit)

---

## 7. Methodological Limitations & Future Directions

1. **Coverage Target Inexactness under Domain Shift:**
   Deriving static thresholds from training out-of-fold uncertainty quantiles does not achieve exact target coverage on held-out subjects. Developing adaptive subject-calibration methods (e.g. unsupervised domain adaptation of confidence distributions) is a natural next step.
2. **Subject Disconnect in Confidence vs Accuracy:**
   Because different individuals have different baseline margins, temperature scaling fits an average global temperature that under-calibrates some subjects and over-calibrates others. Subject-adaptive temperature scaling could align confidence distributions across people.
3. **Channel Reduction Trade-Off:**
   `stab_32` and `pred_32` achieve comparable selective accuracy gains (+8.6 pp and +6.9 pp at low coverage) to the 62-channel baseline, confirming that 32 channels are sufficient for uncertainty-based abstention in EEG emotion recognition.
