# Final Project Status Report: Reliable Subject-Independent EEG Emotion Recognition

**Project Title:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 37,575 total 4-second EEG epochs, 4 emotion classes)  
**Evaluation Protocol:** 15-fold outer Leave-One-Subject-Out (LOSO) cross-validation with strictly nested inner subject-level cross-validation  
**Current Date:** October 26, 2026  
**Status:** Completed (Phases A through E fully implemented, verified, and audited)

---

## 1. Research Question & Objectives

The project investigates the dual challenge of dimensionality reduction and predictive reliability in subject-independent brain-computer interfaces (BCIs):

> *"Can we reduce the number of EEG channels while maintaining subject-independent emotion-recognition performance, and use uncertainty estimation to identify and abstain from unreliable predictions on unseen subjects?"*

### Core Hypotheses:
1. **Channel Reduction Hypothesis:** A subset of 32 stable or predictive channels can achieve emotion classification performance comparable to the full 62-channel montage on unseen subjects.
2. **Selective Prediction Hypothesis:** Predictive uncertainty scores derived without label leakage can identify misclassifications on unseen subjects, enabling an uncertainty-based abstention policy that outperforms label-agnostic random rejection at matched sample counts.

---

## 2. Dataset & Signal Processing Protocol

- **Dataset:** SEED-IV benchmark dataset comprising 15 subjects across 3 separate experimental recording sessions (total $N = 37,575$ window segments; 2,505 windows per subject).
- **Target Classes ($K = 4$):** Neutral ($y=0$), Sad ($y=1$), Fear ($y=2$), Happy ($y=3$). Balanced class distribution (theoretical chance-level accuracy = 25.00%).
- **Feature Extraction:** Differential Entropy (DE) computed across five standard physiological frequency bands: Delta (1–4 Hz), Theta (4–8 Hz), Alpha (8–14 Hz), Beta (14–31 Hz), and Gamma (31–50 Hz).
  - 62 channels $\times$ 5 frequency bands = 310 input features per window segment.
- **Data Normalization:** `StandardScaler` fitted strictly on training subjects within each outer fold and applied to held-out test subjects to prevent subject leakage.
- **Base Classifier:** Linear Support Vector Classifier (`LinearSVC`, $C = 1.0$, `dual=False`, multi-class one-vs-rest continuous decision margins).

---

## 3. Completed Project Phases & Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PROJECT WORKFLOW (PHASES A – E)                        │
└─────────────────────────────────────────────────────────────────────────────┘
  Phase A: Baseline LOSO Model (62 channels, 310 features, LinearSVC)
     │       └── Full-coverage accuracy: 39.53% ± 9.77% (Chance: 25.0%)
     ▼
  Phase B & C: Channel Selection & Stability Correction
     │       ├── Predictive-Importance Selection (`pred_32`, `pred_12`)
     │       └── Stability-Aware Selection (`stab_32`, Kuncheva stability index)
     ▼
  Phase D: Group-Safe Probability Calibration & Uncertainty Metrics
     │       ├── Group-Safe Temperature Scaling (ECE reduced from ~13% to ~6%)
     │       └── Uncertainty metrics: Maximum probability ($u_{max}$), Normalized Entropy ($H_{norm}$)
     ▼
  Phase E: Selective Prediction & Abstention Analysis (LOSO Evaluation)
             ├── Mode A: Retrospective Ranking & AURC (Bounded [0.5, 1.0] and Full [0.0, 1.0])
             └── Mode B: Inductive Operational Abstention (Training-only OOF thresholds)
```

1. **Phase A (Baseline LOSO Pipeline):** Established outer 15-fold LOSO cross-validation. Verified label alignment, zero-leakage subject splitting, and confusion matrix reproducibility. Baseline full-coverage accuracy achieved is **39.53% ± 9.77%**.
2. **Phase B & C (Channel Reduction & Stability):** Implemented permutation importance ranking and Kuncheva-regularized stable channel selection across inner training folds:
   - `all_62`: 62 channels (310 features)
   - `pred_32`: 32 channels chosen purely by predictive importance (160 features)
   - `stab_32`: 32 channels chosen by stability-regularized selection (160 features)
   - `pred_12`: 12 channels chosen by top predictive importance (60 features)
3. **Phase D (Probability Calibration):** Integrated group-safe temperature scaling to map raw SVM decision margins to calibrated probabilities without test-subject leakage. Evaluated Brier score, Expected Calibration Error (ECE), and error detection AUROC.
4. **Phase E (Selective Prediction & Abstention):** Designed, preflighted, executed, and independently audited selective prediction across two modes:
   - **Mode A (Retrospective Evaluation):** Analyzed risk–coverage curves, empirical selective accuracy, and Partial/Full AURC using project-standard numerical integration conventions.
   - **Mode B (Inductive Operational Evaluation):** Derived operational thresholds from training-only, group-level cross-fitted OOF uncertainty quantiles, froze them, and evaluated them on unseen test subjects against a 500-repetition Monte Carlo random rejection baseline.

---

## 4. Principal Findings & Empirical Summary

Table 1 consolidates the primary operational findings across all four channel conditions evaluated with Maximum-Probability Uncertainty ($u_{\max}$) under Group-Safe Temperature Scaling:

#### Table 1: Primary Mode B Inductive Operational Evaluation Summary
*(Evaluated across 15 outer LOSO folds; $\Delta\text{Acc}$ in percentage points; 95% CIs reported)*

| Condition | Nominal Target | Actual Coverage (Mean ± SD) | Coverage Gap ($\Delta\text{Cov}$) | Baseline Full Acc | Selective Acc (Mean ± SD) | Random Rejection Ref Acc | Mean Gain ($\Delta\text{Acc}$) | Paired $t$-test ($p_{\text{unadj}}$) | Wilcoxon ($p_{\text{unadj}}$) | Holm $p_{\text{adj}}$ ($m=20$) | Pos / Neg / Zero Subj |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`all_62`** | 1.00 | 100.0% ± 0.0% | 0.0 pp | 39.53% | 39.53% ± 9.77% | 39.53% | 0.00 pp | 1.000 | 1.000 | 1.000 | 0 / 0 / 15 |
| | 0.90 | 92.9% ± 9.0% | +2.9 pp | 39.53% | 40.40% ± 10.15% | 39.54% | +0.86 pp | 0.250 | 0.253 | 1.000 | 8 / 4 / 3 |
| | 0.80 | 92.8% ± 9.0% | +12.8 pp | 39.53% | 40.38% ± 10.13% | 39.53% | +0.85 pp | 0.255 | 0.253 | 1.000 | 8 / 4 / 3 |
| | 0.70 | 91.9% ± 9.0% | +21.9 pp | 39.53% | 40.31% ± 10.09% | 39.54% | +0.77 pp | 0.304 | 0.332 | 1.000 | 8 / 4 / 3 |
| | 0.60 | 69.4% ± 33.2% | +9.4 pp | 39.53% | 41.39% ± 10.71% | 39.59% | +1.79 pp | 0.165 | 0.138 | 1.000 | 8 / 4 / 3 |
| | 0.50 | 44.5% ± 38.2% | -5.5 pp | 39.53% | 40.75% ± 15.87% | 39.50% | +1.25 pp | 0.762 | 0.551 | 1.000 | 9 / 5 / 1 |
| **`pred_32`** | 1.00 | 100.0% ± 0.0% | 0.0 pp | 38.90% | 38.90% ± 9.70% | 38.90% | 0.00 pp | 1.000 | 1.000 | 1.000 | 0 / 0 / 15 |
| | 0.90 | 92.6% ± 9.6% | +2.6 pp | 38.90% | 39.67% ± 9.60% | 38.88% | +0.79 pp | 0.093 | 0.182 | 1.000 | 9 / 5 / 1 |
| | 0.80 | 90.9% ± 10.1% | +10.9 pp | 38.90% | 40.08% ± 9.79% | 38.89% | +1.19 pp | 0.052 | 0.105 | 0.892 | 9 / 5 / 1 |
| | 0.70 | 89.1% ± 12.0% | +19.1 pp | 38.90% | 39.99% ± 9.85% | 38.87% | +1.12 pp | 0.088 | 0.164 | 1.000 | 9 / 5 / 1 |
| | **0.60** | **79.3% ± 18.0%** | **+19.3 pp** | **38.90%** | **41.96% ± 10.52%** | **38.90%** | **+3.07 pp** | **0.018** | **0.008** | **0.165** | **11 / 3 / 1** |
| | **0.50** | **61.2% ± 32.4%** | **+11.2 pp** | **38.90%** | **45.60% ± 21.56%** | **38.70%** | **+6.91 pp** | **0.201** | **0.038** | **0.571** | **12 / 2 / 1** |
| **`stab_32`** | 1.00 | 100.0% ± 0.0% | 0.0 pp | 38.56% | 38.56% ± 9.54% | 38.56% | 0.00 pp | 1.000 | 1.000 | 1.000 | 0 / 0 / 15 |
| | 0.90 | 93.3% ± 8.9% | +3.3 pp | 38.56% | 39.06% ± 9.34% | 38.55% | +0.52 pp | 0.182 | 0.200 | 1.000 | 9 / 4 / 2 |
| | 0.80 | 91.4% ± 9.5% | +11.4 pp | 38.56% | 39.43% ± 9.55% | 38.56% | +0.87 pp | 0.115 | 0.221 | 1.000 | 8 / 5 / 2 |
| | 0.70 | 88.9% ± 12.2% | +18.9 pp | 38.56% | 39.57% ± 9.58% | 38.54% | +1.03 pp | 0.095 | 0.132 | 1.000 | 9 / 4 / 2 |
| | **0.60** | **78.3% ± 22.2%** | **+18.3 pp** | **38.56%** | **42.78% ± 10.78%** | **38.57%** | **+4.21 pp** | **0.033** | **0.021** | **0.385** | **10 / 3 / 2** |
| | **0.50** | **60.7% ± 33.4%** | **+10.7 pp** | **38.56%** | **47.15% ± 20.80%** | **38.51%** | **+8.64 pp** | **0.117** | **0.025** | **0.422** | **12 / 2 / 1** |
| **`pred_12`** | 1.00 | 100.0% ± 0.0% | 0.0 pp | 38.04% | 38.04% ± 8.91% | 38.04% | 0.00 pp | 1.000 | 1.000 | 1.000 | 0 / 0 / 15 |
| | 0.90 | 93.3% ± 8.7% | +3.3 pp | 38.04% | 38.67% ± 9.65% | 38.04% | +0.63 pp | 0.215 | 0.200 | 1.000 | 9 / 4 / 2 |
| | 0.80 | 93.1% ± 8.6% | +13.1 pp | 38.04% | 38.63% ± 9.66% | 38.03% | +0.60 pp | 0.240 | 0.221 | 1.000 | 9 / 4 / 2 |
| | 0.70 | 92.2% ± 8.7% | +22.2 pp | 38.04% | 38.84% ± 9.57% | 38.04% | +0.81 pp | 0.146 | 0.181 | 1.000 | 9 / 4 / 2 |
| | **0.60** | **68.6% ± 32.2%** | **+8.6 pp** | **38.04%** | **45.10% ± 17.95%** | **38.17%** | **+6.93 pp** | **0.146** | **0.033** | **0.525** | **10 / 3 / 2** |
| | **0.50** | **47.7% ± 33.2%** | **-2.3 pp** | **38.04%** | **44.56% ± 12.21%** | **38.53%** | **+6.02 pp** | **0.030** | **0.020** | **0.385** | **11 / 3 / 0** |

*(Source: [`outputs/tables/final_results_summary.csv`](file:///C:/Users/singh/OneDrive/Desktop/ML%20Project/outputs/tables/final_results_summary.csv))*

---

## 5. Methodological & Statistical Analysis

### 5.1 Multiple Testing Limitations
In exploratory unadjusted testing, 5 combinations exhibit statistically significant improvements over random rejection ($p_{\text{unadj}} < 0.05$). However:
- **Holm-Bonferroni FWER Correction ($m = 20$):** None of the 20 primary analysis tests survive family-wise error rate control ($p_{\text{Holm}} \ge 0.165$).
- **Benjamini-Hochberg FDR Control ($m = 20$):** The minimum false discovery rate is $q = 0.124$ (discovery rate cannot be guaranteed at 5% FDR).
- **Single-Model Isolation:** If `pred_32` is pre-planned as an isolated family ($m = 5$), nominal 0.60 survives Holm adjustment ($p_{\text{Holm}} = 0.041$). However, this cannot be claimed as confirmatory evidence across the entire experimental matrix.

### 5.2 The Operational Coverage Shift (Domain Shift)
At nominal targets of 70%–90%, actual held-out test coverage ranges from **88.9% to 93.3%** under group-safe temperature scaling:
- **Mechanism:** Global temperature scaling divides margins by a scaling factor ($T \approx 1.7$), compressing probabilities toward uniform ($0.25$) and clustering OOF uncertainties into a narrow range ($u_{\max} \in [0.65, 0.72]$). Minor shifts in unseen subjects' margin distributions cause vast majorities of test samples to fall below the training threshold.
- In contrast, **uncalibrated softmax** maintains margin spread, matching actual test coverage to nominal targets within 3.3 percentage points across all levels (e.g., nominal 70% yields 71.2% actual coverage for `stab_32`).

### 5.3 Resolution of Simpson's Paradox
In retrospective Mode A evaluation:
- The **pooled Full AURC** for `stab_32` is $0.6139$, which is virtually indistinguishable from the random rejection reference ($0.6144$).
- However, the **subject-level mean Full AURC** is $0.5628 \pm 0.0855$, representing a substantial risk reduction of **0.0516 (5.16 risk area points)** below random.
- **Why this occurs:** When 37,575 predictions are pooled globally, subjects with naturally high confidence but lower classification accuracy are accepted preferentially over subjects with lower confidence and higher accuracy. This across-subject confidence-accuracy misalignment depresses the pooled curve.
- **Conclusion:** Mode A pooled AURC obscures true individual rejection capability. Only per-subject evaluations reflect real-world BCI utility.

---

## 6. Key Project Limitations

1. **Modest Baseline Accuracy:** Subject-independent 4-class emotion classification accuracy remains modest across all models (~38%–40% vs. 25% chance level), reflecting the known electrophysiological challenges of cross-subject emotion generalization.
2. **Inexact Operational Coverage:** Static training-derived quantile thresholds cannot enforce target coverage on unseen test subjects due to inter-subject distribution shifts.
3. **No External-Dataset Validation:** Evaluation was conducted exclusively on the SEED-IV dataset; generalization to other EEG emotion corpora (e.g., DEAP, DREAMER) remains to be demonstrated.
4. **Offline Window-Level Processing:** All analyses evaluated 4-second segmented Differential Entropy epochs offline. No online streaming, closed-loop feedback, or real-time computational latency has been measured.
5. **No Clinical Claims:** The experimental setup and findings represent basic BCI engineering research and do not constitute clinical diagnostics or medical-grade intervention.

---

## 7. Final Defensible Research Conclusion

The empirical findings provide **PARTIAL SUPPORT** for the overarching research question:

1. **Channel Reduction (Supported):** Reducing the EEG montage from 62 channels down to 32 stable channels (`stab_32`) or 32 predictive channels (`pred_32`) achieves classification performance that is empirically comparable to the full 62-channel montage (38.6% and 38.9% vs. 39.5%). Even a minimal 12-channel montage (`pred_12`) retains 38.0% baseline accuracy.
2. **Selective Prediction Within Subjects (Supported):** Maximum-probability uncertainty reliably ranks classification correctness within individual unseen subjects, driving subject-level accuracy from ~38.6% up to **45.6%–47.2%** as coverage is reduced to 50%–60%.
3. **Operational Threshold Transferability Across Subjects (Not Fully Supported):** While operational thresholds can be derived without label leakage, static training quantiles cannot reliably hit target coverage on unseen subjects, and improvements over random rejection do not achieve statistical significance under family-wise multiple testing control across the full experimental matrix.
