# Experiment D3: Four-Shot Subject Calibration Combined with CORAL

**Project:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 24 trials per session = 1,080 trials total; 4 emotion classes: Neutral, Sad, Fear, Happy)  
**Date:** October 2026  
**Status:** Completed, Validated & Audited  

---

## 1. Executive Summary & Core Results

Experiment D3 investigates whether **combining unsupervised domain adaptation (CORAL) with few-shot subject calibration (four labelled target trials—exactly one per emotion class)** can break the cross-subject accuracy plateau observed in Experiments D1 and D2 on SEED-IV whole-trial EEG emotion recognition.

### Context and Research Questions
1. **D1 Baseline:** Unsupervised CORAL target-to-source alignment achieved **54.00%** on `stab_32` and **56.10%** on `all_62` (evaluated across 68 test trials).
2. **D2 Finding:** Unsupervised class-balanced pseudo-label refinement yielded zero significant gain over CORAL (+0.09 pp, $p=0.582$) due to confirmation bias from noisy pseudo-labels (30%–38% error rate).
3. **D3 Inquiry:** Can ground-truth supervision on a minimal calibration budget ($N_{\text{cal}}=4$ trials, one per emotion class: Neutral, Sad, Fear, Happy) provide the necessary class-anchor to shift the decision boundary in the CORAL-aligned latent feature space?
4. **Target Aspiration:** The aspirational accuracy target is 80%. As mandated by the protocol, **no improvement is assumed or manufactured**.

---

### Primary Benchmark Results (Evaluated on Exactly the Same 68 Test Trials per Subject over 5 Fixed Random Calibration Seeds)

All four methods were evaluated on the **identical 68 held-out test trials** across 15 outer LOSO folds and 5 reproducible calibration seeds (`CALIBRATION_SEEDS = [42, 100, 2024, 7, 999]`), yielding **5,100 evaluated test trials per method** (81,600 total predictions). Statistical tests are strictly conducted at the subject level ($N=15$), where repeated calibration splits are first averaged within subject:

| Channel Condition | Method | Pooled Acc (%) | Subject Mean Acc (%) | Subject SD (%) | 95% CI (Student-$t$) | 95% CI (Bootstrap) | Macro-F1 (%) | Balanced Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_62` (310 feats)** | **D. CORAL + 4-Shot Calibration** | **57.24%** | **57.24%** | **6.47%** | [53.65%, 60.82%] | [54.41%, 60.65%] | **56.88%** | **57.24%** |
| | B. CORAL Unsupervised Baseline | 56.10% | 56.10% | 7.06% | [52.19%, 60.01%] | [52.90%, 59.80%] | 55.59% | 56.10% |
| | C. 4-Shot Calibration Only | 45.06% | 45.06% | 7.46% | [40.93%, 49.19%] | [41.61%, 48.88%] | 43.40% | 45.06% |
| | A. Source-Only Baseline | 41.53% | 41.53% | 6.96% | [37.68%, 45.38%] | [38.29%, 44.96%] | 35.83% | 41.53% |
| **`stab_32` (160 feats)** | **D. CORAL + 4-Shot Calibration** | **55.27%** | **55.27%** | **7.03%** | [51.38%, 59.17%] | [51.90%, 58.88%] | **54.71%** | **55.27%** |
| | B. CORAL Unsupervised Baseline | 54.00% | 54.00% | 8.03% | [49.55%, 58.45%] | [50.20%, 57.88%] | 53.39% | 54.00% |
| | C. 4-Shot Calibration Only | 43.53% | 43.53% | 6.46% | [39.95%, 47.11%] | [40.61%, 46.80%] | 41.31% | 43.53% |
| | A. Source-Only Baseline | 42.18% | 42.18% | 7.06% | [38.27%, 46.09%] | [38.84%, 45.59%] | 36.78% | 42.18% |
| **`pred_32` (160 feats)** | **D. CORAL + 4-Shot Calibration** | **57.00%** | **57.00%** | **6.05%** | [53.65%, 60.35%] | [54.22%, 60.10%] | **56.51%** | **57.00%** |
| | B. CORAL Unsupervised Baseline | 55.71% | 55.71% | 6.10% | [52.33%, 59.08%] | [52.80%, 58.76%] | 55.13% | 55.71% |
| | C. 4-Shot Calibration Only | 44.49% | 44.49% | 7.97% | [40.08%, 48.90%] | [40.82%, 48.43%] | 41.96% | 44.49% |
| | A. Source-Only Baseline | 43.06% | 43.06% | 5.94% | [39.77%, 46.35%] | [40.18%, 45.90%] | 37.77% | 43.06% |
| **`pred_12` (60 feats)** | **D. CORAL + 4-Shot Calibration** | **51.88%** | **51.88%** | **7.72%** | [47.61%, 56.16%] | [48.06%, 55.57%] | **51.22%** | **51.88%** |
| | B. CORAL Unsupervised Baseline | 50.16% | 50.16% | 8.11% | [45.67%, 54.65%] | [46.18%, 54.04%] | 49.23% | 50.16% |
| | C. 4-Shot Calibration Only | 44.27% | 44.27% | 8.36% | [39.64%, 48.90%] | [40.33%, 48.37%] | 41.29% | 44.27% |
| | A. Source-Only Baseline | 40.39% | 40.39% | 7.11% | [36.45%, 44.33%] | [37.02%, 43.88%] | 35.02% | 40.39% |

---

### Key Empirical Findings

1. **Four-Shot Calibration Produces a Measurable, Consistent Boost:**
   - On **`all_62`**, Method D (**CORAL + 4-Shot**) achieves **57.24% ± 6.47%**, an improvement of **+1.14 pp over unsupervised CORAL** ($p = 0.0294$, Wilcoxon $p = 0.0353$, 13 of 15 subjects improved) and **+15.71 pp over the source-only baseline** ($p = 1.07 \times 10^{-6}$, 15 of 15 subjects improved).
   - On **`pred_32`**, Method D reaches **57.00% ± 6.05%**, outperforming unsupervised CORAL by **+1.29 pp** ($p = 0.1269$, 11 of 15 subjects improved) and the source baseline by **+13.94 pp** ($p = 3.86 \times 10^{-7}$, 15 of 15 subjects improved).
   - On **`stab_32`**, Method D reaches **55.27% ± 7.03%**, outperforming unsupervised CORAL by **+1.27 pp** ($p = 0.3013$, 10 of 15 subjects improved) and the source baseline by **+13.10 pp** ($p = 9.84 \times 10^{-6}$, 15 of 15 subjects improved).

2. **Calibration Alone Cannot Substitute for Covariance Alignment:**
   - Method C (**4-Shot Calibration Only**, without CORAL) achieves only **43.53%–45.06%**.
   - While calibration alone provides a modest +1.35 to +3.53 pp increase over the raw source baseline ($p = 0.0307$ on `all_62`), it lags behind CORAL by **~11–12 pp**.
   - **Takeaway:** Four calibration points cannot correct for global inter-subject covariance rotation. Second-order manifold alignment (CORAL) is an indispensable prerequisite before few-shot supervised updates can take effect.

3. **Defensible Conclusion Regarding the 80% Aspirational Target:**
   - **Performance remains at 57.24% on `all_62` (gap of 22.76 pp to 80%) and 55.27% on `stab_32` (gap of 24.73 pp to 80%).**
   - We explicitly declare that four-shot subject calibration combined with linear CORAL does **not** achieve 80% accuracy on SEED-IV whole-trial EEG.
   - Ground-truth labeling of 4 trials provides an anchor against severe centroid drift, but single-trial emotion exemplars cannot resolve intra-subject emotional drift across the 3 sessions, session non-stationarity, or high non-linear manifold curvature.

---

## 2. Evaluation Protocol & Transductive Declaration

### Strict Invariants of the Protocol

```
Held-Out Test Subject (72 Trials Total)
├── Fixed Random Seed (e.g., seed=42)
│   ├── Emotion Class 0 (Neutral): 1 trial selected for Calibration
│   ├── Emotion Class 1 (Sad):     1 trial selected for Calibration
│   ├── Emotion Class 2 (Fear):    1 trial selected for Calibration
│   └── Emotion Class 3 (Happy):   1 trial selected for Calibration
│
├── CALIBRATION SET: Exactly 4 trials (1 per class)
│   └── Used ONLY to update classifier weights w_cal
│   └── STRICT INVARIANT: NEVER SCORED IN TEST EVALUATION
│
└── TEST SET: Exactly 68 remaining trials (17 trials per class)
    └── ALL 4 METHODS SCORED ON IDENTICALLY THESE 68 TRIALS
    └── Evaluated across 5 random calibration seeds: [42, 100, 2024, 7, 999]
    └── Total evaluations per method: 15 subjects × 68 trials × 5 seeds = 5,100
```

### Methods Formulations

#### Method A: Source-Only Baseline (`source_only_baseline`)
- Scaled training features: $\tilde{X}_S \in \mathbb{R}^{1008 \times d}$ from 14 source subjects.
- Model: Linear SVM ($C=0.01$, balanced weights).
- Prediction on held-out subject: $\hat{y}_T = \operatorname{LinearSVM}(\tilde{X}_{T,\text{test}})$, evaluated on the 68 test trials.

#### Method B: CORAL Unsupervised Alignment (`coral_unsupervised`)
- Target covariance estimated transductively from all $n_T = 72$ target feature vectors:
  $$\Sigma_T = \frac{1}{n_T - 1} \sum_{j=1}^{n_T} (x_j^T - \mu_T)(x_j^T - \mu_T)^\top + \lambda I_d$$
- Target trials aligned into source space:
  $$X_T^{\text{coral}} = (X_T - \mathbf{1}\mu_T) \Sigma_T^{-1/2} \Sigma_S^{1/2} + \mathbf{1}\mu_S$$
- Standardized with source scaler: $\tilde{X}_T^{\text{coral}} = \operatorname{StandardScaler}_S(X_T^{\text{coral}})$.
- Evaluated on the 68 test trials without target label information.

#### Method C: Few-Shot Calibration Only (`few_shot_calibration_only`)
- Scaled source features $\tilde{X}_S$ ($n_S = 1,008$) combined with 4 raw scaled calibration trials $\tilde{X}_{T,\text{cal}}$ ($n_{\text{cal}} = 4$).
- Sample weights: Source trials receive weight 1.0; calibration trials receive weight $w_{\text{cal}} = 10.0$ (selected via inner source validation).
- Model trained on augmented dataset: $\mathcal{D}_{\text{aug}} = \{(\tilde{X}_S, y_S), (\tilde{X}_{T,\text{cal}}, y_{T,\text{cal}})\}$.
- Evaluated on the remaining 68 raw test trials $\tilde{X}_{T,\text{test}}$.

#### Method D: CORAL Combined with 4-Shot Calibration (`coral_plus_four_shot`)
- All 72 target trials transformed into source feature space via CORAL alignment and scaled: $\tilde{X}_T^{\text{coral}}$.
- The 4 calibration trials in the aligned space $\tilde{X}_{T,\text{cal}}^{\text{coral}}$ are combined with source trials $\tilde{X}_S$ using calibration sample weight $w_{\text{cal}} = 10.0$.
- Model trained on the aligned-augmented dataset:
  $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \left[ \sum_{i=1}^{n_S} \ell(f(x_i^S), y_i^S) + w_{\text{cal}} \sum_{k=1}^4 \ell(f(x_{T,\text{cal},k}^{\text{coral}}), y_{T,\text{cal},k}) \right]$$
- Evaluated on the remaining 68 aligned test trials $\tilde{X}_{T,\text{test}}^{\text{coral}}$.

### Transductive Evaluation Declaration
As required by the experimental specification:
> **Explicit Transductive Declaration:** Method B and Method D use all 72 target feature vectors ($X_T \in \mathbb{R}^{72 \times d}$) to estimate the target sample mean $\mu_T$ and covariance matrix $\Sigma_T$. This evaluation is **transductive unsupervised covariance alignment** followed by few-shot calibration. **No target emotion labels beyond the four designated calibration trials are ever accessed or observed.**

---

## 3. Leakage Prevention & Inner Validation Tuning

### Inner Validation Protocol
To strictly prevent test-set leakage, all hyperparameters ($C$, $w_{\text{cal}}$) were tuned entirely within the 14 training subjects using simulated few-shot validation folds:
1. For each outer fold, an inner validation loop holds out 5 inner validation subjects from the 14 training subjects.
2. In each inner fold, 4 calibration trials (1 per class) are selected from the inner validation subject, and evaluated on the inner validation subject's remaining 68 trials.
3. Candidate grid:
   - Calibration sample weight: $w_{\text{cal}} \in [2.0, 5.0, 10.0, 20.0]$.
   - SVM Regularization: $C \in [0.005, 0.01, 0.05, 0.1]$.
4. **Validation Findings:**
   - Inner validation accuracy on aligned features:
     - $w_{\text{cal}} = 2.0$: 55.4%
     - $w_{\text{cal}} = 5.0$: 56.1%
     - $w_{\text{cal}} = 10.0$: **56.8%**
     - $w_{\text{cal}} = 20.0$: 56.5% (over-fitting to single calibration trials).
   - Optimal parameters: $C = 0.01, w_{\text{cal}} = 10.0$.
5. **Outer Subject Isolation:** The outer test subject's 68 evaluation trials and 4 calibration labels played zero role in parameter selection.

---

## 4. Detailed Statistical Comparisons

### Paired Subject-Level Differences (15 Subjects)

To avoid pseudo-replication from repeated random calibration seeds, the 5 seed evaluations per subject are averaged to produce a single representative score per subject ($N=15$). All comparisons are paired differences against reference baselines:

```
Table: Paired Statistical Comparison across 15 Subjects
Evaluated Method: D (CORAL + 4-Shot) vs Baselines
```

| Channel Condition | Evaluated Method | Reference Baseline | Delta (pp) | 95% CI (Student-$t$) | 95% CI (Bootstrap) | Paired $t$-test $p$ | Wilcoxon $p$ | Subjects Improved |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_62`** | **CORAL + 4-Shot** | **Source Baseline** | **+15.71 pp** | [+11.58, +19.83] | [+12.00, +19.25] | **$1.07 \times 10^{-6}$** | **$0.00006$** | **15 / 15** |
| | **CORAL + 4-Shot** | **CORAL Unsup.** | **+1.14 pp** | [+0.13, +2.14] | [+0.16, +1.94] | **$0.0294$** | **$0.0353$** | **13 / 15** |
| | **CORAL + 4-Shot** | **4-Shot Only** | **+12.18 pp** | [+9.26, +15.09] | [+9.69, +14.71] | **$3.62 \times 10^{-7}$** | **$0.00006$** | **15 / 15** |
| | CORAL Unsup. | Source Baseline | +14.57 pp | [+10.14, +18.99] | [+10.57, +18.31] | $5.68 \times 10^{-6}$ | $0.00006$ | 15 / 15 |
| | 4-Shot Only | Source Baseline | +3.53 pp | [+0.38, +6.68] | [+0.88, +6.31] | $0.0307$ | $0.0437$ | 12 / 15 |
| **`pred_32`** | **CORAL + 4-Shot** | **Source Baseline** | **+13.94 pp** | [+10.58, +17.30] | [+11.10, +16.98] | **$3.86 \times 10^{-7}$** | **$0.00065$** | **15 / 15** |
| | **CORAL + 4-Shot** | **CORAL Unsup.** | **+1.29 pp** | [-0.42, +3.00] | [-0.29, +2.76] | $0.1269$ | $0.1819$ | 11 / 15 |
| | **CORAL + 4-Shot** | **4-Shot Only** | **+12.51 pp** | [+9.33, +15.69] | [+10.00, +15.55] | **$7.24 \times 10^{-7}$** | **$0.00006$** | **15 / 15** |
| | CORAL Unsup. | Source Baseline | +12.65 pp | [+8.93, +16.36] | [+9.35, +15.92] | $3.90 \times 10^{-6}$ | $0.00006$ | 15 / 15 |
| | 4-Shot Only | Source Baseline | +1.43 pp | [-1.77, +4.63] | [-1.22, +4.49] | $0.3539$ | $0.6788$ | 8 / 15 |
| **`stab_32`** | **CORAL + 4-Shot** | **Source Baseline** | **+13.10 pp** | [+8.92, +17.28] | [+9.55, +16.77] | **$9.84 \times 10^{-6}$** | **$0.00006$** | **15 / 15** |
| | **CORAL + 4-Shot** | **CORAL Unsup.** | **+1.27 pp** | [-1.27, +3.82] | [-0.96, +3.53] | $0.3013$ | $0.2801$ | 10 / 15 |
| | **CORAL + 4-Shot** | **4-Shot Only** | **+11.75 pp** | [+8.94, +14.56] | [+9.29, +14.26] | **$3.55 \times 10^{-7}$** | **$0.00006$** | **15 / 15** |
| | CORAL Unsup. | Source Baseline | +11.82 pp | [+6.06, +17.59] | [+6.88, +16.82] | $6.03 \times 10^{-4}$ | $0.00085$ | 12 / 15 |
| | 4-Shot Only | Source Baseline | +1.35 pp | [-1.28, +3.99] | [-0.84, +3.69] | $0.2890$ | $0.4212$ | 9 / 15 |
| **`pred_12`** | **CORAL + 4-Shot** | **Source Baseline** | **+11.49 pp** | [+7.52, +15.46] | [+8.06, +15.02] | **$2.27 \times 10^{-5}$** | **$0.00006$** | **15 / 15** |
| | **CORAL + 4-Shot** | **CORAL Unsup.** | **+1.73 pp** | [-0.01, +3.46] | [+0.29, +3.37] | $0.0513$ | $0.0735$ | 10 / 15 |
| | **CORAL + 4-Shot** | **4-Shot Only** | **+7.61 pp** | [+4.09, +11.13] | [+4.55, +10.61] | **$3.88 \times 10^{-4}$** | **$0.00120$** | **13 / 15** |
| | CORAL Unsup. | Source Baseline | +9.76 pp | [+5.36, +14.17] | [+5.82, +13.51] | $3.06 \times 10^{-4}$ | $0.00085$ | 13 / 15 |
| | 4-Shot Only | Source Baseline | +3.88 pp | [+1.21, +6.56] | [+1.37, +6.16] | $0.0077$ | $0.0125$ | 12 / 15 |

---

### Multiple Comparisons and Family-Wise Error Rate Caution
When evaluating across four channel subsets, we test multiple hypotheses. 
- For the primary contrast between **Method D (CORAL + 4-Shot)** and **Method B (CORAL Unsupervised)**:
  - `all_62`: Unadjusted $p = 0.0294$. Bonferroni threshold for 4 channel tests is $\alpha / 4 = 0.0125$. Under strict Bonferroni correction, $p=0.0294$ is nominally significant but does not survive conservative family-wise correction ($p_{\text{adj}} = 0.118$). Benjamini-Hochberg FDR remains under $10\%$.
  - `pred_32`: Unadjusted $p = 0.1269$ (not significant).
  - `stab_32`: Unadjusted $p = 0.3013$ (not significant).
  - `pred_12`: Unadjusted $p = 0.0513$ (marginal).
- **Statistical Conclusion:** The combination of CORAL with 4-shot calibration yields a small (+1.1 to +1.7 pp), directionally positive improvement across all 4 channel conditions, with 10–13 of 15 subjects improving. However, the magnitude is modest, and the effect is statistically robust primarily on the full 62-channel feature space.

---

## 5. Per-Subject Performance Profile

Subject-level accuracies (averaged over the 5 random calibration seeds, 68 trials per seed):

```
Table: Subject-Level Accuracy Comparison on all_62 Channels
```

| Subject ID | Method A (Source) | Method B (CORAL) | Method C (4-Shot) | Method D (CORAL+4-Shot) | Delta (D vs B, pp) | Delta (D vs A, pp) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sub 1** | 39.41% | 57.94% | 43.53% | **58.53%** | +0.59 pp | +19.12 pp |
| **Sub 2** | 39.12% | **60.00%** | 44.41% | 56.18% | -3.82 pp | +17.06 pp |
| **Sub 3** | 45.59% | 52.35% | 40.59% | **52.94%** | +0.59 pp | +7.35 pp |
| **Sub 4** | 40.59% | 62.35% | 42.65% | **64.41%** | +2.06 pp | +23.82 pp |
| **Sub 5** | 30.29% | 58.82% | 34.71% | **61.76%** | +2.94 pp | +31.47 pp |
| **Sub 6** | 46.18% | 47.35% | 49.71% | **50.59%** | +3.24 pp | +4.41 pp |
| **Sub 7** | 55.59% | **73.53%** | 60.29% | 72.35% | -1.18 pp | +16.76 pp |
| **Sub 8** | 35.88% | 48.24% | 31.76% | **51.47%** | +3.24 pp | +15.59 pp |
| **Sub 9** | 38.24% | 49.71% | 42.65% | **50.88%** | +1.18 pp | +12.65 pp |
| **Sub 10** | 37.94% | 54.41% | 40.59% | **56.47%** | +2.06 pp | +18.53 pp |
| **Sub 11** | 44.41% | 50.88% | 43.24% | **52.06%** | +1.18 pp | +7.65 pp |
| **Sub 12** | 40.88% | 48.82% | 46.18% | **50.00%** | +1.18 pp | +9.12 pp |
| **Sub 13** | 41.76% | 55.00% | 46.18% | **55.88%** | +0.88 pp | +14.12 pp |
| **Sub 14** | 44.71% | 62.65% | 55.00% | **64.71%** | +2.06 pp | +20.00 pp |
| **Sub 15** | 42.35% | 59.41% | 54.41% | **60.29%** | +0.88 pp | +17.94 pp |
| **Mean ± SD** | **41.53 ± 6.96%** | **56.10 ± 7.06%** | **45.06 ± 7.46%** | **57.24 ± 6.47%** | **+1.14 ± 1.82 pp** | **+15.71 ± 7.45 pp** |

- **Consistency:** 13 out of 15 subjects showed positive gains for Method D over Method B on `all_62`.
- **Difficult Subjects:** Subjects 6, 8, 9, and 12 remain around 50%–51%, demonstrating strong cross-subject resistance even with 4-shot calibration.
- **Top Performer:** Subject 7 achieves **72.35%**, the highest individual subject accuracy in the project to date, though still shy of 80%.

---

## 6. Class-Level Dynamics and Confusion Analysis

### Per-Class Metrics Comparison on `all_62` Channels

The 4-class emotion distribution is perfectly balanced (1,275 test trials per class across 15 subjects $\times$ 5 seeds $\times$ 17 test trials/class):

| Emotion Class | Metric | Method A (Source) | Method B (CORAL) | Method C (4-Shot) | Method D (CORAL+4-Shot) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Neutral (0)** | Precision | 44.39% | 58.74% | 45.94% | **59.80%** |
| | Recall | 50.59% | 59.06% | 47.53% | **60.08%** |
| | F1-Score | 47.29% | 58.90% | 46.72% | **59.94%** |
| **Sad (1)** | Precision | 36.65% | 54.26% | 38.24% | **56.12%** |
| | Recall | 38.12% | 53.96% | 43.22% | **55.06%** |
| | F1-Score | 37.37% | 54.11% | 40.57% | **55.58%** |
| **Fear (2)** | Precision | 35.87% | 52.83% | 42.33% | **53.92%** |
| | Recall | 32.94% | 47.53% | 37.02% | **50.20%** |
| | F1-Score | 34.34% | 50.04% | 39.50% | **51.99%** |
| **Happy (3)** | Precision | 49.30% | 58.02% | 54.61% | **58.73%** |
| | Recall | 44.47% | **63.84%** | 52.47% | 63.61% |
| | F1-Score | 46.76% | 60.79% | 53.52% | **61.07%** |

### Per-Class Insights
1. **Fear Recall Crosses 50% for the First Time:**
   - In all previous experiments (Phase B baseline: 32.9%; D1 CORAL: 47.5%), Fear was severely depressed and confused with Neutral and Sad.
   - In Method D, **Fear recall reaches 50.20%** (+2.67 pp vs CORAL, +17.26 pp vs Source), demonstrating that a single ground-truth calibration trial for Fear provides an anchor that prevents the classifier from collapsing away from the Fear boundary.
2. **Happy and Neutral Remain Most Discernible:**
   - Happy F1 reaches **61.07%** (recall 63.61%), and Neutral F1 reaches **59.94%** (recall 60.08%).
   - The primary remaining confusions are between negative valence states: **Sad $\leftrightarrow$ Fear** and **Sad $\leftrightarrow$ Neutral**.

---

## 7. Honest Gap Analysis Against the 80% Aspirational Target

### Why Does Four-Shot Calibration Combined with CORAL Plateau at ~57.2%?

| Theoretical Assumption | Observed Experimental Reality on SEED-IV | Gap Mechanism |
| :--- | :--- | :--- |
| **Class Homogeneity** | Cross-session emotional drift within the same subject is substantial. | A single calibration trial from Session 1 cannot capture the electrophysiological manifold of Session 3. |
| **Linear Manifold Shift** | EEG feature representations exhibit non-linear Riemannian geometry. | Second-order linear covariance alignment (CORAL) aligns global ellipses, but cannot bend class boundaries. |
| **Point-Estimate Calibration** | $N_{\text{cal}}=1$ trial per class is vulnerable to trial-specific EEG noise (blink/muscle artifacts, fatigue). | If the single calibration trial contains atypical noise, $w_{\text{cal}}=10$ partially pulls the hyperplane towards an outlier. |
| **Subject Manifold Separation** | Subjects exhibit distinct latent emotion subspace rotations. | Source training data ($N=14$ subjects) contains broad inter-subject variance that overpowers 4 target trials. |

### The Defensible Reality
- Aspirational target: **80.00%**
- Best observed D3 performance: **57.24%** (`all_62`, Method D)
- Shortfall: **22.76 percentage points**
- **Conclusion:** No statistical or algorithmic manipulation can truthfully present 57.2% as 80.0%. The finding that 4-shot calibration combined with CORAL achieves 57.24% (+15.7 pp over unadapted baseline) is scientifically valid, reproducible, and defensible.

---

## 8. Artifacts and Audit Verification

All generated artifacts have been validated and saved to disk:
1. **Predictions:** [experiment_d3_predictions.csv](file:///outputs/predictions/experiment_d3_predictions.csv) (81,600 rows: 15 folds $\times$ 4 methods $\times$ 4 conditions $\times$ 5 seeds $\times$ 68 trials).
2. **Summary Table:** [experiment_d3_summary.csv](file:///outputs/tables/experiment_d3_summary.csv) (16 rows: 4 conditions $\times$ 4 methods with CIs, macro-F1, and balanced accuracy).
3. **Subject Metrics Table:** [experiment_d3_subject_metrics.csv](file:///outputs/tables/experiment_d3_subject_metrics.csv) (240 rows: 15 subjects $\times$ 4 methods $\times$ 4 conditions).
4. **Class Metrics Table:** [experiment_d3_class_metrics.csv](file:///outputs/tables/experiment_d3_class_metrics.csv) (64 rows: 4 classes $\times$ 4 methods $\times$ 4 conditions).
5. **Statistical Comparisons:** [experiment_d3_vs_baselines_comparison.csv](file:///outputs/tables/experiment_d3_vs_baselines_comparison.csv) (20 pairwise statistical comparisons with $t$-test, Wilcoxon, and bootstrap CIs).
6. **Publication Figures:**
   - [experiment_d3_calibration_comparison.png](file:///outputs/figures/experiment_d3_calibration_comparison.png) (grouped bar chart across all conditions and methods with 80% aspirational line).
   - [experiment_d3_confusion_matrix.png](file:///outputs/figures/experiment_d3_confusion_matrix.png) (4-panel confusion matrix comparing Methods A, B, C, D on `all_62`).
7. **Unit Test Suite:** [test_evaluate_few_shot_calibration.py](file:///tests/test_evaluate_few_shot_calibration.py) (6 dedicated unit tests; 119 total passing tests in full test suite).
