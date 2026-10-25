# Phase E Methodology: Selective Prediction & Uncertainty-Based Abstention

## 1. Executive Summary & Research Framework

This document provides the formal mathematical and methodological specification for **Phase E** of the project:
**Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction**.

The central research hypothesis investigated in Phase E is:
> *Can uncertainty estimation be used to reliably detect and abstain from low-confidence, error-prone predictions on unseen subjects, yielding monotonic increases in classification accuracy on accepted samples while preserving subject independence?*

To answer this question rigorously without data leakage or overoptimistic evaluation artifacts, Phase E establishes two conceptually and operationally distinct evaluation modes:

1. **Mode A: Retrospective Empirical Evaluation (Ranking Capacity)**
   - Evaluates whether uncertainty monotonically ranks error likelihood on held-out test predictions.
   - Computes empirical risk–coverage curves, partial Area Under the Risk-Coverage Curve ($\text{AURC}_{[0.5, 1.0]}$), and canonical full $\text{AURC}_{[0.0, 1.0]}$.
   - Serves as a descriptive diagnostic of uncertainty calibration and error-separation ability on test samples, *not* as an operational deployment policy.

2. **Mode B: Inductive Operational Evaluation (Deployment Policy)**
   - Derives operational abstention thresholds $\tau_\gamma$ strictly from training-only, out-of-fold (OOF) uncertainty scores across the 14 outer-training subjects.
   - Employs group-level cross-fitted temperature calibration across inner splits so that training calibration parameters are completely independent of validation samples.
   - Freezes $\tau_\gamma$ prior to touching the held-out subject, accepts samples satisfying $u(\mathbf{x}) \le \tau_\gamma$, and reports actual achieved coverage without forcing test coverage through post-hoc sorting.

---

## 2. Mathematical Formulations

### 2.1 Multiclass Probabilistic Prediction
For each EEG window sample $\mathbf{x} \in \mathbb{R}^D$ and label $y \in \{0, 1, \dots, K-1\}$ with $K=4$ emotion classes (neutral, sad, fear, happy), the decision-margin vector $\mathbf{z}(\mathbf{x}) \in \mathbb{R}^K$ from a trained Support Vector Machine (SVM) is mapped to a probability vector $\hat{\mathbf{p}}(\mathbf{x}) \in \Delta^{K-1}$:

$$\hat{p}_k(\mathbf{x}; T) = \frac{\exp(z_k(\mathbf{x}) / T)}{\sum_{j=0}^{K-1} \exp(z_j(\mathbf{x}) / T)}$$

where:
- **Uncalibrated Softmax Regime**: $T = 1.0$ fixed.
- **Group-Safe Temperature Scaling Regime**: $T > 0$ optimized strictly on subject-disjoint out-of-fold decision margins $\mathbf{Z}_{\text{OOF}}$ to minimize negative log-likelihood (NLL).

The predicted class is:
$$\hat{y}(\mathbf{x}) = \arg\max_{k \in \{0, \dots, K-1\}} \hat{p}_k(\mathbf{x})$$

Because temperature scaling is a strictly monotonic transformation of $\mathbf{z}(\mathbf{x})$, the argmax is preserved:
$$\arg\max_k \hat{p}_k(\mathbf{x}; T) = \arg\max_k z_k(\mathbf{x}), \quad \forall T > 0$$

### 2.2 Uncertainty Measures
Phase E evaluates two distinct, mathematically bounded uncertainty measures:

1. **Maximum-Probability Uncertainty ($u_{\max}$)**:
   $$u_{\max}(\mathbf{x}) = 1 - \max_{k \in \{0, \dots, K-1\}} \hat{p}_k(\mathbf{x})$$
   - Range: $u_{\max} \in [0, 1 - 1/K] = [0, 0.75]$.
   - Handled extremes: $u_{\max} = 0.0$ for deterministic predictions ($\max \hat{p} = 1.0$); $u_{\max} = 0.75$ for uniform distribution ($\hat{p}_k = 0.25$).

2. **Normalized Predictive Entropy ($H_{\text{norm}}$)**:
   $$H(\hat{\mathbf{p}}) = -\sum_{k=0}^{K-1} \hat{p}_k \ln \hat{p}_k, \quad \text{with } 0 \ln 0 \equiv 0$$
   $$H_{\text{norm}}(\mathbf{x}) = \frac{H(\hat{\mathbf{p}}(\mathbf{x}))}{\ln K} = \frac{-\sum_{k=0}^{K-1} \hat{p}_k \ln \hat{p}_k}{\ln 4}$$
   - Range: $H_{\text{norm}} \in [0.0, 1.0]$.
   - Handled extremes: $H_{\text{norm}} = 0.0$ when $\max \hat{p} = 1.0$; $H_{\text{norm}} = 1.0$ when $\hat{\mathbf{p}} = [0.25, 0.25, 0.25, 0.25]^\top$.

### 2.3 Selective Prediction Decision Rule
Given an uncertainty function $u(\mathbf{x}) \in \mathbb{R}$ and an operational threshold $\tau \in \mathbb{R} \cup \{\infty\}$, the binary acceptance decision $g(\mathbf{x}) \in \{0, 1\}$ is defined by:

$$g(\mathbf{x}) = \begin{cases} 1, & \text{if } u(\mathbf{x}) \le \tau \\ 0, & \text{if } u(\mathbf{x}) > \tau \text{ (abstain)} \end{cases}$$

- **100% Nominal Coverage Policy**: When full coverage is requested ($\gamma = 1.0$), the threshold is explicitly set to $\tau_{1.0} = \infty$. This guarantees $g(\mathbf{x}) = 1$ unconditionally for all test samples, preventing anomalous abstention caused by test uncertainty exceeding the maximum observed training uncertainty.
- **Tie-Breaking Convention**: If multiple samples possess identical uncertainty equal to $\tau$, all tied samples are accepted: $\{i : u_i \le \tau\}$.

### 2.4 Coverage, Selective Accuracy, and Selective Risk
For a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with predictions $\hat{y}_i$ and decisions $g_i = g(\mathbf{x}_i)$:

1. **Empirical Coverage**:
   $$\text{Coverage}(\tau) = \frac{1}{N} \sum_{i=1}^N g_i = \frac{N_{\text{accepted}}}{N}$$

2. **Selective Accuracy**:
   $$\text{Acc}_{\text{sel}}(\tau) = \begin{cases} \frac{\sum_{i=1}^N g_i \cdot \mathbb{I}(\hat{y}_i = y_i)}{\sum_{i=1}^N g_i}, & \text{if } \sum_{i=1}^N g_i > 0 \\ \text{NaN}, & \text{if } \sum_{i=1}^N g_i = 0 \end{cases}$$

3. **Selective Risk (Error Rate on Accepted)**:
   $$\text{Risk}_{\text{sel}}(\tau) = \begin{cases} \frac{\sum_{i=1}^N g_i \cdot \mathbb{I}(\hat{y}_i \ne y_i)}{\sum_{i=1}^N g_i}, & \text{if } \sum_{i=1}^N g_i > 0 \\ \text{NaN}, & \text{if } \sum_{i=1}^N g_i = 0 \end{cases}$$

Notice the mathematical identity:
$$\text{Acc}_{\text{sel}}(\tau) + \text{Risk}_{\text{sel}}(\tau) = 1.0, \quad \forall \tau \text{ with } N_{\text{accepted}} > 0$$

---

## 3. Random Rejection Baseline

A common methodological fallacy in selective prediction literature is concluding that uncertainty estimation is useful simply because selective accuracy increases as coverage decreases. Even random rejection can show stochastic fluctuations or artifacts if sample sizes are small.

To establish whether uncertainty-based abstention provides genuine information:
1. **Label-Agnostic Random Acceptance**: Acceptance masks are drawn uniformly at random without inspecting labels, predictions, or features:
   $$\mathbf{g}_{\text{rand}} \sim \text{Bernoulli}(c) \quad \text{or uniform subsampling of size } \lfloor c \cdot N \rfloor$$
2. **Analytical Expectation**: The expected accuracy of random rejection at any coverage $c > 0$ strictly equals the full-coverage accuracy:
   $$\mathbb{E}[\text{Acc}_{\text{rand}}(c)] = \text{Acc}_{\text{full}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$$
3. **Monte Carlo Empirical Baseline**: For each subject, condition, and coverage level, 500 independent pseudo-random draws are conducted at identical sample counts:
   $$\text{Acc}_{\text{rand, empirical}} = \frac{1}{500} \sum_{r=1}^{500} \text{Acc}_{\text{sel}}(\mathbf{g}^{(r)})$$
   $$\sigma_{\text{rand}} = \sqrt{\frac{1}{499} \sum_{r=1}^{500} \left(\text{Acc}_{\text{sel}}(\mathbf{g}^{(r)}) - \text{Acc}_{\text{rand, empirical}}\right)^2}$$
4. **Improvement Metric**:
   $$\Delta \text{Acc}_{\text{pp}} = (\text{Acc}_{\text{sel}} - \text{Acc}_{\text{rand, empirical}}) \times 100\%$$
   An uncertainty abstention mechanism is considered informative if and only if $\Delta \text{Acc}_{\text{pp}} > 0$ systematically across subjects.

---

## 4. Integration Conventions for AURC

The Area Under the Risk-Coverage Curve ($\text{AURC}$) integrates selective risk across coverage levels. Lower AURC values indicate superior uncertainty ranking (lower risk at high coverage).

### 4.1 Bounded Partial AURC ($\text{AURC}_{[0.5, 1.0]}$)
In practical clinical or operational EEG systems, rejecting more than 50% of windows is infeasible. Therefore, Phase E defines the primary summary metric as the partial integral over coverage $c \in [0.5, 1.0]$:

$$\text{AURC}_{[0.5, 1.0]} = \int_{0.5}^{1.0} \text{Risk}(c) \, dc$$

- **Domain**: $[0.5, 1.0]$ (width = $0.5$).
- **Bounds**: Since $\text{Risk}(c) \in [0.0, 1.0]$, $\text{AURC}_{[0.5, 1.0]} \in [0.0, 0.50]$.
- **Boundary Clamping**: If empirical coverage points do not hit exactly $0.50$ or $1.00$, boundary risk values are interpolated/clamped to ensure exact integration over $[0.5, 1.0]$.

### 4.2 Canonical Full AURC ($\text{AURC}_{[0.0, 1.0]}$)
To allow direct comparison with general machine learning literature (Geifman & El-Yaniv, 2017), the canonical full-range AURC is also reported:

$$\text{AURC}_{[0.0, 1.0]} = \int_{0.0}^{1.0} \text{Risk}(c) \, dc$$

- **Standard Extension Convention**: Since empirical curves terminate at the minimum achievable non-empty coverage $c_{\min} > 0$ (e.g. 1 sample), the curve is horizontally extended to the left from $(c_{\min}, \text{Risk}(c_{\min}))$ to $(0.0, \text{Risk}(c_{\min}))$.
- **Bounds**: $\text{AURC}_{[0.0, 1.0]} \in [0.0, 1.00]$.

---

## 5. Strict Nested Protocol & Leakage Protection

```mermaid
flowchart TD
    subgraph Outer LOSO Split [Outer LOSO Split: 15 Folds]
        OuterTrain[14 Outer Training Subjects]
        OuterTest[1 Outer Held-Out Test Subject]
    end

    subgraph Inner Nested Cross-Validation [Inner Split: 3 Groups on Outer Train]
        InnerTr[Inner Train: ~9 Subjects]
        InnerVal[Inner Val: ~5 Subjects]
        SVM_fit[Fit SVM on Sliced Features]
        SVM_predict[Predict Decision Margins Z_val]
        Z_assemble[Assemble Z_OOF for 14 Training Subjects]
    end

    subgraph Group-Level Cross-Fitted Calibration [Inner Cross-Fitting]
        GKF_fit[Fit Calibrator on 2/3 of Z_OOF]
        GKF_val[Predict Calibrated Probs on 1/3 of Z_OOF]
        Probs_OOF[Assemble Probs_OOF across all 14 subjects]
        U_OOF[Compute Training OOF Uncertainties]
    end

    subgraph Operational Threshold Determination [Frozen Thresholds]
        Quantile[Compute Quantiles of U_OOF: tau_gamma]
        Freeze[Freeze tau_gamma before touching Test Subject]
    end

    subgraph Outer Test Evaluation [Mode B: Inductive Evaluation]
        Outer_SVM[Fit Final Outer Model on 14 Subjects]
        Outer_Z[Predict Test Z on Held-Out Subject]
        Outer_Cal[Calibrate Test Z with Outer Calibrator]
        Apply_Tau[Apply Frozen tau_gamma: g_i = I(u_i <= tau)]
        Report_Results[Report Actual Achieved Coverage & Selective Accuracy]
    end

    OuterTrain --> InnerTr
    InnerTr --> SVM_fit --> SVM_predict --> Z_assemble
    Z_assemble --> GKF_fit --> GKF_val --> Probs_OOF --> U_OOF
    U_OOF --> Quantile --> Freeze
    OuterTrain --> Outer_SVM --> Outer_Z --> Outer_Cal
    Freeze --> Apply_Tau
    Outer_Cal --> Apply_Tau --> Report_Results
    OuterTest -.->|TOUCHED ONLY HERE| Outer_Z
```

### 5.1 Verification of Calibration Independence
In accordance with Phase E3 Section 1:
- The outer test subject's labels and features are completely isolated from feature selection, SVM fitting, temperature calibration, and threshold derivation.
- For threshold estimation, the calibrator is cross-fitted at the group level across the 14 outer-training subjects.
- For each inner validation group, temperature $T$ is fitted strictly using decision margins $\mathbf{Z}_{\text{inner\_oof}}$ and labels from the *other* inner training subjects.
- Training thresholds $\tau_\gamma$ are derived from these cross-fitted out-of-fold uncertainties, guaranteeing zero self-calibration bias.
- The outer-fold calibrator applied to the test subject is fitted on the pooled out-of-fold scores of the 14 training subjects, as established in Phase D.
