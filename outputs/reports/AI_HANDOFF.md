# AI Handoff: Reliable Subject-Independent EEG Emotion Recognition

**Document Purpose:** Compact, fully verified project handoff for another AI assistant or senior researcher resuming work on this codebase. Paste this entire document into a new conversation for complete, hallucination-free project context.

---

## 1. Project Identity & Research Objectives

- **Project Root Directory:** `C:\Users\singh\OneDrive\Desktop\ML Project`
- **Reference Paper:** Valderrama & Sheoran, *“Identifying relevant EEG channels for subject-independent emotion recognition using attention network layers,”* Frontiers in Psychiatry (2025). Available at `research paper.pdf`.
- **Approved Research Question:**
  > *“Can we reduce the number of EEG channels while maintaining subject-independent emotion-recognition performance, and use uncertainty estimation to identify and abstain from unreliable predictions on unseen subjects?”*
- **Primary Objectives:**
  1. **Channel Reduction:** Reduce 62 physical EEG channels to compact subsets ($k \in \{8, 12, 16, 24, 32\}$) while maintaining classification performance on unseen subjects.
  2. **Uncertainty-Aware Selective Prediction:** Calibrate prediction probabilities, estimate prediction uncertainty ($u_{\max}$, entropy), and abstain from unreliable predictions on unseen subjects to increase accuracy on retained predictions.

---

## 2. Dataset Architecture & Verified Statistics

- **Dataset:** SEED-IV (located at `SEED_IV/`).
- **Subjects:** 15 unique participants (7 male, 8 female).
- **Sessions:** 3 independent sessions per subject (all 3 pooled).
- **Trials per Session:** 24 video-clip trials ($6$ trials per emotion class, perfectly balanced).
- **Emotion Classes ($K=4$):** `0: neutral`, `1: sad`, `2: fear`, `3: happy`. Chance accuracy = $25.0\%$.
- **Physical EEG Channels:** 62 electrodes in standard 10-20 montage (`FP1` to `CB2`).
- **Frequency Bands:** 5 per channel (Delta $1-4$ Hz, Theta $4-8$ Hz, Alpha $8-14$ Hz, Beta $14-31$ Hz, Gamma $31-50$ Hz).
- **Feature Representation:** Precomputed smoothed Differential Entropy (`de_LDS`).
- **Features per Sample Window:** $62 \times 5 = \mathbf{310}$ features.
  - Slicing formula: `feature_index = channel_index * 5 + band_index`.
  - Physical channel indivisibility rule: When selecting channel $c$, all 5 bands $\{5c, \dots, 5c+4\}$ are sliced together.
- **Sample Window Counts:**
  - Session 1: 851 windows/subject ($12,765$ total).
  - Session 2: 832 windows/subject ($12,480$ total).
  - Session 3: 822 windows/subject ($12,330$ total).
  - Windows per Subject: $\mathbf{2,505}$ windows.
  - Total Dataset Windows: $\mathbf{37,575}$ feature windows across all subjects and sessions.
  - Data Quality: Zero NaNs, Infs, or missing values verified.

---

## 3. Evaluation Protocol & Anti-Leakage Safeguards

- **Outer Evaluation:** 15-fold Leave-One-Subject-Out (LOSO) cross-validation.
  - In each outer fold $k \in \{1, \dots, 15\}$, test subject $S_k$ ($2,505$ windows) is held out.
  - The remaining 14 subjects ($35,070$ windows) form the outer training set.
- **Strict Leakage Safeguards (All Verified PASS):**
  1. `StandardScaler` is fitted strictly on the 14 outer-training subjects.
  2. Channel selection rankings ($I_c$, $Q_c$) are computed strictly on the 14 outer-training subjects.
  3. Inner cross-validation splits use `GroupKFold` grouped by `subject_id`, never random window splits.
  4. Probability calibration parameters (temperature $T$ or Platt weights) are fitted strictly on out-of-fold decision scores generated across subject-disjoint inner training folds.
  5. The held-out test subject $S_k$ is evaluated strictly using the final model and calibrator.
  6. The codebase avoids `SVC(probability=True)` to prevent ungrouped random window splits.

---

## 4. Phase Completion Status

| Phase | Description | Status | Primary Output Files |
| :---: | :--- | :---: | :--- |
| **Phase A** | SEED-IV Data Loader & Verification | **COMPLETE** | `outputs/tables/dataset_summary.csv` |
| **Phase B** | All-62 Channel LOSO Baseline | **COMPLETE** | `outputs/predictions/baseline_predictions.csv`, `outputs/metrics/baseline_fold_metrics.csv` |
| **Phase C** | Leakage-Safe Channel Reduction | **COMPLETE & AUDITED** | `outputs/predictions/channel_reduction_predictions.csv`, `outputs/tables/channel_reduction_comparison.csv` |
| **Phase D** | Probability Calibration & Uncertainty | **COMPLETE & AUDITED** | `outputs/predictions/phase_d_calibrated_predictions.csv`, `outputs/tables/phase_d_calibration_summary.csv` |
| **Phase E** | Selective Prediction & Abstention | **NOT STARTED** | To be implemented next |

---

## 5. Verified Empirical Results

### A. Classification Performance Across Channel Budgets (Phase C & D)
*Outer 15-Fold LOSO on Genuine SEED-IV Data ($37,575$ windows):*

| Condition | Physical Channels | Channel % | Pooled Acc | Subject Acc (Mean $\pm$ Std) | Pooled Macro-F1 | Subject Macro-F1 (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_62` (Voting Rule)** | 62 | 100.0% | 39.19% | 39.19% $\pm$ 9.63% | 39.44% | 32.53% $\pm$ 12.68% |
| **`all_62` (Margin Argmax)** | 62 | 100.0% | **39.53%** | **39.53% $\pm$ 9.77%** | **39.73%** | **33.11% $\pm$ 12.92%** |
| **`pred_32` (Predictive)** | 32 | 51.6% | 39.28% | 39.28% $\pm$ 8.88% | 39.49% | 34.34% $\pm$ 11.93% |
| **`stab_32` (Stability-Aware)**| 32 | 51.6% | 38.58% | 38.58% $\pm$ 8.81% | 38.82% | 33.34% $\pm$ 11.82% |
| **`pred_12` (Predictive)** | 12 | 19.4% | 38.31% | 38.31% $\pm$ 9.43% | 38.48% | 32.61% $\pm$ 10.55% |
| **`stab_12` (Stability-Aware)**| 12 | 19.4% | 37.54% | 37.54% $\pm$ 9.02% | 37.76% | 32.14% $\pm$ 10.36% |
| **`pred_8` (Predictive)** | 8 | 12.9% | 37.74% | 37.74% $\pm$ 9.98% | 38.01% | 32.74% $\pm$ 11.14% |
| **`stab_8` (Stability-Aware)** | 8 | 12.9% | 36.74% | 36.74% $\pm$ 9.09% | 36.95% | 30.19% $\pm$ 10.75% |

### B. Probability Calibration & Error Detection (Phase D)
*Evaluating Uncalibrated Softmax vs. Group-Safe Temperature Scaling on Unseen Subjects:*

| Condition | Method | Pooled Acc | Pooled Brier | Pooled Top ECE | Subject Top ECE | Subject AUROC ($u_{\max}$) | Mean $T$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all_62`** | `uncalibrated_softmax` | 39.53% | 0.8285 | 27.85% | 28.99% $\pm$ 10.45% | 0.5891 $\pm$ 0.1070 | — |
| **`all_62`** | `group_safe_temperature`| **39.53%** | **0.7171** | **3.74%** | **8.90% $\pm$ 6.61%** | **0.6048 $\pm$ 0.0884** | **4.045** |
| **`pred_32`** | `group_safe_temperature`| 38.90% | 0.7162 | **1.92%** | **8.92% $\pm$ 6.90%** | **0.6230 $\pm$ 0.0929** | 3.877 |
| **`stab_32`** | `group_safe_temperature`| 38.56% | 0.7193 | **1.56%** | **8.26% $\pm$ 7.16%** | **0.6096 $\pm$ 0.0953** | 3.881 |
| **`pred_12`** | `group_safe_temperature`| 38.04% | 0.7182 | **1.32%** | **9.23% $\pm$ 5.57%** | **0.5788 $\pm$ 0.0640** | 3.765 |

- **Temperature Scaling Impact:** Eliminates overconfidence, reducing top-label ECE by **86.6% to 95.5%** while algebraically preserving classification accuracy ($\arg\max p \equiv \arg\max z$).
- **Top Physical Channels Selected Across Folds:** `F4` (100%), `FT7` (100%), `FPZ` (86.7%), `FC1` (80.0%), `F5`, `P5`, `TP8`, `O1` (60.0%).

---

## 6. Discrepancy Resolution: 39.19% vs. 39.53%

- **Finding:** Phase B reported $39.1883\%$; Phase D reported $39.5263\%$ (+127 correct samples).
- **Forensic Truth:** The underlying models, features, outer splits, and scalers are **100% identical**.
- **Root Cause:** Decision rule:
  - Phase B used LibSVM discrete one-vs-one majority voting (`SVC.predict`).
  - Phase D used continuous margin argmax ($\arg\max_c Z_c = \arg\max_c \text{decision\_function}(X)_c$).
  - On $95.16\%$ of samples (35,758 rows), both rules produce **identical predictions**.
  - On the remaining $4.84\%$ (1,817 rows) where binary votes split or tied, continuous decision margins broke ties by distance to margin, yielding $+127$ net correct samples.
- **Reporting Rule:** Document both numbers transparently. Continuous margin argmax is mathematically required for temperature-scaled probabilities.

---

## 7. Key Source Modules & Responsibilities

| File | Primary Functions & Classes | Key Responsibility |
| :--- | :--- | :--- |
| `src/data_loader.py` | `load_seed_iv_dataset()`, `generate_dataset_summary()` | Ingests `.mat` files, validates shapes, generates 310-feature mappings. |
| `src/subject_split.py` | `get_loso_splits()`, `get_inner_group_splits()` | Yields 15 outer LOSO folds and inner `GroupKFold` partitions. |
| `src/preprocessing.py` | `create_baseline_pipeline()` | Builds `Pipeline([('scaler', StandardScaler()), ('svc', SVC())])`. |
| `src/channel_features.py` | `slice_channel_features()`, `get_channel_feature_indices()` | Extracts indivisible 5-band feature blocks per channel. |
| `src/channel_selection.py` | `select_stability_aware_channels()`, `compute_inner_grouped_permutation_importance()` | Computes permutation importance and stability index $Q_c = 0.5 I_{\text{norm}} + 0.5 S_c$. |
| `src/calibration.py` | `generate_group_safe_oof_decision_scores()`, `GroupSafeTemperatureScalingCalibrator` | Generates inner OOF scores; fits temperature $T > 0$ via NLL minimization. |
| `src/uncertainty_metrics.py`| `compute_uncertainty_scores()`, `compute_probability_metrics()`, `compute_expected_calibration_error()` | Computes Brier, NLL, top/classwise ECE, $u_{\max}$, entropy, AUROC/AUPRC. |
| `src/run_calibration.py` | `run_phase_d_calibration()`, `run_single_fold_calibration()` | Executes Phase D across 15 outer LOSO folds using `joblib.Parallel`. |

---

## 8. Critical Warnings: What Must NOT Be Claimed

1. **Do NOT claim formal statistical non-inferiority or equivalence:** Reducing channels to 32 retains $\approx 99\%$ of baseline accuracy, but formal equivalence was not tested with an equivalence margin hypothesis test.
2. **Do NOT claim high pooled AUROC for error detection:** While subject-level error detection achieves AUROC $\approx 0.605 - 0.623$, pooling all 37,575 predictions across all 15 subjects drops AUROC to $\approx 0.518$ due to inter-subject score scale offsets.
3. **Do NOT claim Platt vector scaling is superior to temperature scaling:** Platt scaling perturbs the decision boundary and risks overfitting on 14 training subjects. Temperature scaling is the validated, preferred regime.
4. **Do NOT use `SVC(probability=True)`:** It introduces random window-level splits that leak temporal and subject correlation.
5. **Do NOT use outer test subjects to select abstention thresholds:** In Phase E, thresholds must be fitted strictly on outer-training out-of-fold data.

---

## 9. Immediate Next Action: Phase E Implementation

Implement **Phase E: Uncertainty-Aware Selective Prediction & Abstention Analysis**:
1. Create `src/selective_prediction.py`:
   - Abstention policy: Predict if uncertainty $u_{\max} \le \tau$ (or confidence $\ge 1 - \tau$), else abstain.
   - Threshold determination: Set threshold $\tau(C)$ on outer training subjects' out-of-fold calibration predictions to match target coverage levels $C \in \{1.0, 0.9, 0.8, 0.7, 0.6, 0.5\}$.
2. Create `src/run_selective_prediction.py`:
   - Apply fixed thresholds $\tau(C)$ to the held-out test subject in each outer fold.
   - Compute selective accuracy, selective balanced accuracy, and selective macro-F1 across coverage levels.
   - Evaluate whether stability-aware channel selection (`stab_32`) retains higher accuracy under abstention than predictive selection (`pred_32`) or the 62-channel baseline.
3. Add unit tests in `tests/test_selective_prediction.py`.
