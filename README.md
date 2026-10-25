# Reliable Subject-Independent EEG Emotion Recognition
### Using Stable Channel Selection and Uncertainty-Aware Prediction

[![Audit Status](https://img.shields.io/badge/Audit-Gate%201%20Passed-brightgreen)](#)
[![Python](https://img.shields.io/badge/Python-3.14.4-blue)](#)
[![Compute](https://img.shields.io/badge/Compute-CPU--first-orange)](#)
[![Dataset](https://img.shields.io/badge/Dataset-SEED--IV-purple)](#)

---

## 1. Project Overview & Research Question

This project addresses the domain shift problem in EEG-based affective computing by investigating:

> **“Can we reduce the number of EEG channels while maintaining subject-independent emotion-recognition performance, and use uncertainty estimation to identify and abstain from unreliable predictions on unseen subjects?”**

Both **channel reduction** and **uncertainty-aware selective prediction** are mandatory core objectives:
1. **Channel Reduction:** Evaluates whether emotion classification performance is preserved when selecting compact electrode subsets ($k \in \{8, 12, 16, 24, 32\}$ out of 62) via stability-aware selection compared to random selection and predictive-importance ranking.
2. **Uncertainty & Abstention:** Evaluates whether model confidence (maximum-probability uncertainty and predictive entropy) identifies likely errors on unseen subjects, and whether abstaining on uncertain predictions improves the selective accuracy of retained samples.

### Reference Paper
- **Citation:** Camilo E. Valderrama and Anshul Sheoran (2025). *Identifying relevant EEG channels for subject-independent emotion recognition using attention network layers*. **Frontiers in Psychiatry**, 16:1494369. [doi:10.3389/fpsyt.2025.1494369](https://doi.org/10.3389/fpsyt.2025.1494369).

---

## 2. Locked Experimental Decisions

- **Dataset:** Verified SEED-IV dataset in `SEED_IV/` (15 subjects, 3 sessions, 24 trials/session, 4 emotion classes: Neutral, Sad, Fear, Happy).
- **Primary Feature Representation:** Precomputed `de_LDS` (Differential Entropy smoothed via Linear Dynamic Systems).
- **Sample Granularity:** Individual 4-second sliding EEG windows ($N = 2,505$ per subject; $37,575$ windows dataset-wide).
- **Session Strategy:** All three sessions (1, 2, 3) pooled for the primary benchmark.
- **Evaluation:** Outer 15-fold Leave-One-Subject-Out (LOSO) cross-validation.
- **Classifier:** Support Vector Machine with Radial Basis Function kernel (RBF-SVM).
- **Channel Budgets ($k$):** $k \in \{8, 12, 16, 24, 32\}$ channels, plus the reference 62-channel model.
- **Uncertainty Quantification:** Maximum-probability uncertainty ($u_{\text{max}} = 1 - \max p_k$) and predictive entropy ($H(p) = -\sum p_k \log p_k$).
- **Selective Retention:** Evaluated at 100%, 90%, 80%, 70%, 60%, and 50% coverage levels.
- **Compute:** CPU-first execution (10 physical cores / 12 logical threads, ~16 GB RAM). Zero GPU requirement.

---

## 3. Strict Leakage Safeguards

1. **Outer Test-Subject Isolation:** No data from the held-out subject enters scaling (`StandardScaler`), feature selection, hyperparameter tuning, probability calibration, or threshold selection.
2. **Subject-Grouped Inner Splits:** Inner cross-validation splits use `GroupKFold` grouped by `subject_id`. Correlated windows from the same subject are never split across train and validation.
3. **Subject-Grouped Out-of-Fold Calibration:** Standard `SVC(probability=True)` Platt scaling is prohibited due to within-subject leakage. A group-safe out-of-fold calibration pipeline (`CalibratedClassifierCV` with `GroupKFold`) is enforced.
4. **Physical Channel Feature Grouping:** Electrodes are selected as physical blocks of 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma). A $k$-channel model uses $k \times 5$ features.
5. **Traceability:** Predictions, probabilities, uncertainties, true labels, and sample identifiers (`subject_id`, `session_id`, `trial_id`, `window_id`) are persisted to disk.

---

## 4. Repository Structure

```text
ML Project/
├── config.yaml               # Master experiment configuration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation and guide
├── PROJECT_RULES.md          # Governance, locked decisions, and integrity rules
├── PROJECT_STATUS.md         # Living audit, state, and checkpoint log
├── IMPLEMENTATION (1).md     # Full step-by-step technical implementation plan
├── research paper.pdf        # Valderrama & Sheoran (2025) reference paper
├── SEED_IV/                  # Genuine SEED-IV dataset
│   ├── eeg_feature_smooth/   # 45 .mat files (de_LDS, de_movingAve, psd_LDS, psd_movingAve)
│   ├── eeg_raw_data/         # Raw EEG recordings
│   ├── Channel Order.xlsx    # Official 62-channel sequence (FP1 to CB2)
│   ├── channel_62_pos.locs   # 2D/3D electrode coordinates
│   └── ReadMe.txt            # Dataset documentation and trial labels
├── src/                      # Source code (Phases A–F)
│   └── (scaffolding created; pending Phase A implementation)
├── tests/                    # Unit tests (Phase G)
│   └── (scaffolding created; pending pytest authorization)
├── outputs/                  # Persistent artifacts
│   ├── tables/               # Dataset summaries and metric tables
│   ├── predictions/          # Out-of-subject predictions and uncertainties
│   ├── metrics/              # Error detection and selective risk metrics
│   └── figures/              # Publication figures and risk-coverage plots
└── report/                   # Final research reports
```

---

## 5. Current Status & Next Step

- **Current State:** Scaffolding complete; `config.yaml`, `requirements.txt`, and directory trees created.
- **Pending Dependency:** `pytest` is required for unit tests, pending user authorization before installation.
- **Immediate Next Step:** Upon user approval of scaffolding, proceed to Phase A: implement `src/data_loader.py` to load and validate SEED-IV `de_LDS` features and produce `outputs/tables/dataset_summary.csv`.
