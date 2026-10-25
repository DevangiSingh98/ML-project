# Project Rules and Governance: Reliable Emotion Recognition with Reduced EEG Channels

## 1. Approved Research Objective

The project must strictly investigate and answer the following core research question:

> **“Can we reduce the number of EEG channels while maintaining subject-independent emotion-recognition performance, and use uncertainty estimation to identify and abstain from unreliable predictions on unseen subjects?”**

- **Mandatory Dual Scope:** Both channel reduction and uncertainty-aware selective prediction are mandatory core research objectives. Neither component may be omitted, replaced, or treated merely as an optional preprocessing or postprocessing detail.
- **Task Definition:** 4-class emotion classification (Neutral, Sad, Fear, Happy) in a strict subject-independent (cross-subject) evaluation protocol.

---

## 2. Locked Primary Experiment Decisions

All experiments in the core pipeline must conform to these twelve locked decisions:

1. **Dataset:** The genuine, verified SEED-IV dataset already present in the workspace (`SEED_IV/`). No synthetic or unapproved datasets may be substituted for research findings.
2. **Features:** Precomputed `de_LDS` (Differential Entropy smoothed via Linear Dynamic Systems) as the primary feature representation.
3. **Sample Granularity:** Individual 4-second EEG window segments, preserving `subject_id`, `session_id`, `trial_id`, and `window_id` for every sample.
4. **Session Handling:** Pool all three sessions (Sessions 1, 2, and 3) for the primary experiment ($851 + 832 + 822 = 2,505$ windows per subject; $37,575$ total windows across 15 subjects).
5. **Evaluation Protocol:** Outer Leave-One-Subject-Out (LOSO) cross-validation across all 15 subjects, holding out all sessions, trials, and windows for one subject at a time in each outer fold.
6. **Classifier Model:** Support Vector Machine with Radial Basis Function kernel (RBF-SVM).
7. **Channel Budgets ($k$):** Predeclared budgets of $k \in \{8, 12, 16, 24, 32\}$ channels, plus the reference model with all 62 channels.
8. **Channel Selection Methods:**
   - *All Channels (62)*: Upper baseline.
   - *Random $k$-channel selection*: Repeated random sampling (distribution baseline).
   - *Predictive-Importance selection*: Grouped permutation importance evaluated on training-subject validation splits.
   - *Stability-Aware selection*: Combined predictive utility and selection stability score ($Q_c = \alpha I_c + \beta S_c$) computed across inner subject-disjoint splits.
9. **Uncertainty Quantification:**
   - Maximum-probability uncertainty: $u_{\text{max}} = 1 - \max_{k} p_k$.
   - Predictive entropy: $H(p) = -\sum_{k=1}^K p_k \log p_k$.
10. **Selective Prediction:** Evaluate abstention at coverage levels of 100% (full retention), 90%, 80%, 70%, 60%, and 50% retained samples based on uncertainty ranking.
11. **Compute Constraint:** CPU-first execution on existing multi-core hardware (10 cores / 12 threads, ~16 GB RAM). No CUDA GPU dependency.
12. **Synthetic Data Constraint:** Synthetic or dummy data may be utilized exclusively for isolated unit test fixtures and smoke tests. Synthetic data must never be presented as empirical research results or benchmark evidence.

---

## 3. Mandatory Leakage Safeguards

To ensure absolute scientific validity and prevent data leakage under distribution shift, the following safeguards are strictly enforced:

1. **Outer Test-Subject Isolation:**
   All data from the held-out subject must remain completely unseen during:
   - Feature scaling and normalization (`StandardScaler`);
   - Channel ranking and channel selection;
   - Hyperparameter optimization and model selection;
   - Probability calibration;
   - Uncertainty threshold determination.
2. **Subject-Grouped Inner Validation:**
   Inner cross-validation splits (used for hyperparameter tuning and stability calculation) must strictly group by `subject_id` (`GroupKFold`). Random splitting of EEG windows across folds is strictly prohibited because windows from the same subject are correlated.
3. **Preservation of Hierarchical Metadata:**
   Every feature vector must retain its complete lineage: `subject_id`, `session_id`, `trial_id`, and `window_id` to enable exact auditability and grouping.
4. **Subject-Grouped Out-of-Fold Probability Calibration:**
   Scikit-learn's standard `SVC(probability=True)` uses internal random `StratifiedKFold`, which leaks correlated windows from the same subject into Platt scaling. Group-safe calibration must be used instead:
   - Decision-function calibration using `CalibratedClassifierCV(estimator=SVC(probability=False), method='sigmoid', cv=GroupKFold(n_splits=...))` with `groups=train_subject_ids`, or an explicit out-of-subject cross-fitted calibrator.
5. **Training-Only Scaling:**
   `StandardScaler` must be fitted exclusively on the outer training subjects and applied to the outer test subject without re-fitting.
6. **Physical Channel Feature Grouping:**
   Channel selection must operate on physical electrodes. Selecting channel $c$ selects all 5 associated frequency bands (Delta, Theta, Alpha, Beta, Gamma) as a single group ($k \times 5$ features). Individual frequency bands must never be selected in isolation under the guise of physical channel reduction.
7. **Complete Output Persistence:**
   All out-of-subject predictions, class probability vectors, uncertainty metrics, true labels, predicted labels, and fold identifiers must be persisted to disk in machine-readable format (`outputs/predictions/`).

---

## 4. Research Integrity and Verification Guidelines

- **No Invention of Results or Citations:** Never fabricate metrics, simulate experiment outputs, or cite non-existent publications. Every claim must trace directly to saved output files or verified peer-reviewed literature.
- **Reporting of Null or Negative Results:** A negative result (e.g., stability-aware selection not outperforming simple ranking, or uncertainty failing to separate errors at low budgets) is scientifically valuable if rigorously obtained. Report the observed evidence without alteration.
- **No Premature Implementation:** Pipeline code scaffolding, model training, and file generation must proceed in accordance with the gated checkpoints defined in the implementation plan.
