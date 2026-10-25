# Implementation Plan: Reliable Emotion Recognition with Reduced EEG Channels

## 1. Purpose and project definition

**Working title:** Reliable Subject-Independent EEG Emotion Recognition
Using Stable Channel Selection and Uncertainty-Aware Prediction

**Reference paper:** *Identifying relevant EEG channels for
subject-independent emotion recognition using attention network layers*
(2025).

### Main research question

Can we reduce the number of EEG channels while maintaining
subject-independent emotion-recognition performance, and use uncertainty
estimation to identify and abstain from unreliable predictions on unseen
subjects?

### What this project combines

This project has two connected components:

1.  **Channel reduction:** Select a compact subset of EEG channels that
    is useful for emotion classification and stable across training
    subjects.
2.  **Uncertainty-aware prediction:** Estimate which predictions are
    likely to be wrong for unseen subjects, and evaluate whether
    abstaining on uncertain predictions improves the reliability of the
    predictions retained.

Emotion classification remains the central task. Channel reduction is
not optional or a future add-on; it is one of the main experimental
contributions. Uncertainty estimation is evaluated both with all
channels and with reduced-channel models.

### Important research-integrity note

There is no honest way to guarantee a specific scientific result or
dataset approval timeline. This plan instead reduces project risk
through early feasibility checks, a minimum complete experiment,
explicit fallback decisions, and strict checkpoints. A negative result
can still be academically valid if the method and evaluation are
correct. Never invent metrics or claim experiments ran when they did
not.

The novelty claim must be checked against prior work on channel
selection, cross-subject EEG emotion recognition, uncertainty
estimation, calibration, and selective prediction. The project should
not claim that these individual techniques are new. The candidate
contribution is their rigorous combination and evaluation under
subject-independent testing.

------------------------------------------------------------------------

## 2. Scope: what must be completed

### Minimum complete project

The project is considered technically complete only when all of the
following have been done on a real, documented EEG dataset:

-   [ ] Load features, emotion labels, subject IDs, and channel-feature
    mappings.
-   [ ] Validate the data and document the sample/session/trial
    structure.
-   [ ] Run a subject-independent all-channel classification baseline.
-   [ ] Implement at least two channel-reduction baselines and the
    proposed stability-aware selection method.
-   [ ] Evaluate several predeclared channel budgets.
-   [ ] Calculate uncertainty scores from held-out predictions.
-   [ ] Evaluate error detection and selective prediction.
-   [ ] Compare all-channel and reduced-channel systems.
-   [ ] Report results by held-out subject.
-   [ ] Run tests for leakage, channel mapping, and metric calculations.
-   [ ] Generate figures, tables, methodology, results, limitations, and
    reproduction instructions.

### Do not add these until the minimum project works

-   Large deep-learning architectures.
-   Multimodal fusion.
-   Real-time EEG acquisition.
-   Clinical claims.
-   A complex custom neural uncertainty model.
-   Extra datasets before the first dataset pipeline works.

------------------------------------------------------------------------

## 3. Dataset decision and feasibility gate

### Preferred dataset: SEED-IV

Use the official dataset page:
https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html

SEED-IV is preferred because it is used by the reference paper and has
four emotion classes. Access may require an application and licence.
Apply for access immediately; do not assume approval will arrive by a
particular date.

### Required data elements

The dataset must provide, or document how to obtain:

-   EEG recordings or precomputed EEG features;
-   emotion labels;
-   subject IDs;
-   session IDs, where available;
-   trial/window identifiers, where available;
-   channel ordering;
-   a mapping from every feature to its source EEG channel;
-   dataset documentation and licence.

**Locked feature choice:** Precomputed Differential Entropy features smoothed via Linear Dynamic Systems (`de_LDS`) from the genuine SEED-IV dataset are locked as the primary representation.

- **Granularity:** Individual 4-second EEG window segments, preserving `subject_id`, `session_id`, `trial_id`, and `window_id` for every sample.
- **Session Pooling:** All three experimental sessions (1, 2, 3) are pooled for the primary experiment ($851 + 832 + 822 = 2,505$ windows per subject; $37,575$ total windows across 15 subjects).
- **Channels & Dimension:** 62 channels $\times$ 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma) = 310 features per sample.

### Feasibility Gate 1 --- Status: PASSED (Verified 2026-10-17)

All six feasibility criteria were verified on real data:
1. Genuine SEED-IV dataset is present locally in `SEED_IV/` and fully accessible.
2. All documentation (`ReadMe.txt`, `Channel Order.xlsx`, `SEED-IV_stimulation.xlsx`, `channel_62_pos.locs`) was audited.
3. All 45 `.mat` feature files across Sessions 1, 2, and 3 loaded cleanly with 24 trials each.
4. Discrete emotion labels (0: Neutral, 1: Sad, 2: Fear, 3: Happy) and subject IDs (1 to 15) are verified and balanced.
5. The 62 physical channels map 1:1 to indices 0..61, with 5 frequency bands per channel.
6. Strict subject-disjoint outer splits (LOSO) across all 15 subjects are supported.

Synthetic data is permitted only for isolated unit test fixtures and must never be reported as research evidence.

------------------------------------------------------------------------

## 4. Experimental design at a glance

The evaluation must answer four questions:

1.  How well does the classifier recognize emotions for unseen subjects
    when using all channels?
2.  Can channel reduction preserve useful classification performance,
    and does stability-aware selection improve on simpler selection
    methods?
3.  Can uncertainty identify likely errors for unseen subjects?
4.  Do uncertainty and abstention remain useful when the classifier uses
    fewer channels?

Use **leave-one-subject-out (LOSO)** evaluation if supported by the
dataset. For each outer fold, one subject is held out entirely. All
channel selection, scaling, hyperparameter tuning, and calibration must
use only the remaining training subjects.

### Critical leakage rules

-   Never use the held-out subject to select channels or the channel
    count.
-   Never fit scaling on the full dataset.
-   Never tune hyperparameters or abstention thresholds on held-out
    labels.
-   Preserve subject, session, trial, and window IDs.
-   Do not treat correlated windows from the same person as independent
    subjects.
-   Fit any probability calibration using training-subject validation
    data only.
-   Record the subject IDs in every train/test fold.

Add an assertion equivalent to:

``` python
assert set(train_subject_ids).isdisjoint(test_subject_ids)
```

------------------------------------------------------------------------

## 5. Tools and implementation choices

Use a CPU-friendly Python stack:

-   Python;
-   NumPy and pandas;
-   scikit-learn;
-   SciPy if required by the data;
-   Matplotlib;
-   pytest;
-   PyYAML for configuration.

Start with an RBF-SVM on CPU. It is suitable for a reproducible, computationally efficient baseline and avoids the overhead of deep neural architectures.

**Mandatory Calibration Safeguard:**
Do NOT use default scikit-learn `SVC(probability=True)` for probability estimation. Default `SVC(probability=True)` uses internal random `StratifiedKFold`, which leaks correlated EEG windows from the same subject across training and calibration folds, producing artificially optimistic calibration parameters.
Instead, use a **subject-grouped out-of-fold calibration procedure**:
- Fit `SVC(probability=False)` using `CalibratedClassifierCV` configured with `cv=GroupKFold(...)` where `groups=train_subject_ids`, or
- Generate out-of-fold decision-function scores across the 14 training subjects via `GroupKFold`, fit Platt scaling / isotonic regression on these out-of-subject scores, and apply the calibrator to the decision function of the final uncalibrated SVM trained on all 14 training subjects.
- Apply this group-calibrated model to the held-out 15th test subject to produce leak-free predicted probabilities and uncertainty metrics.

------------------------------------------------------------------------

## 6. Repository structure

Create a clean, modular repository. Inspect existing files before
creating or overwriting anything.

``` text
eeg-reliable-emotion/
├── README.md
├── IMPLEMENTATION.md
├── requirements.txt
├── config.yaml
├── PROJECT_STATUS.md
├── data/
│   └── README.md
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── subject_split.py
│   ├── channel_features.py
│   ├── channel_selection.py
│   ├── train_baseline.py
│   ├── uncertainty.py
│   ├── selective_prediction.py
│   ├── metrics.py
│   ├── run_experiments.py
│   └── make_figures.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_subject_split.py
│   ├── test_channel_selection.py
│   ├── test_uncertainty.py
│   └── test_metrics.py
├── outputs/
│   ├── predictions/
│   ├── tables/
│   ├── figures/
│   └── logs/
└── report/
    ├── methodology.md
    ├── results.md
    └── research_gap.md
```

Do not place licensed dataset files or private credentials in a public
repository.

------------------------------------------------------------------------

## 7. Implementation Phase A --- data loader and validation

Implement `src/data_loader.py`.

### Tasks

1.  Inspect the actual dataset layout and documentation.
2.  Load features or recordings, labels, subject IDs, session IDs, and
    trial/window IDs where available.
3.  Validate dimensions and class labels.
4.  Count subjects, sessions, trials, samples, channels, and classes.
5.  Check missing values and duplicate identifiers.
6.  Preserve the mapping between each feature and EEG channel.
7.  Save a dataset summary table and a clear report of any exclusions.

If features are grouped by channel and frequency band, document the
exact mapping. Channel reduction is invalid if the implementation cannot
tell which features belong to which channel.

### Deliverables

-   A working loader for the actual dataset.
-   `outputs/tables/dataset_summary.csv`.
-   A dataset inventory in `PROJECT_STATUS.md`.
-   Tests for label validation, IDs, and channel-feature mapping.

**Exit criterion:** A real sample loads successfully and the loader
returns verified labels, subject IDs, and channel mappings. Do not
proceed to claim a valid experiment until this works.

------------------------------------------------------------------------

## 8. Implementation Phase B --- subject-independent baseline

Implement `src/preprocessing.py`, `src/subject_split.py`, and
`src/train_baseline.py`.

### Procedure for each outer LOSO fold (15 folds total)

1.  Hold out one subject entirely (all 3 pooled sessions, 2,505 windows) as the outer test subject.
2.  The remaining 14 training subjects form the outer training set ($14 \times 2,505 = 35,070$ windows).
3.  Fit feature scaling (`StandardScaler`) exclusively on the 14 training subjects.
4.  Tune SVM hyperparameters using inner validation with `GroupKFold` grouped strictly by `subject_id` (never randomly splitting windows across folds).
5.  Perform subject-grouped out-of-fold probability calibration across the 14 training subjects (e.g. `CalibratedClassifierCV` with `GroupKFold`), ensuring zero within-subject calibration leakage.
6.  Fit the selected pipeline and group-safe calibrator on outer training data only.
7.  Predict the held-out subject, generating calibrated class probabilities, predicted labels, true labels, and uncertainty metrics ($u_{\text{max}}$, $H(p)$).
8.  Save out-of-subject predictions and repeat for all 15 subjects.

Use the exact same outer folds and subject groupings for all channel-selection methods and channel budgets.

### Metrics

-   Accuracy.
-   Balanced accuracy.
-   Macro-F1.
-   Per-class precision and recall.
-   Confusion matrix.
-   Per-held-out-subject results.

### Deliverables

-   Out-of-subject predictions for every eligible subject.
-   A baseline metrics CSV.
-   A baseline performance figure.
-   Passing tests for subject-disjoint evaluation.

**Exit criterion:** The full baseline runs on the real dataset, and
there is no overlap between train and test subjects in any outer fold.

------------------------------------------------------------------------

## 9. Implementation Phase C --- channel reduction

Implement `src/channel_features.py` and `src/channel_selection.py`.

Channel reduction must be evaluated as a central part of the project,
not just described in the report.

### 9.1 Channel budgets

Use a predeclared set such as:

`k = 8, 12, 16, 24, 32`, plus all available channels.

Adjust the list only if the dataset has a different channel count. Do
not select the best channel count using the held-out subject.

### 9.2 Required selection methods

Implement at least these methods:

1.  **All channels:** reference model.
2.  **Random k-channel selection:** repeat multiple random selections
    and report the distribution.
3.  **Predictive-importance selection:** rank channels using
    training-only permutation importance or another documented
    training-only method.
4.  **Stability-aware selection:** repeat channel ranking across
    subject-disjoint splits of the outer training data and prioritize
    channels that are both useful and consistently selected.

If time is constrained, implement random selection first, then
predictive-importance selection, then stability-aware selection. Do not
skip the all-channel baseline.

### 9.3 Group features by physical channel

If each EEG channel contributes multiple features, such as several DE
frequency-band values, treat all those features as a single channel
group. Do not select individual frequency features and claim that
physical channels were removed.

### 9.4 Predictive importance

A feasible method is grouped permutation importance: - train the model
on training data; - permute all features associated with one channel
together on validation data; - measure the change in validation
performance; - repeat across training-subject validation splits.

A larger performance decrease indicates greater predictive usefulness in
that fitted model, not causal importance.

### 9.5 Stability score

For channel `c`, calculate: - `I_c`: normalized predictive importance
averaged across training-subject validation splits; - `S_c`: normalized
selection frequency or rank consistency across those splits.

Use a simple initial score:

`Q_c = alpha * I_c + beta * S_c`

Normalize both terms before combining them. Select the top `k` channels
by `Q_c`. Choose the weights using inner validation or predeclare them
before testing. Never use the held-out subject.

Avoid adding more score terms until the simpler version has been
implemented and evaluated.

### 9.6 Fair comparison

For every outer fold and channel budget: - select channels from outer
training data only; - fit preprocessing and the classifier using outer
training data only; - evaluate on the same held-out subject; - use the
same channel budgets and outer folds for every method; - save selected
channel IDs and rankings.

### Deliverables

-   Performance versus channel-count table and figure.
-   Channel-set stability table.
-   Saved channel rankings per outer fold.
-   Comparison of random, predictive-importance, stability-aware, and
    all-channel models.

**Exit criterion:** At least one reduced-channel method runs correctly
end to end, and the full comparison can be extended to the remaining
methods.

------------------------------------------------------------------------

## 10. Implementation Phase D --- uncertainty estimation

Implement `src/uncertainty.py`.

Run uncertainty analysis for both the all-channel model and the
reduced-channel models.

### Method 1: maximum-probability uncertainty

For class probabilities `p_k`:

`u_max = 1 - max(p_k)`

A larger value means lower confidence.

### Method 2: predictive entropy

`H(p) = -sum(p_k * log(p_k))`

Handle zero probabilities safely. Optionally normalize entropy by
`log(K)` for `K > 1`.

For every out-of-subject sample, save: - true label; - predicted
label; - probability vector; - both uncertainty scores; -
subject/session/trial/window IDs where available; - fold ID; -
channel-selection method; - number and IDs of selected channels.

Do not calculate the research metrics on training predictions. Use
predictions for the held-out subjects.

### Deliverables

-   A prediction-level CSV.
-   Tests for uncertainty calculations using hand-checkable probability
    vectors.
-   A report on whether uncertainty scores tend to be higher for errors
    than for correct predictions.

**Exit criterion:** Both uncertainty scores are calculated correctly for
real held-out predictions.

------------------------------------------------------------------------

## 11. Implementation Phase E --- error detection and selective prediction

Implement `src/metrics.py` and `src/selective_prediction.py`.

### 11.1 Error detection

Define an error indicator:

`error_i = 1 if predicted_label_i != true_label_i else 0`

Evaluate whether uncertainty ranks errors above correct predictions.

Report: - AUROC for error detection; - AUPRC for error detection; -
error rate across uncertainty-ranked groups; - per-subject metrics where
defined.

Handle cases where a fold has only correct predictions or only errors;
metrics may be undefined in those cases. Do not silently replace
undefined values with zero.

### 11.2 Selective prediction

Sort held-out predictions from least uncertain to most uncertain.
Evaluate retention at 90%, 80%, 70%, 60%, and 50% coverage, plus the
100% coverage baseline.

For each coverage level report: - number of samples retained; -
coverage; - selective risk, defined as the fraction of retained
predictions that are wrong; - retained accuracy; - retained balanced
accuracy and macro-F1 where defined.

Plot risk versus coverage for both uncertainty measures.

These coverage levels are evaluation points. Do not use held-out labels
to choose an optimal abstention threshold. Any threshold intended for
use beyond the experiment must be selected using training-subject
validation data only.

### 11.3 Joint channel-reduction and uncertainty experiment

This is essential. Compare: 1. All channels, no abstention. 2. Reduced
channels, no abstention. 3. All channels with uncertainty-based
abstention. 4. Reduced channels with uncertainty-based abstention.

This directly evaluates whether uncertainty remains useful when fewer
channels are available.

### Deliverables

-   Error-detection metrics table.
-   Risk--coverage plots.
-   Coverage and selective-risk tables.
-   Comparison across all-channel and reduced-channel models.
-   Per-subject results.

**Exit criterion:** The complete experiment produces real results and
figures from held-out predictions.

------------------------------------------------------------------------

## 12. Implementation Phase F --- analysis and statistical reporting

Implement `src/make_figures.py` and complete the report.

### Required figures

1.  Dataset class distribution.
2.  All-channel versus reduced-channel performance.
3.  Performance versus number of selected channels.
4.  Channel-selection stability across folds.
5.  Error rate by uncertainty quantile.
6.  Risk--coverage curves for both uncertainty measures.
7.  Per-subject performance.
8.  Combined channel-count and coverage comparison.

### Required tables

-   Dataset summary.
-   Baseline metrics.
-   Metrics by channel-selection method and channel budget.
-   Channel stability.
-   Error-detection metrics.
-   Selective-prediction metrics.
-   Per-subject results.

### Statistical care

-   Report pooled sample-level metrics and subject-level summaries where
    appropriate.
-   Use paired subject-level comparisons where feasible.
-   Include confidence intervals when feasible.
-   Do not treat EEG windows as independent subjects.
-   Do not claim statistical significance without an appropriate test.
-   Do not claim the method works better unless the results support that
    conclusion.

### Interpretation questions

The report must answer: 1. How much performance is retained when
channels are reduced? 2. Is stability-aware selection more consistent
than simpler methods? 3. Does uncertainty detect errors on unseen
subjects? 4. Does abstention reduce the error rate of retained
predictions? 5. Does uncertainty remain useful after channel reduction?
6. Which subjects or conditions cause failure? 7. What does the
experiment not establish?

------------------------------------------------------------------------

## 13. Implementation Phase G --- tests and reproducibility

Create tests for: - dataset parsing and label validation; -
channel-feature mapping; - subject-disjoint splitting; - training-only
scaling; - channel selection using only outer training data; - exact
selected-channel counts; - uncertainty formulas; - selective coverage
and risk; - missing classes and undefined metrics; - output generation
and configuration handling.

Synthetic data may be used for unit tests and smoke tests only. Label it
clearly and never report synthetic-data metrics as EEG research results.

Create a configuration file containing: - dataset path; - feature
type; - random seeds; - outer split strategy; - channel budgets; -
selection methods; - classifier and hyperparameters; - coverage
levels; - output directories.

Record configuration and software versions with the results. The final
command should reproduce the pipeline from the documented data location.

------------------------------------------------------------------------

## 14. Research-gap review

Create `report/research_gap.md`.

Review reliable academic work on: - subject-independent EEG emotion
recognition; - channel selection and channel stability; - uncertainty
estimation for EEG classifiers; - calibration under distribution
shift; - selective classification and abstention.

Verify article titles, publication details, and DOI or URLs. Compare the
closest prior methods with the actual proposed experiments. Do not
invent references.

State the contribution narrowly: the study evaluates the interaction
between stable channel reduction and uncertainty-aware selective
prediction in a subject-independent protocol. Revise this framing if the
literature review shows that the exact combination has already been
studied.

------------------------------------------------------------------------

## 15. Risk-Controlled Work Plan

This schedule is a sequence of checkpoints, not a promise that external
access or every experiment will finish on a fixed date. The priority is
to secure a valid complete core experiment before adding optional work.

### Checkpoint 1 --- Data feasibility

-   Confirm data access.
-   Load a real sample.
-   Verify labels, subjects, and channel mapping.

**If this fails:** resolve access or obtain instructor approval for a
documented alternative dataset. Do not spend days building assumptions
about an unknown file structure.

### Checkpoint 2 --- Baseline feasibility

-   Run one subject-disjoint fold.
-   Verify no leakage.
-   Save predictions.

**If this fails:** stop adding research features and fix the loader,
labels, and split logic first.

### Checkpoint 3 --- Minimum channel-reduction result

-   Implement all-channel and random-channel baselines.
-   Implement predictive-importance selection.
-   Evaluate at a small set of channel budgets.

**If time is limited:** complete these correctly before adding
stability-aware scoring.

### Checkpoint 4 --- Minimum uncertainty result

-   Calculate maximum-probability uncertainty and entropy.
-   Evaluate error detection.
-   Generate one risk--coverage curve.

**If time is limited:** retain both required uncertainty measures if
feasible, but prioritize correct held-out predictions and metrics over
elaborate plotting.

### Checkpoint 5 --- Integrated experiment

-   Compare all-channel and reduced-channel models.
-   Evaluate uncertainty and abstention for both.
-   Produce the primary results table.

### Checkpoint 6 --- Final quality assurance

-   Rerun tests.
-   Check the output files.
-   Ensure every result in the report is supported by saved experiment
    outputs.
-   Document unfinished work honestly.

------------------------------------------------------------------------

## 16. Final deliverables

The project is ready for submission when it contains:

-   source code and dependency list;
-   configuration and reproducible run instructions;
-   dataset inventory and documentation of preprocessing;
-   subject-independent baseline results;
-   channel-reduction methods and channel-budget comparisons;
-   uncertainty and error-detection results;
-   selective-prediction and risk--coverage results;
-   joint comparison of all-channel and reduced-channel models;
-   tests and actual test outcomes;
-   figures and machine-readable tables;
-   methodology, results, limitations, and research-gap reports;
-   a clear list of completed experiments and any remaining blockers.

The final report must distinguish between a successful implementation
and a successful hypothesis. The implementation can be correct even if
stability-aware channel selection does not improve accuracy or
uncertainty fails to detect errors reliably. Report the evidence, not
the hoped-for outcome.

## 17. Final principle

Keep the project focused on the combined question:

**Can we reduce the number of EEG channels while maintaining
subject-independent emotion-recognition performance, and use uncertainty
estimation to identify and abstain from unreliable predictions on unseen
subjects?**

The channel-selection experiment, emotion-classification experiment,
uncertainty experiment, and combined experiment must all be present. Do
not let uncertainty estimation replace channel reduction, and do not let
channel reduction become merely a preprocessing detail.
