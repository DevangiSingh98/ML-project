# Final Defense Notes: Reliable Subject-Independent EEG Emotion Recognition

**Project Title:** Reliable Subject-Independent EEG Emotion Recognition Using Stable Channel Selection and Uncertainty-Aware Prediction  
**Dataset:** SEED-IV (15 subjects, 3 sessions, 37,575 epochs, 4 emotion classes)  
**Evaluation Protocol:** 15-fold outer Leave-One-Subject-Out (LOSO) cross-validation  
**Date:** October 26, 2026

---

## 1. Two-Minute Project Pitch (Spoken Defense Summary)

> "Subject-independent EEG emotion recognition faces two fundamental bottlenecks: high-density electrode montages (such as 62 channels) are impractical for real-world deployment, and cross-subject neural variability leads to high error rates on unseen users. Our project addresses both challenges directly.
> 
> Using the benchmark SEED-IV dataset—comprising 37,575 EEG window segments across 15 subjects evaluated under rigorous 15-fold outer Leave-One-Subject-Out cross-validation—we investigated whether we could reduce channel count while maintaining classification performance, and use calibrated uncertainty estimation to identify and abstain from unreliable predictions on unseen subjects.
> 
> First, our channel-reduction findings demonstrate that reducing electrodes from 62 down to 32 stable channels (`stab_32`) or predictive channels (`pred_32`) incurs minimal degradation in baseline accuracy—achieving 38.6% and 38.9% compared to 39.5% for the full 62-channel montage against a 25% chance level. Even a minimal 12-channel montage (`pred_12`) retains 38.0% baseline accuracy.
> 
> Second, we implemented a leakage-safe selective prediction framework. In inductive operational evaluation—where decision thresholds are derived strictly from training out-of-fold uncertainty quantiles and frozen prior to evaluating the held-out subject—we compared uncertainty abstention against 500-repetition Monte Carlo random rejection.
> 
> Our findings are nuanced and scientifically honest: within individual unseen subjects, predictive uncertainty effectively identifies errors, boosting individual mean accuracy from 38.6% up to 47.2% as coverage decreases. However, transferring static threshold policies across subjects encounters significant domain shift: under temperature scaling, nominal 70% coverage actually accepts ~89%–92% of test samples, and improvements over random rejection do not survive family-wise Holm-Bonferroni multiple testing correction across the full 20-condition matrix ($p_{\text{Holm}} \ge 0.165$).
> 
> In conclusion, our evidence demonstrates that channel reduction is viable and uncertainty reliably ranks error risk within individuals, but operational cross-subject thresholding requires subject-adaptive confidence normalization to guarantee target coverage in real-world BCIs."

---

## 2. The Three Strongest Defensible Findings

1. **Electrode Reduction Incurs Minimal Performance Loss:**
   - Reducing the EEG array from 62 channels down to 32 channels (via stability-aware selection `stab_32` or predictive ranking `pred_32`) incurs negligible loss in subject-independent accuracy (38.56% and 38.90% vs. 39.53% for `all_62`).
   - Even a sparse 12-channel montage (`pred_12`) maintains 38.04% baseline accuracy, demonstrating that 50%–80% of electrodes can be eliminated without compromising general classification ability.
2. **Predictive Uncertainty Effectively Identifies Subject-Level Errors:**
   - Within individual unseen subjects, high predictive uncertainty correlates strongly with misclassification.
   - In inductive operational testing (Mode B), abstaining from high-uncertainty samples drives subject-level selective accuracy up from ~38.6% to **45.60%** for `pred_32` and **47.15%** for `stab_32` at low coverage budgets, with 10–12 of the 15 subjects improving over random rejection.
3. **Subject-Level Independence Resolves Simpson's Paradox:**
   - In pooled evaluation (Mode A), global confidence sorting produces an AURC that appears nearly random ($0.6139$ vs. $0.6144$ random reference for `stab_32`).
   - When evaluated at the subject level, mean Full AURC drops to **$0.5628 \pm 0.0855$** (a reduction of 0.0516 in risk area below random), proving that pooling EEG windows across individuals obscures valid within-subject error detection.

---

## 3. The Most Important Limitations

1. **Modest Baseline Subject-Independent Accuracy:**
   - Across all models, 4-class emotion accuracy remains between 38% and 40% (chance level: 25.0%). While substantially above chance, cross-subject electrophysiological variance remains a major barrier.
2. **Cross-Subject Operational Coverage Shift:**
   - Deriving static quantile thresholds from training subjects fails to achieve target coverage on unseen test subjects. Nominal 70% coverage yields actual coverage between 88.9% and 92.2% under temperature scaling due to probability compression.
3. **Multiple Comparisons Constraints:**
   - In unadjusted testing, 5 condition-target combinations achieve $p < 0.05$ against random rejection. However, after formal Holm-Bonferroni correction across the 20 primary evaluation tests, **none remain statistically significant** ($p_{\text{Holm}} \ge 0.165$).
4. **Offline Window-Level Scope:**
   - Analysis was performed on pre-extracted 4-second Differential Entropy features offline. Online computational latency, streaming acquisition, and closed-loop BCI feedback have not been tested.
5. **Single Dataset Validation:**
   - All empirical results derive from SEED-IV; cross-corpus generalization to other datasets (e.g., DEAP, DREAMER) has not been evaluated.

---

## 4. Ten Likely Evaluator Questions and Technically Accurate Answers

### Q1: "Why is your 4-class emotion classification accuracy only ~39%? Isn't that low?"
**Answer:**
"In subject-independent (cross-subject) emotion recognition on SEED-IV with four balanced classes, chance level is 25.0%. Our baseline accuracy of 39.5% is achieved under strict Leave-One-Subject-Out cross-validation using linear classifiers without any subject-specific fine-tuning. Many literature studies reporting >70% accuracy use subject-dependent (within-subject) cross-validation, where segments from the same subject and session appear in both training and test sets, introducing massive identity leakage. Under strict LOSO evaluation, 38%–45% is typical for linear models on 4-class SEED-IV."

### Q2: "How did you ensure there was zero data leakage in your channel selection and threshold estimation?"
**Answer:**
"Every stage of feature scaling, channel selection, model fitting, temperature calibration, and threshold calculation was conducted strictly inside the outer training folds. In Phase B/C, channels were selected using inner cross-validation across the 14 training subjects; the held-out test subject's data was never seen. In Phase E Mode B, operational thresholds were derived using inner group-level cross-fitting across the 14 training subjects, frozen, and then evaluated on the test subject. The test subject's labels and uncertainty scores were never used during parameter or threshold selection."

### Q3: "What is the difference between Mode A and Mode B in your selective prediction evaluation?"
**Answer:**
"Mode A is a **retrospective empirical evaluation** of uncertainty ranking capacity. It sorts all predictions by uncertainty directly on the evaluated test set to plot risk–coverage curves and compute AURC. It measures the theoretical upper-bound ranking quality of uncertainty, but does not represent a deployable policy.  
Mode B is an **inductive operational evaluation**. It estimates a fixed numerical threshold $\tau$ strictly from training out-of-fold uncertainty quantiles, freezes $\tau$, and applies it to the held-out subject without knowing their uncertainty distribution. Mode B evaluates whether a real BCI system can deploy a fixed threshold on a future user."

### Q4: "Why does your nominal 70% coverage target actually accept ~89%–92% of samples on test subjects?"
**Answer:**
"This is caused by cross-subject probability compression under temperature scaling. Temperature scaling divides raw decision margins by a global factor $T \approx 1.7$, which shrinks predicted probabilities toward uniform ($0.25$) and clusters uncertainty values into a narrow range ($u_{\max} \in [0.65, 0.72]$). Minor shifts in an unseen subject's margin distribution cause almost all of their test uncertainty scores to fall below the training threshold. Interestingly, uncalibrated softmax avoids this issue: because raw margins remain dispersed, uncalibrated softmax tracks nominal coverage within 3.3 percentage points across all levels (e.g., nominal 70% achieves 71.2% actual coverage)."

### Q5: "For `stab_32` at nominal 50% coverage, your Wilcoxon test is significant ($p = 0.025$), but the 95% confidence interval includes zero `[-2.44, +19.72]`. How do you explain this contradiction?"
**Answer:**
"This divergence is driven by heavy-tailed subject variance. Inspecting individual subject deltas reveals that 12 out of 15 subjects showed positive improvements over random rejection. However, Subject 6 gained $+66.4\text{ pp}$ while Subject 8 lost $-27.5\text{ pp}$, driving the sample standard deviation to $s = 20.01\text{ pp}$. The parametric Student-$t$ interval depends on $s$, yielding a wide standard error ($5.17\text{ pp}$) and a margin of error ($11.08\text{ pp}$) that crosses zero. In contrast, the non-parametric Wilcoxon test evaluates signed ranks: because 12 positive ranks overwhelmingly dominate the 2 negative ranks, the test is statistically significant ($W = 16.0, p = 0.0248$)."

### Q6: "Did your uncertainty abstention results survive multiple testing corrections?"
**Answer:**
"No, not across the full primary family. We evaluated 4 channel sets across 5 nominal coverage levels, yielding $m = 20$ tests. While 5 tests achieve $p < 0.05$ in unadjusted testing, none survive family-wise Holm-Bonferroni correction ($p_{\text{Holm}} \ge 0.165$), and the minimum Benjamini-Hochberg FDR is $q = 0.124$. If we pre-plan an evaluation restricted to a single model family (`pred_32`, $m = 5$), nominal 60% coverage survives ($p_{\text{Holm}} = 0.041$). However, we report both unadjusted and adjusted statistics transparently and do not present post-hoc single-model significance as confirmatory proof."

### Q7: "Why does pooled Mode A AURC look almost identical to random rejection, while subject-level AURC shows improvement?"
**Answer:**
"This is an empirical manifestation of Simpson's Paradox caused by cross-subject calibration misalignment. Some subjects naturally have high-confidence margins but lower classification accuracy, while others have narrower margins but higher accuracy. When all 37,575 predictions are pooled globally, high-confidence windows from lower-accuracy subjects are accepted first, depressing the pooled curve to random levels ($0.6139$ vs. $0.6144$). When evaluated within each subject, uncertainty consistently separates correct from incorrect predictions, reducing mean subject Full AURC to $0.5628 \pm 0.0855$."

### Q8: "How does your random rejection baseline work, and why is it necessary?"
**Answer:**
"In selective prediction, selective accuracy naturally increases as coverage decreases simply because fewer samples are evaluated. To prove that uncertainty abstention provides genuine utility, one must compare it against a label-agnostic random rejection baseline at the exact same accepted count. We used a 500-repetition Monte Carlo random acceptance procedure for each subject and target. The theoretical expectation of random rejection is exactly the full-coverage accuracy. Comparing against random rejection isolates whether uncertainty selectively eliminates errors rather than randomly dropping data."

### Q9: "Between predictive selection (`pred_32`) and stability-aware selection (`stab_32`), which is better?"
**Answer:**
"Both achieve comparable baseline accuracy (38.90% for `pred_32` vs. 38.56% for `stab_32`) and comparable peak selective prediction gains (+6.9 pp vs. +8.6 pp at low coverage). However, `stab_32` provides higher inter-fold channel stability (Kuncheva index), meaning that the selected electrodes are more consistent across training folds. For physical BCI headset manufacturing, channel stability across populations is a substantial practical advantage even when predictive performance is equivalent."

### Q10: "What is your final verdict on the selective prediction hypothesis?"
**Answer:**
"The hypothesis is **partially supported**. It is supported at the individual subject level: within unseen subjects, predictive uncertainty effectively isolates errors, increasing selective accuracy by 6 to 8 percentage points over full-coverage baseline. However, it is not supported as a fixed cross-subject operational policy: static training thresholds cannot guarantee test coverage on unseen subjects, and improvements over random rejection do not survive rigorous family-wise error rate correction across all conditions."
