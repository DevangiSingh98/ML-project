"""
Phase C Audit: Analyze and summarize random-subset replication across seeds.
Distinguishes:
1. Cross-subject variability (mean +/- std across held-out subjects within a subset)
2. Cross-subset variability (mean +/- std across independent random channel subsets)
3. True between-subset Jaccard similarity vs cross-fold stability
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.channel_selection import select_random_channels, compute_pairwise_jaccard_similarity

def main():
    # 1. Load existing fold metrics
    fold_df = pd.read_csv("outputs/metrics/channel_reduction_fold_metrics.csv")
    
    # Check for additional seed 450 for k=8
    p450 = Path("outputs/metrics/random_k8_seed450_metrics.csv")
    if p450.is_file():
        df_450 = pd.read_csv(p450)
        # Append if not already present
        if not ((fold_df["budget"] == 8) & (fold_df["seed"] == 450)).any():
            fold_df = pd.concat([fold_df, df_450], ignore_index=True)

    # Filter random methods
    rnd_df = fold_df[fold_df["method"].isin(["random", "random_repeat", "random_replicate"])].copy()

    # Replicate-level summary
    replicate_records = []
    for (k, seed), grp in rnd_df.groupby(["budget", "seed"]):
        chs = grp["selected_channels"].iloc[0]
        n_folds = len(grp)
        acc_mean = grp["accuracy"].mean()
        acc_std = grp["accuracy"].std()
        f1_mean = grp["macro_f1"].mean()
        f1_std = grp["macro_f1"].std()
        bacc_mean = grp["balanced_accuracy"].mean()
        bacc_std = grp["balanced_accuracy"].std()
        replicate_records.append({
            "budget": int(k),
            "seed": int(seed),
            "n_folds": n_folds,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "balanced_accuracy_mean": bacc_mean,
            "balanced_accuracy_std": bacc_std,
            "selected_channels": chs,
        })

    replicates_df = pd.DataFrame(replicate_records).sort_values(by=["budget", "seed"]).reset_index(drop=True)
    replicates_path = "outputs/tables/random_subset_replicates.csv"
    replicates_df.to_csv(replicates_path, index=False)
    print(f"Saved: {replicates_path}")

    # Budget-level replication summary
    budget_records = []
    for k, grp in replicates_df.groupby("budget"):
        n_seeds = len(grp)
        mean_of_acc = grp["accuracy_mean"].mean()
        std_of_acc = grp["accuracy_mean"].std() if n_seeds > 1 else 0.0
        min_acc = grp["accuracy_mean"].min()
        max_acc = grp["accuracy_mean"].max()
        avg_subject_std = grp["accuracy_std"].mean()

        mean_of_f1 = grp["macro_f1_mean"].mean()
        std_of_f1 = grp["macro_f1_mean"].std() if n_seeds > 1 else 0.0

        # True pairwise Jaccard across evaluated seeds
        seeds = grp["seed"].tolist()
        if len(seeds) > 1:
            ch_sets = [select_random_channels(budget=k, seed=s) for s in seeds]
            true_jaccard = compute_pairwise_jaccard_similarity(ch_sets)
        else:
            true_jaccard = np.nan

        budget_records.append({
            "budget": int(k),
            "n_replicates_evaluated": n_seeds,
            "seeds_evaluated": ",".join(map(str, seeds)),
            "replicate_mean_accuracy": mean_of_acc,
            "replicate_std_accuracy": std_of_acc,
            "replicate_min_accuracy": min_acc,
            "replicate_max_accuracy": max_acc,
            "mean_subject_std_accuracy": avg_subject_std,
            "replicate_mean_macro_f1": mean_of_f1,
            "replicate_std_macro_f1": std_of_f1,
            "true_between_subset_jaccard": true_jaccard,
        })

    budget_summary_df = pd.DataFrame(budget_records).sort_values(by="budget").reset_index(drop=True)
    summary_path = "outputs/tables/random_subset_variability_summary.csv"
    budget_summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Channel stability comparison table (Stability-Aware vs Predictive vs True Random)
    stab_df = pd.read_csv("outputs/tables/channel_stability_summary.csv")
    
    # Calculate true random Jaccard across 5 arbitrary seeds for comparison
    true_rnd_jaccard = {}
    for k in [8, 12, 16, 24, 32]:
        sets = [select_random_channels(k, seed) for seed in [42+k, 142+k, 242+k, 342+k, 442+k]]
        true_rnd_jaccard[k] = compute_pairwise_jaccard_similarity(sets)

    stab_df["true_random_between_subset_jaccard"] = stab_df["budget"].map(true_rnd_jaccard)
    stab_df.rename(columns={"random_jaccard": "random_fixed_seed_cross_fold_jaccard"}, inplace=True)
    
    comp_path = "outputs/tables/channel_stability_comparison.csv"
    stab_df.to_csv(comp_path, index=False)
    print(f"Saved: {comp_path}")

if __name__ == "__main__":
    main()
