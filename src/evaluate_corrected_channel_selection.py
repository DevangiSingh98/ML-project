"""
Evaluate Corrected Channel Selection Stability across 15 Outer LOSO Folds.

Compares:
1. Corrected Stability-Aware Criterion (Multiplicative variance discount: S_c = R_c * V_c).
2. Legacy Stability-Aware Criterion (Mean rank without variance discount: S_c = R_c).
3. Predictive Importance (Permutation drop without stability: I_norm).

Outputs:
- outputs/tables/corrected_channel_stability_summary.csv
- outputs/tables/corrected_channel_selections.csv
"""

import os
import time
from typing import Dict, List
import numpy as np
import pandas as pd

from src.channel_selection import (
    compute_inner_grouped_permutation_importance,
    compute_channel_stability_scores,
    compute_pairwise_jaccard_similarity,
)
from src.channel_features import get_channel_names
from src.data_loader import load_seed_iv_dataset, load_config
from src.subject_split import get_loso_splits


def run_corrected_stability_experiment(
    config_path: str = "config.yaml",
    output_dir: str = "outputs/tables",
) -> pd.DataFrame:
    """
    Run channel selection across all 15 outer LOSO folds using genuine SEED-IV data.
    Computes pairwise Jaccard stability across folds for each budget and method.
    """
    config = load_config(config_path)
    dataset = load_seed_iv_dataset(config=config)
    X = dataset.X
    y = dataset.y
    metadata = dataset.metadata

    channel_names = dataset.channels
    budgets = [8, 12, 16, 24, 32]
    methods = ["corrected_stability", "legacy_stability", "predictive_importance"]

    # Storage for selected channel indices per method and budget across 15 folds
    selections: Dict[str, Dict[int, List[List[int]]]] = {
        m: {k: [] for k in budgets} for m in methods
    }
    selection_records = []

    print("=" * 70)
    print("RUNNING CORRECTED CHANNEL SELECTION STABILITY EXPERIMENT")
    print(f"Total outer LOSO folds: 15 | Budgets: {budgets}")
    print("=" * 70)

    start_time = time.time()

    for fold_idx, test_sub, train_idx, test_idx in get_loso_splits(metadata, subject_key="subject_id"):
        t0 = time.time()
        train_meta = metadata.iloc[train_idx].reset_index(drop=True)
        assert test_sub not in train_meta["subject_id"].unique(), f"Leakage: subject {test_sub} in train!"

        X_train = X[train_idx]
        y_train = y[train_idx]

        # Inner 3-split grouped permutation importance (RidgeClassifier)
        importance_mat, ranks_mat, avg_imp = compute_inner_grouped_permutation_importance(
            X_train=X_train,
            y_train=y_train,
            train_metadata=train_meta,
            n_splits=3,
            seed=42 + fold_idx,
        )

        # 1. Corrected stability scores (multiplicative variance discount)
        scores_corrected = compute_channel_stability_scores(
            avg_importance=avg_imp,
            ranks_matrix=ranks_mat,
            total_channels=62,
            alpha=0.5,
            beta=0.5,
            method="multiplicative",
        )
        corr_order = scores_corrected["rank_order"]

        # 2. Legacy stability scores (mean rank only)
        scores_legacy = compute_channel_stability_scores(
            avg_importance=avg_imp,
            ranks_matrix=ranks_mat,
            total_channels=62,
            alpha=0.5,
            beta=0.5,
            method="legacy_mean_rank",
        )
        leg_order = scores_legacy["rank_order"]

        # 3. Predictive importance ranking
        pred_order = np.argsort(-avg_imp)

        for k in budgets:
            corr_k = sorted(corr_order[:k].tolist())
            leg_k = sorted(leg_order[:k].tolist())
            pred_k = sorted(pred_order[:k].tolist())

            selections["corrected_stability"][k].append(corr_k)
            selections["legacy_stability"][k].append(leg_k)
            selections["predictive_importance"][k].append(pred_k)

            selection_records.append({
                "fold": fold_idx,
                "test_subject": test_sub,
                "budget": k,
                "corrected_stability_channels": ",".join([channel_names[c] for c in corr_k]),
                "legacy_stability_channels": ",".join([channel_names[c] for c in leg_k]),
                "predictive_importance_channels": ",".join([channel_names[c] for c in pred_k]),
                "corrected_vs_legacy_overlap": len(set(corr_k).intersection(set(leg_k))) / k,
            })

        elapsed_fold = time.time() - t0
        print(f"Fold {fold_idx + 1:2d}/15 (Held-out Sub {test_sub:2d}) completed in {elapsed_fold:.1f}s")

    total_elapsed = time.time() - start_time
    print(f"\nAll 15 folds completed in {total_elapsed:.1f}s ({total_elapsed / 60:.2f} min)")

    # Compute pairwise Jaccard across outer folds for each budget
    summary_records = []
    for k in budgets:
        jaccard_corr = compute_pairwise_jaccard_similarity(selections["corrected_stability"][k])
        jaccard_leg = compute_pairwise_jaccard_similarity(selections["legacy_stability"][k])
        jaccard_pred = compute_pairwise_jaccard_similarity(selections["predictive_importance"][k])

        # Compute average overlap between corrected and legacy methods within same fold
        fold_overlaps = [
            len(set(selections["corrected_stability"][k][f]).intersection(
                set(selections["legacy_stability"][k][f])
            )) / k
            for f in range(15)
        ]
        mean_method_overlap = float(np.mean(fold_overlaps))

        summary_records.append({
            "budget": k,
            "corrected_stability_jaccard": round(jaccard_corr, 6),
            "legacy_stability_jaccard": round(jaccard_leg, 6),
            "predictive_importance_jaccard": round(jaccard_pred, 6),
            "mean_corrected_vs_legacy_overlap": round(mean_method_overlap, 6),
        })

    summary_df = pd.DataFrame(summary_records)
    selections_df = pd.DataFrame(selection_records)

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "corrected_channel_stability_summary.csv")
    selections_path = os.path.join(output_dir, "corrected_channel_selections.csv")

    summary_df.to_csv(summary_path, index=False)
    selections_df.to_csv(selections_path, index=False)

    print("\n" + "=" * 70)
    print("CORRECTED CHANNEL STABILITY EXPERIMENT RESULTS:")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved selections to: {selections_path}")

    return summary_df


if __name__ == "__main__":
    run_corrected_stability_experiment()
