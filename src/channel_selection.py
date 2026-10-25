"""
Leakage-safe EEG channel selection methods and cross-subject stability analysis.

Implements:
1. All-62-channel reference.
2. Repeated random channel subsets (distribution baseline).
3. Predictive-importance selection (grouped permutation importance across inner subject splits).
4. Stability-aware selection (combined predictive utility and cross-subject selection stability).

Guarantees:
- Held-out outer test subject is strictly excluded from all ranking, permutation, and stability calculations.
- Physical channel grouping is preserved (all 5 frequency bands selected together per channel).
- Exact k unique physical channels selected for each budget.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.channel_features import get_channel_feature_indices, slice_channel_features


def select_random_channels(
    budget: int,
    seed: int = 42,
    total_channels: int = 62,
) -> List[int]:
    """
    Select exactly budget unique physical channels uniformly at random without replacement.
    
    Parameters:
        budget: Number of channels to select (k).
        seed: Random seed for reproducibility.
        total_channels: Total available physical channels (62).
        
    Returns:
        Sorted list of k unique channel indices in [0, total_channels - 1].
    """
    if not (1 <= budget <= total_channels):
        raise ValueError(f"Invalid channel budget {budget} for total_channels={total_channels}")

    rng = np.random.RandomState(seed)
    chosen = rng.choice(total_channels, size=budget, replace=False)
    return sorted(chosen.tolist())


def compute_inner_grouped_permutation_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_metadata: pd.DataFrame,
    n_splits: int = 3,
    group_key: str = "subject_id",
    num_bands: int = 5,
    total_channels: int = 62,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute grouped channel permutation importance across subject-disjoint inner splits.
    
    In each inner split, a RidgeClassifier is trained on inner training subjects and
    evaluated on inner validation subjects. Each channel's 5 frequency bands are
    permuted together to measure performance degradation.
    
    Returns:
        (importance_matrix, ranks_matrix, avg_importance)
        where:
            importance_matrix: shape (n_splits, total_channels) with validation score drops
            ranks_matrix: shape (n_splits, total_channels) with rank per channel (1 = best)
            avg_importance: shape (total_channels,) with mean drop across splits
    """
    groups = train_metadata[group_key].to_numpy()
    unique_groups = len(np.unique(groups))
    actual_splits = min(n_splits, unique_groups)

    gkf = GroupKFold(n_splits=actual_splits)
    X_dummy = np.zeros(len(train_metadata))

    importance_list = []
    ranks_list = []

    for split_idx, (inner_tr_idx, inner_val_idx) in enumerate(gkf.split(X_dummy, groups=groups)):
        # Verify strict inner disjointness
        inner_tr_subs = set(train_metadata.iloc[inner_tr_idx][group_key].unique())
        inner_val_subs = set(train_metadata.iloc[inner_val_idx][group_key].unique())
        assert inner_tr_subs.isdisjoint(inner_val_subs), "Inner split subject leakage!"

        X_in_tr, y_in_tr = X_train[inner_tr_idx], y_train[inner_tr_idx]
        X_in_val, y_in_val = X_train[inner_val_idx], y_train[inner_val_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(alpha=10.0, random_state=seed + split_idx)),
        ])
        pipe.fit(X_in_tr, y_in_tr)

        base_score = float(balanced_accuracy_score(y_in_val, pipe.predict(X_in_val)))

        rng = np.random.RandomState(seed + split_idx * 100)
        split_drops = np.zeros(total_channels, dtype=np.float64)

        for ch in range(total_channels):
            col_indices = [ch * num_bands + b for b in range(num_bands)]
            X_val_perm = X_in_val.copy()
            perm_order = rng.permutation(len(X_val_perm))
            X_val_perm[:, col_indices] = X_in_val[perm_order, :][:, col_indices]

            perm_score = float(balanced_accuracy_score(y_in_val, pipe.predict(X_val_perm)))
            split_drops[ch] = base_score - perm_score

        # Rank channels in descending order of performance drop (1 = best)
        # Higher drop = more useful
        rank_order = np.argsort(-split_drops)
        ranks = np.zeros(total_channels, dtype=np.int64)
        for rank_pos, ch_idx in enumerate(rank_order, start=1):
            ranks[ch_idx] = rank_pos

        importance_list.append(split_drops)
        ranks_list.append(ranks)

    importance_matrix = np.array(importance_list)  # (n_splits, 62)
    ranks_matrix = np.array(ranks_list)            # (n_splits, 62)
    avg_importance = np.mean(importance_matrix, axis=0)  # (62,)

    return importance_matrix, ranks_matrix, avg_importance


def select_predictive_importance_channels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_metadata: pd.DataFrame,
    budget: int,
    n_splits: int = 3,
    group_key: str = "subject_id",
    num_bands: int = 5,
    total_channels: int = 62,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Select top k channels ranked by grouped permutation importance averaged across inner splits.
    
    Returns:
        (selected_channels, avg_importance, rank_order)
    """
    _, _, avg_importance = compute_inner_grouped_permutation_importance(
        X_train=X_train,
        y_train=y_train,
        train_metadata=train_metadata,
        n_splits=n_splits,
        group_key=group_key,
        num_bands=num_bands,
        total_channels=total_channels,
        seed=seed,
    )

    rank_order = np.argsort(-avg_importance)
    selected_channels = sorted(rank_order[:budget].tolist())
    return selected_channels, avg_importance, rank_order


def compute_channel_stability_scores(
    avg_importance: np.ndarray,
    ranks_matrix: np.ndarray,
    total_channels: int = 62,
    alpha: float = 0.5,
    beta: float = 0.5,
    method: str = "multiplicative",
) -> Dict[str, np.ndarray]:
    """
    Compute stability-aware channel scores from permutation importance drops and split ranks.
    
    Components:
    1. Normalized Predictive Importance (I_norm in [0, 1]):
       Cardinal performance drop averaged across inner validation splits.
       I_norm = (avg_importance - min(avg_importance)) / (max(avg_importance) - min(avg_importance))
       Safeguard: If max == min, I_norm = 1.0 for all channels.
       
    2. Normalized Rank Quality (R_c in [0, 1]):
       Ordinal performance based on mean rank across inner splits.
       R_c = 1.0 - (mean_rank - 1.0) / (total_channels - 1.0)
       Where mean_rank in [1, total_channels]. R_c = 1.0 for best mean rank, 0.0 for worst.
       
    3. Normalized Rank Consistency (V_c in [0, 1]):
       Standard deviation of ranks across inner splits.
       std_ranks = std(ranks_matrix, axis=0)
       Normalized against theoretical maximum standard deviation (Popoviciu's inequality):
       sigma_max = (total_channels - 1.0) / 2.0
       V_c = 1.0 - (std_ranks / sigma_max)
       Safeguard: V_c is clipped to [0, 1]. For n_splits < 2 or constant ranks, V_c = 1.0.
       
    4. Rank Stability Score (S_c in [0, 1]):
       - "multiplicative" (recommended / default):
         S_c = R_c * V_c
         Multiplicatively discounts rank quality by rank volatility across splits.
         Guarantees:
         * If std_ranks == 0 (perfect stability across splits), S_c = R_c.
         * If R_c == 0 (consistently worst rank), S_c = 0.
         * Volatile channels with equal mean rank receive lower scores than stable channels.
       - "additive":
         S_c = 0.5 * R_c + 0.5 * V_c
       - "legacy_mean_rank":
         S_c = R_c (Historical Phase C definition, purely mean rank without variability).
         
    5. Combined Selection Score (Q_c in [0, 1]):
       Q_c = alpha * I_norm + beta * S_c
       
    Tie-breaking:
       Determined by tuple (-Q_c, -I_norm, -S_c, channel_index) ensuring strict determinism.
       
    Returns:
       Dictionary containing:
         'Q_c': Final combined score array of shape (total_channels,)
         'I_norm': Normalized importance array of shape (total_channels,)
         'S_c': Stability score array of shape (total_channels,)
         'R_c': Rank quality array of shape (total_channels,)
         'V_c': Rank consistency array of shape (total_channels,)
         'mean_ranks': Mean rank array of shape (total_channels,)
         'std_ranks': Rank standard deviation array of shape (total_channels,)
         'rank_order': Sorted channel indices in descending order of selection priority
    """
    if total_channels < 1:
        raise ValueError(f"total_channels must be >= 1, got {total_channels}")
    if avg_importance.shape[0] != total_channels:
        raise ValueError(
            f"avg_importance length {avg_importance.shape[0]} != total_channels {total_channels}"
        )
    if ranks_matrix.shape[1] != total_channels:
        raise ValueError(
            f"ranks_matrix columns {ranks_matrix.shape[1]} != total_channels {total_channels}"
        )

    # 1. Normalized Predictive Importance I_norm in [0, 1]
    min_i = float(np.min(avg_importance))
    max_i = float(np.max(avg_importance))
    if max_i - min_i > 1e-8:
        I_norm = (avg_importance - min_i) / (max_i - min_i)
    else:
        I_norm = np.ones(total_channels, dtype=np.float64)
    I_norm = np.clip(I_norm, 0.0, 1.0)

    # 2. Normalized Rank Quality R_c in [0, 1]
    mean_ranks = np.mean(ranks_matrix, axis=0)
    denom_r = float(total_channels - 1) if total_channels > 1 else 1.0
    R_c = 1.0 - (mean_ranks - 1.0) / denom_r
    R_c = np.clip(R_c, 0.0, 1.0)

    # 3. Normalized Rank Consistency V_c in [0, 1]
    n_splits = ranks_matrix.shape[0]
    if n_splits < 2:
        std_ranks = np.zeros(total_channels, dtype=np.float64)
        V_c = np.ones(total_channels, dtype=np.float64)
    else:
        std_ranks = np.std(ranks_matrix, axis=0, ddof=0)
        sigma_max = (total_channels - 1.0) / 2.0 if total_channels > 1 else 1.0
        V_c = 1.0 - (std_ranks / sigma_max)
        V_c = np.clip(V_c, 0.0, 1.0)

    # 4. Rank Stability Score S_c
    if method == "multiplicative":
        S_c = np.clip(R_c * V_c, 0.0, 1.0)
    elif method == "additive":
        S_c = np.clip(0.5 * R_c + 0.5 * V_c, 0.0, 1.0)
    elif method == "legacy_mean_rank":
        S_c = R_c
    else:
        raise ValueError(
            f"Unknown stability scoring method: {method}. "
            "Supported: 'multiplicative', 'additive', 'legacy_mean_rank'"
        )

    # 5. Combined Selection Score Q_c
    Q_c = alpha * I_norm + beta * S_c

    # Deterministic tie-breaking: primary -Q_c, secondary -I_norm, tertiary -S_c, quaternary ch_idx
    rank_order = sorted(
        range(total_channels),
        key=lambda c: (-float(Q_c[c]), -float(I_norm[c]), -float(S_c[c]), int(c)),
    )
    rank_order_arr = np.array(rank_order, dtype=np.int64)

    return {
        "Q_c": Q_c,
        "I_norm": I_norm,
        "S_c": S_c,
        "R_c": R_c,
        "V_c": V_c,
        "mean_ranks": mean_ranks,
        "std_ranks": std_ranks,
        "rank_order": rank_order_arr,
    }


def select_stability_aware_channels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_metadata: pd.DataFrame,
    budget: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    n_splits: int = 3,
    group_key: str = "subject_id",
    num_bands: int = 5,
    total_channels: int = 62,
    seed: int = 42,
    method: str = "multiplicative",
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """
    Select top k channels using the corrected stability-aware criterion:
        Q_c = alpha * I_c_norm + beta * S_c
        
    where:
        I_c_norm: Normalized predictive importance across inner validation splits in [0, 1].
        S_c: Normalized rank consistency score across inner validation splits in [0, 1].
             In "multiplicative" mode (default): S_c = R_c * V_c, discounting mean rank quality
             by rank volatility across splits.
             In "legacy_mean_rank" mode: S_c = R_c = 1 - (mean_rank - 1) / (total_channels - 1).
             
    Returns:
        (selected_channels, Q_c, I_norm, S_c)
    """
    if not (1 <= budget <= total_channels):
        raise ValueError(f"Invalid budget {budget} for total_channels={total_channels}")

    importance_matrix, ranks_matrix, avg_importance = compute_inner_grouped_permutation_importance(
        X_train=X_train,
        y_train=y_train,
        train_metadata=train_metadata,
        n_splits=n_splits,
        group_key=group_key,
        num_bands=num_bands,
        total_channels=total_channels,
        seed=seed,
    )

    scores = compute_channel_stability_scores(
        avg_importance=avg_importance,
        ranks_matrix=ranks_matrix,
        total_channels=total_channels,
        alpha=alpha,
        beta=beta,
        method=method,
    )

    Q_c = scores["Q_c"]
    I_norm = scores["I_norm"]
    S_c = scores["S_c"]
    rank_order = scores["rank_order"]
    selected_channels = sorted(rank_order[:budget].tolist())

    return selected_channels, Q_c, I_norm, S_c


def select_channels_for_loso_fold(
    X_full: np.ndarray,
    y_full: np.ndarray,
    metadata_full: pd.DataFrame,
    test_subject: int,
    budget: int,
    method_name: str = "stability_aware",
    stability_method: str = "multiplicative",
    alpha: float = 0.5,
    beta: float = 0.5,
    n_splits: int = 3,
    seed: int = 42,
    group_key: str = "subject_id",
    num_bands: int = 5,
    total_channels: int = 62,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Select channels for a single LOSO fold with explicit outer test subject isolation.
    
    Guarantees:
    - Held-out subject is strictly quarantined before any processing.
    - An explicit assertion verifies that test_subject is absent from training metadata.
    """
    train_mask = (metadata_full[group_key] != test_subject).to_numpy()
    assert not np.any(metadata_full[group_key].iloc[train_mask] == test_subject), (
        f"Leakage detected: test subject {test_subject} present in training set!"
    )
    X_tr = X_full[train_mask]
    y_tr = y_full[train_mask]
    meta_tr = metadata_full.iloc[train_mask].reset_index(drop=True)

    if method_name == "random":
        selected_channels = select_random_channels(
            budget=budget, seed=seed, total_channels=total_channels
        )
        info = {"method": "random", "budget": budget, "seed": seed}
    elif method_name == "predictive_importance":
        selected_channels, avg_imp, rank_order = select_predictive_importance_channels(
            X_train=X_tr,
            y_train=y_tr,
            train_metadata=meta_tr,
            budget=budget,
            n_splits=n_splits,
            group_key=group_key,
            num_bands=num_bands,
            total_channels=total_channels,
            seed=seed,
        )
        info = {
            "method": "predictive_importance",
            "budget": budget,
            "avg_importance": avg_imp,
            "rank_order": rank_order,
        }
    elif method_name == "stability_aware":
        selected_channels, Q_c, I_norm, S_c = select_stability_aware_channels(
            X_train=X_tr,
            y_train=y_tr,
            train_metadata=meta_tr,
            budget=budget,
            alpha=alpha,
            beta=beta,
            n_splits=n_splits,
            group_key=group_key,
            num_bands=num_bands,
            total_channels=total_channels,
            seed=seed,
            method=stability_method,
        )
        info = {
            "method": "stability_aware",
            "stability_method": stability_method,
            "budget": budget,
            "Q_c": Q_c,
            "I_norm": I_norm,
            "S_c": S_c,
        }
    else:
        raise ValueError(f"Unknown channel selection method: {method_name}")

    return selected_channels, info


def compute_pairwise_jaccard_similarity(channel_sets: List[List[int]]) -> float:
    """
    Compute mean pairwise Jaccard similarity across folds for a fixed channel budget.
    
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    Values range from 0.0 (completely disjoint) to 1.0 (identical channel subsets).
    """
    n = len(channel_sets)
    if n < 2:
        return 1.0

    similarities = []
    for i in range(n):
        set_i = set(channel_sets[i])
        for j in range(i + 1, n):
            set_j = set(channel_sets[j])
            union_len = len(set_i.union(set_j))
            if union_len == 0:
                sim = 1.0
            else:
                sim = len(set_i.intersection(set_j)) / union_len
            similarities.append(sim)

    return float(np.mean(similarities))
