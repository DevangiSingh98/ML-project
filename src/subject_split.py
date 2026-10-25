"""
Subject-independent Leave-One-Subject-Out (LOSO) and inner GroupKFold split utilities.

Guarantees:
- Strict subject disjointness between training and test sets in all outer folds.
- Each subject is held out exactly once across outer folds.
- Grouped inner validation splits strictly respect subject boundaries.
"""

from typing import Dict, Generator, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def get_loso_splits(
    metadata: pd.DataFrame,
    subject_key: str = "subject_id"
) -> Generator[Tuple[int, int, np.ndarray, np.ndarray], None, None]:
    """
    Generate 15 outer Leave-One-Subject-Out (LOSO) folds.
    
    Parameters:
        metadata: DataFrame containing sample-level metadata including subject_key.
        subject_key: Column name identifying unique subjects.
        
    Yields:
        Tuple of (fold_idx, test_subject_id, train_indices, test_indices)
        where train_indices and test_indices are NumPy integer index arrays.
        
    Raises:
        AssertionError if subject disjointness is violated or if a subject is duplicated.
    """
    if subject_key not in metadata.columns:
        raise KeyError(f"Metadata DataFrame missing required column: '{subject_key}'")

    unique_subjects = sorted(metadata[subject_key].unique())
    seen_test_subjects: Set[int] = set()

    for fold_idx, test_sub in enumerate(unique_subjects, start=1):
        if test_sub in seen_test_subjects:
            raise AssertionError(f"Subject {test_sub} was already held out in a previous fold!")
        seen_test_subjects.add(test_sub)

        # Boolean masks
        test_mask = (metadata[subject_key] == test_sub).to_numpy()
        train_mask = ~test_mask

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        train_subjects = set(metadata.iloc[train_indices][subject_key].unique())
        test_subjects = set(metadata.iloc[test_indices][subject_key].unique())

        # Mandatory leakage assertions
        assert train_subjects.isdisjoint(test_subjects), (
            f"Leakage detected in Fold {fold_idx}: "
            f"Intersection between train and test subjects: {train_subjects.intersection(test_subjects)}"
        )
        assert len(test_subjects) == 1, (
            f"Fold {fold_idx} expected exactly 1 held-out subject, found: {test_subjects}"
        )
        assert test_sub in test_subjects, (
            f"Fold {fold_idx} test set does not contain target subject {test_sub}"
        )
        assert len(test_indices) > 0, f"Fold {fold_idx} test set is empty!"
        assert len(train_indices) > 0, f"Fold {fold_idx} train set is empty!"

        yield fold_idx, test_sub, train_indices, test_indices

    assert len(seen_test_subjects) == len(unique_subjects), (
        f"Expected {len(unique_subjects)} subjects held out, but only held out {len(seen_test_subjects)}"
    )


def get_inner_group_splits(
    train_metadata: pd.DataFrame,
    n_splits: int = 5,
    group_key: str = "subject_id"
) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
    """
    Generate inner validation splits strictly grouped by subject.
    
    Prevents temporal window autocorrelation leakage during inner hyperparameter tuning.
    """
    if group_key not in train_metadata.columns:
        raise KeyError(f"train_metadata missing required grouping key: '{group_key}'")

    groups = train_metadata[group_key].to_numpy()
    n_unique_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_unique_groups)

    gkf = GroupKFold(n_splits=actual_splits)
    X_dummy = np.zeros(len(train_metadata))

    for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(gkf.split(X_dummy, groups=groups), start=1):
        inner_train_subs = set(train_metadata.iloc[inner_train_idx][group_key].unique())
        inner_val_subs = set(train_metadata.iloc[inner_val_idx][group_key].unique())

        assert inner_train_subs.isdisjoint(inner_val_subs), (
            f"Inner fold {inner_fold_idx} leakage: inner train and val subjects overlap!"
        )

        yield inner_fold_idx, inner_train_idx, inner_val_idx
