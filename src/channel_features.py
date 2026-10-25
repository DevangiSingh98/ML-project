"""
Channel-to-feature grouping, slicing, and validation utilities.

Each physical EEG channel is represented as a single indivisible block of 5 frequency-band
differential entropy features (Delta, Theta, Alpha, Beta, Gamma):
    feature_index = channel_index * 5 + band_index

Enforces:
- Physical electrode-level selection (never individual frequency features).
- Deterministic column mapping and index extraction.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


def get_channel_feature_indices(
    channel_indices: Union[List[int], np.ndarray],
    num_bands: int = 5,
    total_channels: int = 62,
) -> np.ndarray:
    """
    Given a list/array of physical channel indices (0..total_channels-1),
    return a sorted 1D array of flat feature column indices for all associated frequency bands.
    
    Parameters:
        channel_indices: Sequence of unique physical channel indices in [0, total_channels - 1].
        num_bands: Number of frequency bands per physical channel (default: 5).
        total_channels: Total physical channels in the montage (default: 62).
        
    Returns:
        Sorted 1D np.ndarray of shape (len(channel_indices) * num_bands,) containing column indices.
        
    Raises:
        ValueError if channel_indices contains duplicates, invalid indices, or is empty.
    """
    ch_list = list(channel_indices)
    if len(ch_list) == 0:
        raise ValueError("channel_indices cannot be empty.")

    if len(set(ch_list)) != len(ch_list):
        raise ValueError(f"channel_indices contains duplicates: {ch_list}")

    for ch in ch_list:
        if not (0 <= ch < total_channels):
            raise ValueError(
                f"Channel index {ch} out of bounds for total_channels={total_channels} (expected 0..{total_channels-1})"
            )

    feat_indices: List[int] = []
    for ch in sorted(ch_list):
        start_idx = ch * num_bands
        feat_indices.extend(range(start_idx, start_idx + num_bands))

    return np.array(feat_indices, dtype=np.int64)


def slice_channel_features(
    X: np.ndarray,
    channel_indices: Union[List[int], np.ndarray],
    num_bands: int = 5,
    total_channels: int = 62,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature submatrix corresponding to the specified physical channels.
    
    Parameters:
        X: Feature matrix of shape (N, total_channels * num_bands).
        channel_indices: Physical channel indices to select.
        num_bands: Frequency bands per channel.
        total_channels: Total physical channels.
        
    Returns:
        (X_sliced, feature_indices) where X_sliced has shape (N, len(channel_indices) * num_bands).
    """
    feat_indices = get_channel_feature_indices(
        channel_indices=channel_indices,
        num_bands=num_bands,
        total_channels=total_channels,
    )
    expected_dim = total_channels * num_bands
    if X.shape[1] != expected_dim:
        raise ValueError(f"Expected input X to have {expected_dim} columns, got {X.shape[1]}")

    X_sliced = X[:, feat_indices]
    return X_sliced, feat_indices


def get_channel_names(
    channel_indices: Union[List[int], np.ndarray],
    all_channel_names: List[str],
) -> List[str]:
    """Return physical electrode names corresponding to channel indices."""
    return [all_channel_names[idx] for idx in channel_indices]
