"""
SEED-IV EEG Emotion Recognition Dataset Loader.

This module loads, validates, and structures the precomputed Differential Entropy (de_LDS)
features from the genuine SEED-IV dataset, producing model-ready matrices with deterministic
channel-to-feature mappings and metadata tracking.
"""

from dataclasses import dataclass
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import openpyxl
import pandas as pd
import scipy.io as sio
import yaml


# Official SEED-IV Trial Labels per Session (0: Neutral, 1: Sad, 2: Fear, 3: Happy)
OFFICIAL_LABELS: Dict[int, List[int]] = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}

LABEL_NAMES: Dict[int, str] = {
    0: "neutral",
    1: "sad",
    2: "fear",
    3: "happy",
}

SUBJECT_GENDERS: Dict[int, str] = {
    1: "male", 2: "male", 3: "female", 4: "female", 5: "female",
    6: "male", 7: "male", 8: "female", 9: "female", 10: "female",
    11: "female", 12: "male", 13: "male", 14: "female", 15: "female",
}

DEFAULT_FREQUENCY_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


@dataclass
class DatasetContainer:
    """Container holding feature matrix, labels, metadata, and mapping schema."""
    X: np.ndarray                  # Shape: (N_samples, N_features = 310)
    y: np.ndarray                  # Shape: (N_samples,)
    metadata: pd.DataFrame         # Columns: subject_id, session_id, trial_id, window_id, label, label_name
    channels: List[str]            # 62 channel names in exact physical order
    frequency_bands: List[str]     # 5 frequency band names
    feature_names: List[str]       # 310 column names (e.g. 'FP1_delta', 'FP1_theta', ...)
    channel_to_indices: Dict[str, List[int]]  # Map channel name -> list of 5 column indices


def load_config(config_path: Union[str, Path] = "config.yaml") -> dict:
    """Load project YAML configuration file."""
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_channel_order(data_dir: Union[str, Path]) -> List[str]:
    """
    Extract the official 62-channel sequence from Channel Order.xlsx or channel_62_pos.locs.
    
    Returns uppercase channel names in standard 10-20 order (FP1 to CB2).
    """
    data_path = Path(data_dir)
    excel_path = data_path / "Channel Order.xlsx"
    locs_path = data_path / "channel_62_pos.locs"

    if excel_path.is_file():
        wb = openpyxl.load_workbook(excel_path)
        ws = wb.active
        channels = [
            str(r[0]).strip().upper()
            for r in ws.iter_rows(values_only=True)
            if r[0] is not None
        ]
        if len(channels) == 62:
            return channels

    if locs_path.is_file():
        channels = []
        with open(locs_path, "r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 4:
                    channels.append(parts[3].strip().upper())
        if len(channels) == 62:
            return channels

    raise FileNotFoundError(
        f"Unable to load 62-channel sequence from {excel_path} or {locs_path}"
    )


def generate_feature_mappings(
    channels: List[str],
    bands: Optional[List[str]] = None
) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Generate deterministic feature column names and grouped channel-to-index mappings.
    
    Each channel contains len(bands) contiguous features:
        Feature index = channel_index * num_bands + band_index
    """
    if bands is None:
        bands = DEFAULT_FREQUENCY_BANDS

    feature_names: List[str] = []
    channel_to_indices: Dict[str, List[int]] = {}

    idx = 0
    for ch in channels:
        ch_indices = []
        for band in bands:
            feature_names.append(f"{ch}_{band}")
            ch_indices.append(idx)
            idx += 1
        channel_to_indices[ch] = ch_indices

    return feature_names, channel_to_indices


def find_subject_file(session_dir: Path, subject_id: int) -> Path:
    """Find the .mat file corresponding to a subject ID in a session directory."""
    patterns = [
        f"{subject_id}_*.mat",
        f"sub{subject_id}_*.mat",
        f"{subject_id}.mat",
    ]
    for pat in patterns:
        matches = list(session_dir.glob(pat))
        if matches:
            # Filter strictly by prefix before '_'
            for m in matches:
                prefix = m.name.split("_")[0].replace("sub", "")
                if prefix.isdigit() and int(prefix) == subject_id:
                    return m
    raise FileNotFoundError(
        f"No .mat file found for subject {subject_id} in session directory: {session_dir}"
    )


def load_seed_iv_dataset(
    config: Optional[dict] = None,
    config_path: Union[str, Path] = "config.yaml",
    data_dir_override: Optional[Union[str, Path]] = None,
    sessions_override: Optional[List[int]] = None,
    feature_type_override: Optional[str] = None,
) -> DatasetContainer:
    """
    Load genuine SEED-IV Differential Entropy (de_LDS) features across subjects and sessions.
    
    Validates:
    - Existence and readability of all expected session files.
    - Trial counts and dimensions per file: (62 channels, T windows, 5 frequency bands).
    - Absence of NaN, Inf, and empty arrays.
    - Alignment of trial labels against official session-specific ground truth.
    - Deterministic 310-feature mapping.
    
    Returns:
        DatasetContainer with feature matrix X (float32), label vector y (int64),
        metadata DataFrame, channel list, band list, and channel-feature mappings.
    """
    if config is None:
        config = load_config(config_path)

    data_dir = Path(data_dir_override or config.get("dataset", {}).get("data_dir", "SEED_IV"))
    feature_type = feature_type_override or config.get("dataset", {}).get("feature_type", "de_LDS")
    sessions = sessions_override or config.get("dataset", {}).get("sessions", [1, 2, 3])
    num_subjects = config.get("dataset", {}).get("num_subjects", 15)
    expected_trials = config.get("dataset", {}).get("trials_per_session", 24)
    expected_channels = config.get("dataset", {}).get("total_channels", 62)
    expected_bands = config.get("dataset", {}).get("num_frequency_bands", 5)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {data_dir.resolve()}")

    feature_smooth_dir = data_dir / "eeg_feature_smooth"
    if not feature_smooth_dir.is_dir():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_smooth_dir.resolve()}")

    # Load 62 channels & generate deterministic mappings
    channels = load_channel_order(data_dir)
    if len(channels) != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, found {len(channels)}")

    bands = config.get("dataset", {}).get("frequency_bands", DEFAULT_FREQUENCY_BANDS)
    if len(bands) != expected_bands:
        raise ValueError(f"Expected {expected_bands} frequency bands, found {len(bands)}")

    feature_names, channel_to_indices = generate_feature_mappings(channels, bands)
    expected_feature_dim = expected_channels * expected_bands

    feature_list: List[np.ndarray] = []
    metadata_rows: List[dict] = []

    total_files_loaded = 0

    for session_id in sorted(sessions):
        session_path = feature_smooth_dir / str(session_id)
        if not session_path.is_dir():
            raise FileNotFoundError(f"Session directory not found: {session_path}")

        session_labels = OFFICIAL_LABELS.get(session_id)
        if session_labels is None or len(session_labels) != expected_trials:
            raise ValueError(f"Invalid official labels for session {session_id}")

        for subject_id in range(1, num_subjects + 1):
            mat_file = find_subject_file(session_path, subject_id)
            try:
                mat_data = sio.loadmat(mat_file)
            except Exception as exc:
                raise IOError(f"Failed to load .mat file: {mat_file}") from exc

            total_files_loaded += 1

            for trial_id in range(1, expected_trials + 1):
                key = f"{feature_type}{trial_id}"
                if key not in mat_data:
                    raise KeyError(f"Missing expected key '{key}' in file: {mat_file}")

                trial_arr = mat_data[key]

                # Expected shape in SEED-IV .mat files: (62 channels, T_windows, 5 frequency bands)
                if trial_arr.ndim != 3:
                    raise ValueError(
                        f"Malformed array shape {trial_arr.shape} in {mat_file.name} (trial {trial_id}); expected 3D"
                    )

                n_ch, n_win, n_b = trial_arr.shape
                if n_ch != expected_channels or n_b != expected_bands:
                    raise ValueError(
                        f"Unexpected dimension in {mat_file.name} trial {trial_id}: shape={trial_arr.shape}, "
                        f"expected ({expected_channels}, T, {expected_bands})"
                    )

                if n_win == 0:
                    raise ValueError(f"Empty window sequence in {mat_file.name} trial {trial_id}")

                if np.isnan(trial_arr).any() or np.isinf(trial_arr).any():
                    raise FloatingPointError(f"NaN or Inf encountered in {mat_file.name} trial {trial_id}")

                # Transpose from (62, T, 5) to (T, 62, 5) then reshape to (T, 310)
                trial_win_ch_band = np.transpose(trial_arr, (1, 0, 2))
                trial_matrix = trial_win_ch_band.reshape(n_win, expected_feature_dim).astype(np.float32)

                label = session_labels[trial_id - 1]
                label_name = LABEL_NAMES[label]

                feature_list.append(trial_matrix)

                for w_idx in range(n_win):
                    metadata_rows.append({
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "trial_id": trial_id,
                        "window_id": w_idx,
                        "label": label,
                        "label_name": label_name,
                    })

    if not feature_list:
        raise ValueError("No feature arrays were loaded from dataset.")

    X = np.vstack(feature_list)
    metadata_df = pd.DataFrame(metadata_rows)
    y = metadata_df["label"].to_numpy(dtype=np.int64)

    return DatasetContainer(
        X=X,
        y=y,
        metadata=metadata_df,
        channels=channels,
        frequency_bands=bands,
        feature_names=feature_names,
        channel_to_indices=channel_to_indices,
    )


def generate_dataset_summary(
    container: DatasetContainer,
    output_csv_path: Union[str, Path] = "outputs/tables/dataset_summary.csv"
) -> pd.DataFrame:
    """
    Generate dataset summary report and save to outputs/tables/dataset_summary.csv.
    
    Includes:
    - Overall dataset counts.
    - Breakdown per subject (sessions, trials, windows, label counts, gender).
    - Breakdown per session (trials, windows, label counts).
    - Breakdown per class (window counts, percentages).
    """
    output_file = Path(output_csv_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = container.metadata

    # Subject-level summary
    records = []
    for sub_id, sub_df in df.groupby("subject_id"):
        rec = {
            "category": "subject",
            "identifier": f"subject_{sub_id}",
            "gender": SUBJECT_GENDERS.get(sub_id, "unknown"),
            "sessions": sub_df["session_id"].nunique(),
            "trials": len(sub_df.drop_duplicates(subset=["session_id", "trial_id"])),
            "total_windows": len(sub_df),
            "neutral_windows": int((sub_df["label"] == 0).sum()),
            "sad_windows": int((sub_df["label"] == 1).sum()),
            "fear_windows": int((sub_df["label"] == 2).sum()),
            "happy_windows": int((sub_df["label"] == 3).sum()),
        }
        records.append(rec)

    # Session-level summary
    for sess_id, sess_df in df.groupby("session_id"):
        rec = {
            "category": "session",
            "identifier": f"session_{sess_id}",
            "gender": "all",
            "sessions": 1,
            "trials": len(sess_df.drop_duplicates(subset=["subject_id", "trial_id"])),
            "total_windows": len(sess_df),
            "neutral_windows": int((sess_df["label"] == 0).sum()),
            "sad_windows": int((sess_df["label"] == 1).sum()),
            "fear_windows": int((sess_df["label"] == 2).sum()),
            "happy_windows": int((sess_df["label"] == 3).sum()),
        }
        records.append(rec)

    # Class-level summary
    for label_id in sorted(LABEL_NAMES.keys()):
        label_df = df[df["label"] == label_id]
        rec = {
            "category": "class",
            "identifier": f"class_{label_id}_{LABEL_NAMES[label_id]}",
            "gender": "all",
            "sessions": label_df["session_id"].nunique(),
            "trials": len(label_df.drop_duplicates(subset=["subject_id", "session_id", "trial_id"])),
            "total_windows": len(label_df),
            "neutral_windows": int((label_df["label"] == 0).sum()),
            "sad_windows": int((label_df["label"] == 1).sum()),
            "fear_windows": int((label_df["label"] == 2).sum()),
            "happy_windows": int((label_df["label"] == 3).sum()),
        }
        records.append(rec)

    # Dataset Total row
    records.append({
        "category": "total",
        "identifier": "dataset_total",
        "gender": "all",
        "sessions": df["session_id"].nunique(),
        "trials": len(df.drop_duplicates(subset=["subject_id", "session_id", "trial_id"])),
        "total_windows": len(df),
        "neutral_windows": int((df["label"] == 0).sum()),
        "sad_windows": int((df["label"] == 1).sum()),
        "fear_windows": int((df["label"] == 2).sum()),
        "happy_windows": int((df["label"] == 3).sum()),
    })

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(output_file, index=False)
    return summary_df


if __name__ == "__main__":
    print("=" * 70)
    print("SEED-IV EEG Emotion Recognition: Loading and Validating Data...")
    print("=" * 70)

    cfg = load_config()
    data = load_seed_iv_dataset(config=cfg)

    print(f"Feature matrix X shape: {data.X.shape} (dtype: {data.X.dtype})")
    print(f"Label vector y shape:    {data.y.shape} (dtype: {data.y.dtype})")
    print(f"Metadata rows:          {len(data.metadata)}")
    print(f"Channels verified:      {len(data.channels)} ({data.channels[0]} to {data.channels[-1]})")
    print(f"Frequency bands:        {data.frequency_bands}")
    print(f"Features per channel:   5")
    print(f"Total feature columns:  {len(data.feature_names)}")

    summary = generate_dataset_summary(data)
    print(f"Dataset summary successfully saved to outputs/tables/dataset_summary.csv")
    print("\nOverall Dataset Summary:")
    print(summary[summary["category"] == "total"].to_string(index=False))
    print("\nSession Breakdown:")
    print(summary[summary["category"] == "session"].to_string(index=False))
    print("\nClass Breakdown:")
    print(summary[summary["category"] == "class"].to_string(index=False))
    print("=" * 70)
    print("Phase A Data Loader execution completed successfully.")
    print("=" * 70)
