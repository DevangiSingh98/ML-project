"""
Classification metrics, confusion matrix, and reporting utilities for LOSO evaluation.

Provides:
- Sample-level and subject-level accuracy, balanced accuracy, macro-F1.
- Per-class precision, recall, F1, and support.
- Confusion matrix computation and visualization via matplotlib.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


DEFAULT_CLASS_NAMES = {
    0: "neutral",
    1: "sad",
    2: "fear",
    3: "happy",
}


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics handling undefined cases explicitly.
    
    Zero division is set to 0.0 to prevent silent NaN propagation or crashes on empty classes.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if len(y_t) == 0 or len(y_p) == 0:
        raise ValueError("Cannot compute metrics on empty true or predicted labels.")

    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))

    metrics: Dict[str, float] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "total_samples": len(y_t),
    }

    # Per-class metrics
    for c_id in sorted(class_names.keys()):
        c_name = class_names[c_id]
        binary_true = (y_t == c_id)
        binary_pred = (y_p == c_id)

        support = int(binary_true.sum())
        prec = float(precision_score(binary_true, binary_pred, zero_division=0))
        rec = float(recall_score(binary_true, binary_pred, zero_division=0))
        f1 = float(f1_score(binary_true, binary_pred, zero_division=0))

        metrics[f"precision_{c_name}"] = prec
        metrics[f"recall_{c_name}"] = rec
        metrics[f"f1_{c_name}"] = f1
        metrics[f"support_{c_name}"] = support

    return metrics


def compute_confusion_matrix_df(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Compute confusion matrix formatted as a pandas DataFrame.
    
    Rows = True labels, Columns = Predicted labels.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    labels = sorted(class_names.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    names = [class_names[c] for c in labels]

    cm_df = pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names])
    return cm_df


def plot_and_save_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    output_path: Union[str, Path],
    class_names: Optional[Dict[int, str]] = None,
    title: str = "Confusion Matrix (Pooled 15-Fold LOSO)",
) -> None:
    """
    Render and save a labeled confusion matrix figure.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    labels = sorted(class_names.keys())
    names = [class_names[c] for c in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    # Annotate counts and percentages
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.divide(cm.astype(float), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0) * 100

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            perc = cm_perc[i, j]
            text_color = "white" if cm[i, j] > (cm.max() / 2) else "black"
            ax.text(
                j, i, f"{count:,}\n({perc:.1f}%)",
                ha="center", va="center", color=text_color, fontsize=10, fontweight="medium"
            )

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([n.capitalize() for n in names], fontsize=11)
    ax.set_yticklabels([n.capitalize() for n in names], fontsize=11)

    ax.set_xlabel("Predicted Emotion", fontsize=12, labelpad=10)
    ax.set_ylabel("True Emotion", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=15, fontweight="bold")
    plt.tight_layout()

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
