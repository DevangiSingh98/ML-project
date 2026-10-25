"""
Plotting and figure generation for EEG channel reduction and evaluation.

Generates:
1. Performance vs Channel Count curve (Accuracy and Balanced Accuracy).
2. Channel Selection Stability and Frequency heatmap/barchart.
3. Confusion matrices.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_performance_vs_channel_count(
    summary_df: pd.DataFrame,
    output_path: Union[str, Path] = "outputs/figures/channel_reduction_performance.png",
    metric_name: str = "accuracy",
) -> None:
    """
    Plot performance vs number of channels for All-62, Random, Predictive, and Stability-Aware.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    # All-62 reference baseline
    all_62_row = summary_df[(summary_df["method"] == "all") & (summary_df["budget"] == 62)]
    if not all_62_row.empty:
        ref_val = all_62_row[metric_name].values[0] * 100
        ax.axhline(
            y=ref_val,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=f"All 62 Channels Baseline ({ref_val:.2f}%)",
            alpha=0.85
        )

    # Styles for each method
    styles = {
        "stability_aware": {"color": "#1f77b4", "marker": "o", "label": "Stability-Aware Selection (Q_c)", "lw": 2.2},
        "predictive_importance": {"color": "#2ca02c", "marker": "s", "label": "Predictive-Importance (Permutation)", "lw": 2.0},
        "random": {"color": "#d62728", "marker": "^", "label": "Random Selection (Mean ± Std)", "lw": 1.8},
    }

    budgets = sorted([b for b in summary_df["budget"].unique() if b != 62])

    for method, style in styles.items():
        m_df = summary_df[(summary_df["method"] == method) & (summary_df["budget"] != 62)]
        if m_df.empty:
            continue

        m_df = m_df.sort_values("budget")
        b_vals = m_df["budget"].values
        acc_vals = m_df[metric_name].values * 100

        if f"{metric_name}_std" in m_df.columns:
            err_vals = m_df[f"{metric_name}_std"].values * 100
            ax.errorbar(
                b_vals, acc_vals, yerr=err_vals,
                color=style["color"], marker=style["marker"], label=style["label"],
                linewidth=style["lw"], capsize=4, capthick=1.2, alpha=0.9
            )
        else:
            ax.plot(
                b_vals, acc_vals,
                color=style["color"], marker=style["marker"], label=style["label"],
                linewidth=style["lw"], alpha=0.9
            )

    ax.set_xlabel("Number of Selected Physical EEG Channels (k)", fontsize=12, labelpad=10)
    metric_label = "Classification Accuracy (%)" if metric_name == "accuracy" else "Balanced Accuracy (%)"
    ax.set_ylabel(metric_label, fontsize=12, labelpad=10)
    ax.set_title("SEED-IV LOSO Emotion Recognition: Performance vs. Channel Count", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(budgets + [62])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_channel_selection_stability(
    selection_freq_df: pd.DataFrame,
    all_channel_names: List[str],
    output_path: Union[str, Path] = "outputs/figures/channel_selection_stability.png",
    top_n: int = 25,
) -> None:
    """
    Plot top selected channels by selection frequency across 15 outer folds.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Sort channels by stability-aware selection frequency
    if "stability_aware_freq" in selection_freq_df.columns:
        df_sorted = selection_freq_df.sort_values(by="stability_aware_freq", ascending=False).head(top_n)
    else:
        df_sorted = selection_freq_df.head(top_n)

    channels = df_sorted["channel_name"].tolist()
    stab_freq = df_sorted["stability_aware_freq"].values * 100 if "stability_aware_freq" in df_sorted.columns else None
    pred_freq = df_sorted["predictive_freq"].values * 100 if "predictive_freq" in df_sorted.columns else None

    x = np.arange(len(channels))
    width = 0.38

    if stab_freq is not None and pred_freq is not None:
        ax.bar(x - width/2, stab_freq, width, label="Stability-Aware (Q_c)", color="#1f77b4", alpha=0.85)
        ax.bar(x + width/2, pred_freq, width, label="Predictive Importance", color="#2ca02c", alpha=0.85)
    elif stab_freq is not None:
        ax.bar(x, stab_freq, width * 1.5, label="Stability-Aware (Q_c)", color="#1f77b4", alpha=0.85)

    ax.set_ylabel("Selection Frequency Across 15 Folds (%)", fontsize=12, labelpad=10)
    ax.set_xlabel("Physical EEG Electrode Channel", fontsize=12, labelpad=10)
    ax.set_title(f"Top {top_n} Most Consistently Selected Channels Across Outer Folds", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=9, fontweight="medium")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
