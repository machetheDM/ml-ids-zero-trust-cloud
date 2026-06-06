"""
visualise.py
============
Visualisation utilities for ML-IDS Zero Trust Cloud Research

Generates all figures referenced in the research paper:
  - Figure 2: End-to-end ML-IDS pipeline diagram
  - Figure 3: Model performance comparison chart (paper Table 5 data)
  - ZTA pillar integration diagram (5 pillars × 6 phases)
  - Confusion matrices, ROC curves, feature importance plots

Author: Dingaan Mahlatse Machethe — EC-Council University, ECCU500
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

# ---------------------------------------------------------------------------
# Paper data — Figure 3 model performance comparison (Table 5 values)
# ---------------------------------------------------------------------------

PAPER_RESULTS = {
    "Model":     ["LSTM",  "XGBoost", "Random Forest", "SVM",  "Autoencoder", "Isolation Forest"],
    "Accuracy":  [98.1,    97.3,      96.8,            94.2,   91.5,          89.3],
    "Precision": [98.3,    97.5,      97.1,            None,   90.8,          88.5],
    "Recall":    [97.9,    97.1,      96.2,            None,   92.3,          90.1],
    "F1":        [98.1,    97.3,      96.6,            None,   91.5,          89.3],
    "FPR":       [1.8,     1.9,       2.1,             3.4,    4.2,           5.1],
}

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "figures",
)

# Colour scheme: green=excellent (≥97%), amber=good (94–97%), red=baseline (<94%)
PERFORMANCE_COLOURS = {
    "LSTM":             "#2ecc71",
    "XGBoost":          "#2ecc71",
    "Random Forest":    "#f39c12",
    "SVM":              "#f39c12",
    "Autoencoder":      "#e74c3c",
    "Isolation Forest": "#e74c3c",
}


def plot_paper_performance_comparison(save: bool = True) -> plt.Figure:
    """
    Reproduce Figure 3 from the paper — model performance comparison.

    Accuracy and F1 bar chart for all 6 models on NSL-KDD dataset.
    Colour-coded: Green ≥97% (Excellent), Amber 94–97% (Good), Red <94% (Baseline).
    """
    models = PAPER_RESULTS["Model"]
    accuracy = PAPER_RESULTS["Accuracy"]
    f1 = [v if v else 0 for v in PAPER_RESULTS["F1"]]
    colours = [PERFORMANCE_COLOURS[m] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "ML Algorithm Performance Comparison — NSL-KDD Dataset\n"
        "Source: Machethe (2026), Table 5, ECCU500, EC-Council University",
        fontsize=13, fontweight="bold",
    )

    # Accuracy bars
    bars = axes[0].barh(models, accuracy, color=colours, edgecolor="white", height=0.6)
    axes[0].set_xlabel("Accuracy (%)")
    axes[0].set_title("Detection Accuracy")
    axes[0].set_xlim(85, 100)
    axes[0].axvline(x=97, color="grey", linestyle="--", alpha=0.5, label="Excellent threshold (97%)")
    axes[0].axvline(x=94, color="grey", linestyle=":", alpha=0.5, label="Good threshold (94%)")
    for bar, val in zip(bars, accuracy):
        axes[0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9)
    axes[0].legend(fontsize=8)

    # F1 bars
    bars_f1 = axes[1].barh(models, f1, color=colours, edgecolor="white", height=0.6)
    axes[1].set_xlabel("F1-Score (%)")
    axes[1].set_title("F1-Score")
    axes[1].set_xlim(85, 100)
    axes[1].axvline(x=97, color="grey", linestyle="--", alpha=0.5)
    axes[1].axvline(x=94, color="grey", linestyle=":", alpha=0.5)
    for bar, val, model in zip(bars_f1, f1, models):
        label = f"{val:.1f}%" if val > 0 else "N/A"
        axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     label, va="center", fontsize=9)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Excellent (≥97%)"),
        mpatches.Patch(color="#f39c12", label="Good (94–97%)"),
        mpatches.Patch(color="#e74c3c", label="Baseline (<94%)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9, frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "figure3_model_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


def plot_zta_pillar_framework(save: bool = True) -> plt.Figure:
    """
    Visualise the ZTA 5-pillar × 6-phase integration framework.

    ZTA Pillars (NIST SP 800-207): Identity, Devices, Networks, Applications, Data
    Implementation Phases: Foundation, Data Collection, Model Development,
                           Integration, Validation, Operationalisation
    """
    pillars = ["Identity", "Devices", "Networks", "Applications", "Data"]
    phases = [
        "Phase 1\nFoundation",
        "Phase 2\nData Collection",
        "Phase 3\nModel Dev.",
        "Phase 4\nIntegration",
        "Phase 5\nValidation",
        "Phase 6\nOperationalisation",
    ]

    # ML-ZTA mapping (which models active in which pillar/phase cells)
    # Based on paper Table 2 and Section 7 recommendations
    cell_labels = [
        ["IAM baseline",   "Identity logs",   "UEBA model",      "Auth anomaly",    "Red team ID",  "SOAR playbook"],
        ["Device inv.",    "Endpoint logs",   "Behaviour model", "Trust scoring",   "Device tests", "MDM dashboards"],
        ["Micro-seg.",     "Flow capture",    "RF + LSTM train", "SIEM deploy",     "Pen testing",  "SOC monitoring"],
        ["App inventory",  "API logs",        "AE model",        "WAF integration", "App scanning", "API analytics"],
        ["Data classify",  "DLP baseline",    "Exfil. model",    "DLP auto-resp.",  "Data leakage", "Compliance rpt"],
    ]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, len(phases))
    ax.set_ylim(0, len(pillars))
    ax.set_aspect("auto")

    pillar_colours = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]

    for i, pillar in enumerate(pillars):
        for j, phase in enumerate(phases):
            colour = pillar_colours[i]
            rect = plt.Rectangle((j, i), 1, 1, linewidth=1,
                                  edgecolor="white", facecolor=colour, alpha=0.25)
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, cell_labels[i][j],
                    ha="center", va="center", fontsize=7.5, color="#2c3e50",
                    wrap=True)

    # Pillar labels (y-axis)
    ax.set_yticks([i + 0.5 for i in range(len(pillars))])
    ax.set_yticklabels(
        [f"  {p}" for p in pillars],
        fontsize=11, fontweight="bold",
    )
    for label, colour in zip(ax.get_yticklabels(), pillar_colours):
        label.set_color(colour)

    # Phase labels (x-axis)
    ax.set_xticks([j + 0.5 for j in range(len(phases))])
    ax.set_xticklabels(phases, fontsize=9, ha="center")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_title(
        "ZTA Six-Phase ML-IDS Implementation Framework\n"
        "Five Pillars (NIST SP 800-207) × Six Implementation Phases\n"
        "Source: Machethe (2026), Table 4, ECCU500, EC-Council University",
        fontsize=11, pad=40,
    )
    ax.tick_params(left=False, top=False)
    ax.spines[:].set_visible(False)

    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "zta_framework_grid.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save: bool = True,
) -> plt.Figure:
    """Plot a labelled confusion matrix for a binary classifier."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fname = model_name.lower().replace(" ", "_")
        path = os.path.join(FIGURES_DIR, f"confusion_matrix_{fname}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(
    roc_data: dict,
    save: bool = True,
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on one figure.

    roc_data: {model_name: {"fpr": array, "tpr": array, "auc": float}}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#e74c3c"]

    for (name, data), colour in zip(roc_data.items(), colours):
        ax.plot(data["fpr"], data["tpr"], color=colour, lw=2,
                label=f"{name} (AUC = {data['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(
        "ROC Curves — All Models (NSL-KDD)\n"
        "Source: Machethe (2026), ECCU500, EC-Council University"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "roc_curves_all_models.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 15,
    model_name: str = "Random Forest",
    save: bool = True,
) -> plt.Figure:
    """
    Plot top-N feature importances (Random Forest or XGBoost).

    Top 15 features from Appendix B of the research paper.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ["#2ecc71" if i < 5 else "#3498db" if i < 10 else "#f39c12"
               for i in range(top_n)]
    ax.barh(top_features[::-1], top_importances[::-1], color=colours[::-1])
    ax.set_xlabel("Feature Importance Score")
    ax.set_title(
        f"Top {top_n} Feature Importances — {model_name}\n"
        "NSL-KDD Dataset | Ref: Appendix B, Machethe (2026)"
    )
    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fname = model_name.lower().replace(" ", "_")
        path = os.path.join(FIGURES_DIR, f"feature_importance_{fname}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
