import matplotlib.pyplot as plt
import numpy as np


# =========================
# MODEL NAMES
# =========================

models = [
    "CNN",
    "LSTM",
    "BiLSTM",
    "CNN-BiLSTM"
]


# =========================
# FINAL METRICS
# =========================

accuracy = [
    0.7105,
    0.4095,
    0.4210,
    0.4585
]

precision = [
    0.7144,
    0.2568,
    0.4013,
    0.4068
]

recall = [
    0.7105,
    0.4095,
    0.4210,
    0.4585
]

f1_scores = [
    0.7093,
    0.2536,
    0.2791,
    0.3401
]

specificity = [
    0.8485,
    0.6701,
    0.6777,
    0.7020
]

auc_scores = [
    0.8532,
    0.5067,
    0.5643,
    0.5959
]


# =========================
# CREATE FIGURE
# =========================

x = np.arange(len(models))
width = 0.13

fig, ax = plt.subplots(figsize=(15, 7))


# =========================
# BAR CHARTS
# =========================

ax.bar(
    x - 2.5 * width,
    accuracy,
    width,
    label='Accuracy'
)

ax.bar(
    x - 1.5 * width,
    precision,
    width,
    label='Precision'
)

ax.bar(
    x - 0.5 * width,
    recall,
    width,
    label='Recall/Sensitivity'
)

ax.bar(
    x + 0.5 * width,
    f1_scores,
    width,
    label='F1-Score'
)

ax.bar(
    x + 1.5 * width,
    specificity,
    width,
    label='Specificity'
)

ax.bar(
    x + 2.5 * width,
    auc_scores,
    width,
    label='AUC'
)


# =========================
# LABELS
# =========================

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)


# =========================
# SAVE FIGURE
# =========================

plt.tight_layout()

plt.savefig(
    "../docs/model_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

print("\nComparison chart saved successfully.")

plt.show()
