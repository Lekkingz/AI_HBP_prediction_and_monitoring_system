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
# METRICS
# =========================

accuracy = [
    0.6615,
    0.3995,
    0.4070,
    0.4645
]

f1_scores = [
    0.6572,
    0.2583,
    0.2355,
    0.3715
]

auc_scores = [
    0.8212,
    0.5485,
    0.5213,
    0.6199
]


# =========================
# CREATE FIGURE
# =========================

x = np.arange(len(models))

width = 0.25


fig, ax = plt.subplots(
    figsize=(10,6)
)


# =========================
# BAR CHARTS
# =========================

ax.bar(
    x - width,
    accuracy,
    width,
    label='Accuracy'
)

ax.bar(
    x,
    f1_scores,
    width,
    label='F1-Score'
)

ax.bar(
    x + width,
    auc_scores,
    width,
    label='AUC'
)


# =========================
# LABELS
# =========================

ax.set_ylabel(
    'Scores'
)

ax.set_title(
    'Model Performance Comparison'
)

ax.set_xticks(x)

ax.set_xticklabels(models)

ax.legend()


# =========================
# SAVE FIGURE
# =========================

plt.savefig(
    "../docs/model_comparison.png"
)

print(
    "\nComparison chart saved successfully."
)

plt.show()
