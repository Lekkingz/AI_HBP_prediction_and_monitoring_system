import h5py
import numpy as np

# =========================
# Load Dataset
# =========================

file = h5py.File(
    "../dataset/processed/ppg_subset.h5",
    "r"
)

ppg = file["ppg"][:]

labels = file["label"][:]

# =========================
# Extract Systolic BP
# =========================

systolic_bp = labels[:,0]

# =========================
# Convert to Classes
# =========================

classes = []

for bp in systolic_bp:

    if bp < 120:
        classes.append(0)   # NORMAL

    elif bp < 140:
        classes.append(1)   # ELEVATED

    else:
        classes.append(2)   # HIGH

classes = np.array(classes)

# =========================
# Save Classification Data
# =========================

np.save(
    "../dataset/processed/X_class.npy",
    ppg
)

np.save(
    "../dataset/processed/y_class.npy",
    classes
)

print("\n===== CLASSIFICATION DATASET CREATED =====\n")

print("PPG Shape:", ppg.shape)

print("Class Shape:", classes.shape)

print("\nClass Distribution:")

unique, counts = np.unique(
    classes,
    return_counts=True
)

for u, c in zip(unique, counts):

    print(f"Class {u}: {c}")
