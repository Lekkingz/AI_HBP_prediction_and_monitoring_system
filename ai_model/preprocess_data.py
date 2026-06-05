import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# =========================
# Load Dataset
# =========================

file_path = "../dataset/processed/ppg_subset.h5"

with h5py.File(file_path, 'r') as f:

    X = f['ppg'][:]
    y = f['label'][:]

# =========================
# Use ONLY systolic BP
# =========================

y = y[:, 0]

# =========================
# Normalize Signals
# =========================

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

# =========================
# Reshape for CNN-BiLSTM
# =========================

X = X.reshape((X.shape[0], X.shape[1], 1))

# =========================
# Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Save Preprocessed Data
# =========================

np.save("../dataset/processed/X_train.npy", X_train)
np.save("../dataset/processed/X_test.npy", X_test)

np.save("../dataset/processed/y_train.npy", y_train)
np.save("../dataset/processed/y_test.npy", y_test)

# =========================
# Print Shapes
# =========================

print("\n===== PREPROCESSING COMPLETE =====\n")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
