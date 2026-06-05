import h5py
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical


# =========================
# LOAD DATASET
# =========================

with h5py.File(
    "../dataset/processed/ppg_subset.h5",
    "r"
) as f:

    X = f["ppg"][:]

    y = f["label"][:]


# =========================
# USE ONLY SYSTOLIC LABEL
# =========================

y = y[:, 0]


# =========================
# CREATE CLASSES
# =========================

classes = []

for bp in y:

    if bp < 120:
        classes.append("NORMAL")

    elif bp < 140:
        classes.append("ELEVATED")

    else:
        classes.append("HIGH")

classes = np.array(classes)


# =========================
# ENCODE LABELS
# =========================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(
    classes
)

y_categorical = to_categorical(
    y_encoded
)


# =========================
# RESHAPE INPUT
# =========================

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_categorical,

    test_size=0.2,

    random_state=42
)


# =========================
# BUILD CNN MODEL
# =========================

model = Sequential([

    Conv1D(
        32,
        kernel_size=3,
        activation='relu',
        input_shape=(875,1)
    ),

    MaxPooling1D(pool_size=2),

    Conv1D(
        64,
        kernel_size=3,
        activation='relu'
    ),

    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.3),

    Dense(
        3,
        activation='softmax'
    )
])


# =========================
# COMPILE MODEL
# =========================

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']
)


# =========================
# TRAIN MODEL
# =========================

history = model.fit(

    X_train,
    y_train,

    epochs=10,

    batch_size=32,

    validation_split=0.2
)


# =========================
# PREDICTIONS
# =========================

y_pred_probs = model.predict(
    X_test
)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)


# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_true,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_true,
    y_pred,
    average='weighted'
)

auc = roc_auc_score(
    y_test,
    y_pred_probs,
    multi_class='ovr'
)


# =========================
# PRINT RESULTS
# =========================

print("\n===== CNN RESULTS =====\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")


# =========================
# SAVE MODEL
# =========================

model.save(
    "../dataset/models/cnn_model.h5"
)

print("\nCNN Model Saved Successfully")
