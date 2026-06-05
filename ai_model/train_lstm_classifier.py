import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================
# Load Dataset
# =========================

X = np.load(
    "../dataset/processed/X_class.npy"
)

y = np.load(
    "../dataset/processed/y_class.npy"
)

# =========================
# Reshape Input
# =========================

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

# =========================
# One-Hot Encode Labels
# =========================

y_cat = to_categorical(
    y,
    num_classes=3
)

# =========================
# Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_cat,

    test_size=0.2,

    random_state=42
)

# =========================
# Build LSTM Model
# =========================

model = Sequential()

model.add(

    LSTM(
        64,
        input_shape=(875,1)
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        64,
        activation='relu'
    )
)

model.add(
    Dense(
        3,
        activation='softmax'
    )
)

# =========================
# Compile Model
# =========================

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# =========================
# Train Model
# =========================

history = model.fit(

    X_train,
    y_train,

    epochs=10,

    batch_size=32,

    validation_split=0.2
)

# =========================
# Predictions
# =========================

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(
    y_pred,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)

# =========================
# Metrics
# =========================

accuracy = accuracy_score(
    y_true,
    y_pred_classes
)

precision = precision_score(
    y_true,
    y_pred_classes,
    average='weighted'
)

recall = recall_score(
    y_true,
    y_pred_classes,
    average='weighted'
)

f1 = f1_score(
    y_true,
    y_pred_classes,
    average='weighted'
)

auc = roc_auc_score(
    y_test,
    y_pred,
    multi_class='ovr'
)

# =========================
# Print Results
# =========================

print("\n===== LSTM RESULTS =====\n")

print(f"Accuracy   : {accuracy:.4f}")

print(f"Precision  : {precision:.4f}")

print(f"Recall     : {recall:.4f}")

print(f"F1-Score   : {f1:.4f}")

print(f"AUC        : {auc:.4f}")

# =========================
# Save Model
# =========================

model.save(
    "../dataset/models/lstm_classifier.h5"
)

print("\nLSTM Model Saved Successfully")
