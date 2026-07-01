import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# =========================
# Generate Sample Dataset
# =========================

np.random.seed(42)

samples = 1000

heart_rate = np.random.randint(60, 130, samples)
temperature = np.random.uniform(35.5, 39.5, samples)

# Risk labeling
risk = []

for hr, temp in zip(heart_rate, temperature):

    if hr > 100 or temp > 37.5:
        risk.append(1)
    else:
        risk.append(0)

# =========================
# Create DataFrame
# =========================

df = pd.DataFrame({
    'heart_rate': heart_rate,
    'temperature': temperature,
    'risk': risk
})

# =========================
# Features & Labels
# =========================

X = df[['heart_rate', 'temperature']].values
y = df['risk'].values

# Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Reshape for CNN-LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# Build CNN-BiLSTM Model
# =========================

model = Sequential()

model.add(
    Conv1D(
        filters=32,
        kernel_size=1,
        activation='relu',
        input_shape=(X.shape[1], 1)
    )
)

model.add(
    Bidirectional(
        LSTM(32)
    )
)

model.add(Dense(1, activation='sigmoid'))

# =========================
# Compile
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# Train
# =========================

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# =========================
# Evaluate
# =========================

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# =========================
# Save Model
# =========================

model.save("hbp_model.h5")

print("\nModel Saved Successfully")
