import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

def load_dataset(base_dir, target_len=300):
    X, y = [], []
    for label, category in enumerate(["legit", "cheater"]):
        path = os.path.join(base_dir, category)
        for file in os.listdir(path):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(path, file))

                    # Drop non-numeric / irrelevant columns
                    df = df.drop(columns=["tick", "steamid", "label", "weapon_name", "weapon_type"], errors="ignore")

                    if df.shape[0] >= 290:
                        if df.shape[0] < target_len:
                            pad_rows = target_len - df.shape[0]
                            last_row = df.iloc[[-1]].copy()
                            padding = pd.concat([last_row] * pad_rows, ignore_index=True)
                            df = pd.concat([df, padding], ignore_index=True)

                        X.append(df.values[:target_len])
                        y.append(label)

                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")
    return np.array(X), np.array(y)


# Load dataset
print("📥 Loading dataset...")
X, y = load_dataset("data/processed/features")
print(f"📊 Feature shape: {X.shape}, Labels: {y.shape}")

# Normalize features
print("🧪 Scaling features...")
X_flat = X.reshape(-1, X.shape[-1])
scaler = StandardScaler()
X_scaled_flat = scaler.fit_transform(X_flat)
X_scaled = X_scaled_flat.reshape(X.shape)

# Train/val/test split
print("🧪 Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"✅ Train: {X_train.shape}")
print(f"✅ Val:   {X_val.shape}")
print(f"✅ Test:  {X_test.shape}")

# Show class balance
counts = np.bincount(y)
print(f"\nClass Distribution:")
print(f"  Legit: {counts[0]}")
print(f"  Cheat: {counts[1]}\n")

# Build model
print("🧠 Building model...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(300, X.shape[2])),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
print("🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8
)

# Evaluate model
print("\n📈 Evaluating on test set...")
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Cheat"])
plt.figure(figsize=(4, 4))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix on Test Set")
plt.tight_layout()
plt.show()

