import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --------------------- Load and Normalize ---------------------
def load_legit_cheat(base_dir, target_len=300):
    X, y = [], []
    for label, category in enumerate(["legit", "cheater"]):
        path = os.path.join(base_dir, category)
        for file in os.listdir(path):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(path, file))
                    features = df.drop(columns=["tick", "steamid", "label"], errors="ignore")

                    if features.shape[0] >= 290:
                        if features.shape[0] < target_len:
                            pad_rows = target_len - features.shape[0]
                            padding = pd.DataFrame(np.zeros((pad_rows, features.shape[1])), columns=features.columns)
                            features = pd.concat([features, padding], ignore_index=True)

                        X.append(features.values[:target_len])
                        y.append(label)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    return np.array(X), np.array(y)

# --------------------- Build Autoencoder ---------------------
def build_autoencoder(seq_len, n_features):
    inp = Input(shape=(seq_len, n_features))
    encoded = LSTM(64, activation='tanh')(inp)
    decoded = RepeatVector(seq_len)(encoded)
    decoded = LSTM(n_features, activation='tanh', return_sequences=True)(decoded)

    model = Model(inputs=inp, outputs=decoded)
    opt = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=opt, loss='mse')
    return model

# --------------------- Main ---------------------
if __name__ == "__main__":
    base_dir = "data/processed/features"
    X, y = load_legit_cheat(base_dir)
    print("✅ Loaded dataset:", X.shape, y.shape)

    # Normalize features
    X_flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(X.shape)

    # Use only legit for training
    X_legit = X_scaled[y == 0]
    X_train, X_val = train_test_split(X_legit, test_size=0.2, random_state=42)

    print("🚀 Training LSTM Autoencoder...")
    model = build_autoencoder(seq_len=300, n_features=X.shape[2])
    model.summary()

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=20,
        batch_size=16,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    print("📈 Evaluating...")
    X_pred = model.predict(X_scaled)
    mse = np.mean(np.square(X_scaled - X_pred), axis=(1, 2))

    threshold = np.percentile(mse[y == 0], 80)
    print(f"🔎 Threshold (95th percentile of legit): {threshold:.6f}")

    y_pred = (mse > threshold).astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(y, y_pred, target_names=["Legit", "Cheat"]))

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Cheat"])
    disp.plot(cmap="Blues")
    plt.title("Autoencoder Confusion Matrix")
    plt.tight_layout()
    plt.show()
