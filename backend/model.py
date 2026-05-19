import numpy as np
import librosa
import os
import pickle

# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────

def extract_features(audio_data: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract MFCC + spectral features from raw audio."""
    features = []

    # MFCCs (40 coefficients, mean + std)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
    features.extend(np.mean(contrast, axis=1))

    # Mel spectrogram (mean per band)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    features.extend(np.mean(mel, axis=1)[:20])

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)
    features.append(float(np.mean(zcr)))
    features.append(float(np.std(zcr)))

    # RMS energy
    rms = librosa.feature.rms(y=audio_data)
    features.append(float(np.mean(rms)))
    features.append(float(np.std(rms)))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
    features.append(float(np.mean(rolloff)))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    features.append(float(np.mean(centroid)))

    return np.array(features, dtype=np.float32)


# ──────────────────────────────────────────────
# Synthetic dataset generation
# (replaces real dataset when none is available)
# ──────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 600, sr: int = 22050):
    """
    Generate synthetic audio signals that mimic different cough categories.
    Categories:
      0 – Healthy / No respiratory issue
      1 – Mild risk  (possible upper respiratory)
      2 – High risk  (possible lower respiratory / COVID-like pattern)
    """
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    X, y = [], []

    rng = np.random.default_rng(42)

    for _ in range(n_samples):
        label = rng.integers(0, 3)

        if label == 0:
            # Healthy: clean tone, low noise
            freq = rng.uniform(200, 400)
            signal = 0.5 * np.sin(2 * np.pi * freq * t)
            signal += rng.normal(0, 0.02, len(t))

        elif label == 1:
            # Mild: multi-harmonic, moderate noise
            freq = rng.uniform(150, 300)
            signal = (0.4 * np.sin(2 * np.pi * freq * t) +
                      0.2 * np.sin(2 * np.pi * freq * 2 * t) +
                      0.1 * np.sin(2 * np.pi * freq * 3 * t))
            signal += rng.normal(0, 0.08, len(t))

        else:
            # High risk: irregular, burst-like, high noise
            freq = rng.uniform(80, 200)
            signal = 0.3 * np.sin(2 * np.pi * freq * t)
            signal += 0.4 * rng.uniform(-1, 1, len(t))
            # Add burst pattern
            burst_start = rng.integers(0, len(t) // 2)
            signal[burst_start:burst_start + sr // 4] *= 2.5

        signal = signal.astype(np.float32)
        # Clip to [-1, 1]
        signal = np.clip(signal / (np.max(np.abs(signal)) + 1e-6), -1, 1)
        features = extract_features(signal, sr)
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
# Simple MLP model (no TensorFlow dependency)
# ──────────────────────────────────────────────

class SimpleMLP:
    """Lightweight multi-layer perceptron trained with numpy."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3):
        rng = np.random.default_rng(0)
        # Layer 1
        self.W1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        # Layer 2
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, 64)).astype(np.float32)
        self.b2 = np.zeros(64, dtype=np.float32)
        # Output
        self.W3 = rng.normal(0, np.sqrt(2 / 64), (64, output_dim)).astype(np.float32)
        self.b3 = np.zeros(output_dim, dtype=np.float32)

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, X):
        h1 = self._relu(X @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        return self._softmax(h2 @ self.W3 + self.b3)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1), probs

    def _cross_entropy(self, probs, y):
        n = len(y)
        correct_log_probs = -np.log(probs[np.arange(n), y] + 1e-9)
        return np.mean(correct_log_probs)

    def train(self, X, y, epochs=200, lr=0.01, batch_size=64):
        n = len(X)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]

            for start in range(0, n, batch_size):
                Xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                m = len(yb)

                # Forward
                h1 = self._relu(Xb @ self.W1 + self.b1)
                h2 = self._relu(h1 @ self.W2 + self.b2)
                out = self._softmax(h2 @ self.W3 + self.b3)

                # Loss grad
                dout = out.copy()
                dout[np.arange(m), yb] -= 1
                dout /= m

                # Backprop layer 3
                dW3 = h2.T @ dout
                db3 = dout.sum(axis=0)
                dh2 = dout @ self.W3.T

                # Backprop layer 2
                dh2_relu = dh2 * (h2 > 0)
                dW2 = h1.T @ dh2_relu
                db2 = dh2_relu.sum(axis=0)
                dh1 = dh2_relu @ self.W2.T

                # Backprop layer 1
                dh1_relu = dh1 * (h1 > 0)
                dW1 = Xb.T @ dh1_relu
                db1 = dh1_relu.sum(axis=0)

                # Update
                self.W3 -= lr * dW3
                self.b3 -= lr * db3
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

            if epoch % 50 == 0:
                preds_all = self.forward(X)
                loss = self._cross_entropy(preds_all, y)
                acc = np.mean(np.argmax(preds_all, axis=1) == y)
                print(f"  Epoch {epoch:3d} | loss={loss:.4f} | acc={acc:.3f}")


# ──────────────────────────────────────────────
# Feature normalizer
# ──────────────────────────────────────────────

class Normalizer:
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cough_model.pkl")

CLASSES = ["Healthy", "Mild Risk", "High Risk"]
DESCRIPTIONS = {
    "Healthy": "No significant respiratory distress patterns detected. Your cough sound appears normal.",
    "Mild Risk": "Some irregular patterns detected that may indicate upper respiratory involvement. Consider monitoring your symptoms.",
    "High Risk": "Significant acoustic irregularities detected, consistent with patterns associated with lower respiratory conditions. Recommend consulting a healthcare professional.",
}
COLORS = {
    "Healthy": "green",
    "Mild Risk": "orange",
    "High Risk": "red",
}
RECOMMENDATIONS = {
    "Healthy": ["Stay hydrated", "Maintain good ventilation", "Continue regular health monitoring"],
    "Mild Risk": ["Rest and stay hydrated", "Monitor temperature", "Avoid strenuous activity", "Consult a doctor if symptoms persist > 3 days"],
    "High Risk": ["Seek medical attention promptly", "Isolate as a precaution", "Monitor oxygen levels if possible", "Keep emergency contacts ready"],
}


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading saved model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    print("Training new model on synthetic dataset...")
    X, y = generate_synthetic_dataset(n_samples=800)

    normalizer = Normalizer()
    X_norm = normalizer.fit_transform(X)

    mlp = SimpleMLP(input_dim=X_norm.shape[1], hidden_dim=128, output_dim=3)
    mlp.train(X_norm, y, epochs=300, lr=0.005, batch_size=64)

    bundle = {"model": mlp, "normalizer": normalizer}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model saved to {MODEL_PATH}")
    return bundle


def predict_cough(bundle, audio_data: np.ndarray, sr: int):
    mlp: SimpleMLP = bundle["model"]
    normalizer: Normalizer = bundle["normalizer"]

    features = extract_features(audio_data, sr)
    features_norm = normalizer.transform(features.reshape(1, -1))

    labels_idx, probs = mlp.predict(features_norm)
    label_idx = int(labels_idx[0])
    probs_list = probs[0].tolist()

    label = CLASSES[label_idx]
    confidence = float(probs_list[label_idx])

    # Audio analytics
    duration = len(audio_data) / sr
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    db_level = float(20 * np.log10(rms + 1e-9))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)))

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 1),
        "risk_level": label,
        "color": COLORS[label],
        "description": DESCRIPTIONS[label],
        "recommendations": RECOMMENDATIONS[label],
        "probabilities": {
            CLASSES[i]: round(probs_list[i] * 100, 1) for i in range(3)
        },
        "audio_stats": {
            "duration_seconds": round(duration, 2),
            "rms_energy": round(rms, 4),
            "db_level": round(db_level, 1),
            "zero_crossing_rate": round(zcr, 4),
        },
    }
