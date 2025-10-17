import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import librosa
import matplotlib.pyplot as plt

# === 1️⃣ Veri seti === #
DATA_PATH = "cats_dogs/ses_ozellikleri.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop("label", axis=1).values
y = df["label"].values

# Etiketleri sayısal hale getir
le = LabelEncoder()
y = le.fit_transform(y)

# Özellikleri ölçekle
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], X.shape[1], 1, 1)

# Eğitim ve test böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 2️⃣ Veri artırımı (Data Augmentation) === #
def augment_audio(y, sr):
    augmented = []
    # Gürültü ekle
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise
    augmented.append(y_noise)
    # Hız değiştir
    y_fast = librosa.effects.time_stretch(y, rate=1.2)
    y_slow = librosa.effects.time_stretch(y, rate=0.8)
    augmented.extend([y_fast, y_slow])
    # Pitch değiştir
    y_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    y_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    augmented.extend([y_up, y_down])
    return augmented

# === 3️⃣ CNN Modeli === #
model = models.Sequential([
    layers.Conv2D(32, (3,1), activation='relu', input_shape=(X.shape[1], 1, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,1)),

    layers.Conv2D(64, (3,1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,1)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # daha dengeli dropout
    layers.Dense(1, activation='sigmoid')
])

# === 4️⃣ Modeli derle === #
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# === 5️⃣ Early Stopping === #
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# === 6️⃣ Model eğitimi === #
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# === 7️⃣ Değerlendirme === #
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Final CNN modeli eğitimi tamamlandı.")
print(f"🎯 Test Doğruluğu: {acc*100:.2f}%")

# === 8️⃣ Grafik === #
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Test Doğruluğu')
plt.title("Final CNN Modeli Eğitim Süreci")
plt.xlabel("Epoch")
plt.ylabel("Doğruluk")
plt.legend()
plt.tight_layout()
plt.show()

# === 9️⃣ Modeli kaydet === #
os.makedirs("cats_dogs", exist_ok=True)
model.save("cats_dogs/cnn_model_final.keras")
joblib.dump(scaler, "cats_dogs/cnn_scaler.pkl")
joblib.dump(le, "cats_dogs/cnn_label_encoder.pkl")
