import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import joblib
import os

# === Veri setini oku === #
DATA_PATH = "cats_dogs/ses_ozellikleri.csv"
df = pd.read_csv(DATA_PATH)

# Özellik ve etiketleri ayır
X = df.drop("label", axis=1).values
y = df["label"].values

# Etiketleri sayısal hale getir
le = LabelEncoder()
y = le.fit_transform(y)

# Özellikleri ölçekle
scaler = StandardScaler()
X = scaler.fit_transform(X)

# CNN 2D giriş beklediği için yeniden şekillendir
# (örnek_sayısı, zaman, özellik_sayısı, kanal)
X = X.reshape(X.shape[0], X.shape[1], 1, 1)

# Eğitim/Test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === CNN Modeli === #
model = models.Sequential([
    layers.Conv2D(32, (3,1), activation='relu', input_shape=(X.shape[1], 1, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,1), activation='relu'),
    layers.MaxPooling2D((2,1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# === Eğitimi başlat === #
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    verbose=1
)

# === Model değerlendirme === #
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ CNN modeli eğitimi tamamlandı.")
print(f"🎯 Test Doğruluğu: {acc*100:.2f}%")

# === Eğitim grafiği === #
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Test Doğruluğu')
plt.title("CNN Modeli Eğitim Süreci")
plt.xlabel("Epoch")
plt.ylabel("Doğruluk")
plt.legend()
plt.tight_layout()
plt.show()

# === Model ve scaler'ı kaydet === #
os.makedirs("cats_dogs", exist_ok=True)
model.save("cats_dogs/cnn_model.keras")
joblib.dump(scaler, "cats_dogs/cnn_scaler.pkl")
joblib.dump(le, "cats_dogs/cnn_label_encoder.pkl")
