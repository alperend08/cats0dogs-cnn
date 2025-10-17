import os
import librosa
import numpy as np
import pandas as pd

# === Klasör yolları === #
BASE_DIR = "cats_dogs"
INPUT_DIR = os.path.join(BASE_DIR, "islenmis_sesler")
OUTPUT_CSV = os.path.join(BASE_DIR, "ses_ozellikleri.csv")


# === Özellik çıkarma fonksiyonu === #
def ozellik_cikar(dosya_yolu):
    y, sr = librosa.load(dosya_yolu, sr=None)

    # MFCC (ortalama + std)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Mel Spectrogram (ortalama)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)

    # Zero Crossing Rate (ortalama)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Tüm özellikleri birleştir
    features = np.hstack([mfcc_mean, mfcc_std, mel_mean, zcr_mean])

    return features


# === Tüm dosyaları işle === #
ozellik_listesi = []
etiket_listesi = []

for dosya in os.listdir(INPUT_DIR):
    if dosya.lower().endswith(".wav"):
        yol = os.path.join(INPUT_DIR, dosya)
        ozellik = ozellik_cikar(yol)
        ozellik_listesi.append(ozellik)

        # Etiket tahmini: dosya adında "cat" veya "dog" varsa
        if "cat" in dosya.lower():
            etiket_listesi.append("cat")
        elif "dog" in dosya.lower():
            etiket_listesi.append("dog")
        else:
            etiket_listesi.append("unknown")

# === DataFrame olarak kaydet === #
df = pd.DataFrame(ozellik_listesi)
df["label"] = etiket_listesi
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Özellik çıkarımı tamamlandı! {OUTPUT_CSV} dosyası oluşturuldu.")
