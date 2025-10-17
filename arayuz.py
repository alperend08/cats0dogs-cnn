import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
import joblib
from pygame import mixer
import threading
import os

# === MODEL YOLLARI === #
BASE_DIR = "cats_dogs"
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# === MODEL YÜKLE === #
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)

# === SEÇİLEN DOSYA === #
secili_dosya = None


# === ÖN İŞLEME + ÖZELLİK ÇIKARIMI === #
def ozellik_cikar(dosya_yolu):
    y, sr = librosa.load(dosya_yolu, sr=None)

    # 1️⃣ Sessizlikleri kes
    y, _ = librosa.effects.trim(y)

    # 2️⃣ Normalize et
    y = librosa.util.normalize(y)

    # 3️⃣ Süreyi sabitle (model 3 saniyelik veriye alışık)
    hedef_uzunluk = int(3 * sr)
    y = librosa.util.fix_length(y, size=hedef_uzunluk)

    # 4️⃣ Özellikleri çıkar
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    features = np.hstack([mfcc_mean, mfcc_std, mel_mean, zcr])
    return features.reshape(1, -1)


# === SESİ TAHMİN ET === #
def ses_tahmin_et():
    global secili_dosya
    dosya_yolu = filedialog.askopenfilename(
        title="Ses Dosyası Seç",
        filetypes=[("WAV files", "*.wav")]
    )
    if not dosya_yolu:
        return

    secili_dosya = dosya_yolu

    try:
        ozellik = ozellik_cikar(dosya_yolu)
        ozellik = scaler.transform(ozellik)
        tahmin = model.predict(ozellik)
        etiket = le.inverse_transform(tahmin)[0]

        if etiket == "cat":
            sonuc_label.config(text="🐱 Bu ses: KEDİ", fg="purple")
        elif etiket == "dog":
            sonuc_label.config(text="🐶 Bu ses: KÖPEK", fg="darkgreen")
        else:
            sonuc_label.config(text="❓ Tanınamadı", fg="red")

        cal_buton.config(state="normal")

    except Exception as e:
        messagebox.showerror("Hata", str(e))


# === SESİ ÇAL === #
def sesi_cal():
    global secili_dosya
    if not secili_dosya:
        messagebox.showwarning("Uyarı", "Önce bir ses dosyası seçin.")
        return

    def oynat():
        mixer.init()
        mixer.music.load(secili_dosya)
        mixer.music.play()

    threading.Thread(target=oynat).start()


# === ARAYÜZ === #
pencere = tk.Tk()
pencere.title("Cats & Dogs Ses Sınıflandırıcı")
pencere.geometry("370x250")
pencere.resizable(False, False)

baslik = tk.Label(pencere, text="🐾 Cats & Dogs Sınıflandırıcı", font=("Arial", 14, "bold"))
baslik.pack(pady=10)

sec_buton = tk.Button(pencere, text="🎵 Ses Dosyası Seç", command=ses_tahmin_et, font=("Arial", 11))
sec_buton.pack(pady=5)

cal_buton = tk.Button(pencere, text="▶️ Sesi Çal", command=sesi_cal, font=("Arial", 11), state="disabled")
cal_buton.pack(pady=5)

sonuc_label = tk.Label(pencere, text="", font=("Arial", 13))
sonuc_label.pack(pady=20)

pencere.mainloop()
