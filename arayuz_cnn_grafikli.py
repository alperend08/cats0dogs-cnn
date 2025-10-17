import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from pygame import mixer
import threading
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === MODEL YOLLARI === #
BASE_DIR = "cats_dogs"
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model_final.keras")
SCALER_PATH = os.path.join(BASE_DIR, "cnn_scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "cnn_label_encoder.pkl")

# === MODELİ YÜKLE === #
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)

secili_dosya = None

# === ÖZELLİK ÇIKARIM === #
def ozellik_cikar(dosya_yolu):
    y, sr = librosa.load(dosya_yolu, sr=None)
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)
    hedef_uzunluk = int(3 * sr)
    y = librosa.util.fix_length(y, size=hedef_uzunluk)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    features = np.hstack([mfcc_mean, mfcc_std, mel_mean, zcr])
    features = scaler.transform(features.reshape(1, -1))
    features = features.reshape(features.shape[0], features.shape[1], 1, 1)
    return y, sr, features

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

# === GÖRSEL GRAFİK OLUŞTUR === #
def grafikleri_goster(y, sr):
    fig, axs = plt.subplots(2, 1, figsize=(6, 4), dpi=100)
    fig.tight_layout(pad=3.0)

    # 1️⃣ Dalga formu (waveform)
    librosa.display.waveshow(y, sr=sr, ax=axs[0], color="royalblue")
    axs[0].set_title("Dalga Formu (Waveform)")
    axs[0].set_xlabel("Zaman (s)")
    axs[0].set_ylabel("Genlik")

    # 2️⃣ Mel Spektrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=axs[1], cmap="magma")
    axs[1].set_title("Mel Spektrogram")
    fig.colorbar(img, ax=axs[1], format="%+2.f dB")

    # Önceki grafiği temizle
    for widget in frame_grafik.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame_grafik)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# === TAHMİN === #
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
        y, sr, ozellik = ozellik_cikar(dosya_yolu)
        tahmin = model.predict(ozellik)
        prob = float(tahmin[0])  # sigmoid çıkışı (0-1 arası olasılık)

        # 🔁 Olasılığı ve etiketi doğru şekilde hesapla
        if prob > 0.5:
            etiket = "dog"
            olasilik = prob * 100
        else:
            etiket = "cat"
            olasilik = (1 - prob) * 100

        # 🎨 Grafikleri güncelle
        grafikleri_goster(y, sr)

        # 🔤 Sonucu yazdır
        if etiket == "cat":
            sonuc_label.config(
                text=f"🐱 Bu ses: KEDİ\n(%{olasilik:.2f} olasılıkla)",
                fg="purple"
            )
        elif etiket == "dog":
            sonuc_label.config(
                text=f"🐶 Bu ses: KÖPEK\n(%{olasilik:.2f} olasılıkla)",
                fg="darkgreen"
            )
        else:
            sonuc_label.config(text="❓ Tanınamadı", fg="red")

        # ▶️ Ses çal butonunu aktif et
        cal_buton.config(state="normal")

    except Exception as e:
        messagebox.showerror("Hata", str(e))


# === TKINTER ARAYÜZ === #
pencere = tk.Tk()
pencere.title("Cats & Dogs CNN Grafikli Sınıflandırıcı")
pencere.geometry("800x600")
pencere.resizable(False, False)

# Başlık
baslik = tk.Label(pencere, text="🐾 Cats & Dogs (CNN) - Grafikli Arayüz", font=("Arial", 16, "bold"))
baslik.pack(pady=10)

# Butonlar
sec_buton = tk.Button(pencere, text="🎵 Ses Dosyası Seç", command=ses_tahmin_et, font=("Arial", 12))
sec_buton.pack(pady=5)

cal_buton = tk.Button(pencere, text="▶️ Sesi Çal", command=sesi_cal, font=("Arial", 12), state="disabled")
cal_buton.pack(pady=5)

# Sonuç etiketi
sonuc_label = tk.Label(pencere, text="", font=("Arial", 13))
sonuc_label.pack(pady=10)

# Grafik alanı
frame_grafik = tk.Frame(pencere, width=760, height=400)
frame_grafik.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

pencere.mainloop()
