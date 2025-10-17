import os
import librosa
import soundfile as sf

# === Sabitler === #
BASE_DIR = "cats_dogs"  # Proje alt klasörün
DATA_DIR = os.path.join(BASE_DIR, "dataset")  # Ham sesler
OUTPUT_DIR = os.path.join(BASE_DIR, "islenmis_sesler")  # İşlenmiş sesler
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_DURATION = 3.0  # Sabit süre (saniye)


# === Fonksiyonlar === #
def ses_isle(file_path, output_path, duration=TARGET_DURATION):
    """Ses dosyasını kırp, normalize et ve sabit süreye getir."""
    y, sr = librosa.load(file_path, sr=None)

    # Sessizlikleri kes
    y, _ = librosa.effects.trim(y)

    # Normalize et
    y = librosa.util.normalize(y)

    # Süreyi sabitle (örnek: 3 saniye)
    hedef_uzunluk = int(duration * sr)
    if len(y) < hedef_uzunluk:
        y = librosa.util.fix_length(y, size=hedef_uzunluk)
    else:
        y = y[:hedef_uzunluk]

    # Kaydet
    sf.write(output_path, y, sr)
    print(f"✅ İşlendi: {os.path.basename(file_path)}")


def tum_sesleri_isle():
    """Klasördeki tüm .wav dosyalarını sırayla işler."""
    for dosya in os.listdir(DATA_DIR):
        if dosya.lower().endswith(".wav"):
            giris_yolu = os.path.join(DATA_DIR, dosya)
            cikis_yolu = os.path.join(OUTPUT_DIR, dosya)
            ses_isle(giris_yolu, cikis_yolu)


if __name__ == "__main__":
    print("🎵 SesArıtıcı başlatılıyor...")
    tum_sesleri_isle()
    print("🏁 Tüm sesler başarıyla işlendi!")
