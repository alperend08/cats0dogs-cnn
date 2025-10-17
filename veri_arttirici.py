import os
import librosa
import soundfile as sf
import numpy as np

# === Yollar === #
BASE_DIR = "cats_dogs"
INPUT_DIR = os.path.join(BASE_DIR, "islenmis_sesler")   # işlenmiş seslerin olduğu klasör
OUTPUT_DIR = os.path.join(BASE_DIR, "arttirilmis_sesler")  # yeni oluşturulacak klasör
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Ayarlar === #
TARGET_DURATION = 3.0  # saniye
AUG_PER_FILE = 5       # her ses için kaç varyasyon üretilecek

# === Yardımcı fonksiyon === #
def pad_or_trim(y, sr, target_duration=TARGET_DURATION):
    """Ses uzunluğunu sabitle (örnek: 3 sn)."""
    hedef_uzunluk = int(target_duration * sr)
    return librosa.util.fix_length(y, size=hedef_uzunluk)

# === Ana artırma fonksiyonu === #
def augment_audio(file_path, output_dir):
    y, sr = librosa.load(file_path, sr=None)
    y = librosa.util.normalize(y)
    y = pad_or_trim(y, sr)

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # 1️⃣ Gürültü ekleme
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise
    sf.write(os.path.join(output_dir, f"{base_name}_noise.wav"), y_noise, sr)

    # 2️⃣ Hız değiştirme (hızlı)
    y_fast = librosa.effects.time_stretch(y, rate=1.2)
    y_fast = pad_or_trim(y_fast, sr)
    sf.write(os.path.join(output_dir, f"{base_name}_fast.wav"), y_fast, sr)

    # 3️⃣ Hız değiştirme (yavaş)
    y_slow = librosa.effects.time_stretch(y, rate=0.8)
    y_slow = pad_or_trim(y_slow, sr)
    sf.write(os.path.join(output_dir, f"{base_name}_slow.wav"), y_slow, sr)

    # 4️⃣ Pitch yukarı (2 yarım nota)
    y_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    sf.write(os.path.join(output_dir, f"{base_name}_up.wav"), y_up, sr)

    # 5️⃣ Pitch aşağı (2 yarım nota)
    y_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    sf.write(os.path.join(output_dir, f"{base_name}_down.wav"), y_down, sr)

    print(f"✅ {base_name} için varyasyonlar oluşturuldu.")

# === Tüm dosyaları işle === #
def main():
    print("🎛️ Veri artırma işlemi başlatıldı...")
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(".wav"):
            augment_audio(os.path.join(INPUT_DIR, file), OUTPUT_DIR)
    print("🏁 Tüm sesler artırıldı! Yeni dosyalar 'arttirilmis_sesler' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
