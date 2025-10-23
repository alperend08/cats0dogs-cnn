# 🐾 Cats & Dogs CNN - Ses Sınıflandırıcı

Bu proje, **kedi ve köpek seslerini ayırt etmek** için geliştirilen bir **Convolutional Neural Network (CNN)** modelini ve bu modelin sonuçlarını görselleştiren bir **grafik arayüzü (GUI)** içerir.  
Uygulama, seçilen `.wav` dosyasını analiz eder, sesin dalga formu ve Mel-spektrogramını gösterir, ardından sesin **kedi mi yoksa köpek mi olduğunu** yüksek doğrulukla tahmin eder.

---

## 🎯 Amaç

Hayvan seslerini otomatik olarak tanıyabilen, yapay zeka tabanlı bir sistem geliştirmek.  
Bu sistem; akıllı ev asistanları, veterinerlik uygulamaları, ses tabanlı izleme sistemleri ve eğitim projelerinde kullanılabilir.

---

## 🧠 Model Özeti

Model, **Librosa** kütüphanesiyle çıkarılan ses özelliklerini kullanır:

- **MFCC (Mel-Frequency Cepstral Coefficients)**
- **Mel-spektrogram Enerjileri**
- **Zero Crossing Rate (ZCR)**
- Ortalama ve standart sapma değerleri

Bu özellikler **MinMaxScaler** ile normalize edilir ve bir **CNN (Convolutional Neural Network)** modeline beslenir.  
Modelin çıkışı `sigmoid` aktivasyonu ile `[0,1]` aralığında olasılık değeri üretir:  
- `0` → Cat 🐱  
- `1` → Dog 🐶

---

## 📂 Proje Yapısı

cats&dogs/
│
├── cats_dogs/
│ ├── cnn_model_final.keras → Eğitilmiş CNN modeli
│ ├── cnn_scaler.pkl → Özellik ölçekleyici
│ ├── cnn_label_encoder.pkl → Etiket dönüştürücü
│ ├── dataset/ → Ham ses verileri
│ ├── train/ test/ → Eğitim ve test setleri
│ ├── newcnn_gui.py → CNN tabanlı GUI arayüzü
│ └── ses_ozellikleri.csv → Özellik tablosu
│
├── CatsDogsClass/
│ ├── model_egitici.py → CNN eğitimi
│ ├── ozellik_cikarici.py → Ses özellik çıkarımı
│ ├── sesaritici.py → Ses oynatma modülü
│ └── utils.py → Yardımcı fonksiyonlar
│
└── README.md


| Özellik            | Açıklama                                                 |
| ------------------ | -------------------------------------------------------- |
| **Kütüphaneler**   | TensorFlow, Librosa, Matplotlib, Joblib, Pygame, Tkinter |
| **Model Türü**     | CNN (Convolutional Neural Network)                       |
| **Ses Uzunluğu**   | 3 saniyeye normalize edilir                              |
| **Özellik Sayısı** | 13 MFCC + Mel Enerjileri + ZCR                           |
| **Çıktı**          | 0 (Cat) – 1 (Dog)                                        |
| **Tahmin Süresi**  | ~0.3 saniye / dosya                                      |


🐾 Cats & Dogs (CNN) - Grafikli Arayüz
-------------------------------------
🎵 Ses Dosyası Seç
▶️ Sesi Çal

🐱 Bu ses: KEDİ (%91.4 olasılıkla)

[ Dalga Formu ve Mel-Spektrogram Görseli ]

👨‍💻 Geliştirici

Alperen D
💡 Yapay Zeka • Otomasyon • Enerji Sistemleri
