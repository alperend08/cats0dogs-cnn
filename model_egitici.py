import pandas as pd
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Veri setini oku === #
DATA_PATH = "cats_dogs/ses_ozellikleri.csv"
df = pd.read_csv(DATA_PATH)

# Etiket ve özellikleri ayır
X = df.drop("label", axis=1)
y = df["label"]

# Etiketleri sayısal hale getir (cat -> 0, dog -> 1)
le = LabelEncoder()
y = le.fit_transform(y)

# Veriyi eğitim ve test olarak böl (örnek: %80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Özellikleri ölçekle (normalize et)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Model: Logistic Regression === #
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === Tahmin ve değerlendirme === #
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Model eğitimi tamamlandı!")
print(f"🎯 Doğruluk (Accuracy): {acc*100:.2f}%")

# Confusion Matrix göster
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Cats vs Dogs")
plt.show()
# === Analiz ve kayıt === #


cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)



plt.figure(figsize=(6,5))
sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
plt.title(f"Confusion Matrix - Doğruluk: {acc*100:.2f}%")
plt.xlabel("Tahmin Edilen Etiket")
plt.ylabel("Gerçek Etiket")
plt.tight_layout()
plt.savefig("cats_dogs/confusion_matrix.png")
plt.show()


joblib.dump(model, "cats_dogs/model.pkl")
joblib.dump(scaler, "cats_dogs/scaler.pkl")
joblib.dump(le, "cats_dogs/label_encoder.pkl")


