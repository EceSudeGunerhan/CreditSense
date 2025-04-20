# CreditSense
Yapay Zeka Destekli Kredi Uygunluğu Tahmincisi

# Kredi Onayı Tahmin API'si (SVM + FastAPI)

Bu proje, kullanıcıdan alınan finansal bilgiler doğrultusunda kredi başvurusunun onaylanıp onaylanmayacağını tahmin eden bir makine öğrenmesi sistemidir. Model olarak SVM (Support Vector Machine) algoritması kullanılmıştır ve FastAPI ile servis edilmiştir.

## Proje Özellikleri
- **Makine Öğrenmesi Algoritması:** Hiperparametre optimizasyonu yapılmış SVM
- **Veri Önişleme:** One-hot encoding, standardizasyon
- **Model Kaydetme:** `joblib` ile `svm_model.pkl`, `scaler.pkl`, `feature_order.pkl`
- **API Teknolojisi:** FastAPI
- **Test Arayüzü:** Swagger (http://localhost:8000/docs)

## Kullanım

### 1. Ortam Kurulumu
```bash
pip install fastapi uvicorn pandas scikit-learn joblib
```

### 2. Uygulamayı Başlat
```bash
uvicorn app:app --reload
```

### 3. Swagger Arayüzü
Tarayıcıdan aç:
```
http://localhost:8000/docs
```

### 4. Tahmin Endpoints
#### `POST /predict`
**Açıklama:** Kullanıcıdan alınan bilgilerle kredi başvurusunun onaylanıp onaylanmayacağı tahmin edilir.

Örnek girdi:
```json
{
  "LOAN": 1200.0,
  "MORTDUE": 25000.0,
  "VALUE": 40000.0,
  "YOJ": 6.0,
  "DEROG": 0.0,
  "DELINQ": 0.0,
  "CLAGE": 100.0,
  "NINQ": 1.0,
  "CLNO": 20.0,
  "DEBTINC": 30.0,
  "REASON": "DebtCon",
  "JOB": "Mgr"
}
```

Çıktı:
```json
{
  "credit_approved": true,
  "risk_score": 0.0,
  "message": "Credit approved"
}
```

#### `POST /retrain`
**Açıklama:** `output.csv` verisiyle modeli yeniden eğitir, .pkl dosyalarını günceller.

#### `GET /features`
**Açıklama:** Model tarafından kullanılan öznitelikleri döner.

#### `GET /healthcheck`
**Açıklama:** API'nin ve modelin yüklü olup olmadığını test eder.

##  Model Dosyaları
Model eğitildikten sonra aşağıdaki dosyalar oluşur:
- `svm_model.pkl`
- `scaler.pkl`
- `feature_order.pkl`

Bu dosyalar, `main.py` ile aynı dizine kaydedilir.

##  Dosya Yapısı
```
FastAPI/
├── app.py
├── output.csv
└── svm_model.pkl / scaler.pkl / feature_order.pkl  (retrain ile oluşur)
```

##  Notlar
- `REASON` değeri yalnızca: `DebtCon`, `HomeImp`
- `JOB` değeri yalnızca: `Mgr`, `Office`, `Other`, `ProfExe`, `Sales`, `Self`
- Model doğru çalışması için `feature_order.pkl` kesinlikle eğitimde kaydedilmelidir.

---

Hazırlayan: Ece Sude GÜNERHAN

