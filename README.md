 #  CreditSense - Kredi Riski Tahmin ve Açıklama Sistemi

CreditSense, bireysel kredi başvurularının risk durumunu değerlendiren, kararların nedenlerini açıklayan ve kullanıcıya özel öneriler sunan yapay zeka destekli bir web uygulamasıdır. FastAPI, Streamlit, SHAP ve LLM teknolojilerinin birleşimiyle oluşturulmuştur.

---
[![Streamlit Uygulamasını Görüntüle](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen?logo=streamlit)](https://creditsense.streamlit.app/)
---

##  Proje Bileşenleri

### 1.  Veri Seti

* Kaynak: `hmeq.csv`
* Kredi başvuru bilgileri (Gelir, Borç, Ev Durumu, Kredili Borç vb.)
* Hedef değişken: `BAD` (1 = kredi geri ödenmedi, 0 = kredi ödendi)

### 2.  Makine Öğrenmesi

* Model: `Support Vector Machine (SVM)`
* Aşamalar:

  * Veri temizleme (eksik değerler, aykırılıklar)
  * Özellik mühendisliği
  * Model eğitimi & test değerlendirmesi (accuracy, recall, precision)
  * Model dosyası: `final_model.pkl`

### 3.  API Servisi (FastAPI)

* Ana uç nokta: `/predict`

  * Girdi: Başvuru bilgileri (JSON formatında)
  * Çıktı: Onay durumu, risk oranı ve mesaj
* Diğer uç noktalar:

  * `/explain`: SHAP ile karar açıklamaları (özellik bazlı katkılar)

### 4.  SHAP Görselleştirme

* Her tahminin nedenlerini grafiksel olarak açıklayan SHAP değerleri
* Kullanıcılar için modelin "neden bu kararı verdiğini" açıklama

### 5.  Doğal Dil Destekli Kredi Asistanı

* LLM tabanlı chatbot 
* Prompt tabanlı açıklama: "Kredim neden onaylanmadı?", "Riskim yüksek mi?"

### 6.  Streamlit Arayüzü

* 3 Sekmeli yapı:

  1. **Tahmin Sonucu:** Model çıktısı ve karar
  2. **Karar Açıklaması:** SHAP görselleştirmesi
  3. **Kredi Asistanı:** Soru-cevap sistemi (LLM tabanlı)

---

##  Proje Dosya Yapısı

```
CreditSense/
├── requirements.txt
├── streamlit_app.py
├── api/
│   ├── agent.py
│   ├── app.py
│   ├── model_api.py
│   ├── shap_api.py
│   └── requirements.txt
├── credit_model_repo/
│   ├── pytorch_model.bin
│   └── README.md
├── data/
│   ├── outliers_removed.csv
│   ├── raw/hmeq.csv
│   └── processed/
│       ├── cleaned_data.csv
│       └── final_scaled_data.csv
├── models/
│   ├── feature_columns.pkl
│   ├── final_model.pkl
│   └── scaler.pkl
├── scripts/
│   ├── encode_scale.py
│   ├── preprocess.py
│   ├── save_final_model.py
│   └── train_model.py
```

---

##  Yayınlama

* Hugging Face Spaces (Streamlit tabanlı web arayüzü)
* GitHub Proje Linki: [https://github.com/EceSudeGunerhan](https://github.com/EceSudeGunerhan)

---

##  Güvenlik

* `.env` dosyasında gizli API anahtarları
* LLM çağrıları güvenli ve sınırlı istek üzerinden yapılır

---


##  Geliştirici

**Ece Sude GÜNERHAN**
Süleyman Demirel Üniversitesi - Bilgisayar Mühendisliği

---

CreditSense ile kredi değerlendirmelerini daha şeffaf, erişilebilir ve kullanıcı dostu hale getirmeyi amaçlıyoruz. 
