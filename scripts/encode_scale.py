import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# === Ana işlem: CSV dosyasını oku ve tam veri setini hazırla ===

INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_PATH = "data/processed/final_scaled_data.csv"

df = pd.read_csv(INPUT_PATH)

target_col = 'BAD'
y = df[target_col]
X = df.drop(columns=[target_col])

# Sayısal ve kategorik sütunları belirle
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# One-hot encoding
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Scaler fit ve uygulama
scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

# Kaydet: modelin beklediği sütun sırasını unutma
feature_columns = X_encoded.columns.tolist()

# Scaler ve sütunları kayıt et
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

# Final dataframe (y hedefle birlikte)
final_df = pd.concat([X_encoded, y], axis=1)
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"Encoding ve scaling işlemleri tamamlandı.\n→ Kaydedilen dosya: {OUTPUT_PATH}")

# === Fonksiyonel kullanım (tek örnek için dışarıdan çağrılabilir) ===
def encode_and_scale(input_df: pd.DataFrame) -> pd.DataFrame:
    # Aynı işlemleri tek satırda uygula (SHAP gibi yerlerde kullanılır)

    # Dışarıdan yükle
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    # Kategorik sütunları encode et (drop_first aynı kalsın)
    df_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

    # Eksik sütunları tamamla
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Sıralama
    df_encoded = df_encoded[feature_columns]

    # Ölçekleme
    df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])

    return df_encoded





'''
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


burada One-hot encoding (kategorik sütunu sayısala çevirme)yapıldı standardScaler yapıldı


INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_PATH = "data/processed/final_scaled_data.csv"

df = pd.read_csv(INPUT_PATH)

target_col = 'BAD'
y = df[target_col]
X = df.drop(columns=[target_col])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# One-hot encoding (kategorik sütunlar)
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# StandardScaler (sayısal sütunlar)
scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

# Hedefi geri ekledik
final_df = pd.concat([X_encoded, y], axis=1)

# Sonuçları kaydet
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"Encoding ve scaling işlemleri tamamlandı.\n→ Kaydedilen dosya: {OUTPUT_PATH}")
'''

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

'''
burada One-hot encoding (kategorik sütunu sayısala çevirme)yapıldı standardScaler yapıldı
'''

INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_PATH = "data/processed/final_scaled_data.csv"

df = pd.read_csv(INPUT_PATH)

target_col = 'BAD'
y = df[target_col]
X = df.drop(columns=[target_col])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# One-hot encoding (kategorik sütunlar)
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# StandardScaler (sayısal sütunlar)
scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

# Hedefi geri ekledik
final_df = pd.concat([X_encoded, y], axis=1)

# Sonuçları kaydet
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"Encoding ve scaling işlemleri tamamlandı.\n→ Kaydedilen dosya: {OUTPUT_PATH}")