import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from scipy.stats import zscore

RAW_DATA_PATH = "data/raw/hmeq.csv"
PROCESSED_PATH = "data/processed/cleaned_data.csv"
OUTLIERS_REMOVED_PATH = "data/outliers_removed.csv"

df = pd.read_csv(RAW_DATA_PATH)

target_col = 'BAD'
features = df.columns.drop(target_col)

num_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df[features].select_dtypes(include=["object"]).columns.tolist()

num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

os.makedirs("data/processed", exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)

df_numeric = df[num_cols]
z_scores = np.abs(zscore(df_numeric))
df_clean = df[(z_scores < 3).all(axis=1)]
df_clean.to_csv(OUTLIERS_REMOVED_PATH, index=False)

print(f"Veri temizleme tamamlandı.\n→ İşlenmiş veri: {PROCESSED_PATH}\n→ Aykırılar çıkarıldı: {OUTLIERS_REMOVED_PATH}")
