import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_PATH = "data/processed/final_scaled_data.csv"

df = pd.read_csv(INPUT_PATH)

target_col = 'BAD'
y = df[target_col]
X = df.drop(columns=[target_col])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

feature_columns = X_encoded.columns.tolist()

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

final_df = pd.concat([X_encoded, y], axis=1)
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"Encoding ve scaling işlemleri tamamlandı.\n→ Kaydedilen dosya: {OUTPUT_PATH}")

def encode_and_scale(input_df: pd.DataFrame) -> pd.DataFrame:

    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    df_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[feature_columns]

    df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])

    return df_encoded
