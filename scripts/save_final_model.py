import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

'''
burada fastapide kullanacağımız model kaydını yaptık. standadScaler ve one-hot encoding ile.
'''

DATA_PATH = "data/processed/cleaned_data.csv"
df = pd.read_csv(DATA_PATH)

target_col = "BAD"
X = df.drop(columns=[target_col])
y = df[target_col]

# One-hot encoding
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# StandardScaler
num_cols = X_encoded.select_dtypes(include=["int64", "float64"]).columns.tolist()
scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model ve scaler kaydet
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/final_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Kullanılan encoding sütunlarını da kaydettik (FastAPI'de uyum için)
joblib.dump(X_encoded.columns.tolist(), "models/feature_columns.pkl")

print("Model ve scaler başarıyla kaydedildi.")
