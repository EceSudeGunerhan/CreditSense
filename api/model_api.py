import joblib
import pandas as pd

model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

def preprocess_input(data: dict) -> pd.DataFrame:
    
    df = pd.DataFrame([data])

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[feature_columns]

    scaled_cols = scaler.feature_names_in_  
    df_encoded[scaled_cols] = scaler.transform(df_encoded[scaled_cols])

    return df_encoded

def predict_credit_risk(data: dict) -> dict:
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0][1]

    if prediction == 1:
        label = "Kredi Verilemez"
        approved = False
        message = "Bu müşteri yüksek riskli görülüyor, kredi önerilmez."
    else:
        label = "Kredi Verilebilir"
        approved = True
        message = "Bu müşteri düşük riskli görülüyor, kredi verilebilir."

    return {
        "approved": approved,
        "prediction_label": label,
        "risk_probability": round(proba, 4),
        "message": message
    }

# app.py tarafından erişilebilmesi için dışa aktardık
__all__ = ["predict_credit_risk", "feature_columns"]
