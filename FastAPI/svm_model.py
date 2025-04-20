from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for credit risk prediction using SVM model",
    version="1.0.0"
)

# Model ve scaler yükleniyor
try:
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_order = joblib.load('feature_order.pkl')  # Model eğitilirken kaydedilen sütun sırası
except:
    model = None
    scaler = None
    feature_order = None

# Giriş şeması
class CreditApplication(BaseModel):
    LOAN: float
    MORTDUE: float
    VALUE: float
    YOJ: float
    DEROG: float
    DELINQ: float
    CLAGE: float
    NINQ: float
    CLNO: float
    DEBTINC: float
    REASON: str
    JOB: str

    class Config:
        schema_extra = {
            "example": {
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
        }

# Sağlık durumu
@app.get("/healthcheck")
async def healthcheck():
    if model is None or scaler is None or feature_order is None:
        return {"status": "error", "message": "Model or scaler not loaded"}
    return {"status": "ok", "message": "API is running and model is loaded"}

# Özellik listesi
@app.get("/features")
async def get_features():
    try:
        return {"features": feature_order}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Tahmin uç noktası
@app.post("/predict")
async def predict(application: CreditApplication):
    if model is None or scaler is None or feature_order is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")

    try:
        df = pd.DataFrame([application.dict()])
        df_encoded = pd.get_dummies(df, columns=["REASON", "JOB"])

        # Eksik sütunları sıfırla
        for col in feature_order:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[feature_order]
        X_scaled = scaler.transform(df_encoded)

        prediction = model.predict(X_scaled)[0]

        return {
            "credit_approved": bool(prediction == 0),
            "risk_score": float(prediction),
            "message": "Credit approved" if prediction == 0 else "Credit not approved"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Modeli yeniden eğit
@app.post("/retrain")
async def retrain():
    try:
        df = pd.read_csv("output.csv")
        X = df.drop("BAD", axis=1)
        y = df["BAD"]

        # One-hot encoding
        X_encoded = pd.get_dummies(X, columns=["REASON", "JOB"])

        global feature_order
        feature_order = X_encoded.columns.tolist()

        # Standardize
        global scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Grid Search + Model fit
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
        grid.fit(X_scaled, y)

        global model
        model = grid.best_estimator_

        save_path = os.path.dirname(os.path.abspath(__file__))

        joblib.dump(model, os.path.join(save_path, 'svm_model.pkl'))
        joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))
        joblib.dump(feature_order, os.path.join(save_path, 'feature_order.pkl'))

        return {
            "status": "success",
            "best_params": grid.best_params_,
            "message": "Model retrained and saved successfully."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uygulama başlatma
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
