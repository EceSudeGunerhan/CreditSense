from fastapi import FastAPI, Body
from pydantic import BaseModel
from api.model_api import predict_credit_risk, feature_columns
from api.shap_api import get_shap_explanation  
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import requests
import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI(
    title="CreditSense API",
    description="Makine öğrenmesi ile kredi riski tahmini yapan servis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CreditInput(BaseModel):
    LOAN: float
    MORTDUE: float
    VALUE: float
    REASON: str
    JOB: str
    YOJ: float
    DEROG: float
    DELINQ: float
    CLAGE: float
    NINQ: float
    CLNO: float
    DEBTINC: float

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

@app.get("/model_info")
def model_info():
    return {
        "model": "Random Forest",
        "scaler": "StandardScaler",
        "feature_count": len(feature_columns),
        "status": "ready"
    }

@app.get("/")
def root():
    return {"message": "CreditSense API yayında."}

@app.get("/features")
def get_features():
    return {"expected_features": feature_columns}

@app.post("/predict")
def predict(input_data: CreditInput):
    result = predict_credit_risk(input_data.dict())
    return result

@app.post("/explain", response_model=Dict[str, Any])
def explain(input_data: CreditInput):
    return get_shap_explanation(input_data.dict())

@app.post("/ask")
def ask_question(input_data: dict = Body(...)):
    import json
    question = input_data.get("question", "")
    features = input_data.get("features", {})

    prompt = f"""
Kullanıcının kredi başvuru verileri: {features}
Model sonucu: {predict_credit_risk(features)['prediction_label']}
Soru: {question}
Cevabı sade ve kullanıcı dostu bir şekilde ver.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://creditsense.local",
        "X-Title": "CreditSense Assistant"
    }

    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Sen bir kredi başvuru asistanısın. Kullanıcının verilerine göre soruları cevapla."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        res = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        res.raise_for_status()
        response = res.json()
        return {"response": response["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"response": f"OpenRouter API hatası: {str(e)}"}

@app.get("/test_env")
def test_env():
    return {
        "api_key": OPENROUTER_API_KEY,
        "api_url": OPENROUTER_API_URL
    }

# uvicorn api.app:app --reload
