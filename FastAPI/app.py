from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Dict, Any
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for credit risk prediction using SVM model",
    version="1.0.0"
)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model, scaler and feature order
try:
    model = joblib.load(os.path.join(current_dir, 'svm_model.pkl'))
    scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
    feature_order = joblib.load(os.path.join(current_dir, 'feature_order.pkl'))
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    scaler = None
    feature_order = None

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

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "LOAN": 1000.0,
                "MORTDUE": 30000.0,
                "VALUE": 40000.0,
                "YOJ": 5.0,
                "DEROG": 0.0,
                "DELINQ": 0.0,
                "CLAGE": 100.0,
                "NINQ": 1.0,
                "CLNO": 20.0,
                "DEBTINC": 35.0,
                "REASON": "DebtCon",
                "JOB": "Mgr"
            }]
        }
    }

class PredictionResponse(BaseModel):
    credit_approved: bool
    risk_score: float
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "credit_approved": True,
                "risk_score": 0.0,
                "message": "Credit approved"
            }]
        }
    }

class RetrainInput(BaseModel):
    file_path: str = "output.csv"

@app.get("/healthcheck")
async def healthcheck():
    """Check if the API is running and the model is loaded"""
    if model is None or scaler is None or feature_order is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "API is running and model is loaded"}

@app.get("/features")
async def get_features():
    """Get the list of features used by the model"""
    if feature_order is None:
        raise HTTPException(status_code=500, detail="Feature order not loaded")
    return {"features": feature_order}

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: CreditApplication):
    """Make credit approval prediction using the SVM model"""
    if model is None or scaler is None or feature_order is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert application data to DataFrame with correct feature order
        app_data = application.model_dump()
        
        # Create one-hot encoded features for REASON and JOB
        reason_debtcon = 1 if app_data['REASON'] == 'DebtCon' else 0
        reason_homeimp = 1 if app_data['REASON'] == 'HomeImp' else 0
        
        job_mgr = 1 if app_data['JOB'] == 'Mgr' else 0
        job_office = 1 if app_data['JOB'] == 'Office' else 0
        job_other = 1 if app_data['JOB'] == 'Other' else 0
        job_profexe = 1 if app_data['JOB'] == 'ProfExe' else 0
        job_sales = 1 if app_data['JOB'] == 'Sales' else 0
        job_self = 1 if app_data['JOB'] == 'Self' else 0
        
        # Create feature vector in the correct order
        features = [
            app_data['LOAN'],
            app_data['MORTDUE'],
            app_data['VALUE'],
            app_data['YOJ'],
            app_data['DEROG'],
            app_data['DELINQ'],
            app_data['CLAGE'],
            app_data['NINQ'],
            app_data['CLNO'],
            app_data['DEBTINC'],
            reason_debtcon,
            reason_homeimp,
            job_mgr,
            job_office,
            job_other,
            job_profexe,
            job_sales,
            job_self
        ]
        
        # Scale the features
        X_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        # Return result
        return {
            "credit_approved": bool(prediction == 0),
            "risk_score": float(prediction),
            "message": "Credit approved" if prediction == 0 else "Credit not approved"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
async def retrain(input_data: RetrainInput):
    """Retrain the model with new data"""
    try:
        # Load the dataset
        data = pd.read_csv(input_data.file_path)
        
        # Separate features and target variable
        X = data.drop('BAD', axis=1)
        y = data['BAD']
        
        # Save feature order
        global feature_order
        feature_order = X.columns.tolist()
        joblib.dump(feature_order, os.path.join(current_dir, 'feature_order.pkl'))
        
        # Standardize the features
        global scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define the parameter grid for hyperparameter optimization
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        # Create the SVM model
        svm = SVC()
        
        # Perform grid search with cross-validation
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_scaled, y)
        
        # Create the final model with best parameters
        global model
        model = SVC(**grid_search.best_params_)
        model.fit(X_scaled, y)
        
        # Save the model and scaler
        joblib.dump(model, os.path.join(current_dir, 'svm_model.pkl'))
        joblib.dump(scaler, os.path.join(current_dir, 'scaler.pkl'))
        
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "best_parameters": grid_search.best_params_
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 