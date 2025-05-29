import shap
import joblib
import pandas as pd
from api.model_api import feature_columns
from scripts.encode_scale import encode_and_scale 

model = joblib.load("models/final_model.pkl")

explainer = shap.Explainer(model)

def get_shap_explanation(user_input: dict):
    
    input_df = pd.DataFrame([user_input])

    input_processed = encode_and_scale(input_df)

    input_processed = input_processed[feature_columns]

    shap_values = explainer(input_processed)

    shap_array = shap_values.values[0]
    if shap_array.ndim == 2:
        shap_array = shap_array[:, 0]  

    shap_series = pd.Series(shap_array, index=feature_columns)

    shap_df = pd.DataFrame({
        "feature": shap_series.index,
        "shap_value": shap_series.values
    }).sort_values(by="shap_value", key=lambda x: abs(x), ascending=False).head(5)

    explanation = "\n".join(
        f"- {row['feature']} model kararını etkiledi (etki: {row['shap_value']:.2f})"
        for _, row in shap_df.iterrows()
    )

    return {
        "explanation": explanation,
        "shap_chart": {
            "features": shap_df["feature"].tolist(),
            "values": shap_df["shap_value"].tolist()
        }
    }
