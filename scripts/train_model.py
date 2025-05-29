import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

'''
burada birkaç ML algoritmasını deneyerek en optimum olanı seçtim.
'''

DATA_PATH = "data/processed/final_scaled_data.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["BAD"])
y = df["BAD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modeller
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

print("Model Karşılaştırmaları:\n")

model_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    print(f"\n🔹 {name}")
    print(classification_report(y_test, preds))
    auc = roc_auc_score(y_test, probas)
    print(f"ROC AUC: {auc:.4f}")

    model_scores[name] = auc

# Sonuç özeti
print("\nModel Performans Özeti (ROC AUC):")
for model_name, score in model_scores.items():
    print(f" - {model_name}: {score:.4f}")


# BU METRİKLERİN SONUCUNA GÖRE SEÇTİĞİM ALGORİTMA RANDOM FOREST.