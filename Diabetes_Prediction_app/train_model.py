import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
from pathlib import Path
import json
import numpy as np

DATA_PATH = Path("data/diabetes.csv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

def load_data(path = DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.Please download the Pima Indians dataset and save as data/diabetes.csv")
    df = pd.read_csv(path)
    required = {'Glucose', 'BMI', 'Age', 'Outcome'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Dataset must contain columns: {required}. Found: {set(df.columns)}")
    # Replace zeros in Glucose/BMI with median (zeros often mean missing in Pima)
    for col in ['Glucose', 'BMI', 'BloodPressure']:
        df[col] = df[col].replace(0, pd.NA)
        df[col] = df[col].fillna(df[col].median())
    # Create synthetic Systolic: Systolic = round(1.30 * Diastolic)
    df['Systolic'] = (df['BloodPressure'] * 1.30).round().astype(int)
    # Keep naming consistent: Diastolic = BloodPressure
    df = df.rename(columns={'BloodPressure': 'Diastolic'})
    return df

def train_and_save(path = DATA_PATH, model_path = MODEL_PATH, metrics_path=METRICS_PATH):
    df = load_data(path)
    X = df[['Glucose', 'BMI', 'Age', 'Systolic', 'Diastolic']].astype(float)
    y = df['Outcome'].astype(int)

    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter = 1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    MODEL_DIR.mkdir(parents=True, exist_ok = True)
    joblib.dump({'model' : model, 'scaler' : scaler}, model_path)

    metrics = {"accuracy": float(accuracy_score(y_test, y_pred)), "roc_auc": float(roc_auc_score(y_test, y_proba))}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":

    train_and_save()
