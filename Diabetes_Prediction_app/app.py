import streamlit as st
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from utils import init_db, save_prediction, load_history
import plotly.express as px
import train_model
import base64
import io
import json
import os

# Paths
MODEL_PATH = Path("model/diabetes_model.pkl")
DATA_PATH = Path("data/diabetes.csv")
METRICS_PATH = Path("model/metrics.json")

# Initialize DB
init_db()

# Page Config
st.set_page_config(
    page_title=" Diabetes Health Predictor",
    page_icon="💚",
    layout="wide",
)

# ------------------ GLOBAL STYLE (BEAUTIFIED UI) ------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header Title */
.title-box {
    text-align: center;
    padding: 30px 0 10px;
}
.title-main {
    font-size: 42px;
    font-weight: 800;
    color: #0b6e4f;
}
.title-sub {
    font-size: 18px;
    color: #3c6652;
    margin-top: -8px;
}

/* Soft Card */
.card {
    border-radius: 16px;
    padding: 22px;
    background: #ffffff;
    box-shadow: 0 8px 25px rgba(0,0,0,0.06);
    margin-bottom: 22px;
}

/* Buttons */
div.stButton > button {
    width: 70%;
    background-color:#0b6e4f !important;
    color:white !important;
    border-radius: 10px !important;
    padding:12px 20px !important;
    font-size:18px !important;
    border: none !important;
    display: block;
    margin: auto;
}
div.stButton > button:hover {
    background-color:#0a5c44 !important;
}

/* DataFrame */
.dataframe {
    border-radius: 10px;
    overflow: hidden;
}

/* Section headers */
.section-header {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 15px;
    color: #0b6e4f;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div class='title-box'>
    <div class='title-main'>💚 Diabetes Health Prediction System</div>
    <div class='title-sub'>Early Detection for Better Health Outcomes</div>
</div>
""", unsafe_allow_html=True)


# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_model_and_metrics():
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        model = bundle['model']
        scaler = bundle['scaler']
        metrics = {}
        if METRICS_PATH.exists():
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        return model, scaler, metrics
    return None, None, {}

model, scaler, metrics = load_model_and_metrics()

if model is None:
    st.warning("⚠️ Trained model not found. Please run `python train_model.py`.")
    if st.button("Run Training Now (In-App)"):
        try:
            train_model.train_and_save()
            st.success("Training completed. Reload page.")
            model, scaler, metrics = load_model_and_metrics()
        except Exception as e:
            st.error(f"Training error: {e}")

# ------------------ INPUT FORM ------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>🧍 Patient Information</div>", unsafe_allow_html=True)

with st.form("predict_form"):
    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=35)
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0)
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
    with right:
        bp_text = st.text_input("Blood Pressure (e.g., 120/80)", value="120/80")

        systolic = diastolic = None
        if "/" in bp_text:
            try:
                systolic, diastolic = map(int, bp_text.split("/"))
            except:
                st.error("Enter BP correctly (e.g., 120/80)")

        glucose = st.number_input("Glucose (Fasting mg/dL)", min_value=10.0, max_value=600.0, value=95.0)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=20, max_value=200, value=72)

    submit = st.form_submit_button("🔮 Predict Health Status")

st.markdown("</div>", unsafe_allow_html=True)


# ------------------ HELPER FUNCTIONS ------------------
def compute_bmi(weight, height_cm):
    if height_cm <= 0:
        return 0.0
    height_m = height_cm / 100
    return round(weight / (height_m ** 2), 2)

def interpret_bmi(bmi):
    if bmi < 18.5:
        return "Underweight", "low"
    elif bmi < 25:
        return "Normal", "normal"
    elif bmi < 30:
        return "Overweight", "high"
    else:
        return "Obese", "high"

def interpret_bp(sys, dia):
    if sys < 90 or dia < 60:
        return "Low Blood Pressure", "low"
    if sys < 120 and dia < 80:
        return "Normal Blood Pressure", "normal"
    if sys < 130 and dia < 80:
        return "Elevated", "elevated"
    if sys < 140 or dia < 90:
        return "High Blood Pressure (Stage 1)", "high"
    return "High Blood Pressure (Stage 2)", "high"

NORMAL_RANGES = {
    'BMI': "Normal BMI: 18.5 - 24.9 kg/m²",
    'BP': "Normal: <120 systolic and <80 diastolic",
    'HeartRate': "Normal: 60 - 100 bpm",
    'GlucoseFBS': "Normal: 70 - 99 mg/dL",
}


# ------------------ PREDICTION ------------------
if submit:
    bmi = compute_bmi(weight_kg, height_cm)
    bmi_label, bmi_level = interpret_bmi(bmi)
    bp_label, bp_level = interpret_bp(systolic, diastolic)

    # PATIENT SUMMARY CARD
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📄 Patient Summary</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Age:** {age}")
        st.markdown(f"**Gender:** {gender}")
        st.markdown(f"**Height:** {height_cm} cm")
        st.markdown(f"**Weight:** {weight_kg} kg")
        st.markdown(f"**BMI:** {bmi} kg/m² — **{bmi_label}**")

    with c2:
        st.markdown(f"**Blood Pressure:** {systolic}/{diastolic} — **{bp_label}**")
        st.markdown(f"**Heart Rate:** {heart_rate} bpm")
        st.markdown(f"**Glucose (FBS):** {glucose} mg/dL")

    st.markdown("</div>", unsafe_allow_html=True)

    # NORMAL RANGE CARD
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📘 Normal Ranges</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown(f"**BMI:** {NORMAL_RANGES['BMI']}")
        st.markdown(f"Interpretation: {bmi_label}")

        st.write("---")

        st.markdown(f"**Blood Pressure:** {NORMAL_RANGES['BP']}")
        st.markdown(f"Interpretation: {bp_label}")

    with right:
        st.markdown(f"**Heart Rate:** {NORMAL_RANGES['HeartRate']}")
        st.markdown(f"**Glucose:** {NORMAL_RANGES['GlucoseFBS']}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ML PREDICTION
    if model is None or scaler is None:
        st.error("❌ Model unavailable.")
    else:
        X = np.array([[glucose, bmi, age, systolic, diastolic]], dtype=float)
        X_scaled = scaler.transform(X)
        proba = float(model.predict_proba(X_scaled)[0, 1])
        pred = int(model.predict(X_scaled)[0])
        risk_pct = proba * 100

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🤖 ML Prediction</div>", unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f"<div style='font-size:24px; color:#dc2626; font-weight:700;'>Likely Diabetic — {risk_pct:.1f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='font-size:24px; color:#0b6e4f; font-weight:700;'>Unlikely Diabetic — {risk_pct:.1f}%</div>", unsafe_allow_html=True)

        if metrics:
            st.markdown(f"**Model Accuracy:** {metrics.get('accuracy',0):.3f}")
            st.markdown(f"**ROC AUC:** {metrics.get('roc_auc',0):.3f}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Save to history
        save_prediction({
            'gender': gender,
            'age': age,
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'bmi': bmi,
            'systolic': systolic,
            'diastolic': diastolic,
            'glucose': glucose,
            'prediction': pred,
            'probability': proba
        })
        st.success("✔ Prediction saved to history.")


# ------------------ DASHBOARD ------------------
px.defaults.template = "plotly_dark"
px.defaults.width = 450
px.defaults.height = 350

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📊 Prediction Dashboard</div>", unsafe_allow_html=True)

history = load_history()
if history.empty:
    st.info("No history yet.")
else:
    history['label'] = history['prediction'].map({0: 'Unlikely', 1: 'Likely'})

    left, right = st.columns(2)
    with left:
        fig = px.pie(history, names='label', title='Prediction Share', hole=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig2 = px.histogram(history, x='bmi', nbins=20, title='BMI Distribution')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📅 Recent Predictions")
    st.dataframe(history[['timestamp', 'age', 'bmi', 'glucose', 'bp', 'label', 'probability']].head(30))

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center;padding:10px;color:#3e6d52;'>Made with ❤️</p>", unsafe_allow_html=True)

