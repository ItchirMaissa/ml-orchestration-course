"""Frontend Streamlit - Détection de maladie cardiaque.

Appelle l'API FastAPI pour prédire si un patient est à risque.
Lancement : PYTHONPATH=src streamlit run frontend/app.py
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Heart Disease Classifier", layout="wide")
st.title("🫀 Détection de maladie cardiaque")

api_url = st.text_input("URL de l'API", value=API_URL)

predict_tab, info_tab = st.tabs(["Prédiction", "À propos"])

with predict_tab:
    st.subheader("Renseignez les données du patient")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Âge", min_value=0.0, max_value=120.0, value=54.0)
            trestbps = st.number_input("Pression artérielle au repos (mm Hg)", min_value=0.0, value=130.0)
            chol = st.number_input("Cholestérol (mg/dl)", min_value=0.0, value=250.0)
            thalch = st.number_input("Fréquence cardiaque max", min_value=0.0, value=150.0)
            oldpeak = st.number_input("Dépression ST", value=1.5)

        with col2:
            sex = st.selectbox("Sexe", ["Male", "Female"])
            cp = st.selectbox("Type de douleur thoracique", ["asymptomatic", "atypical angina", "non-anginal", "typical angina"])
            restecg = st.selectbox("ECG au repos", ["normal", "lv hypertrophy", "st-t abnormality"])
            slope = st.selectbox("Pente ST", ["flat", "downsloping", "upsloping"])
            thal = st.selectbox("Thalassémie", ["normal", "fixed defect", "reversable defect"])
            exang = st.selectbox("Angine à l'effort", [True, False])
            fbs = st.selectbox("Glycémie à jeun > 120 mg/dl", [True, False])

        submitted = st.form_submit_button("Prédire")

    if submitted:
        payload = {
            "age": age, "trestbps": trestbps, "chol": chol,
            "thalch": thalch, "oldpeak": oldpeak,
            "sex": sex, "cp": cp, "restecg": restecg,
            "slope": slope, "thal": thal,
            "exang": exang, "fbs": fbs,
        }
        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel à l'API impossible : {exc}")
        else:
            prediction = result["prediction"]
            probability = result["probability"]

            if prediction == 1:
                st.error(f"⚠️ Maladie cardiaque détectée")
            else:
                st.success(f"✅ Pas de maladie cardiaque détectée")

            st.metric("Probabilité d'être malade", f"{probability * 100:.1f}%")
            st.progress(probability)

with info_tab:
    st.subheader("À propos")
    st.markdown("""
    - **Dataset** : Heart Disease UCI
    - **Modèle** : Logistic Regression (baseline)
    - **Métriques** : f1=0.856, roc_auc=0.911
    """)
