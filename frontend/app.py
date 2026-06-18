"""Frontend Streamlit - Détection de maladie cardiaque."""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
VPS_IP = os.environ.get("VPS_IP", "34.140.229.122")

st.set_page_config(
    page_title="Heart Disease Classifier",
    layout="wide",
    page_icon="🫀",
)

# Sidebar avec liens vers tous les services
with st.sidebar:
    st.markdown("## 🔗 Services")
    st.markdown(f"""
    | Service | Lien |
    |---|---|
    | 🫀 Frontend | [localhost:8501](http://localhost:8501) |
    | ⚡ API docs | [:{8000}/docs](http://{VPS_IP}:8000/docs) |
    | 📊 MLflow | [:{5001}](http://{VPS_IP}:5001) |
    | 🌀 Airflow | [:{8080}](http://{VPS_IP}:8080) |
    """)
    st.divider()
    st.markdown("**Identifiants Airflow**")
    st.code("login : admin\npassword : airflow")
    st.divider()
    st.caption("Cours MLOps — ESGI 2025/2026")

st.title("🫀 Détection de maladie cardiaque")
st.caption("Heart Disease UCI — Modèle : Logistic Regression baseline")

api_url = st.text_input("URL de l'API", value=API_URL, label_visibility="collapsed")

info_tab, predict_tab = st.tabs(["📖 À propos", "🔬 Prédiction"])

with info_tab:
    st.header("À propos du projet")

    st.markdown("""
    ## 🫀 Détection de maladie cardiaque — Heart Disease UCI

    Ce projet a été réalisé dans le cadre d'un cours **MLOps** à l'ESGI.
    Il illustre le cycle complet d'un projet de machine learning en production :
    des données brutes jusqu'au déploiement d'une API avec interface utilisateur.
    """)

    st.divider()

    st.subheader("🎯 La problématique")
    st.markdown("""
    Les maladies cardiovasculaires sont la **première cause de mortalité mondiale** selon l'OMS.
    Un diagnostic précoce est crucial mais nécessite des examens médicaux coûteux et l'expertise d'un cardiologue.

    **L'objectif** : construire un modèle de machine learning capable de prédire si un patient
    est à risque de maladie cardiaque à partir de données cliniques simples (âge, pression artérielle,
    cholestérol, résultat d'ECG...).

    C'est un problème de **classification binaire** : malade (1) ou non malade (0).
    """)

    st.divider()

    st.subheader("📊 Le dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Source", "UCI ML Repository")
    col2.metric("Patients", "920")
    col3.metric("Variables", "13 + 1 cible")

    st.markdown("""
    Le dataset **Heart Disease UCI** combine des données issues de 4 hôpitaux
    (Cleveland, Hongrie, Suisse, Long Beach VA).

    | Variable | Description |
    |---|---|
    | `age` | Âge du patient |
    | `sex` | Sexe (Male / Female) |
    | `cp` | Type de douleur thoracique |
    | `trestbps` | Pression artérielle au repos (mm Hg) |
    | `chol` | Cholestérol sérique (mg/dl) |
    | `fbs` | Glycémie à jeun > 120 mg/dl |
    | `restecg` | Résultat de l'ECG au repos |
    | `thalch` | Fréquence cardiaque maximale atteinte |
    | `exang` | Angine induite par l'effort |
    | `oldpeak` | Dépression du segment ST |
    | `slope` | Pente du segment ST |
    | `thal` | Résultat du test à la thallium |
    | `num` | **Cible** : présence de maladie (0 = non, 1 = oui) |
    """)

    st.divider()

    st.subheader("🏗️ Ce qu'on a mis en place")
    st.markdown("""
    **1. Entraînement & Suivi — MLflow**
    Enregistrement automatique de chaque entraînement, comparaison des modèles, Model Registry.

    **2. Optimisation — Optuna**
    Recherche intelligente des meilleurs hyperparamètres (RF, XGBoost, LightGBM).

    **3. Porte qualité**
    Le modèle est validé automatiquement avant déploiement (ROC AUC > 0.85, F1 > 0.80).

    **4. API d'inférence — FastAPI**
    Endpoint `/predict` pour faire des prédictions en temps réel.

    **5. Conteneurisation — Docker & Docker Compose**
    4 services : MLflow, entraînement, API, frontend.

    **6. Orchestration — Airflow**
    DAG automatique : préparation → entraînement → contrôle qualité, tous les lundis à 3h.

    **7. Pipeline CI/CD — GitHub Actions**
    À chaque push : lint, entraînement baseline, build et push image Docker sur GHCR.

    **8. Déploiement — GCP VPS**
    Stack complète déployée sur une VM Google Cloud (e2-standard-2, 8 Go RAM).
    """)

    st.divider()

    st.subheader("📈 Performances du modèle")
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC AUC", "0.911", "Baseline LR")
    col2.metric("F1 Score", "0.856", "Baseline LR")
    col3.metric("Accuracy", "~84%", "Baseline LR")

    st.info(
        "💡 Le modèle en production est la **baseline (Régression Logistique)** entraînée automatiquement "
        "par le pipeline CI. Des modèles plus complexes (XGBoost, LightGBM) ont été explorés avec Optuna."
    )

    st.divider()

    st.subheader("🛠️ Stack technique")
    st.markdown("""
    | Composant | Technologie |
    |---|---|
    | Langage | Python 3.13 |
    | ML | scikit-learn, XGBoost, LightGBM |
    | Tracking | MLflow |
    | Optimisation | Optuna |
    | Orchestration | Apache Airflow |
    | API | FastAPI + Uvicorn |
    | Frontend | Streamlit |
    | Conteneurs | Docker + Docker Compose |
    | CI/CD | GitHub Actions |
    | Registre images | GitHub Container Registry (GHCR) |
    | Cloud | Google Cloud Platform (GCP) |
    """)

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

        submitted = st.form_submit_button("🔍 Prédire", use_container_width=True)

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

            st.divider()
            if prediction == 1:
                st.error("⚠️ Maladie cardiaque détectée")
            else:
                st.success("✅ Pas de maladie cardiaque détectée")

            col1, col2 = st.columns(2)
            col1.metric("Probabilité d'être malade", f"{probability * 100:.1f}%")
            col2.metric("Résultat", "Positif" if prediction == 1 else "Négatif")
            st.progress(probability)
