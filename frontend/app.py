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

info_tab, predict_tab = st.tabs(["À propos", "Prédiction"])

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
    (Cleveland, Hongrie, Suisse, Long Beach VA). Chaque ligne correspond à un patient avec :

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
    Ce projet suit une approche **MLOps complète** avec les outils suivants :

    **1. Entraînement & Suivi des expériences — MLflow**
    - Enregistrement automatique de chaque entraînement (paramètres, métriques, modèle)
    - Comparaison de plusieurs modèles (Régression Logistique, Random Forest, XGBoost, LightGBM)
    - Model Registry pour versionner et promouvoir les modèles

    **2. Optimisation des hyperparamètres — Optuna**
    - Recherche intelligente des meilleurs hyperparamètres (meilleure que GridSearchCV)
    - Intégration avec MLflow pour tracer chaque essai

    **3. Porte qualité**
    - Le modèle est validé automatiquement avant déploiement
    - Si le ROC AUC ou le F1 sont insuffisants, le déploiement est bloqué

    **4. API d'inférence — FastAPI**
    - Endpoint `/predict` pour faire des prédictions en temps réel
    - Validation automatique des données d'entrée
    - Documentation interactive sur `/docs`

    **5. Conteneurisation — Docker & Docker Compose**
    - Chaque service (MLflow, entraînement, API, frontend) tourne dans son propre conteneur
    - Reproductible sur n'importe quelle machine

    **6. Pipeline CI/CD — GitHub Actions**
    - À chaque push : vérification du code (ruff), entraînement de la baseline, publication de l'image Docker
    """)

    st.divider()

    st.subheader("📈 Performances du modèle")
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC AUC", "0.911", "Baseline LR")
    col2.metric("F1 Score", "0.856", "Baseline LR")
    col3.metric("Accuracy", "~84%", "Baseline LR")

    st.info(
        "💡 Le modèle en production est la **baseline (Régression Logistique)** entraînée automatiquement "
        "par le pipeline CI. Des modèles plus complexes (XGBoost, LightGBM) ont été explorés avec Optuna "
        "et sont disponibles dans le Model Registry MLflow."
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
    | API | FastAPI + Uvicorn |
    | Frontend | Streamlit |
    | Conteneurs | Docker + Docker Compose |
    | CI/CD | GitHub Actions |
    | Registre images | GitHub Container Registry (GHCR) |
    """)
