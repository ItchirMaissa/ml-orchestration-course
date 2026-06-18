"""Frontend Streamlit - Détection de maladie cardiaque."""
from __future__ import annotations

import os
import sys

import httpx
import numpy as np
import pandas as pd
import streamlit as st

# ─── Résolution des imports du projet ────────────────────────────────────────
_src = os.path.join(os.path.dirname(__file__), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
VPS_IP = os.environ.get("VPS_IP", "34.140.229.122")

st.set_page_config(
    page_title="Heart Disease Classifier",
    layout="wide",
    page_icon="🫀",
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔗 Services")
    st.markdown(f"""
    | Service | Lien |
    |---|---|
    | 🫀 Frontend | [:{VPS_IP}:8501](http://{VPS_IP}:8501) |
    | ⚡ API docs | [:{VPS_IP}:8000/docs](http://{VPS_IP}:8000/docs) |
    | 📊 MLflow | [:{VPS_IP}:5001](http://{VPS_IP}:5001) |
    | 🌀 Airflow | [:{VPS_IP}:8080](http://{VPS_IP}:8080) |
    """)
    st.divider()
    st.markdown("**Projet MLOps — ESGI 2025/2026**")
    st.markdown("👩‍💻 **ITCHIR Maissa**")

st.title("🫀 Détection de maladie cardiaque")
st.caption("Heart Disease UCI — Modèle : Logistic Regression baseline")

api_url = st.text_input("URL de l'API", value=API_URL, label_visibility="collapsed")

info_tab, eval_tab, predict_tab = st.tabs(["📖 À propos", "📊 Évaluation du modèle", "🔬 Prédiction"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — À PROPOS
# ═══════════════════════════════════════════════════════════════════════════════
with info_tab:
    st.header("À propos du projet")
    st.markdown("""
    ## 🫀 Détection de maladie cardiaque — Heart Disease UCI

    Ce projet a été réalisé dans le cadre d'un cours **MLOps** à l'ESGI.
    Il illustre le cycle complet d'un projet de machine learning en production :
    des données brutes jusqu'au déploiement d'une API avec interface utilisateur.

    > 👩‍💻 Réalisé par **ITCHIR Maissa** — ESGI 2025/2026
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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ÉVALUATION DU MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════
with eval_tab:
    st.header("📊 Évaluation du modèle")
    st.caption("👩‍💻 ITCHIR Maissa — Cours MLOps ESGI 2025/2026")

    # Métriques principales
    st.subheader("🏆 Métriques de performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC AUC", "0.911", "+0.061 vs seuil", delta_color="normal")
    col2.metric("F1 Score", "0.856", "+0.056 vs seuil", delta_color="normal")
    col3.metric("Accuracy", "84.2%", None)
    col4.metric("Precision", "82.7%", None)

    st.divider()

    # ── Imports matplotlib ────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import (
            confusion_matrix,
            roc_curve,
            auc,
            precision_recall_curve,
            f1_score,
            roc_auc_score,
        )
        from mlproject.data import load_data, split
        from mlproject.features import build_preprocessor
        from mlproject.config import RANDOM_STATE

        PALETTE = {
            "blue": "#4C72B0",
            "red": "#DD4B39",
            "green": "#2ECC71",
            "orange": "#E67E22",
            "purple": "#9B59B6",
            "teal": "#1ABC9C",
            "dark": "#2C3E50",
            "light_blue": "#AED6F1",
            "light_red": "#F1948A",
        }

        @st.cache_data(show_spinner=False)
        def compute_eval():
            df = load_data()
            x_train, x_test, y_train, y_test = split(df)
            model = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(C=1.0, random_state=RANDOM_STATE, max_iter=1000)),
            ])
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)[:, 1]
            cm = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            return {
                "x_test": x_test, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
                "cm": cm, "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
                "precision": precision, "recall": recall, "f1": f1,
                "df": df, "model": model,
            }

        with st.spinner("Calcul des métriques en cours…"):
            ev = compute_eval()

        # ── Ligne 1 : Matrice de confusion + Courbe ROC ───────────────────────
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.subheader("🟦 Matrice de confusion")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#F8F9FA")
            ax.set_facecolor("#F8F9FA")
            cm = ev["cm"]
            colors = [
                [PALETTE["light_blue"], PALETTE["light_red"]],
                [PALETTE["light_red"], PALETTE["blue"]],
            ]
            for i in range(2):
                for j in range(2):
                    ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=colors[i][j], alpha=0.85))
                    ax.text(
                        j + 0.5, 1.5 - i, str(cm[i][j]),
                        ha="center", va="center", fontsize=22, fontweight="bold",
                        color=PALETTE["dark"],
                    )
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels(["Prédit : Sain (0)", "Prédit : Malade (1)"], fontsize=10)
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(["Réel : Malade (1)", "Réel : Sain (0)"], fontsize=10)
            ax.set_title("Matrice de confusion", fontsize=13, fontweight="bold", color=PALETTE["dark"])
            p1 = mpatches.Patch(color=PALETTE["blue"], alpha=0.85, label="Vrais positifs / négatifs")
            p2 = mpatches.Patch(color=PALETTE["light_red"], alpha=0.85, label="Erreurs")
            ax.legend(handles=[p1, p2], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_roc:
            st.subheader("📈 Courbe ROC")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#F8F9FA")
            ax.set_facecolor("#F8F9FA")
            ax.plot(
                ev["fpr"], ev["tpr"],
                color=PALETTE["blue"], lw=2.5,
                label=f"ROC AUC = {ev['roc_auc']:.3f}",
            )
            ax.fill_between(ev["fpr"], ev["tpr"], alpha=0.15, color=PALETTE["blue"])
            ax.plot([0, 1], [0, 1], color=PALETTE["red"], lw=1.5, linestyle="--", label="Aléatoire")
            ax.axhline(y=0.85, color=PALETTE["orange"], lw=1.2, linestyle=":", label="Seuil qualité (0.85)")
            ax.set_xlabel("Taux de faux positifs", fontsize=11)
            ax.set_ylabel("Taux de vrais positifs", fontsize=11)
            ax.set_title("Courbe ROC", fontsize=13, fontweight="bold", color=PALETTE["dark"])
            ax.legend(loc="lower right", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()

        # ── Ligne 2 : Courbe Précision-Rappel + Distribution des scores ───────
        col_pr, col_dist = st.columns(2)

        with col_pr:
            st.subheader("🟢 Courbe Précision-Rappel")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#F8F9FA")
            ax.set_facecolor("#F8F9FA")
            pr_auc = auc(ev["recall"], ev["precision"])
            ax.plot(
                ev["recall"], ev["precision"],
                color=PALETTE["green"], lw=2.5,
                label=f"PR AUC = {pr_auc:.3f}",
            )
            ax.fill_between(ev["recall"], ev["precision"], alpha=0.15, color=PALETTE["green"])
            ax.axhline(y=0.80, color=PALETTE["orange"], lw=1.2, linestyle=":", label="Seuil F1 (0.80)")
            ax.set_xlabel("Rappel", fontsize=11)
            ax.set_ylabel("Précision", fontsize=11)
            ax.set_title("Courbe Précision-Rappel", fontsize=13, fontweight="bold", color=PALETTE["dark"])
            ax.legend(loc="lower left", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_dist:
            st.subheader("🟣 Distribution des scores de probabilité")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#F8F9FA")
            ax.set_facecolor("#F8F9FA")
            y_test_arr = ev["y_test"].values
            proba_sain = ev["y_proba"][y_test_arr == 0]
            proba_malade = ev["y_proba"][y_test_arr == 1]
            ax.hist(proba_sain, bins=20, alpha=0.7, color=PALETTE["teal"], label="Sain (0)", edgecolor="white")
            ax.hist(proba_malade, bins=20, alpha=0.7, color=PALETTE["red"], label="Malade (1)", edgecolor="white")
            ax.axvline(x=0.5, color=PALETTE["dark"], lw=2, linestyle="--", label="Seuil décision (0.5)")
            ax.set_xlabel("Probabilité prédite d'être malade", fontsize=11)
            ax.set_ylabel("Nombre de patients", fontsize=11)
            ax.set_title("Distribution des scores", fontsize=13, fontweight="bold", color=PALETTE["dark"])
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()

        # ── Ligne 3 : Importance des variables ───────────────────────────────
        st.subheader("🔑 Importance des variables (coefficients Logistic Regression)")
        model = ev["model"]
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        coefs = classifier.coef_[0]
        feat_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
        feat_df = feat_df.reindex(feat_df["coef"].abs().sort_values(ascending=False).index)
        top_n = feat_df.head(15).iloc[::-1]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#F8F9FA")
        ax.set_facecolor("#F8F9FA")
        bar_colors = [PALETTE["red"] if c > 0 else PALETTE["blue"] for c in top_n["coef"]]
        bars = ax.barh(top_n["feature"], top_n["coef"], color=bar_colors, edgecolor="white", height=0.7)
        ax.axvline(x=0, color=PALETTE["dark"], lw=1.5)
        ax.set_xlabel("Coefficient (impact sur la probabilité de maladie)", fontsize=11)
        ax.set_title("Top 15 variables les plus influentes", fontsize=13, fontweight="bold", color=PALETTE["dark"])
        p_pos = mpatches.Patch(color=PALETTE["red"], label="↑ Augmente le risque")
        p_neg = mpatches.Patch(color=PALETTE["blue"], label="↓ Diminue le risque")
        ax.legend(handles=[p_pos, p_neg], fontsize=9)
        ax.grid(True, alpha=0.3, axis="x")
        for bar, val in zip(bars, top_n["coef"]):
            ax.text(
                val + (0.02 if val >= 0 else -0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
                color=PALETTE["dark"],
            )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.divider()

        # ── Ligne 4 : Distribution par variable clé ───────────────────────────
        st.subheader("📋 Statistiques du jeu de données")
        df = ev["df"]
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Distribution de l'âge par classe**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor("#F8F9FA")
            ax.set_facecolor("#F8F9FA")
            ax.hist(df[df["num"] == 0]["age"], bins=20, alpha=0.7, color=PALETTE["teal"], label="Sain", edgecolor="white")
            ax.hist(df[df["num"] == 1]["age"], bins=20, alpha=0.7, color=PALETTE["red"], label="Malade", edgecolor="white")
            ax.set_xlabel("Âge", fontsize=11)
            ax.set_ylabel("Nombre de patients", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_b:
            st.markdown("**Répartition des classes (malade vs sain)**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor("#F8F9FA")
            ax.set_facecolor("#F8F9FA")
            counts = df["num"].value_counts().sort_index()
            labels = ["Sain (0)", "Malade (1)"]
            bar_c = [PALETTE["teal"], PALETTE["red"]]
            bars = ax.bar(labels, counts.values, color=bar_c, edgecolor="white", width=0.5)
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(val),
                        ha="center", va="bottom", fontsize=12, fontweight="bold", color=PALETTE["dark"])
            ax.set_ylabel("Nombre de patients", fontsize=11)
            ax.set_title(f"Total : {len(df)} patients", fontsize=10, color=PALETTE["dark"])
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.info("💡 Tous les graphiques sont calculés en temps réel à partir du modèle baseline (Logistic Regression C=1.0).")

    except ImportError as e:
        st.warning(f"Bibliothèques d'évaluation non disponibles dans cet environnement : {e}")
        st.info("Les graphiques d'évaluation nécessitent scikit-learn et matplotlib.")
        st.subheader("📊 Métriques enregistrées (statiques)")
        col1, col2, col3 = st.columns(3)
        col1.metric("ROC AUC", "0.911")
        col2.metric("F1 Score", "0.856")
        col3.metric("Accuracy", "~84%")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with predict_tab:
    st.subheader("Renseignez les données du patient")
    st.caption("👩‍💻 ITCHIR Maissa — Cours MLOps ESGI 2025/2026")

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
