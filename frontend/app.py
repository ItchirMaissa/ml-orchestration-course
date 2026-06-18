"""Frontend Streamlit - Détection de maladie cardiaque."""
from __future__ import annotations

import os
import sys

import httpx
import pandas as pd
import streamlit as st

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
VPS_IP = os.environ.get("VPS_IP", "34.140.229.122")

st.set_page_config(
    page_title="Heart Disease Classifier — ITCHIR Maissa",
    layout="wide",
    page_icon="🫀",
)

# ─── CSS custom ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Header hero */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero p  { font-size: 1rem; opacity: 0.75; margin: 0.3rem 0 0; }
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.82rem;
    margin-top: 0.8rem;
    margin-right: 0.4rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1.2rem;
    color: white;
    text-align: center;
}
.metric-card.green  { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
.metric-card.orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
.metric-card.blue   { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
.metric-card h2 { font-size: 2rem; margin: 0; font-weight: 800; }
.metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }

/* Section titles */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a2e;
    border-left: 4px solid #667eea;
    padding-left: 0.6rem;
    margin: 1.5rem 0 1rem;
}

/* Result cards */
.result-positive {
    background: linear-gradient(135deg, #ff6b6b, #ee5a24);
    color: white; border-radius: 12px; padding: 1.5rem; text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white; border-radius: 12px; padding: 1.5rem; text-align: center;
}
.result-positive h2, .result-negative h2 { margin: 0; font-size: 1.8rem; }
.result-positive p,  .result-negative p  { margin: 0.3rem 0 0; opacity: 0.9; }

/* Sidebar */
[data-testid="stSidebar"] { background: #1a1a2e; }
[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─── HERO HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🫀 Détection de maladie cardiaque</h1>
    <p>Heart Disease UCI — Classification binaire (malade / sain)</p>
    <span class="badge">👩‍💻 ITCHIR Maissa</span>
    <span class="badge">🏫 ESGI — Cours MLOps 2025/2026</span>
    <span class="badge">🤖 Logistic Regression baseline</span>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Heart Disease")
    st.markdown("**👩‍💻 ITCHIR Maissa**")
    st.markdown("🏫 ESGI — MLOps 2025/2026")
    st.divider()
    st.markdown("### 🔗 Services")
    st.markdown(f"[🫀 Frontend](http://{VPS_IP}:8501)")
    st.markdown(f"[⚡ API docs](http://{VPS_IP}:8000/docs)")
    st.markdown(f"[📊 MLflow](http://{VPS_IP}:5001)")
    st.markdown(f"[🌀 Airflow](http://{VPS_IP}:8080)")
    st.divider()
    st.markdown("### 📈 Métriques")
    st.metric("ROC AUC", "0.911", "+0.061")
    st.metric("F1 Score", "0.856", "+0.056")
    st.metric("Accuracy", "84.2%")

api_url = st.text_input("🔌 URL de l'API", value=API_URL)

# ─── TABS ─────────────────────────────────────────────────────────────────────
info_tab, eval_tab, predict_tab = st.tabs([
    "📖 À propos",
    "📊 Évaluation du modèle",
    "🔬 Prédiction",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with predict_tab:
    st.markdown('<div class="section-title">Renseignez les données cliniques du patient</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🔢 Données numériques**")
            age = st.number_input("Âge", min_value=0.0, max_value=120.0, value=54.0)
            trestbps = st.number_input("Pression artérielle (mm Hg)", min_value=0.0, value=130.0)
            chol = st.number_input("Cholestérol (mg/dl)", min_value=0.0, value=250.0)
            thalch = st.number_input("Fréquence cardiaque max", min_value=0.0, value=150.0)
            oldpeak = st.number_input("Dépression ST", value=1.5)

        with col2:
            st.markdown("**🔤 Données catégorielles**")
            sex = st.selectbox("Sexe", ["Male", "Female"])
            cp = st.selectbox("Douleur thoracique", ["asymptomatic", "atypical angina", "non-anginal", "typical angina"])
            restecg = st.selectbox("ECG au repos", ["normal", "lv hypertrophy", "st-t abnormality"])
            slope = st.selectbox("Pente ST", ["flat", "downsloping", "upsloping"])

        with col3:
            st.markdown("**⚕️ Autres indicateurs**")
            thal = st.selectbox("Thalassémie", ["normal", "fixed defect", "reversable defect"])
            exang = st.selectbox("Angine à l'effort", [True, False])
            fbs = st.selectbox("Glycémie à jeun > 120", [True, False])
            st.markdown("")
            st.markdown("")
            submitted = st.form_submit_button("🔍 Lancer la prédiction", use_container_width=True)

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
            st.error(f"❌ Appel à l'API impossible : {exc}")
        else:
            prediction = result["prediction"]
            probability = result["probability"]
            st.divider()

            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <h2>⚠️ Maladie cardiaque détectée</h2>
                    <p>Probabilité : <strong>{probability * 100:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h2>✅ Pas de maladie détectée</h2>
                    <p>Probabilité de maladie : <strong>{probability * 100:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            col1, col2, col3 = st.columns(3)
            col1.metric("Probabilité", f"{probability * 100:.1f}%")
            col2.metric("Résultat", "Positif ⚠️" if prediction == 1 else "Négatif ✅")
            col3.metric("Seuil décision", "50%")
            st.progress(probability)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
with eval_tab:
    st.markdown('<div class="section-title">Performance et analyse du modèle baseline</div>', unsafe_allow_html=True)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import (
            confusion_matrix, roc_curve, auc,
            precision_recall_curve, f1_score,
        )
        from mlproject.data import load_data, split
        from mlproject.features import build_preprocessor
        from mlproject.config import RANDOM_STATE

        COLORS = {
            "blue": "#4C72B0", "red": "#E74C3C", "green": "#2ECC71",
            "orange": "#E67E22", "purple": "#9B59B6", "teal": "#1ABC9C",
            "dark": "#1a1a2e", "bg": "#F8FAFC",
            "light_blue": "#AED6F1", "light_red": "#F1948A",
            "grad1": "#667eea", "grad2": "#764ba2",
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
            fpr, tpr, thresh_roc = roc_curve(y_test, y_proba)
            roc_auc_val = auc(fpr, tpr)
            precision, recall, thresh_pr = precision_recall_curve(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            return {
                "x_test": x_test, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
                "cm": cm, "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_val,
                "precision": precision, "recall": recall, "f1": f1,
                "df": df, "model": model, "thresh_roc": thresh_roc,
            }

        with st.spinner("⏳ Calcul des métriques en cours…"):
            ev = compute_eval()

        # ── KPIs ──────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="display:flex;gap:1rem;margin-bottom:1.5rem;">
            <div style="flex:1;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:12px;padding:1.2rem;color:white;text-align:center;">
                <div style="font-size:2rem;font-weight:800;">0.911</div>
                <div style="opacity:.85;font-size:.9rem;">ROC AUC</div>
            </div>
            <div style="flex:1;background:linear-gradient(135deg,#11998e,#38ef7d);border-radius:12px;padding:1.2rem;color:white;text-align:center;">
                <div style="font-size:2rem;font-weight:800;">0.856</div>
                <div style="opacity:.85;font-size:.9rem;">F1 Score</div>
            </div>
            <div style="flex:1;background:linear-gradient(135deg,#f093fb,#f5576c);border-radius:12px;padding:1.2rem;color:white;text-align:center;">
                <div style="font-size:2rem;font-weight:800;">84.2%</div>
                <div style="opacity:.85;font-size:.9rem;">Accuracy</div>
            </div>
            <div style="flex:1;background:linear-gradient(135deg,#4facfe,#00f2fe);border-radius:12px;padding:1.2rem;color:white;text-align:center;">
                <div style="font-size:2rem;font-weight:800;">82.7%</div>
                <div style="opacity:.85;font-size:.9rem;">Precision</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        def styled_fig():
            fig, ax = plt.subplots(figsize=(5.5, 4))
            fig.patch.set_facecolor(COLORS["bg"])
            ax.set_facecolor(COLORS["bg"])
            for spine in ax.spines.values():
                spine.set_color("#E0E0E0")
            return fig, ax

        # ── Ligne 1 : Confusion + ROC ─────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Matrices & Courbes</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = styled_fig()
            cm = ev["cm"]
            cell_colors = [[COLORS["light_blue"], COLORS["light_red"]], [COLORS["light_red"], COLORS["blue"]]]
            for i in range(2):
                for j in range(2):
                    ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=cell_colors[i][j], alpha=0.9))
                    ax.text(j + 0.5, 1.5 - i, str(cm[i][j]), ha="center", va="center",
                            fontsize=26, fontweight="bold", color=COLORS["dark"])
            labels_x = ["Prédit : Sain", "Prédit : Malade"]
            labels_y = ["Réel : Malade", "Réel : Sain"]
            ax.set_xlim(0, 2); ax.set_ylim(0, 2)
            ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(labels_x, fontsize=10)
            ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(labels_y, fontsize=10)
            ax.set_title("🟦 Matrice de confusion", fontsize=12, fontweight="bold", color=COLORS["dark"], pad=12)
            p1 = mpatches.Patch(color=COLORS["blue"], alpha=0.9, label="Bonne prédiction")
            p2 = mpatches.Patch(color=COLORS["light_red"], alpha=0.9, label="Erreur")
            ax.legend(handles=[p1, p2], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with c2:
            fig, ax = styled_fig()
            ax.plot(ev["fpr"], ev["tpr"], color=COLORS["grad1"], lw=2.5, label=f"AUC = {ev['roc_auc']:.3f}")
            ax.fill_between(ev["fpr"], ev["tpr"], alpha=0.18, color=COLORS["grad1"])
            ax.plot([0, 1], [0, 1], color=COLORS["red"], lw=1.5, linestyle="--", label="Aléatoire")
            ax.axhline(0.85, color=COLORS["orange"], lw=1.2, linestyle=":", label="Seuil qualité (0.85)")
            ax.set_xlabel("Taux faux positifs", fontsize=10); ax.set_ylabel("Taux vrais positifs", fontsize=10)
            ax.set_title("📈 Courbe ROC", fontsize=12, fontweight="bold", color=COLORS["dark"], pad=12)
            ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.25)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        # ── Ligne 2 : PR + Distribution scores ───────────────────────────────
        c3, c4 = st.columns(2)

        with c3:
            fig, ax = styled_fig()
            pr_auc = auc(ev["recall"], ev["precision"])
            ax.plot(ev["recall"], ev["precision"], color=COLORS["green"], lw=2.5, label=f"PR AUC = {pr_auc:.3f}")
            ax.fill_between(ev["recall"], ev["precision"], alpha=0.18, color=COLORS["green"])
            ax.axhline(0.80, color=COLORS["orange"], lw=1.2, linestyle=":", label="Seuil F1 (0.80)")
            ax.set_xlabel("Rappel", fontsize=10); ax.set_ylabel("Précision", fontsize=10)
            ax.set_title("🟢 Courbe Précision-Rappel", fontsize=12, fontweight="bold", color=COLORS["dark"], pad=12)
            ax.legend(loc="lower left", fontsize=9); ax.grid(True, alpha=0.25)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with c4:
            fig, ax = styled_fig()
            y_arr = ev["y_test"].values
            ax.hist(ev["y_proba"][y_arr == 0], bins=20, alpha=0.75, color=COLORS["teal"], label="Sain (0)", edgecolor="white")
            ax.hist(ev["y_proba"][y_arr == 1], bins=20, alpha=0.75, color=COLORS["red"], label="Malade (1)", edgecolor="white")
            ax.axvline(0.5, color=COLORS["dark"], lw=2, linestyle="--", label="Seuil 0.5")
            ax.set_xlabel("Probabilité prédite", fontsize=10); ax.set_ylabel("Patients", fontsize=10)
            ax.set_title("🟣 Distribution des scores", fontsize=12, fontweight="bold", color=COLORS["dark"], pad=12)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        # ── Ligne 3 : Seuil F1 + Importance variables ─────────────────────────
        st.markdown('<div class="section-title">🔑 Importance des variables & Optimisation du seuil</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)

        with c5:
            # Courbe F1 en fonction du seuil
            fig, ax = styled_fig()
            thresholds = ev["thresh_roc"][1:]
            tpr_vals = ev["tpr"][1:]
            fpr_vals = ev["fpr"][1:]
            precision_at_thresh = tpr_vals / (tpr_vals + fpr_vals + 1e-9)
            recall_at_thresh = tpr_vals
            f1_vals = 2 * precision_at_thresh * recall_at_thresh / (precision_at_thresh + recall_at_thresh + 1e-9)
            ax.plot(thresholds, f1_vals, color=COLORS["purple"], lw=2.5, label="F1 Score")
            best_idx = f1_vals.argmax()
            ax.axvline(thresholds[best_idx], color=COLORS["orange"], lw=1.5, linestyle="--",
                       label=f"Meilleur seuil = {thresholds[best_idx]:.2f}")
            ax.axvline(0.5, color=COLORS["red"], lw=1.2, linestyle=":", label="Seuil standard (0.5)")
            ax.set_xlabel("Seuil de décision", fontsize=10); ax.set_ylabel("F1 Score", fontsize=10)
            ax.set_title("🎯 F1 selon le seuil de décision", fontsize=12, fontweight="bold", color=COLORS["dark"], pad=12)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with c6:
            model = ev["model"]
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]
            feature_names = preprocessor.get_feature_names_out()
            coefs = classifier.coef_[0]
            feat_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
            feat_df = feat_df.reindex(feat_df["coef"].abs().sort_values(ascending=False).index).head(12).iloc[::-1]
            fig, ax = styled_fig()
            fig.set_figheight(4.5)
            bar_colors = [COLORS["red"] if c > 0 else COLORS["blue"] for c in feat_df["coef"]]
            bars = ax.barh(feat_df["feature"], feat_df["coef"], color=bar_colors, edgecolor="white", height=0.65)
            ax.axvline(0, color=COLORS["dark"], lw=1.5)
            ax.set_xlabel("Coefficient", fontsize=10)
            ax.set_title("🔑 Variables les plus influentes", fontsize=12, fontweight="bold", color=COLORS["dark"], pad=12)
            p_pos = mpatches.Patch(color=COLORS["red"], label="↑ Augmente le risque")
            p_neg = mpatches.Patch(color=COLORS["blue"], label="↓ Diminue le risque")
            ax.legend(handles=[p_pos, p_neg], fontsize=8)
            ax.grid(True, alpha=0.25, axis="x")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        # ── Ligne 4 : Dataset stats ───────────────────────────────────────────
        st.markdown('<div class="section-title">📋 Analyse du dataset</div>', unsafe_allow_html=True)
        c7, c8, c9 = st.columns(3)
        df = ev["df"]

        with c7:
            fig, ax = styled_fig()
            counts = df["num"].value_counts().sort_index()
            wedge_colors = [COLORS["teal"], COLORS["red"]]
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=["Sain (0)", "Malade (1)"],
                colors=wedge_colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            for at in autotexts:
                at.set_fontsize(11); at.set_fontweight("bold"); at.set_color("white")
            ax.set_title("Répartition des classes", fontsize=11, fontweight="bold", color=COLORS["dark"])
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with c8:
            fig, ax = styled_fig()
            ax.hist(df[df["num"] == 0]["age"], bins=18, alpha=0.75, color=COLORS["teal"], label="Sain", edgecolor="white")
            ax.hist(df[df["num"] == 1]["age"], bins=18, alpha=0.75, color=COLORS["red"], label="Malade", edgecolor="white")
            ax.set_xlabel("Âge", fontsize=10); ax.set_ylabel("Patients", fontsize=10)
            ax.set_title("Distribution de l'âge", fontsize=11, fontweight="bold", color=COLORS["dark"])
            ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with c9:
            fig, ax = styled_fig()
            ax.hist(df[df["num"] == 0]["thalch"], bins=18, alpha=0.75, color=COLORS["teal"], label="Sain", edgecolor="white")
            ax.hist(df[df["num"] == 1]["thalch"], bins=18, alpha=0.75, color=COLORS["red"], label="Malade", edgecolor="white")
            ax.set_xlabel("Fréquence cardiaque max", fontsize=10); ax.set_ylabel("Patients", fontsize=10)
            ax.set_title("Fréquence cardiaque max", fontsize=11, fontweight="bold", color=COLORS["dark"])
            ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        # ── Ligne 5 : Cholestérol + Pression + Corrélation ───────────────────
        c10, c11 = st.columns(2)

        with c10:
            fig, ax = styled_fig()
            ax.scatter(
                df[df["num"] == 0]["age"], df[df["num"] == 0]["chol"],
                alpha=0.5, color=COLORS["teal"], label="Sain", s=25, edgecolors="none",
            )
            ax.scatter(
                df[df["num"] == 1]["age"], df[df["num"] == 1]["chol"],
                alpha=0.5, color=COLORS["red"], label="Malade", s=25, edgecolors="none",
            )
            ax.set_xlabel("Âge", fontsize=10); ax.set_ylabel("Cholestérol (mg/dl)", fontsize=10)
            ax.set_title("Âge vs Cholestérol", fontsize=11, fontweight="bold", color=COLORS["dark"])
            ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with c11:
            fig, ax = styled_fig()
            numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
            corr = df[numeric_cols].corr()
            import numpy as np
            im = ax.imshow(corr.values, cmap="RdYlBu", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(numeric_cols))); ax.set_xticklabels(numeric_cols, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(numeric_cols))); ax.set_yticklabels(numeric_cols, fontsize=8)
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="black" if abs(corr.values[i, j]) < 0.6 else "white")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Corrélation entre variables", fontsize=11, fontweight="bold", color=COLORS["dark"])
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.success("✅ Tous les graphiques sont calculés en temps réel — modèle Logistic Regression (C=1.0)")

    except ImportError as e:
        st.warning(f"Bibliothèques non disponibles : {e}")
        st.metric("ROC AUC", "0.911")
        st.metric("F1 Score", "0.856")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — À PROPOS
# ═══════════════════════════════════════════════════════════════════════════════
with info_tab:
    st.markdown('<div class="section-title">À propos du projet</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("""
        ### 🎯 La problématique
        Les maladies cardiovasculaires sont la **première cause de mortalité mondiale** selon l'OMS.
        Un diagnostic précoce est crucial mais nécessite des examens coûteux et l'expertise d'un cardiologue.

        **L'objectif** : prédire si un patient est à risque à partir de données cliniques simples
        (âge, pression artérielle, cholestérol, ECG…).

        C'est un problème de **classification binaire** : malade (1) ou sain (0).
        """)

    with col_b:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a1a2e,#0f3460);border-radius:12px;padding:1.5rem;color:white;text-align:center;">
            <div style="font-size:1rem;opacity:.7;">Projet réalisé par</div>
            <div style="font-size:1.4rem;font-weight:800;margin:.3rem 0;">ITCHIR Maissa</div>
            <div style="font-size:.85rem;opacity:.7;">ESGI — MLOps 2025/2026</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("📊 Le dataset — Heart Disease UCI")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Source", "UCI ML Repo")
    col2.metric("Patients", "920")
    col3.metric("Variables", "13 + 1 cible")
    col4.metric("Hôpitaux", "4")

    st.divider()

    st.subheader("🏗️ Stack technique")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Composant | Technologie |
        |---|---|
        | Langage | Python 3.13 |
        | ML | scikit-learn, XGBoost, LightGBM |
        | Tracking | MLflow |
        | Optimisation | Optuna |
        | Orchestration | Apache Airflow |
        """)
    with col2:
        st.markdown("""
        | Composant | Technologie |
        |---|---|
        | API | FastAPI + Uvicorn |
        | Frontend | Streamlit |
        | Conteneurs | Docker + Compose |
        | CI/CD | GitHub Actions |
        | Cloud | Google Cloud Platform |
        """)
