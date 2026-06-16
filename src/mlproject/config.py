"""Configuration centrale du projet de classification.

C'est le SEUL fichier a adapter pour brancher votre propre jeu de donnees :
data.py, features.py et les scripts d'entrainement lisent toutes leurs
colonnes via ces constantes. Voir tp/TP_S0_projet_personnel.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# TODO (S0-1) : chemin vers votre fichier de donnees (CSV) place dans data/
DATA_PATH = ROOT / "data" / "heart_disease_uci.csv"
MODEL_DIR = ROOT / "models"

# TODO (S0-2) : nom de la colonne cible binaire (valeurs 0/1)
TARGET = "num"

# TODO (S0-3) : colonnes numeriques de votre dataset
NUMERIC_FEATURES: list[str] = ["age", "trestbps", "chol", "thalch", "oldpeak"]

# TODO (S0-4) : colonnes categorielles (peut rester vide : [])
CATEGORICAL_FEATURES: list[str] = ["sex", "cp", "restecg", "slope", "thal", "exang", "fbs"]

RANDOM_STATE = 42

# Surcouche via variables d'environnement (principe 12-factor)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "heart-disease-baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "heart-disease-classifier")

# Description et tags de l'experience MLflow (utilises par tracking.py, TP S5)
MLFLOW_EXPERIMENT_DESCRIPTION = os.getenv(
    "MLFLOW_EXPERIMENT_DESCRIPTION",
    "Détection de maladie cardiaque - Heart Disease UCI - cours MLOps ESGI",
)


def _parse_tags(raw: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key, value = key.strip(), value.strip()
        if key:
            tags[key] = value
    return tags


MLFLOW_EXPERIMENT_TAGS = _parse_tags(
    os.getenv("MLFLOW_EXPERIMENT_TAGS", "course=mlops,dataset=heart-disease-uci,model=xgboost")
)

# Seuils de la porte qualite (evaluate.py, TP S11)
EVAL_ROC_AUC_MIN = float(os.getenv("EVAL_ROC_AUC_MIN", "0.85"))
EVAL_F1_MIN = float(os.getenv("EVAL_F1_MIN", "0.80"))

# URL de l'API FastAPI
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
