"""API d'inference du modèle de détection de maladie cardiaque (FastAPI).

Séance 12 - TP FastAPI
    Expose /health, /predict et /model-info pour le modèle Heart Disease UCI.
    Lancement : `uvicorn mlproject.api:app --reload`
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproject.config import MODEL_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"
    logger.info("Chargement du modèle depuis %s", model_path)
    ml["model"] = joblib.load(model_path)
    logger.info("Modèle chargé avec succès")
    yield
    ml.clear()
    logger.info("Modèle déchargé")


app = FastAPI(
    title="Heart Disease Classifier API",
    description="Prédit si un patient est atteint d'une maladie cardiaque.",
    version="0.1.0",
    lifespan=lifespan,
)


class Features(BaseModel):
    # Colonnes numériques
    age: float = Field(..., ge=0, description="Âge du patient")
    trestbps: float = Field(
        ..., ge=0, description="Pression artérielle au repos (mm Hg)"
    )
    chol: float = Field(..., ge=0, description="Cholestérol sérique (mg/dl)")
    thalch: float = Field(
        ..., ge=0, description="Fréquence cardiaque maximale atteinte"
    )
    oldpeak: float = Field(..., description="Dépression ST induite par l'exercice")
    # Colonnes catégorielles (None possible dans le dataset pour slope et thal)
    sex: str | None = Field(None, description="Sexe : Male / Female")
    cp: str | None = Field(None, description="Type de douleur thoracique")
    restecg: str | None = Field(None, description="Résultats ECG au repos")
    slope: str | None = Field(None, description="Pente du segment ST à l'effort")
    thal: str | None = Field(None, description="Thalassémie")
    exang: bool | str | None = Field(None, description="Angine induite par l'exercice")
    fbs: bool | str | None = Field(None, description="Glycémie à jeun > 120 mg/dl")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 54.0,
                    "trestbps": 130.0,
                    "chol": 250.0,
                    "thalch": 150.0,
                    "oldpeak": 1.5,
                    "sex": "Male",
                    "cp": "asymptomatic",
                    "restecg": "normal",
                    "slope": "flat",
                    "thal": "fixed defect",
                    "exang": "True",
                    "fbs": "False",
                }
            ]
        }
    }


class PredictionOut(BaseModel):
    prediction: int = Field(..., description="Classe prédite : 0 = sain, 1 = malade")
    probability: float = Field(..., description="Probabilité d'être malade (classe 1)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}
