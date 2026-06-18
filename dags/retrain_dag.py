"""DAG Airflow - pipeline de ré-entraînement du modèle Heart Disease.

Pipeline : préparation des données → entraînement → contrôle qualité.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    from mlproject.data import load_data
    df = load_data()
    logger.info("Données chargées : %d lignes", len(df))


def task_train(**context) -> None:
    from mlproject.train import train
    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    context["ti"].xcom_push(key="roc_auc", value=metrics["roc_auc"])
    logger.info("Entraînement terminé : f1=%.3f roc_auc=%.3f", metrics["f1"], metrics["roc_auc"])


def task_check_quality(**context) -> None:
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(f"f1={f1:.3f} < seuil {QUALITY_THRESHOLD} — modèle rejeté")
    logger.info("Qualité OK : f1=%.3f >= seuil %.2f", f1, QUALITY_THRESHOLD)


with DAG(
    dag_id="model_retraining",
    description="Prépare les données, réentraîne le modèle et contrôle sa qualité",
    schedule="0 3 * * 1",  # tous les lundis à 3h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    prepare >> train_task >> check
