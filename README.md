# Projet MLOps - Détection de maladie cardiaque

## Problématique

Prédire si un patient est atteint d'une maladie cardiaque à partir de ses données médicales.

- **0** = pas de maladie cardiaque
- **1** = maladie cardiaque détectée

Prédire cette cible est utile pour aider les médecins à identifier rapidement les patients à risque et prioriser leur prise en charge.

## Dataset

- **Source** : Heart Disease UCI (Kaggle)
- **Fichier** : `data/heart_disease_uci.csv`
- **Colonne cible** : `num` (convertie en binaire : 0 / 1)
- **Colonnes numériques** : `age`, `trestbps`, `chol`, `thalch`, `oldpeak`
- **Colonnes catégorielles** : `sex`, `cp`, `restecg`, `slope`, `thal`, `exang`, `fbs`

## Stack technique

- Python 3.13 (géré par uv)
- MLflow (tracking + registry)
- FastAPI (API de prédiction)
- Airflow (orchestration du ré-entraînement)
- Docker + docker-compose
- GitHub Actions (CI/CD)

## Feuille de route

| Séance | TP | Objectif | Statut |
|--------|----|----------|--------|
| S0 | Projet personnel | Choisir dataset + configurer environnement | ✅ |
| S5 | MLflow | Suivi d'expériences MLflow | ⬜ |
| S6 | Optuna | Optimisation Optuna + Model Registry | ⬜ |
| S7 | AutoML + SHAP | Comparaison de modèles + explicabilité | ⬜ |
| S8 | Docker | Conteneuriser l'entraînement | ⬜ |
| S12 | FastAPI | Exposer le modèle via une API | ⬜ |
| S14 | Docker Compose | Orchestrer la stack | ⬜ |
| S17 | Airflow | Planifier le ré-entraînement | ⬜ |

## Installation

```bash
make -C todo install
export PYTHONPATH=todo
```
