# Projet MLOps - Détection de maladie cardiaque

## Problématique

Prédire si un patient est atteint d'une maladie cardiaque à partir de ses données médicales.

- **0** = pas de maladie cardiaque
- **1** = maladie cardiaque détectée

Prédire cette cible est utile pour aider les médecins à identifier rapidement les patients à risque et prioriser leur prise en charge.

## Dataset

- **Source** : Heart Disease UCI (Kaggle)
- **Fichier** : `data/heart_disease_uci.csv`
- **Colonne cible** : `num` (convertie en binaire : 0 = sain, 1 = malade)
- **Colonnes numériques** : `age`, `trestbps`, `chol`, `thalch`, `oldpeak`
- **Colonnes catégorielles** : `sex`, `cp`, `restecg`, `slope`, `thal`, `exang`, `fbs`

## Structure du projet

```
ml-orchestration-course/
  README.md          ce fichier
  Makefile           commandes du projet
  pyproject.toml     dépendances Python
  data/              dataset CSV
  src/mlproject/     code source à compléter séance par séance
    config.py        configuration du projet (dataset, colonnes, MLflow)
    data.py          chargement et préparation des données
    features.py      construction du pré-processing
  todo/              squelette de référence du prof (non pushé sur git)
```

## Ce qu'on a fait dans chaque fichier (S0)

### `src/mlproject/config.py`
Configure tous les paramètres du projet :
- Le chemin vers le fichier CSV
- La colonne cible (`num`)
- Les colonnes numériques et catégorielles
- Le nom de l'expérience MLflow

### `src/mlproject/data.py`
Charge le CSV et prépare les données :
- Lit le fichier `heart_disease_uci.csv`
- **Convertit la colonne `num` en binaire** : 0 reste 0, toute valeur > 0 devient 1
- Découpe les données en train/test (80% / 20%)

### `src/mlproject/features.py`
Dans le fichier features.py on effectue la normalisation des variables numériques et l'encodage des variables catégorielles 
- **Colonnes numériques** → StandardScaler (normalisation)
- **Colonnes catégorielles** → OneHotEncoder (encodage)

## Installation

```bash
make install
export PYTHONPATH=src
```

## Commandes disponibles

```bash
make help          # liste toutes les commandes
make install       # installe les dépendances
make data          # vérifie que le dataset est en place
make train         # entraîne le modèle 
make train-optuna  # optimisation Optuna 
make train-models  # comparaison de modèles + SHAP 
make api           # lance l'API FastAPI 
make frontend      # lance le frontend Streamlit 
make docker-up     # démarre toute la stack Docker 
```

## Feuille de route

| Séance | TP | Objectif | Statut |
|--------|----|----------|--------|
| S0 | Projet personnel | Choisir dataset + configurer environnement | ✅ |
| S5 | MLflow | Suivi d'expériences MLflow | ✅ |
| S6 | Optuna | Optimisation Optuna + Model Registry | ⬜ |
| S7 | AutoML + SHAP | Comparaison de modèles + explicabilité | ✅ |
| S8 | Docker | Conteneuriser l'entraînement | ⬜ |
| S12 | FastAPI | Exposer le modèle via une API | ⬜ |
| S14 | Docker Compose | Orchestrer la stack | ⬜ |
| S17 | Airflow | Planifier le ré-entraînement | ⬜ |

## Résultats obtenus

| Modèle | f1 | roc_auc |
|--------|----|---------|
| Logistic Regression (baseline) | 0.856 | 0.911 |
| Random Forest | - | - |
| **XGBoost** ← meilleur | - | **0.926** |
| LightGBM | - | - |
