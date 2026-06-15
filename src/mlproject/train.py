"""Baseline simple : Logistic Regression."""
from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from mlproject.config import MODEL_DIR, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor


def train():
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ])

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"f1={f1:.3f}  roc_auc={roc_auc:.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    print(f"Modele sauvegarde dans {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    train()
