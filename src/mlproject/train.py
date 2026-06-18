"""Baseline : Logistic Regression avec suivi MLflow."""

from __future__ import annotations

import argparse

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from mlproject.config import MODEL_DIR, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor
from mlproject.tracking import log_dataset, setup_experiment


def train(c: float = 1.0, max_iter: int = 1000):
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    setup_experiment()

    with mlflow.start_run(run_name=f"logreg-c{c}"):
        log_dataset(df, context="training")

        model = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c, random_state=RANDOM_STATE, max_iter=max_iter
                    ),
                ),
            ]
        )

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params({"c": c, "max_iter": max_iter, "model": "logreg"})
        mlflow.log_metrics({"f1": f1, "roc_auc": roc_auc})
        mlflow.sklearn.log_model(
            model, name="model", skops_trusted_types=["numpy.dtype"]
        )

        print(f"f1={f1:.3f}  roc_auc={roc_auc:.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    return {"f1": f1, "roc_auc": roc_auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
