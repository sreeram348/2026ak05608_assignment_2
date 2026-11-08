"""Train and evaluate the five required classifiers on the mobile-price dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "dataset.csv"
MODEL_DIR = ROOT_DIR / "model"
TARGET_COLUMN = "price_range"
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_SPECS: dict[str, tuple[str, Any]] = {
    "Logistic Regression": (
        "logistic_regression",
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    ),
    "Decision Tree": (
        "decision_tree",
        DecisionTreeClassifier(random_state=RANDOM_STATE),
    ),
    "KNN": ("knn", KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": ("naive_bayes", GaussianNB()),
    "SVM": (
        "svm",
        CalibratedClassifierCV(
            SVC(random_state=RANDOM_STATE),
            method="sigmoid",
            cv=5,
            ensemble=False,
        ),
    ),
}


def validate_dataset(data: pd.DataFrame) -> list[str]:
    """Validate the source dataset and return its ordered feature names."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain the target column '{TARGET_COLUMN}'.")

    feature_columns = [column for column in data.columns if column != TARGET_COLUMN]
    if len(data) < 500 or len(feature_columns) < 12:
        raise ValueError(
            "The assignment requires at least 500 records and 12 features."
        )
    if data.empty or data.isna().any().any():
        raise ValueError("Dataset must not be empty or contain missing values.")
    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Target column must contain at least two classes.")

    try:
        numeric_features = data[feature_columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("All feature columns must contain numeric values.") from error
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be finite numbers.")

    return feature_columns


def calculate_metrics(
    y_true: pd.Series,
    y_pred: Any,
    y_prob: Any,
    class_labels: list[Any],
) -> dict[str, float | None]:
    """Calculate classification metrics, omitting AUC when a class is absent."""
    all_classes_present = set(y_true.unique()) == set(class_labels)
    auc = (
        float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=class_labels,
                multi_class="ovr",
                average="weighted",
            )
        )
        if all_classes_present
        else None
    )

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc,
        "Precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def train() -> pd.DataFrame:
    """Fit models, save artifacts, and export a reproducible holdout set."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    feature_columns = validate_dataset(data)
    features = data[feature_columns]
    target = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(ROOT_DIR / "test_data.csv", index=False)
    sample_data = (
        test_data.groupby(TARGET_COLUMN, group_keys=False)
        .sample(n=25, random_state=RANDOM_STATE)
        .sample(frac=1, random_state=RANDOM_STATE)
    )
    sample_data.to_csv(ROOT_DIR / "sample_test_data.csv", index=False)

    rows: list[dict[str, Any]] = []
    class_labels = sorted(target.unique().tolist())

    for display_name, (file_name, model) in MODEL_SPECS.items():
        model.fit(x_train_scaled, y_train)
        predictions = model.predict(x_test_scaled)
        probabilities = model.predict_proba(x_test_scaled)
        metrics = calculate_metrics(y_test, predictions, probabilities, class_labels)
        rows.append({"Model": display_name, **metrics})
        joblib.dump(model, MODEL_DIR / f"{file_name}.pkl")

    results = pd.DataFrame(rows)
    results.to_csv(ROOT_DIR / "results.csv", index=False)
    return results


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    print(train().to_string(index=False, float_format=lambda value: f"{value:.4f}"))
