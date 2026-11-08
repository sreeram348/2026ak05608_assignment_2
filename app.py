"""Interactive classification benchmark and assignment model demonstration."""

from __future__ import annotations

import html
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from datetime import datetime
from time import perf_counter
from typing import Any

import joblib
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_classification
from sklearn.metrics import classification_report, confusion_matrix

from benchmark import MODEL_NAMES, compare_classifiers
from train_models import MODEL_SPECS, TARGET_COLUMN, calculate_metrics, validate_dataset


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PROJECT_NAME = "EvalSphere"
PROJECT_TAGLINE = "Benchmark Smarter. Decide Faster."
PROJECT_DESCRIPTION = "The unified workspace for model evaluation and insight"

DATASET_CATALOG: list[dict[str, Any]] = [
    {
        "id": "mobile-price",
        "name": "Mobile Price Classification",
        "rows": "2,000",
        "features": "20",
        "task": "Multiclass",
        "difficulty": "Beginner",
        "description": "Compare device specifications across four price-range classes.",
        "source": "Assignment dataset",
        "synthetic": False,
    },
    {
        "id": "titanic",
        "name": "Titanic Survival",
        "rows": "800 sample",
        "features": "12",
        "task": "Binary",
        "difficulty": "Beginner",
        "description": "Explore passenger-profile signals in a survival classification example.",
        "source": "Synthetic instructional sample",
        "synthetic": True,
    },
    {
        "id": "heart-disease",
        "name": "Heart Disease Prediction",
        "rows": "800 sample",
        "features": "13",
        "task": "Binary",
        "difficulty": "Intermediate",
        "description": "Practice binary classification with familiar health-screening feature names.",
        "source": "Synthetic instructional sample",
        "synthetic": True,
    },
    {
        "id": "customer-churn",
        "name": "Customer Churn",
        "rows": "800 sample",
        "features": "12",
        "task": "Binary",
        "difficulty": "Beginner",
        "description": "Explore customer-retention patterns using a generated tabular example.",
        "source": "Synthetic instructional sample",
        "synthetic": True,
    },
    {
        "id": "breast-cancer",
        "name": "Breast Cancer Classification",
        "rows": "569",
        "features": "30",
        "task": "Binary",
        "difficulty": "Intermediate",
        "description": "Use the bundled scikit-learn dataset to compare diagnostic classifiers.",
        "source": "scikit-learn dataset",
        "synthetic": False,
    },
    {
        "id": "diabetes",
        "name": "Diabetes Prediction",
        "rows": "800 sample",
        "features": "12",
        "task": "Binary",
        "difficulty": "Intermediate",
        "description": "Benchmark a generated health-screening classification example.",
        "source": "Synthetic instructional sample",
        "synthetic": True,
    },
    {
        "id": "bank-marketing",
        "name": "Bank Marketing",
        "rows": "800 sample",
        "features": "16",
        "task": "Binary",
        "difficulty": "Intermediate",
        "description": "Explore a generated campaign-response classification example.",
        "source": "Synthetic instructional sample",
        "synthetic": True,
    },
    {
        "id": "iris",
        "name": "Iris",
        "rows": "150",
        "features": "4",
        "task": "Multiclass",
        "difficulty": "Beginner",
        "description": "A compact, classic flower-species classification benchmark.",
        "source": "scikit-learn dataset",
        "synthetic": False,
    },
    {
        "id": "wine-quality",
        "name": "Wine Quality",
        "rows": "800 sample",
        "features": "11",
        "task": "Multiclass",
        "difficulty": "Intermediate",
        "description": "Practice quality-band classification with generated chemistry-style features.",
        "source": "Synthetic instructional sample",
        "synthetic": True,
    },
    {
        "id": "wine",
        "name": "Wine Classification",
        "rows": "178",
        "features": "13",
        "task": "Multiclass",
        "difficulty": "Beginner",
        "description": "Use the bundled scikit-learn wine chemistry dataset for a compact multiclass benchmark.",
        "source": "scikit-learn dataset",
        "synthetic": False,
    },
    {
        "id": "credit-card-fraud",
        "name": "Credit Card Fraud Detection",
        "rows": "1,000 sample",
        "features": "12",
        "task": "Imbalanced binary",
        "difficulty": "Advanced",
        "description": "Inspect an imbalanced fraud-detection example with generated transactions.",
        "source": "Synthetic instructional sample",
        "synthetic": True,
    },
    {
        "id": "employee-attrition",
        "name": "Employee Attrition",
        "rows": "800 sample",
        "features": "12",
        "task": "Binary",
        "difficulty": "Intermediate",
        "description": "Explore workforce-retention patterns using a generated example.",
        "source": "Synthetic instructional sample",
        "synthetic": True,
    },
]

SYNTHETIC_DATASET_FEATURES: dict[str, list[str]] = {
    "titanic": [
        "passenger_class", "age", "fare", "family_size", "siblings_spouses",
        "parents_children", "travel_group_size", "deck_score", "ticket_group",
        "port_code", "title_code", "is_alone",
    ],
    "heart-disease": [
        "age", "resting_bp", "cholesterol", "max_heart_rate", "oldpeak",
        "sex_code", "chest_pain", "fasting_sugar", "ecg_result", "exercise_angina",
        "slope", "vessels", "thal",
    ],
    "customer-churn": [
        "tenure_months", "monthly_charges", "total_charges", "support_calls",
        "contract_months", "data_usage", "late_payments", "device_age",
        "international_plan", "autopay", "streaming_usage", "service_count",
    ],
    "diabetes": [
        "age", "bmi", "glucose", "blood_pressure", "insulin", "skin_thickness",
        "family_history", "activity_score", "pregnancies", "diet_score",
        "sleep_hours", "waist_circumference",
    ],
    "bank-marketing": [
        "age", "account_balance", "campaign_contacts", "days_since_contact",
        "previous_contacts", "call_duration", "job_code", "marital_code",
        "education_code", "default_flag", "housing_loan", "personal_loan",
        "contact_code", "month_code", "weekday_code", "poutcome_code",
    ],
    "wine-quality": [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "ph", "sulphates", "alcohol",
    ],
    "credit-card-fraud": [
        "transaction_amount", "hour_of_day", "merchant_risk", "distance_from_home",
        "distance_from_last_transaction", "repeat_merchant", "chip_used",
        "online_order", "international", "account_age_days", "velocity_24h",
        "device_trust_score",
    ],
    "employee-attrition": [
        "age", "monthly_income", "years_at_company", "years_in_role",
        "job_satisfaction", "work_life_balance", "distance_from_home",
        "training_hours", "salary_growth", "overtime_flag", "promotion_gap",
        "team_size",
    ],
}


@st.cache_resource
def load_artifacts(model_name: str) -> tuple[Any, Any]:
    """Load an assignment classifier and its fitted feature scaler."""
    model_file = MODEL_SPECS[model_name][0]
    model_path = MODEL_DIR / f"{model_file}.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    if not model_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError(
            "Assignment model artifacts are missing. Run "
            "`python train_models.py` first."
        )
    return joblib.load(model_path), joblib.load(scaler_path)


@st.cache_data(show_spinner=False)
def measure_assignment_prediction_times(raw_data: bytes) -> pd.DataFrame:
    """Measure bundled models on the same local assignment reference holdout."""
    data = pd.read_csv(BytesIO(raw_data), encoding="utf-8-sig")
    first_model_name = next(iter(MODEL_SPECS))
    _, scaler = load_artifacts(first_model_name)
    feature_columns = list(scaler.feature_names_in_)
    validated = validate_assignment_upload(data, feature_columns)
    scaled_features = scaler.transform(validated[feature_columns])
    measurements = []
    for model_name in MODEL_SPECS:
        model, _ = load_artifacts(model_name)
        started = perf_counter()
        model.predict(scaled_features)
        measurements.append(
            {
                "Model": model_name,
                "Prediction Time (s)": perf_counter() - started,
            }
        )
    return pd.DataFrame(measurements)


def validate_assignment_upload(
    data: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Validate the labeled Mobile Price CSV used by the assignment models."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"CSV must include the '{TARGET_COLUMN}' target column.")

    uploaded_features = [column for column in data.columns if column != TARGET_COLUMN]
    missing = [column for column in feature_columns if column not in uploaded_features]
    unexpected = [column for column in uploaded_features if column not in feature_columns]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError(
            "CSV feature columns do not match the assignment dataset ("
            + ", ".join(details)
            + ")."
        )
    if data.empty:
        raise ValueError("Uploaded CSV contains no rows.")
    if len(data) > 50_000:
        raise ValueError("Assignment test CSV is limited to 50,000 rows.")
    if data.isna().any().any():
        raise ValueError("Uploaded CSV must not contain missing values.")

    try:
        features = data[feature_columns].apply(pd.to_numeric, errors="raise")
        target = pd.to_numeric(data[TARGET_COLUMN], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Assignment CSV feature and target values must be numeric.") from error
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Assignment CSV features must contain only finite numbers.")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Assignment CSV target values must contain only finite numbers.")

    return features.assign(**{TARGET_COLUMN: target})


@st.cache_data(show_spinner=False)
def load_demo_dataset(dataset_id: str) -> pd.DataFrame:
    """Load a verified local dataset or a deterministic instructional sample."""
    legacy_names = {
        "Breast Cancer · binary": "breast-cancer",
        "Wine · multiclass": "wine",
    }
    dataset_id = legacy_names.get(dataset_id, dataset_id)

    if dataset_id == "mobile-price":
        return pd.read_csv(ROOT_DIR / "dataset.csv")

    sklearn_loaders = {
        "breast-cancer": load_breast_cancer,
        "iris": load_iris,
        "wine": load_wine,
    }
    if dataset_id in sklearn_loaders:
        bundle = sklearn_loaders[dataset_id](as_frame=True)
        demo = bundle.data.copy()
        demo["target"] = bundle.target.map(dict(enumerate(bundle.target_names)))
        return demo

    if dataset_id not in SYNTHETIC_DATASET_FEATURES:
        raise ValueError(f"Unknown sample dataset: {dataset_id}")

    classes = 3 if dataset_id == "wine-quality" else 2
    weights = [0.98, 0.02] if dataset_id == "credit-card-fraud" else None
    feature_names = SYNTHETIC_DATASET_FEATURES[dataset_id]
    feature_matrix, labels = make_classification(
        n_samples=1_000 if dataset_id == "credit-card-fraud" else 800,
        n_features=len(feature_names),
        n_informative=min(7, len(feature_names) - 1),
        n_redundant=min(2, len(feature_names) - 7),
        n_repeated=0,
        n_classes=classes,
        n_clusters_per_class=1,
        weights=weights,
        flip_y=0.025,
        random_state=42,
    )
    demo = pd.DataFrame(feature_matrix, columns=feature_names)
    if dataset_id == "wine-quality":
        label_names = {0: "Standard", 1: "Good", 2: "Premium"}
    else:
        label_names = {0: "No", 1: "Yes"}
    demo["target"] = pd.Series(labels).map(label_names)
    return demo


def render_confusion_and_report(
    y_true: pd.Series,
    y_pred: Any,
    class_labels: list[Any],
    title: str,
) -> None:
    """Render a labeled confusion matrix and per-class classification report."""
    matrix, report_column = st.columns([1.1, 1])
    with matrix:
        st.markdown("#### Confusion matrix")
        values = confusion_matrix(y_true, y_pred, labels=class_labels)
        confusion_data = pd.DataFrame(
            [
                {"Actual": str(actual), "Predicted": str(predicted), "Count": int(values[row, column])}
                for row, actual in enumerate(class_labels)
                for column, predicted in enumerate(class_labels)
            ]
        )
        base = alt.Chart(confusion_data).encode(
            x=alt.X(
                "Predicted:N",
                sort=[str(label) for label in class_labels],
                axis=alt.Axis(title="Predicted class", labelAngle=-30),
            ),
            y=alt.Y(
                "Actual:N",
                sort=[str(label) for label in class_labels],
                axis=alt.Axis(title="Actual class"),
            ),
        )
        color_threshold = float(values.max()) / 2 if values.size else 0
        matrix_chart = (
            base.mark_rect(stroke="white", strokeWidth=2)
            .encode(
                color=alt.Color(
                    "Count:Q",
                    title="Rows",
                    scale=alt.Scale(scheme="blues"),
                ),
                tooltip=[
                    alt.Tooltip("Actual:N", title="Actual"),
                    alt.Tooltip("Predicted:N", title="Predicted"),
                    alt.Tooltip("Count:Q", title="Records", format=","),
                ],
            )
            +
            base.mark_text(fontSize=13, fontWeight="bold")
            .encode(
                text=alt.Text("Count:Q", format=","),
                color=alt.condition(
                    alt.datum.Count > color_threshold,
                    alt.value("white"),
                    alt.value("#172554"),
                ),
            )
        ).properties(
            height=max(240, min(440, 36 * len(class_labels))),
            title=title,
        )
        st.altair_chart(
            matrix_chart,
            width="stretch",
            key=f"confusion_{sha256(title.encode()).hexdigest()[:10]}",
        )
        st.caption("Hover a cell to inspect exact counts. Correct predictions run along the diagonal.")

    with report_column:
        st.markdown("#### Classification report")
        report = classification_report(
            y_true,
            y_pred,
            labels=class_labels,
            output_dict=True,
            zero_division=0,
        )
        report_frame = pd.DataFrame(report).transpose()
        report_formatters = {
            column: lambda value: "—" if pd.isna(value) else f"{value:.3f}"
            for column in report_frame.columns
        }
        report_html = report_frame.to_html(
            border=0,
            classes="ml-comparison-table ml-report-table",
            formatters=report_formatters,
            escape=True,
        )
        st.markdown(
            '<div class="ml-table-wrap ml-report-wrap" role="region" '
            'aria-label="Classification report">'
            f"{report_html}</div>",
            unsafe_allow_html=True,
        )


def render_metric_table(results: pd.DataFrame, label: str) -> None:
    """Render a compact, readable table without virtualized hidden columns."""
    formatters = {
        metric: lambda value: "N/A" if pd.isna(value) else f"{value:.4f}"
        for metric in METRIC_NAMES
    }
    table_html = results.to_html(
        index=False,
        border=0,
        classes="ml-comparison-table",
        formatters=formatters,
        escape=True,
    )
    st.markdown(
        f'<div class="ml-table-wrap" role="region" aria-label="{label}">'
        f"{table_html}</div>",
        unsafe_allow_html=True,
    )


def render_metric_cards(values: pd.Series) -> None:
    """Display all six metric values in roomy three-column rows."""
    for metric_row in (METRIC_NAMES[:3], METRIC_NAMES[3:]):
        columns = st.columns(3)
        for column, metric_name in zip(columns, metric_row):
            value = values[metric_name]
            column.metric(
                metric_name,
                "N/A" if pd.isna(value) else f"{value:.4f}",
            )


def lucide_icon(name: str, size: int = 18) -> str:
    """Render a small inline icon from the consistent Lucide stroke style."""
    paths = {
        "chart": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
        "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
        "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>',
        "sparkles": '<path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z"/><path d="m19 14 1.2 2.8L23 18l-2.8 1.2L19 22l-1.2-2.8L15 18l2.8-1.2L19 14Z"/>',
        "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
        "trophy": '<path d="M8 21h8m-4-4v4"/><path d="M7 4h10v5a5 5 0 0 1-10 0V4Z"/><path d="M7 6H4v2a4 4 0 0 0 4 4m9-6h3v2a4 4 0 0 1-4 4"/>',
    }
    svg = paths.get(name, paths["chart"])
    return (
        f'<svg aria-hidden="true" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round">{svg}</svg>'
    )


def select_gallery_dataset(dataset_id: str) -> None:
    """Remember the selected sample before Streamlit reruns the page."""
    st.session_state["selected_gallery_dataset"] = dataset_id
    st.session_state["generic_data_source"] = "Sample gallery"


def render_dataset_gallery() -> Any:
    """Display a searchable gallery with transparent source and difficulty labels."""
    st.markdown(
        f'<h2 id="dataset-library" class="mb-section-title">'
        f'{lucide_icon("database", 21)} Dataset library</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="mb-section-copy">Choose a ready-to-run sample or bring a labeled CSV. '
        'Each sample is available immediately without an external download.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div id="dataset-upload"></div>', unsafe_allow_html=True)
    source = st.radio(
        "Data source",
        ["Sample gallery", "Upload a CSV"],
        horizontal=True,
        key="generic_data_source",
    )
    search_col, upload_col = st.columns([1.55, 1])
    query = search_col.text_input(
        "Search datasets",
        placeholder="Search by name, task, or level",
        key="dataset_gallery_search",
    ).strip().casefold()
    uploaded_file = None
    if source == "Upload a CSV":
        uploaded_file = upload_col.file_uploader(
            "Upload your labeled CSV",
            type=["csv"],
            max_upload_size=20,
            help="Choose a classification CSV with at least one feature and one target column.",
            key="generic_csv_upload",
        )
    filter_columns = st.columns(2)
    task_filter = filter_columns[0].selectbox(
        "Problem type",
        ["All problem types"]
        + sorted({str(item["task"]) for item in DATASET_CATALOG}),
        key="dataset_gallery_task_filter",
    )
    difficulty_filter = filter_columns[1].selectbox(
        "Difficulty",
        ["All levels"] + sorted({str(item["difficulty"]) for item in DATASET_CATALOG}),
        key="dataset_gallery_difficulty_filter",
    )
    filtered_datasets = [
        item
        for item in DATASET_CATALOG
        if (task_filter == "All problem types" or item["task"] == task_filter)
        and (difficulty_filter == "All levels" or item["difficulty"] == difficulty_filter)
        if query
        in " ".join(
            str(item[key])
            for key in ("name", "task", "difficulty", "description", "source")
        ).casefold()
    ]
    if not filtered_datasets:
        st.info("No datasets match these filters. Adjust your filters or upload a CSV.")
    else:
        st.caption(f"{len(filtered_datasets)} datasets")
    for start in range(0, len(filtered_datasets), 3):
        columns = st.columns(3)
        for column, item in zip(columns, filtered_datasets[start : start + 3]):
            selected = (
                source == "Sample gallery"
                and
                st.session_state.get("selected_gallery_dataset", "mobile-price")
                == item["id"]
            )
            card_class = "mb-dataset-card is-selected" if selected else "mb-dataset-card"
            feature_count = int(item["features"])
            synthetic_badge = (
                '<span class="mb-data-source synthetic">Illustrative sample</span>'
                if item["synthetic"]
                else f'<span class="mb-data-source">{html.escape(item["source"])}</span>'
            )
            card_html = (
                f'<article class="{card_class}" title="{html.escape(str(item["description"]), quote=True)}">'
                f'<div class="mb-dataset-card-top"><span class="mb-icon">'
                f'{lucide_icon("database", 17)}</span>{synthetic_badge}</div>'
                f'<h3 title="{html.escape(str(item["name"]), quote=True)}">{html.escape(item["name"])}</h3>'
                f'<p title="{html.escape(str(item["description"]), quote=True)}">{html.escape(item["description"])}</p>'
                f'<div class="mb-badges"><span>{html.escape(item["task"])}</span>'
                f'<span>{html.escape(item["difficulty"])}</span></div>'
                f'<div class="mb-dataset-facts"><span><strong>{html.escape(item["rows"])}</strong> rows</span>'
                f'<span><strong>{feature_count + 1}</strong> columns</span>'
                f'<span><strong>{feature_count}</strong> features</span></div>'
                '</article>'
            )
            with column:
                st.markdown(card_html, unsafe_allow_html=True)
                st.button(
                    "Loaded" if selected else "Load sample",
                    key=f"load_sample_{item['id']}",
                    type="primary" if selected else "secondary",
                    on_click=select_gallery_dataset,
                    args=(item["id"],),
                    use_container_width=True,
                )
    if any(item["synthetic"] for item in DATASET_CATALOG):
        st.caption(
            "Samples marked illustrative are generated, deterministic examples. "
            "They are not real passenger, patient, customer, banking, or transaction records."
        )
    return uploaded_file


def render_model_scorecards(
    results: pd.DataFrame,
    ranking_metric: str,
    *,
    rank_by: str | None = None,
    prediction_rows: int | None = None,
) -> None:
    """Show comparable per-model metric cards with accessible progress bars."""
    score_column = rank_by or ranking_metric
    ranked = results.sort_values(score_column, ascending=False, na_position="last").reset_index(
        drop=True
    )
    cards: list[str] = []
    for index, row in ranked.iterrows():
        rank = index + 1
        winner = rank == 1
        name = html.escape(str(row["Model"]))
        medals = {1: "Gold", 2: "Silver", 3: "Bronze"}
        rank_label = f"{medals[rank]} · Rank {rank}" if rank in medals else f"Rank {rank}"
        rank_class = f"mb-rank rank-{rank}" if rank <= 3 else "mb-rank"
        training_time = row.get("Training Time (s)", np.nan)
        training_label = (
            f"{training_time:.2f} s"
            if pd.notna(training_time)
            else "Pre-trained · not recorded"
        )
        prediction_time = row.get("Prediction Time (s)", np.nan)
        prediction_label = (
            f"{prediction_time * 1000:.2f} ms"
            if pd.notna(prediction_time)
            else "Not recorded"
        )
        if pd.notna(prediction_time) and prediction_rows:
            prediction_label += f" / {prediction_rows:,} rows"
        progress_metrics = []
        for metric in ("Accuracy", "F1", "Precision", "Recall"):
            raw_value = row.get(metric, np.nan)
            display_value = "—" if pd.isna(raw_value) else f"{raw_value:.1%}"
            width = 0 if pd.isna(raw_value) else min(100, max(0, raw_value * 100))
            progress_metrics.append(
                f'<div class="mb-score-line"><div><span>{metric}</span>'
                f'<strong>{display_value}</strong></div>'
                f'<div class="mb-progress" role="progressbar" aria-label="{metric}" '
                f'aria-valuenow="{width:.0f}" aria-valuemin="0" aria-valuemax="100">'
                f'<span style="width:{width:.2f}%"></span></div></div>'
            )
        auc = row.get("AUC", np.nan)
        mcc = row.get("MCC", np.nan)
        cards.append(
            f'<article class="mb-model-card {"is-winner" if winner else ""}">'
            f'<div class="mb-model-card-heading"><div><span class="mb-model-icon">'
            f'{lucide_icon("chart", 17)}</span><h3 title="{name}">{name}</h3></div>'
            f'<span class="{rank_class}">{rank_label}</span></div>'
            f'{"".join(progress_metrics)}'
            f'<div class="mb-model-foot"><span>{lucide_icon("clock", 14)} '
            f'Training <strong>{training_label}</strong></span>'
            f'<span>Prediction <strong>{prediction_label}</strong></span>'
            f'<span>AUC <strong>{"—" if pd.isna(auc) else f"{auc:.3f}"}</strong>'
            f'</span><span>MCC <strong>{"—" if pd.isna(mcc) else f"{mcc:.3f}"}</strong>'
            '</span></div></article>'
        )
    st.markdown(
        '<div class="mb-model-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def render_model_guide() -> None:
    """Explain each classifier and when to use it in the comparison."""
    model_guidance = [
        (
            "Logistic Regression",
            "A fast, interpretable baseline. Start here for roughly linear class "
            "boundaries. Scaled numeric features help optimization.",
        ),
        (
            "Decision Tree",
            "Learns nonlinear rules and feature interactions. Useful for an "
            "explainable rule-like comparison, while deep trees can overfit.",
        ),
        (
            "K-Nearest Neighbors",
            "Classifies from nearby training rows. Scaling matters because "
            "distances drive predictions. Performance can fall with many features.",
        ),
        (
            "Gaussian Naive Bayes",
            "A lightweight probabilistic baseline with a conditional-independence "
            "assumption. Often fast, but correlated features can limit it.",
        ),
        (
            "Support Vector Machine",
            "Finds separating boundaries and can model complex patterns. Scaling "
            "is important. Probability calibration can take longer on large data.",
        ),
    ]
    with st.expander("Model playbook — when to use each algorithm", expanded=False):
        st.caption(
            "These are practical starting points, not guarantees. Compare all five "
            "on the same holdout and use the metric that matches your task."
        )
        for start in range(0, len(model_guidance), 3):
            cards = st.columns(3)
            for card, (name, guidance) in zip(cards, model_guidance[start : start + 3]):
                with card.container(border=True):
                    st.markdown(f"**{name}**")
                    st.caption(guidance)


def render_generic_benchmark() -> None:
    """Render the dataset library and mixed-type classification workflow."""
    uploaded_file = render_dataset_gallery()
    data_source = st.session_state.get("generic_data_source", "Sample gallery")
    if data_source == "Upload a CSV" and uploaded_file is None:
        st.info("Upload a labeled CSV to preview its columns and run a benchmark.")
        return
    expected_target: str | None = None
    selected_id = st.session_state.get("selected_gallery_dataset", "mobile-price")
    selected_dataset = next(
        (item for item in DATASET_CATALOG if item["id"] == selected_id),
        DATASET_CATALOG[0],
    )
    if uploaded_file is not None:
        raw_data = uploaded_file.getvalue()
        if len(raw_data) > MAX_UPLOAD_BYTES:
            st.error("CSV exceeds the 20 MB upload limit. Please use a smaller test dataset.")
            return
        upload_id = sha256(raw_data).hexdigest()
        try:
            header = pd.read_csv(
                BytesIO(raw_data),
                encoding="utf-8-sig",
                nrows=0,
                on_bad_lines="error",
            )
            if len(header.columns) > 101:
                st.error("CSV has too many columns. The general workbench allows at most 100 features plus a target.")
                return
            dataset = pd.read_csv(
                BytesIO(raw_data),
                encoding="utf-8-sig",
                low_memory=False,
                on_bad_lines="error",
                nrows=15_001,
            )
            if len(dataset) > 15_000:
                st.error("CSV exceeds the 15,000-row interactive benchmark limit.")
                return
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.error(f"Could not read the CSV: {error}")
            return
        expected_target = None
        dataset_label = uploaded_file.name
    else:
        try:
            dataset = load_demo_dataset(selected_id)
        except (OSError, ValueError) as error:
            st.error(f"Could not load the selected sample: {error}")
            return
        upload_id = f"sample:{selected_id}"
        expected_target = (
            TARGET_COLUMN if selected_id == "mobile-price" else "target"
        )
        dataset_label = selected_dataset["name"]

    if st.session_state.get("generic_upload_id") != upload_id:
        st.session_state["generic_upload_id"] = upload_id
        st.session_state.pop("generic_result", None)

    if dataset.empty or len(dataset.columns) < 2:
        st.error("CSV must have at least one data row, one feature, and a target column.")
        return
    if dataset.columns.duplicated().any():
        st.error("CSV column names must be unique.")
        return

    st.markdown(
        f'<h2 id="dataset-overview" class="mb-section-title">'
        f'{lucide_icon("database", 21)} Dataset overview</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**{html.escape(dataset_label)}**")
    control_columns = st.columns([2, 1, 1])
    target_column = control_columns[0].selectbox(
        "Target column",
        list(dataset.columns),
        index=(
            list(dataset.columns).index(expected_target)
            if expected_target in dataset.columns
            else len(dataset.columns) - 1
        ),
        key=f"generic_target_column_{sha256(upload_id.encode()).hexdigest()[:10]}",
        help="Choose the column containing the class label to predict.",
    )
    st.caption(f"Selected target column: {target_column}")
    test_size = control_columns[1].select_slider(
        "Holdout size",
        options=[0.1, 0.2, 0.25, 0.3, 0.4],
        value=0.2,
        format_func=lambda value: f"{value:.0%}",
        key="generic_test_size",
    )
    if uploaded_file is None:
        control_columns[2].download_button(
            "Download sample CSV",
            data=dataset.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_id}_sample.csv",
            mime="text/csv",
            use_container_width=True,
        )

    feature_columns = [column for column in dataset.columns if column != target_column]
    profile_columns = st.columns(4)
    profile_columns[0].metric("Rows", f"{len(dataset):,}")
    profile_columns[1].metric("Features", f"{len(feature_columns):,}")
    profile_columns[2].metric(
        "Classes",
        f"{dataset[target_column].nunique(dropna=True):,}",
    )
    profile_columns[3].metric(
        "Missing feature cells",
        f"{int(dataset[feature_columns].isna().sum().sum()):,}",
    )
    dataset_info = (
        f'<div class="mb-dataset-note"><span>{lucide_icon("database", 16)}</span>'
        f'<span><strong>{html.escape(str(selected_dataset["task"]))}</strong> task'
        f'<span class="mb-dot">·</span><strong>{len(feature_columns)}</strong> input features'
        f'<span class="mb-dot">·</span><strong>{html.escape(str(dataset[target_column].nunique(dropna=True)))}</strong> classes</span></div>'
    )
    st.markdown(dataset_info, unsafe_allow_html=True)
    if uploaded_file is None and selected_dataset["synthetic"]:
        st.info(
            "This is generated sample data for interface and workflow practice. "
            "It is not a real-world dataset and should not be used for decisions."
        )
    with st.expander("Preview dataset", expanded=False):
        st.dataframe(dataset.head(10), hide_index=True, width="stretch")
    render_model_guide()

    configuration = (upload_id, target_column, test_size)
    st.markdown(
        f'<h2 id="model-comparison" class="mb-section-title">'
        f'{lucide_icon("chart", 21)} Model comparison</h2>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Five classifiers share the same stratified holdout. Preprocessing is fitted "
        "on training data only."
    )
    if st.button("Compare classification algorithms", type="primary"):
        try:
            with st.spinner("Training five models and evaluating the holdout..."):
                (
                    results,
                    predictions,
                    y_test,
                    class_labels,
                    feature_relevance,
                ) = compare_classifiers(dataset, target_column, test_size=test_size)
        except (ValueError, TypeError, MemoryError) as error:
            st.session_state.pop("generic_result", None)
            st.error(f"Benchmark could not be completed: {error}")
            return
        st.session_state["generic_result"] = {
            "configuration": configuration,
            "results": results,
            "predictions": predictions,
            "y_test": y_test,
            "class_labels": class_labels,
            "feature_relevance": feature_relevance,
        }

    benchmark = st.session_state.get("generic_result")
    if benchmark is None or benchmark["configuration"] != configuration:
        st.info("Choose the target column, then run the comparison.")
        return

    results = benchmark["results"]
    y_test = benchmark["y_test"]
    class_labels = benchmark["class_labels"]
    predictions = benchmark["predictions"]

    ranking_metric = st.selectbox(
        "Rank models by",
        ["Accuracy", "F1", "AUC", "MCC", "Precision", "Recall"],
        key="generic_ranking_metric",
    )
    ranked_results = results.sort_values(
        ranking_metric, ascending=False, na_position="last"
    ).reset_index(drop=True)
    winner = ranked_results.iloc[0]
    winner_value = winner[ranking_metric]
    st.markdown(
        f'<div class="mb-winner-banner"><span class="mb-winner-icon">'
        f'{lucide_icon("trophy", 20)}</span><div><strong>{html.escape(str(winner["Model"]))}</strong>'
        f'<span> leads on {html.escape(ranking_metric)} at {winner_value:.1%}</span></div></div>',
        unsafe_allow_html=True,
    )
    fastest_model = results.loc[results["Prediction Time (s)"].idxmin()]
    summary = st.columns(3)
    summary[0].metric(
        f"Best {ranking_metric}",
        str(winner["Model"]),
        f"{winner_value:.1%}",
    )
    summary[1].metric(
        "Fastest prediction",
        str(fastest_model["Model"]),
        f"{fastest_model['Prediction Time (s)'] * 1000:.2f} ms",
        help="Total time to predict the held-out rows. Timing varies by machine.",
    )
    summary[2].metric(
        "Models compared",
        str(len(results)),
        f"{len(y_test):,} holdout rows",
    )
    render_model_scorecards(
        results,
        ranking_metric,
        prediction_rows=len(y_test),
    )
    st.caption(
        "Training time includes pipeline preprocessing and fitting. Prediction "
        "time measures the holdout prediction call and varies by machine."
    )

    st.markdown(
        f'<h2 id="feature-insights" class="mb-section-title">'
        f'{lucide_icon("sparkles", 21)} Feature insights</h2>',
        unsafe_allow_html=True,
    )
    insight_columns = st.columns([1, 1.4])
    with insight_columns[0]:
        distribution = (
            dataset[target_column]
            .dropna()
            .astype(str)
            .value_counts()
            .sort_index()
            .rename("Records")
        )
        st.markdown("#### Class balance")
        st.bar_chart(distribution, y_label="Records")
    with insight_columns[1]:
        st.markdown("#### Dataset profile")
        profile_metrics = st.columns(3)
        profile_metrics[0].metric(
            "Numeric features",
            f"{dataset[feature_columns].select_dtypes(include=np.number).shape[1]:,}",
        )
        profile_metrics[1].metric(
            "Categorical features",
            f"{dataset[feature_columns].select_dtypes(exclude=np.number).shape[1]:,}",
        )
        profile_metrics[2].metric(
            "Missing feature cells",
            f"{int(dataset[feature_columns].isna().sum().sum()):,}",
        )
    with insight_columns[0]:
        st.markdown("#### Training-set feature relevance")
        relevance = benchmark["feature_relevance"].head(10).sort_values(
            "Importance", ascending=True
        )
        st.bar_chart(
            relevance.set_index("Feature")["Importance"],
            x_label="Mutual information",
        )
        st.caption(
            "Estimated from the training partition only. Mutual information measures "
            "association, not causation or model-specific importance."
        )

    st.markdown(
        f'<h2 id="results-predictions" class="mb-section-title">'
        f'{lucide_icon("chart", 21)} Results &amp; predictions</h2>',
        unsafe_allow_html=True,
    )
    selected_model = st.selectbox(
        "Select a model for the confusion matrix and detailed report",
        MODEL_NAMES,
        key="generic_confusion_model",
    )
    chosen_row = results.loc[results["Model"] == selected_model].iloc[0]
    render_metric_cards(chosen_row)
    render_confusion_and_report(
        y_test,
        predictions[selected_model],
        class_labels,
        f"{selected_model} · holdout",
    )
    st.markdown(
        f'<h2 id="export-center" class="mb-section-title">'
        f'{lucide_icon("upload", 21)} Export center</h2>',
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download comparison metrics",
        data=ranked_results.to_csv(index=False).encode("utf-8"),
        file_name="classification_model_comparison.csv",
        mime="text/csv",
    )
    st.caption(
        "Metrics reflect one reproducible holdout and do not guarantee future performance."
    )


def render_dataset_hub() -> dict[str, Any] | None:
    """Load and profile the active dataset for every workspace."""
    uploaded_file = render_dataset_gallery()
    data_source = st.session_state.get("generic_data_source", "Sample gallery")
    if data_source == "Upload a CSV" and uploaded_file is None:
        st.info("Upload a labeled CSV to preview it and start a benchmark.")
        return None

    selected_id = st.session_state.get("selected_gallery_dataset", "mobile-price")
    selected_dataset = next(
        (item for item in DATASET_CATALOG if item["id"] == selected_id),
        DATASET_CATALOG[0],
    )
    expected_target: str | None = None
    try:
        if uploaded_file is not None:
            raw_data = uploaded_file.getvalue()
            if len(raw_data) > MAX_UPLOAD_BYTES:
                st.error("CSV exceeds the 20 MB upload limit. Please use a smaller dataset.")
                return None
            upload_id = sha256(raw_data).hexdigest()
            header = pd.read_csv(
                BytesIO(raw_data),
                encoding="utf-8-sig",
                nrows=0,
                on_bad_lines="error",
            )
            if len(header.columns) > 101:
                st.error(
                    "CSV has too many columns. The workbench allows at most "
                    "100 features plus a target."
                )
                return None
            dataset = pd.read_csv(
                BytesIO(raw_data),
                encoding="utf-8-sig",
                low_memory=False,
                on_bad_lines="error",
                nrows=15_001,
            )
            if len(dataset) > 15_000:
                st.error("CSV exceeds the 15,000-row interactive benchmark limit.")
                return None
            dataset_label = uploaded_file.name
        else:
            dataset = load_demo_dataset(selected_id)
            upload_id = f"sample:{selected_id}"
            expected_target = TARGET_COLUMN if selected_id == "mobile-price" else "target"
            dataset_label = str(selected_dataset["name"])
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(f"Could not load the selected dataset: {error}")
        return None

    if st.session_state.get("generic_upload_id") != upload_id:
        st.session_state["generic_upload_id"] = upload_id
        st.session_state.pop("generic_result", None)
    if dataset.empty or len(dataset.columns) < 2:
        st.error("Dataset must contain at least one row, one feature, and a target.")
        return None
    if dataset.columns.duplicated().any():
        st.error("Dataset column names must be unique.")
        return None

    st.markdown(
        '<h2 class="ml-section-title">Dataset profile</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**{html.escape(dataset_label)}**")
    controls = st.columns([2, 1, 1])
    target_column = controls[0].selectbox(
        "Target column",
        list(dataset.columns),
        index=(
            list(dataset.columns).index(expected_target)
            if expected_target in dataset.columns
            else len(dataset.columns) - 1
        ),
        key=f"generic_target_column_{sha256(upload_id.encode()).hexdigest()[:10]}",
        help="Choose the label your classifiers should predict.",
    )
    test_size = controls[1].select_slider(
        "Holdout size",
        options=[0.1, 0.2, 0.25, 0.3, 0.4],
        value=0.2,
        format_func=lambda value: f"{value:.0%}",
        key="generic_test_size",
    )
    feature_columns = [column for column in dataset.columns if column != target_column]
    if uploaded_file is None:
        controls[2].download_button(
            "Download sample CSV",
            data=dataset.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_id}_sample.csv",
            mime="text/csv",
            use_container_width=True,
        )

    dataset_metrics = [
        ("Records", f"{len(dataset):,}"),
        ("Features", f"{len(feature_columns):,}"),
        ("Classes", f"{dataset[target_column].nunique(dropna=True):,}"),
        (
            "Missing feature cells",
            f"{int(dataset[feature_columns].isna().sum().sum()):,}",
        ),
        (
            "Problem type",
            "Binary"
            if dataset[target_column].nunique(dropna=True) == 2
            else "Multiclass",
        ),
    ]
    for start in range(0, len(dataset_metrics), 3):
        metric_row = dataset_metrics[start : start + 3]
        for column, (label, value) in zip(st.columns(len(metric_row)), metric_row):
            column.metric(label, value)
    st.caption(f"Selected target: {target_column}")
    if uploaded_file is None and selected_dataset["synthetic"]:
        st.info(
            "This is deterministic generated sample data for workflow practice, "
            "not a real-world dataset."
        )
    with st.expander("Preview dataset", expanded=False):
        st.dataframe(dataset.head(10), hide_index=True, width="stretch")
    return {
        "dataset": dataset,
        "dataset_id": upload_id,
        "dataset_label": dataset_label,
        "selected_dataset": selected_dataset,
        "target_column": target_column,
        "test_size": test_size,
        "feature_columns": feature_columns,
        "configuration": (upload_id, target_column, test_size),
    }


def render_overview(context: dict[str, Any] | None) -> None:
    """Summarize the selected dataset and latest benchmark at a glance."""
    benchmark = st.session_state.get("generic_result")
    if context is None:
        st.info("Open Dataset Hub to choose a sample or upload a labeled CSV.")
        return
    is_current = (
        benchmark is not None
        and benchmark.get("configuration") == context["configuration"]
    )
    results = benchmark["results"] if is_current else None
    winner = (
        results.sort_values("Accuracy", ascending=False).iloc[0]
        if results is not None
        else None
    )
    kpis = [
        ("Models tested", "5" if is_current else "—"),
        (
            "Best accuracy",
            f"{winner['Accuracy']:.1%}" if winner is not None else "—",
        ),
        ("Selected dataset", context["dataset_label"]),
        (
            "Training duration",
            f"{results['Training Time (s)'].sum():.2f} s" if is_current else "—",
        ),
        ("Feature count", f"{len(context['feature_columns']):,}"),
    ]
    for start in range(0, len(kpis), 3):
        kpi_row = kpis[start : start + 3]
        for column, (label, value) in zip(st.columns(len(kpi_row)), kpi_row):
            column.metric(label, value)

    st.markdown("#### Benchmark summary")
    if winner is None:
        st.info("Your benchmark summary will appear here after the first comparison.")
    else:
        st.success(
            f"Current champion: {winner['Model']} with "
            f"{winner['Accuracy']:.1%} accuracy and {winner['F1']:.1%} F1."
        )
        st.dataframe(
            results[["Model", "Accuracy", "AUC", "F1", "MCC"]]
            .sort_values("Accuracy", ascending=False)
            .reset_index(drop=True),
            hide_index=True,
            width="stretch",
        )
    summary, activity = st.columns([1.25, 1])
    with summary:
        st.markdown("#### Dataset snapshot")
        st.write(
            f"{len(context['dataset']):,} records · "
            f"{len(context['feature_columns']):,} features · "
            f"{context['dataset'][context['target_column']].nunique(dropna=True):,} classes"
        )
    with activity:
        st.markdown("#### Recent activity")
        if is_current:
            st.caption(
                f"Benchmark completed · "
                f"{benchmark.get('completed_at', 'Earlier this session')}"
            )
        else:
            st.caption(f"Dataset ready · {context['dataset_label']}")
        st.markdown("#### Workspace guide")
        st.caption(
            "Dataset Hub → Benchmark Lab → Feature Studio / Prediction Center → Reports / Assignment"
        )
        st.caption("Evaluation uses a stratified holdout and never retrains on upload data.")


def render_benchmark_lab(context: dict[str, Any] | None) -> None:
    """Run and display the five-model benchmark with comparison visuals."""
    st.markdown("#### Champion model")
    if context is None:
        st.info("Choose a dataset in Dataset Hub before starting a benchmark.")
        return
    configuration = context["configuration"]
    st.caption(
        "Five classifiers share the same stratified holdout. Preprocessing is fitted "
        "on training data only."
    )
    if st.button("Run benchmark", type="primary", key="evalsphere_run_benchmark"):
        try:
            with st.spinner("Training and evaluating five classifiers..."):
                output = compare_classifiers(
                    context["dataset"],
                    context["target_column"],
                    test_size=context["test_size"],
                    include_probabilities=True,
                )
        except (ValueError, TypeError, MemoryError) as error:
            st.session_state.pop("generic_result", None)
            st.error(f"Benchmark could not be completed: {error}")
            return
        (
            results,
            predictions,
            y_test,
            class_labels,
            feature_relevance,
            probabilities,
        ) = output
        st.session_state["generic_result"] = {
            "configuration": configuration,
            "results": results,
            "predictions": predictions,
            "y_test": y_test,
            "class_labels": class_labels,
            "feature_relevance": feature_relevance,
            "probabilities": probabilities,
            "completed_at": datetime.now().astimezone().strftime(
                "%b %d, %Y · %H:%M %Z"
            ),
        }

    benchmark = st.session_state.get("generic_result")
    if benchmark is None or benchmark.get("configuration") != configuration:
        st.info("Run the benchmark to compare model performance on a shared holdout.")
        return

    results = benchmark["results"]
    ranking_metric = st.selectbox(
        "Rank models by",
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        key="generic_ranking_metric",
    )
    ranked = results.sort_values(
        ranking_metric, ascending=False, na_position="last"
    ).reset_index(drop=True)
    winner = ranked.iloc[0]
    winner_score = (
        f"{winner[ranking_metric]:.3f}"
        if ranking_metric == "MCC"
        else f"{winner[ranking_metric]:.1%}"
    )
    st.markdown(
        f'<div class="mb-winner-banner"><span class="mb-winner-icon">'
        f'{lucide_icon("trophy", 20)}</span><div><strong>{html.escape(str(winner["Model"]))}</strong>'
        f'<span> leads on {html.escape(ranking_metric)} at {winner_score}</span></div></div>',
        unsafe_allow_html=True,
    )
    metrics = st.columns(4)
    metrics[0].metric("Accuracy", f"{winner['Accuracy']:.1%}")
    metrics[1].metric("ROC AUC", f"{winner['AUC']:.3f}")
    metrics[2].metric("F1 score", f"{winner['F1']:.3f}")
    metrics[3].metric("Training time", f"{winner['Training Time (s)']:.2f} s")
    st.caption(
        f"Why it leads: highest {ranking_metric} on this holdout. Use it as a benchmark, "
        f"then validate for your deployment requirements."
    )

    render_model_scorecards(
        results,
        ranking_metric,
        prediction_rows=len(benchmark["y_test"]),
    )
    leaderboard = ranked[
        [
            "Model",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC",
            "Training Time (s)",
            "Prediction Time (s)",
        ]
    ].copy()
    leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
    st.markdown("#### Benchmark leaderboard")
    st.dataframe(leaderboard, hide_index=True, width="stretch")

    chart, heatmap = st.columns([1.2, 1])
    with chart:
        st.markdown("#### Metric comparison")
        metric_to_plot = st.selectbox(
            "Visualize metric",
            ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            key="benchmark_chart_metric",
        )
        chart_data = ranked[["Model", metric_to_plot]].copy()
        chart_data["Display"] = chart_data[metric_to_plot].map(
            lambda value: f"{value:.3f}" if metric_to_plot == "MCC" else f"{value:.1%}"
        )
        metric_domain = [-1, 1.18] if metric_to_plot == "MCC" else [0, 1.18]
        bars = (
            alt.Chart(chart_data)
            .mark_bar(color="#4f46e5", cornerRadiusEnd=6, height=22)
            .encode(
                x=alt.X(
                    f"{metric_to_plot}:Q",
                    title=metric_to_plot,
                    scale=alt.Scale(domain=metric_domain),
                    axis=alt.Axis(format=".2f" if metric_to_plot == "MCC" else ".0%"),
                ),
                y=alt.Y("Model:N", sort=ranked["Model"].tolist(), title=None),
                tooltip=[
                    alt.Tooltip("Model:N", title="Model"),
                    alt.Tooltip(
                        f"{metric_to_plot}:Q",
                        title=metric_to_plot,
                        format=".3f",
                    ),
                ],
            )
        )
        labels = bars.mark_text(
            align="left",
            baseline="middle",
            dx=6,
            fontWeight="bold",
            color="#334155",
        ).encode(text="Display:N")
        st.altair_chart(
            (bars + labels).properties(
                height=230,
                padding={"left": 4, "right": 38, "top": 8, "bottom": 4},
            ).interactive(bind_y=False),
            width="stretch",
            key="benchmark_metric_chart",
        )
    with heatmap:
        st.markdown("#### Performance matrix")
        heatmap_data = results[
            ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        ].melt(id_vars="Model", var_name="Metric", value_name="Score")
        heatmap_base = alt.Chart(heatmap_data).encode(
            x=alt.X("Metric:N", sort=METRIC_NAMES, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("Model:N", sort=ranked["Model"].tolist(), title=None),
        )
        score_cells = heatmap_base.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color(
                "Score:Q",
                title="Score",
                scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
            ),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Score:Q", format=".3f"),
            ],
        )
        score_labels = heatmap_base.mark_text(fontSize=11, fontWeight="bold").encode(
            text=alt.Text("Score:Q", format=".2f"),
            color=alt.condition(
                (alt.datum.Score > 0.6) | (alt.datum.Score < -0.6),
                alt.value("white"),
                alt.value("#172554"),
            ),
        )
        st.altair_chart(
            (score_cells + score_labels)
            .properties(height=230),
            width="stretch",
            key="benchmark_performance_heatmap",
        )
    st.caption(
        "Training time includes preprocessing and model fitting. Prediction time is "
        "measured on the holdout and varies by machine."
    )


def render_feature_studio(
    context: dict[str, Any] | None,
    benchmark: dict[str, Any] | None,
) -> None:
    """Explore training-only feature relevance and dataset distributions."""
    if context is None:
        st.info("Choose a dataset in Dataset Hub to explore its features.")
        return
    dataset = context["dataset"]
    features = context["feature_columns"]
    target = context["target_column"]
    st.markdown("#### Target distribution")
    balance = (
        dataset[target]
        .dropna()
        .astype(str)
        .value_counts()
        .sort_index()
        .rename_axis("Class")
        .reset_index(name="Records")
    )
    target_bars = alt.Chart(balance).mark_bar(color="#0891b2", cornerRadiusEnd=5).encode(
        x=alt.X("Class:N", sort=balance["Class"].tolist(), title="Class"),
        y=alt.Y(
            "Records:Q",
            title="Records",
            scale=alt.Scale(domainMax=float(balance["Records"].max()) * 1.18),
        ),
        tooltip=["Class:N", alt.Tooltip("Records:Q", format=",")],
    )
    target_labels = target_bars.mark_text(dy=-8, fontWeight="bold").encode(
        text=alt.Text("Records:Q", format=","),
    )
    st.altair_chart(
        (target_bars + target_labels)
        .properties(height=260)
        .interactive(bind_x=False),
        width="stretch",
        key="feature_target_distribution",
    )
    st.markdown("#### Data quality indicators")
    quality_metrics = [
        (
            "Numeric features",
            f"{dataset[features].select_dtypes(include=np.number).shape[1]:,}",
        ),
        (
            "Categorical features",
            f"{dataset[features].select_dtypes(exclude=np.number).shape[1]:,}",
        ),
        (
            "Missing feature cells",
            f"{int(dataset[features].isna().sum().sum()):,}",
        ),
    ]
    for column, (label, value) in zip(st.columns(3), quality_metrics):
        column.metric(label, value)

    if benchmark is not None and benchmark.get("configuration") == context["configuration"]:
        relevance = benchmark["feature_relevance"].head(15).sort_values(
            "Importance", ascending=False
        )
        st.markdown("#### Feature influence ranking")
        relevance = relevance.sort_values("Importance", ascending=True)
        relevance_bars = alt.Chart(relevance).mark_bar(
            color="#7c3aed",
            cornerRadiusEnd=5,
            height=15,
        ).encode(
            x=alt.X(
                "Importance:Q",
                title="Mutual information",
                scale=alt.Scale(
                    domainMax=max(
                        float(relevance["Importance"].max()) * 1.25,
                        0.01,
                    )
                ),
            ),
            y=alt.Y("Feature:N", sort=relevance["Feature"].tolist(), title=None),
            tooltip=[
                alt.Tooltip("Feature:N"),
                alt.Tooltip("Importance:Q", format=".4f"),
            ],
        )
        relevance_labels = relevance_bars.mark_text(
            align="left",
            baseline="middle",
            dx=5,
            fontSize=10,
            color="#334155",
        ).encode(text=alt.Text("Importance:Q", format=".3f"))
        st.altair_chart(
            (relevance_bars + relevance_labels)
            .properties(height=max(240, 24 * len(relevance)))
            .interactive(bind_y=False),
            width="stretch",
            key="feature_influence_chart",
        )
        st.caption(
            "Mutual information is estimated on training data only. It measures "
            "association, not causation or model-specific feature importance."
        )
    else:
        st.info("Run a benchmark to calculate training-partition feature relevance.")

    numeric_features = list(dataset[features].select_dtypes(include=np.number).columns)
    if not numeric_features:
        st.info("Correlation and numeric distribution views require numeric features.")
        return
    default_features = (
        benchmark["feature_relevance"]["Feature"].tolist()[:12]
        if benchmark is not None and benchmark.get("configuration") == context["configuration"]
        else numeric_features[:12]
    )
    selected_features = [
        feature for feature in default_features if feature in numeric_features
    ][:12]
    if len(selected_features) > 1:
        st.markdown("#### Correlation analysis")
        correlation_data = (
            dataset[selected_features]
            .corr(numeric_only=True)
            .rename_axis("Feature")
            .reset_index()
            .melt(id_vars="Feature", var_name="Related feature", value_name="Correlation")
        )
        correlation_base = alt.Chart(correlation_data).encode(
            x=alt.X(
                "Related feature:N",
                sort=selected_features,
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("Feature:N", sort=selected_features),
        )
        correlation_cells = correlation_base.mark_rect(
            stroke="white",
            strokeWidth=1,
        ).encode(
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
            ),
            tooltip=[
                alt.Tooltip("Feature:N"),
                alt.Tooltip("Related feature:N"),
                alt.Tooltip("Correlation:Q", format=".3f"),
            ],
        )
        correlation_labels = correlation_base.mark_text(
            fontSize=10,
            fontWeight="bold",
        ).encode(
            text=alt.Text("Correlation:Q", format=".2f"),
            color=alt.condition(
                (alt.datum.Correlation > 0.55) | (alt.datum.Correlation < -0.55),
                alt.value("white"),
                alt.value("#172554"),
            ),
        )
        st.altair_chart(
            (correlation_cells + correlation_labels)
            .properties(
                height=max(300, min(600, len(selected_features) * 34)),
            ),
            width="stretch",
            key="feature_correlation_heatmap",
        )
    selected_feature = st.selectbox(
        "Inspect a feature distribution",
        numeric_features,
        key="feature_studio_distribution",
    )
    feature_values = dataset[selected_feature].dropna().to_numpy(dtype=float)
    bin_count = min(30, max(1, dataset[selected_feature].nunique()))
    distribution_counts, bin_edges = np.histogram(feature_values, bins=bin_count)
    distribution_data = pd.DataFrame(
        {
            "Value range": [
                f"{bin_edges[index]:.3g}–{bin_edges[index + 1]:.3g}"
                for index in range(len(distribution_counts))
            ],
            "Records": distribution_counts,
        }
    )
    distribution_bars = alt.Chart(distribution_data).mark_bar(
        color="#2563eb",
        opacity=0.85,
        cornerRadiusEnd=3,
    ).encode(
        x=alt.X(
            "Value range:N",
            sort=distribution_data["Value range"].tolist(),
            title=selected_feature,
            axis=alt.Axis(labelAngle=-35),
        ),
        y=alt.Y("Records:Q", title="Records"),
        tooltip=[
            alt.Tooltip("Value range:N"),
            alt.Tooltip("Records:Q", format=","),
        ],
    )
    st.altair_chart(
        distribution_bars.properties(height=280).interactive(bind_x=False),
        width="stretch",
        key="feature_distribution_chart",
    )


def render_prediction_center(benchmark: dict[str, Any] | None) -> None:
    """Inspect held-out predictions with probability confidence and batch export."""
    if benchmark is None:
        st.info("Run a benchmark to explore single and batch holdout predictions.")
        return
    model_name = st.selectbox(
        "Prediction model",
        MODEL_NAMES,
        key="prediction_center_model",
    )
    predictions = benchmark["predictions"][model_name]
    probabilities = benchmark["probabilities"][model_name]
    actual = benchmark["y_test"].reset_index(drop=True)
    predicted = pd.Series(predictions, name="Prediction")
    confidence = pd.Series(probabilities.max(axis=1), name="Confidence")
    class_labels = benchmark["class_labels"]
    mode = st.radio(
        "Prediction view",
        ["Single record", "Batch results"],
        horizontal=True,
        key="prediction_center_mode",
    )
    if mode == "Single record":
        row_index = st.number_input(
            "Holdout record",
            min_value=1,
            max_value=len(actual),
            value=1,
            step=1,
            key="prediction_center_row",
        ) - 1
        result_columns = st.columns(3)
        result_columns[0].metric("Predicted class", str(predicted.iloc[row_index]))
        result_columns[1].metric("Confidence", f"{confidence.iloc[row_index]:.1%}")
        result_columns[2].metric("Actual class", str(actual.iloc[row_index]))
        probability_row = pd.Series(
            probabilities[row_index],
            index=[str(label) for label in class_labels],
            name="Class probability",
        )
        st.bar_chart(probability_row, y_label="Probability")
        st.caption(
            "Confidence is the selected model's highest predicted class probability. "
            "It is not a guarantee of correctness."
        )
    else:
        batch = pd.DataFrame(
            {
                "Actual": actual.astype(str),
                "Prediction": predicted.astype(str),
                "Confidence": confidence,
            }
        )
        for class_index, class_label in enumerate(class_labels):
            batch[f"Probability · {class_label}"] = probabilities[:, class_index]
        st.dataframe(batch, hide_index=True, width="stretch")
        st.download_button(
            "Export batch predictions",
            data=batch.to_csv(index=False).encode("utf-8"),
            file_name=f"{model_name.lower().replace(' ', '_')}_predictions.csv",
            mime="text/csv",
            key="prediction_center_export",
        )
    st.markdown("#### Model-level feature context")
    st.caption(
        "The strongest dataset associations from training data are shown for context; "
        "they are not per-record explanations."
    )
    st.dataframe(
        benchmark["feature_relevance"].head(5),
        hide_index=True,
        width="stretch",
    )


def _build_benchmark_pdf(results: pd.DataFrame, dataset_label: str) -> bytes:
    """Create a concise executive benchmark PDF in memory."""
    from matplotlib.backends.backend_pdf import PdfPages

    buffer = BytesIO()
    report_columns = [
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "Training Time (s)",
        "Prediction Time (s)",
    ]
    report = results[report_columns].sort_values("Accuracy", ascending=False)
    with PdfPages(buffer) as pdf:
        figure, axis = plt.subplots(figsize=(11.7, 8.3))
        axis.axis("off")
        axis.text(
            0.02,
            0.94,
            "EvalSphere · Benchmark Report",
            fontsize=22,
            fontweight="bold",
            color="#312E81",
            transform=axis.transAxes,
        )
        axis.text(
            0.02,
            0.87,
            f"Dataset: {dataset_label}  |  Champion: {report.iloc[0]['Model']}",
            fontsize=12,
            color="#475569",
            transform=axis.transAxes,
        )
        formatted = report.copy()
        for column in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
        for column in ["Training Time (s)", "Prediction Time (s)"]:
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f} s")
        table = axis.table(
            cellText=formatted.values,
            colLabels=formatted.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        axis.text(
            0.02,
            0.08,
            "Metrics describe one stratified holdout and do not guarantee future performance.",
            fontsize=9,
            color="#64748B",
            transform=axis.transAxes,
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    return buffer.getvalue()


def render_reports(
    context: dict[str, Any] | None,
    benchmark: dict[str, Any] | None,
) -> None:
    """Export dataset and benchmark reports."""
    if context is not None:
        dataset = context["dataset"]
        features = context["feature_columns"]
        target_column = context["target_column"]
        dataset_profile = pd.DataFrame(
            {
                "Feature": features,
                "Type": [str(dataset[column].dtype) for column in features],
                "Missing": [int(dataset[column].isna().sum()) for column in features],
                "Unique values": [
                    int(dataset[column].nunique(dropna=True)) for column in features
                ],
            }
        )
        st.markdown("#### Dataset report")
        st.caption(
            f"{context['dataset_label']} · {len(dataset):,} records · "
            f"{len(features):,} features · "
            f"{dataset[target_column].nunique(dropna=True):,} target classes"
        )
        st.dataframe(dataset_profile.head(20), hide_index=True, width="stretch")
        st.download_button(
            "Export dataset profile as CSV",
            data=dataset_profile.to_csv(index=False).encode("utf-8"),
            file_name="evalsphere_dataset_profile.csv",
            mime="text/csv",
            key="dataset_profile_export",
        )
        st.markdown("#### Benchmark summary")
    if (
        context is not None
        and benchmark is not None
        and benchmark.get("configuration") == context["configuration"]
    ):
        results = benchmark["results"].sort_values("Accuracy", ascending=False)
        st.dataframe(results, hide_index=True, width="stretch")
        downloads = st.columns(2)
        downloads[0].download_button(
            "Export results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="evalsphere_benchmark_report.csv",
            mime="text/csv",
            key="report_csv_export",
        )
        downloads[1].download_button(
            "Export executive PDF",
            data=_build_benchmark_pdf(results, context["dataset_label"]),
            file_name="evalsphere_benchmark_report.pdf",
            mime="application/pdf",
            key="report_pdf_export",
        )
    else:
        st.info("Run a benchmark to generate downloadable CSV and PDF reports.")

def render_evalsphere_workspace() -> None:
    """Present the complete application as a seven-workspace product."""
    tabs = st.tabs(
    [
        "Overview",
        "Dataset Hub",
        "Benchmark Lab",
        "Feature Studio",
        "Prediction Center",
        "Reports",
        "Assignment",
    ]
    )
    with tabs[1]:
        context = render_dataset_hub()
    with tabs[2]:
        render_benchmark_lab(context)
    benchmark = st.session_state.get("generic_result")
    with tabs[0]:
        render_overview(context)
    with tabs[3]:
        render_feature_studio(context, benchmark)
    with tabs[4]:
        current_benchmark = (
            benchmark
            if context is not None
            and benchmark is not None
            and benchmark.get("configuration") == context["configuration"]
            else None
        )
        render_prediction_center(current_benchmark)
    with tabs[5]:
        current_benchmark = (
            benchmark
            if context is not None
            and benchmark is not None
            and benchmark.get("configuration") == context["configuration"]
            else None
        )
        render_reports(context, current_benchmark)
    with tabs[6]:
        render_assignment_demo()


def render_assignment_demo() -> None:
    """Keep the original saved-model assignment workflow available."""
    st.subheader("Mobile Price assignment models")
    st.write(
        "These saved models use the included Mobile Price dataset and fixed "
        "training split. Upload a matching labeled test CSV to compare them."
    )

    try:
        assignment_dataset = pd.read_csv(ROOT_DIR / "dataset.csv")
        feature_columns = validate_dataset(assignment_dataset)
        record_count = len(assignment_dataset)
        feature_count = len(feature_columns)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not verify the included assignment dataset: {error}")
        return

    class_count = assignment_dataset[TARGET_COLUMN].nunique()
    missing_count = int(assignment_dataset.isna().sum().sum())
    verification = st.columns(4)
    verification[0].metric("Dataset check", "Passed")
    verification[1].metric("Records", f"{record_count:,}")
    verification[2].metric("Input features", f"{feature_count}")
    verification[3].metric("Target", TARGET_COLUMN)
    st.caption(
        f"Verified from dataset.csv: {record_count:,} instances, "
        f"{feature_count} numeric features, {class_count} classes, and "
        f"{missing_count:,} missing values. Assignment minimum: 500 instances "
        "and 12 features."
    )
    render_model_guide()
    with st.expander("Sample files and how to view their metrics", expanded=False):
        st.markdown(
            """
            1. Download **Balanced sample (100 rows)** for a quick check, or the
               **Full holdout (400 rows)** for the assignment evaluation.
            2. Upload that CSV below. It must include all 20 Mobile Price features
               and the `price_range` target. The app validates the columns.
            3. Choose a classifier to inspect its metrics, confusion matrix, and
               classification report. The comparison evaluates every saved
               model on the same uploaded rows without retraining.
            4. In **Explore datasets**, try samples from the searchable gallery.
               Generic mode trains on a
               stratified training partition and evaluates on its holdout.

            **Metric reminder:** Accuracy is overall correctness. AUC measures
            class ranking. Precision measures reliability of positive/class
            predictions. Recall measures coverage of actual examples. F1 balances
            Precision and Recall. MCC summarizes the confusion matrix from -1 to 1.
            For multiclass tasks, AUC is weighted one-vs-rest and Precision,
            Recall, and F1 use weighted averaging.
            """
        )

    selected_model = st.selectbox(
        "Assignment model",
        list(MODEL_SPECS),
        key="assignment_model",
    )

    try:
        baseline_results = pd.read_csv(ROOT_DIR / "results.csv")
        reference_data = (ROOT_DIR / "test_data.csv").read_bytes()
        prediction_times = measure_assignment_prediction_times(reference_data)
        baseline_results = baseline_results.merge(
            prediction_times,
            on="Model",
            how="left",
            validate="one_to_one",
        )
        reference_rows = len(pd.read_csv(BytesIO(reference_data), nrows=50_001))
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not load the included holdout comparison: {error}")
    else:
        st.markdown("#### Saved model comparison")
        render_model_scorecards(
            baseline_results,
            "Accuracy",
            prediction_rows=reference_rows,
        )
        st.caption(
            "The saved models' original training times were not recorded. "
            f"Prediction time measures their call on the {reference_rows:,}-row reference holdout."
        )
        reference_data_frame = pd.read_csv(BytesIO(reference_data), encoding="utf-8-sig")
        selected_artifact, selected_scaler = load_artifacts(selected_model)
        reference_validated = validate_assignment_upload(
            reference_data_frame,
            list(selected_scaler.feature_names_in_),
        )
        reference_predictions = selected_artifact.predict(
            selected_scaler.transform(
                reference_validated[list(selected_scaler.feature_names_in_)]
            )
        )
        st.markdown("#### Reference holdout diagnostics")
        render_confusion_and_report(
            reference_validated[TARGET_COLUMN],
            reference_predictions,
            list(selected_artifact.classes_),
            f"{selected_model} · {reference_rows:,}-row reference holdout",
        )

    for filename, label in [
        ("test_data.csv", "Download full holdout (400 rows)"),
        ("sample_test_data.csv", "Download balanced sample (100 rows)"),
    ]:
        sample_path = ROOT_DIR / filename
        if sample_path.is_file():
            st.download_button(
                label,
                data=sample_path.read_bytes(),
                file_name=filename,
                mime="text/csv",
                key=f"assignment_download_{filename}",
            )

    uploaded_file = st.file_uploader(
        "Upload Mobile Price test CSV",
        type=["csv"],
    max_upload_size=20,
    key="assignment_csv_upload",
    )
    if uploaded_file is None:
        return

    try:
        raw_data = uploaded_file.getvalue()
        if len(raw_data) > MAX_UPLOAD_BYTES:
            st.error("CSV exceeds the 20 MB upload limit. Please use a smaller test dataset.")
            return
        uploaded_data = pd.read_csv(
            BytesIO(raw_data),
            encoding="utf-8-sig",
            low_memory=False,
            on_bad_lines="error",
            nrows=50_001,
        )
        if len(uploaded_data) > 50_000:
            st.error("Assignment test CSV is limited to 50,000 rows.")
            return
        model, scaler = load_artifacts(selected_model)
        feature_columns = list(scaler.feature_names_in_)
        validated = validate_assignment_upload(uploaded_data, feature_columns)
    except (
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError,
        FileNotFoundError,
    ) as error:
        st.error(str(error))
        return

    features = validated[feature_columns]
    target = validated[TARGET_COLUMN]
    known_classes = list(model.classes_)
    unknown = sorted(set(target.unique()) - set(known_classes))
    if unknown:
        st.error(f"Target contains unknown class labels: {unknown}")
        return

    scaled_features = scaler.transform(features)
    rows = []
    predictions_by_model = {}
    for model_name in MODEL_SPECS:
        try:
            candidate = (
                model
                if model_name == selected_model
                else load_artifacts(model_name)[0]
            )
        except (OSError, ValueError, EOFError) as error:
            st.error(f"Could not load {model_name}: {error}")
            return
        prediction_started = perf_counter()
        predictions = candidate.predict(scaled_features)
        prediction_time = perf_counter() - prediction_started
        probabilities = candidate.predict_proba(scaled_features)
        rows.append(
            {
                "Model": model_name,
                **calculate_metrics(
                    target,
                    predictions,
                    probabilities,
                    known_classes,
                ),
                "Prediction Time (s)": prediction_time,
            }
        )
        predictions_by_model[model_name] = predictions

    assignment_results = pd.DataFrame(rows)
    st.subheader("Comparison on uploaded test data")
    winner = assignment_results.loc[assignment_results["Accuracy"].idxmax()]
    st.success(
        f"Best on this upload by accuracy: {winner['Model']} "
        f"({winner['Accuracy']:.2%})"
    )
    render_model_scorecards(
        assignment_results,
        "Accuracy",
        prediction_rows=len(target),
    )
    st.caption(
        "Saved-model prediction timing excludes CSV loading and feature scaling."
    )
    if assignment_results["AUC"].isna().any():
        st.info(
            "AUC is unavailable when the uploaded test data does not contain "
            "every target class."
        )

    chosen_metrics = assignment_results.loc[
        assignment_results["Model"] == selected_model
    ].iloc[0]
    render_metric_cards(chosen_metrics)

    render_confusion_and_report(
        target,
        predictions_by_model[selected_model],
        known_classes,
        f"{selected_model} · uploaded test data",
    )


st.set_page_config(
    page_title=f"{PROJECT_NAME} | {PROJECT_TAGLINE}",
    page_icon=str(ROOT_DIR / "assets" / "evalsphere.svg"),
    layout="wide",
)
dark_mode = bool(st.session_state.get("evalsphere_dark_mode", False))
if dark_mode:
    theme_tokens = """
        --ml-page: #0b1020; --ml-surface: #111a2e; --ml-surface-raised: #17223a;
        --ml-ink: #edf2ff; --ml-muted: #a7b3cc; --ml-brand: #c5c2ff;
        --ml-accent: #60a5fa; --ml-line: #2a3752; --ml-row: #141f35;
        --ml-soft: #192641; --ml-shadow: rgba(0,0,0,.28);
        --ml-gold-ink: #f3d982; --ml-gold-bg: #3a321d; --ml-gold-line: #766637;
        --ml-silver-ink: #d5dce7; --ml-silver-bg: #283548; --ml-silver-line: #48576a;
        --ml-bronze-ink: #e4b895; --ml-bronze-bg: #392a23; --ml-bronze-line: #76533f;
    """
else:
    theme_tokens = """
        --ml-page: #f5f7fc; --ml-surface: #ffffff; --ml-surface-raised: #ffffff;
        --ml-ink: #18233c; --ml-muted: #5c6881; --ml-brand: #312e81;
        --ml-accent: #2563eb; --ml-line: #dce3f0; --ml-row: #f7f8fc;
        --ml-soft: #eff1ff; --ml-shadow: rgba(34,39,92,.08);
        --ml-gold-ink: #704b00; --ml-gold-bg: #fff0bd; --ml-gold-line: #e8ca73;
        --ml-silver-ink: #495466; --ml-silver-bg: #edf0f5; --ml-silver-line: #cbd2dc;
        --ml-bronze-ink: #754321; --ml-bronze-bg: #f6e1d0; --ml-bronze-line: #e1b794;
    """
st.markdown(
    """
    <style>
    :root {
        __ML_THEME_TOKENS__
    }
    .stApp { background: var(--ml-page) !important; color: var(--ml-ink) !important; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--ml-page) !important; }
    [data-testid="stTabs"] [role="tablist"] {
        gap: .35rem; padding: .35rem; border: 1px solid var(--ml-line);
        border-radius: 14px; background: var(--ml-surface);
    }
    [data-testid="stTabs"] button[role="tab"] {
        min-height: 2.8rem; padding: .55rem .85rem; border-radius: 10px;
        color: var(--ml-muted); transition: color .18s ease, background .18s ease;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--ml-brand); background: var(--ml-soft);
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }
    .eval-header {
        display:flex; align-items:center; justify-content:space-between; gap:1rem;
        padding:.9rem 1.15rem; margin:.1rem 0 1.1rem;
        border:1px solid var(--ml-line); border-radius:16px;
        background:var(--ml-surface); box-shadow:0 6px 20px var(--ml-shadow);
    }
    .eval-lockup { display:flex; align-items:center; gap:.85rem; min-width:0; }
    .eval-logo {
        display:grid; place-items:center; width:2.8rem; height:2.8rem; flex:none;
        border-radius:13px; background:linear-gradient(145deg,#312e81,#2563eb);
        box-shadow:0 7px 18px rgba(37,99,235,.24);
    }
    .eval-wordmark { color:var(--ml-brand); font-weight:850; font-size:1.04rem; letter-spacing:.075em; }
    .eval-tagline { color:var(--ml-muted); font-size:.78rem; margin-top:.12rem; }
    .eval-hero {
        display:flex; align-items:center; justify-content:space-between; gap:1.5rem;
        padding:clamp(1.3rem,3vw,2.1rem); margin-bottom:1.15rem;
        border:1px solid var(--ml-line); border-radius:20px;
        background:
          radial-gradient(ellipse at 87% 15%,rgba(124,58,237,.14),transparent 33%),
          linear-gradient(120deg,var(--ml-surface),var(--ml-soft));
        box-shadow:0 10px 30px var(--ml-shadow);
    }
    .eval-hero h1 {
        margin:.55rem 0 .45rem; color:var(--ml-brand); font-size:clamp(1.85rem,4vw,3rem);
        line-height:1.08; letter-spacing:-.05em; font-weight:850;
    }
    .eval-hero p { max-width:760px; margin:0; color:var(--ml-muted); font-size:1rem; line-height:1.6; }
    .eval-kicker { color:var(--ml-accent); font-size:.72rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
    .eval-orbit { flex:none; width:clamp(5rem,12vw,8rem); height:auto; opacity:.92; }
    .eval-section-kicker { color:var(--ml-accent); font-size:.72rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
    .eval-card-note { color:var(--ml-muted); font-size:.83rem; }
    [data-testid="stMetric"] {
        border-radius:14px; border-color:var(--ml-line);
        box-shadow:0 4px 14px var(--ml-shadow);
    }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
        border-radius:10px; transition:transform .18s ease, box-shadow .18s ease;
    }
    [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
        transform:translateY(-1px); box-shadow:0 6px 16px var(--ml-shadow);
    }
    .block-container { max-width:1480px; padding-top:4.25rem!important; }
    @media (max-width:760px) {
        .eval-hero { align-items:flex-start; }
        .eval-orbit { width:4.5rem; }
        [data-testid="stTabs"] [role="tablist"] { overflow-x:auto; }
        [data-testid="stTabs"] button[role="tab"] { padding:.5rem .65rem; }
    }
    @media (max-width:520px) {
        .eval-hero { padding:1.2rem; }
        .eval-orbit { display:none; }
        .eval-header { padding:.75rem; }
        .eval-tagline { font-size:.7rem; }
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--ml-page) !important; color: var(--ml-ink) !important; }
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"] { color: var(--ml-ink) !important; }
    [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p { color: var(--ml-ink) !important; }
    [data-testid="stAlert"] > div, [data-testid="stNotification"] {
        background: var(--ml-surface-raised) !important;
        color: var(--ml-ink) !important;
        border-color: var(--ml-line) !important;
    }
    [data-testid="stAlert"] p, [data-testid="stNotification"] * { color: var(--ml-ink) !important; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--ml-muted) !important; }
    .block-container { max-width: 1480px; padding-top: .8rem; padding-bottom: 3rem; }
    [data-testid="stHeader"] { background: var(--ml-surface); border-bottom: 1px solid var(--ml-line); }
    [data-testid="stSidebar"] { background: var(--ml-surface); border-right: 1px solid var(--ml-line); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--ml-ink); }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {
        padding: .48rem .6rem; border: 1px solid transparent; border-radius: 7px;
        transition: background .18s ease, border-color .18s ease, transform .18s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
        background: var(--ml-soft); border-color: var(--ml-line); transform: translateX(2px);
    }
    .ml-topbar {
        display: flex; align-items: center; justify-content: space-between; gap: 1rem;
        background: var(--ml-surface); border: 1px solid var(--ml-line); border-radius: 12px;
        padding: .8rem 1.15rem; margin: .25rem 0 1rem;
        box-shadow: 0 2px 8px var(--ml-shadow);
        transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
    }
    .ml-brand { color: var(--ml-brand) !important; font-weight: 800; font-size: 1.05rem; }
    .ml-brand small { color: var(--ml-muted) !important; font-weight: 500; margin-left: .55rem; }
    .ml-owner, .ml-owner strong { color: var(--ml-ink) !important; font-size: .82rem; text-align: right; }
    .ml-hero {
        position: relative; overflow: hidden; padding: 1.65rem 1.9rem;
        margin: .25rem 0 1rem; border-radius: 12px; color: var(--ml-ink);
        background: var(--ml-surface); border: 1px solid var(--ml-line);
        border-left: 5px solid var(--ml-accent);
        box-shadow: 0 3px 12px var(--ml-shadow);
    }
    .stMarkdownContainer .ml-hero { background: var(--ml-surface) !important; color: var(--ml-ink) !important; }
    .ml-eyebrow { color: var(--ml-accent) !important; font-weight: 750; letter-spacing: .10em; text-transform: uppercase; font-size: .73rem; }
    .ml-hero h1 { color: var(--ml-brand) !important; font-size: clamp(1.8rem, 3vw, 2.65rem); letter-spacing: -.035em; margin: .45rem 0 .4rem; }
    .ml-hero p { color: var(--ml-muted) !important; font-size: 1rem; max-width: 850px; margin: 0; }
    .ml-pill-row { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: 1rem; }
    .ml-pill { border: 1px solid var(--ml-line); background: var(--ml-soft); border-radius: 6px; padding: .35rem .65rem; color: var(--ml-brand); font-size: .8rem; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
    [data-testid="stTabs"] button { font-weight: 700; color: var(--ml-muted); }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--ml-accent); }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--ml-accent); }
    [data-testid="stTabs"] [role="tablist"] {
        display: grid !important;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        align-items: stretch;
        gap: .45rem;
        width: 100%;
        padding: .45rem;
        border: 1px solid var(--ml-line);
        border-radius: 14px;
        background: var(--ml-surface);
        box-shadow: 0 3px 12px var(--ml-shadow);
    }
    [data-testid="stTabs"] button[role="tab"] {
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 0;
        min-height: 3rem;
        padding: .6rem .5rem;
        border: 1px solid transparent;
        border-radius: 10px;
        color: var(--ml-muted);
        font-size: .9rem;
        font-weight: 700;
        line-height: 1.25;
        text-align: center;
        white-space: normal;
        overflow-wrap: anywhere;
        transition: color .18s ease, background .18s ease, border-color .18s ease,
            box-shadow .18s ease, transform .18s ease;
    }
    [data-testid="stTabs"] button[role="tab"]:hover {
        background: var(--ml-soft);
        border-color: var(--ml-line);
        transform: translateY(-1px);
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--ml-brand) !important;
        background: var(--ml-soft);
        border-color: var(--ml-line);
        box-shadow: 0 2px 8px var(--ml-shadow);
    }
    [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        padding-top: 1.2rem;
    }
    @media (max-width: 1024px) {
        [data-testid="stTabs"] [role="tablist"] {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
    }
    @media (max-width: 680px) {
        [data-testid="stTabs"] [role="tablist"] {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .35rem;
            padding: .4rem;
        }
        [data-testid="stTabs"] button[role="tab"] {
            min-height: 2.85rem;
            padding: .55rem .4rem;
            font-size: .84rem;
        }
        [data-testid="stTabs"] [data-baseweb="tab-panel"] {
            padding-top: 1rem;
        }
    }
    [data-testid="stMetric"] {
        background: var(--ml-surface); border: 1px solid var(--ml-line); border-radius: 9px;
        padding: .85rem 1rem; box-shadow: 0 2px 7px var(--ml-shadow);
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: var(--ml-brand) !important; }
    [data-testid="stMetricLabel"] { color: var(--ml-muted) !important; }
    .ml-table-wrap {
        width: 100%; overflow-x: auto; margin: .55rem 0 1rem;
        border: 1px solid var(--ml-line); border-radius: 9px;
        box-shadow: 0 2px 7px var(--ml-shadow); background: var(--ml-surface);
        transition: box-shadow .2s ease, border-color .2s ease;
    }
    .ml-comparison-table {
        width: 100%; min-width: 720px; border-collapse: separate;
        border-spacing: 0; background: var(--ml-surface);
        font-variant-numeric: tabular-nums;
    }
    .ml-comparison-table th {
        padding: .8rem .85rem; text-align: right; white-space: nowrap;
        color: var(--ml-brand); background: var(--ml-soft);
        font-size: .82rem; letter-spacing: .02em;
        border-bottom: 1px solid var(--ml-line);
    }
    .ml-comparison-table th:first-child { text-align: left; }
    .ml-comparison-table td {
        padding: .76rem .85rem; border-bottom: 1px solid var(--ml-line);
        text-align: right; white-space: nowrap; color: var(--ml-ink);
        font-size: .9rem;
    }
    .ml-comparison-table td:first-child {
        text-align: left; font-weight: 700; color: var(--ml-brand);
    }
    .ml-comparison-table tbody tr:nth-child(even) td {
        background: var(--ml-row);
    }
    .ml-comparison-table tbody tr:last-child td { border-bottom: 0; }
    [data-testid="stButton"] button[kind="primary"] {
        border: 0; border-radius: 6px; font-weight: 700;
        background: var(--ml-accent); color: #fff; box-shadow: none;
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    }
    [data-testid="stDownloadButton"] button { border-radius: 6px; transition: transform .18s ease, box-shadow .18s ease; }
    [data-testid="stExpander"] { border-color: var(--ml-line); border-radius: 8px; background: var(--ml-surface); color: var(--ml-ink); }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--ml-line); border-radius: 8px; background: var(--ml-surface); transition: transform .18s ease, box-shadow .18s ease; }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div { background: var(--ml-surface); color: var(--ml-ink); border-color: var(--ml-line); }
    [data-baseweb="select"] input, [data-baseweb="input"] input, textarea { color: var(--ml-ink); }
    [data-testid="stFileUploader"] section { background: var(--ml-surface); border-color: var(--ml-line); color: var(--ml-ink); }
    [data-testid="stFileUploader"] button { background: var(--ml-soft); border-color: var(--ml-line); color: var(--ml-ink); }
    [data-testid="stDataFrame"] { background: var(--ml-surface); }
    .ml-topbar:hover, .ml-table-wrap:hover { border-color: var(--ml-accent); box-shadow: 0 7px 20px var(--ml-shadow); }
    .ml-pill:hover { transform: translateY(-2px); border-color: var(--ml-accent); box-shadow: 0 4px 10px var(--ml-shadow); }
    [data-testid="stMetric"]:hover, [data-testid="stVerticalBlockBorderWrapper"]:hover { transform: translateY(-2px); border-color: var(--ml-accent); box-shadow: 0 7px 16px var(--ml-shadow); }
    .ml-comparison-table tbody tr:hover td { background: var(--ml-soft); }
    [data-testid="stButton"] button[kind="primary"]:hover, [data-testid="stDownloadButton"] button:hover { transform: translateY(-1px); box-shadow: 0 5px 13px var(--ml-shadow); filter: brightness(1.04); }
    :focus-visible { outline: 3px solid var(--ml-accent) !important; outline-offset: 2px !important; }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; }
    }
    .stApp :not(code):not(pre):not(.material-icons):not(.material-symbols-rounded):not([data-testid="stIconMaterial"]) {
        font-family: Inter, "Segoe UI", ui-sans-serif, system-ui, sans-serif !important;
    }
    [data-testid="stIconMaterial"] {
        font-family:"Material Symbols Rounded";
        font-feature-settings:"liga";
    }
    .block-container { max-width: 1440px; padding-top: 1.15rem; }
    .ml-topbar {
        border-radius: 16px; padding: .8rem 1.2rem; margin-bottom: 1rem;
        box-shadow: 0 5px 18px var(--ml-shadow);
    }
    .ml-brand-lockup { display:flex; align-items:center; gap:.75rem; }
    .ml-brand-mark {
        display:grid; place-items:center; width:2.55rem; height:2.55rem;
        color:#fff; border-radius:12px;
        background:linear-gradient(145deg,#1267d6,#13b8a6);
    }
    .ml-brand-title { color:var(--ml-brand); font-weight:800; font-size:1.04rem; letter-spacing:-.02em; }
    .ml-brand-caption { color:var(--ml-muted); font-size:.78rem; margin-top:.08rem; }
    .ml-hero {
        display:flex; align-items:center; justify-content:space-between; gap:2rem;
        min-height:270px; padding:clamp(1.5rem,4vw,3rem);
        border:1px solid var(--ml-line); border-radius:22px;
        background:
          radial-gradient(ellipse at 86% 20%,rgba(35,183,187,.17),transparent 35%),
          linear-gradient(120deg,var(--ml-surface),var(--ml-soft));
        box-shadow:0 12px 35px var(--ml-shadow);
        border-left:0;
    }
    .stMarkdownContainer .ml-hero {
        background:
          radial-gradient(ellipse at 86% 20%,rgba(35,183,187,.17),transparent 35%),
          linear-gradient(120deg,var(--ml-surface),var(--ml-soft)) !important;
    }
    .ml-hero-copy { max-width:730px; }
    .ml-eyebrow {
        display:flex; align-items:center; gap:.55rem;
        color:var(--ml-accent)!important; font-size:.75rem; font-weight:750;
        letter-spacing:.12em; text-transform:uppercase;
    }
    .ml-hero h1 {
        max-width:720px; margin:.75rem 0 .65rem;
        color:var(--ml-brand)!important;
        font-size:clamp(2rem,4.7vw,3.8rem); line-height:1.05;
        letter-spacing:-.055em; font-weight:800;
    }
    .ml-hero-copy > p {
        max-width:620px; color:var(--ml-muted)!important;
        font-size:clamp(1rem,1.4vw,1.15rem); line-height:1.65;
    }
    .ml-hero-actions { display:flex; gap:.7rem; flex-wrap:wrap; margin-top:1.4rem; }
    .ml-hero-cta {
        display:inline-flex; align-items:center; justify-content:center; gap:.5rem;
        padding:.72rem 1rem; border:1px solid var(--ml-line); border-radius:10px;
        color:var(--ml-brand)!important; background:var(--ml-surface);
        font-size:.9rem; font-weight:700; text-decoration:none!important;
        transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    .ml-hero-cta.primary { color:#fff!important; border-color:#1267d6; background:#1267d6; }
    .ml-hero-cta:hover { transform:translateY(-2px); border-color:var(--ml-accent); box-shadow:0 8px 18px var(--ml-shadow); }
    .ml-hero-visual {
        flex:0 0 min(34%,350px); min-width:245px; padding:1.15rem;
        border:1px solid var(--ml-line); border-radius:18px;
        background:color-mix(in srgb,var(--ml-surface) 86%,transparent);
        box-shadow:0 10px 28px var(--ml-shadow);
    }
    .ml-visual-head { display:flex; align-items:center; justify-content:space-between; color:var(--ml-muted); font-size:.76rem; }
    .ml-visual-dot { width:.55rem; height:.55rem; border-radius:50%; background:#18a884; box-shadow:0 0 0 4px rgba(24,168,132,.13); }
    .ml-visual-bars { height:130px; display:flex; align-items:end; gap:.65rem; padding:.7rem .25rem .25rem; }
    .ml-visual-bars span { flex:1; min-height:20%; border-radius:7px 7px 3px 3px; background:linear-gradient(180deg,#2b8ef0,#19b6a5); animation:mb-rise .7s ease both; transform-origin:bottom; }
    .ml-visual-bars span:nth-child(2) { animation-delay:.08s; }
    .ml-visual-bars span:nth-child(3) { animation-delay:.16s; }
    .ml-visual-bars span:nth-child(4) { animation-delay:.24s; }
    .ml-visual-bars span:nth-child(5) { animation-delay:.32s; }
    @keyframes mb-rise { from { transform:scaleY(.25); opacity:.45; } to { transform:scaleY(1); opacity:1; } }
    .ml-visual-footer { display:flex; justify-content:space-between; padding-top:.7rem; border-top:1px solid var(--ml-line); color:var(--ml-muted); font-size:.73rem; }
    .ml-visual-footer strong { color:var(--ml-brand); }
    .ml-section-title {
        display:flex; align-items:center; gap:.65rem; margin:2.2rem 0 .35rem;
        color:var(--ml-brand); font-size:1.45rem; line-height:1.25;
        letter-spacing:-.035em; font-weight:780;
    }
    .ml-section-title svg { color:var(--ml-accent); flex:none; }
    .mb-section-copy { color:var(--ml-muted); margin-bottom:1.1rem; }
    .mb-dataset-card {
        min-height:231px; display:flex; flex-direction:column; gap:.65rem;
        margin:.15rem 0 .4rem; padding:1.15rem;
        background:var(--ml-surface); border:1px solid var(--ml-line);
        border-radius:15px; box-shadow:0 4px 14px var(--ml-shadow);
        transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease;
    }
    .mb-dataset-card:hover { transform:translateY(-3px); border-color:var(--ml-accent); box-shadow:0 12px 28px var(--ml-shadow); }
    .mb-dataset-card.is-selected { border-color:var(--ml-accent); box-shadow:0 0 0 2px color-mix(in srgb,var(--ml-accent) 18%,transparent); }
    .mb-dataset-card-top { display:flex; justify-content:space-between; align-items:center; }
    .mb-icon,.mb-model-icon {
        display:grid; place-items:center; width:2.1rem; height:2.1rem;
        color:var(--ml-accent); background:var(--ml-soft); border-radius:10px;
    }
    .mb-data-source,.mb-badges span {
        padding:.24rem .52rem; border:1px solid var(--ml-line); border-radius:999px;
        color:var(--ml-muted); background:var(--ml-soft); font-size:.68rem; font-weight:650;
    }
    .mb-data-source.synthetic { color:#8a5a00; background:#fff5d8; border-color:#f0d998; }
    .mb-dataset-card h3 { margin:.12rem 0 0; color:var(--ml-brand); font-size:1rem; font-weight:760; }
    .mb-dataset-card p { margin:0; color:var(--ml-muted); font-size:.82rem; line-height:1.45; }
    .mb-badges { display:flex; gap:.4rem; flex-wrap:wrap; }
    .mb-dataset-facts { display:flex; gap:1.1rem; margin-top:auto; color:var(--ml-muted); font-size:.76rem; }
    .mb-dataset-facts strong { color:var(--ml-brand); font-variant-numeric:tabular-nums; }
    .mb-dataset-note {
        display:flex; align-items:center; gap:.65rem; margin:1rem 0;
        padding:.7rem .9rem; color:var(--ml-muted); background:var(--ml-soft);
        border:1px solid var(--ml-line); border-radius:11px; font-size:.82rem;
    }
    .mb-dataset-note svg { color:var(--ml-accent); flex:none; }
    .mb-dot { padding:0 .4rem; color:var(--ml-line); }
    .mb-winner-banner {
        display:flex; align-items:center; gap:.75rem; margin:.9rem 0 1.1rem;
        padding:.85rem 1rem; border:1px solid #b9e8d8; border-radius:13px;
        color:var(--ml-ink); background:linear-gradient(100deg,rgba(24,168,132,.12),var(--ml-surface));
    }
    .mb-winner-icon { display:grid; place-items:center; width:2.2rem; height:2.2rem; color:#087a61; background:rgba(24,168,132,.15); border-radius:10px; }
    .mb-winner-banner strong { color:var(--ml-brand); }
    .mb-winner-banner span { color:var(--ml-muted); }
    .mb-model-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr)); gap:1rem; margin:1rem 0 1.6rem; }
    .mb-model-card {
        padding:1rem; border:1px solid var(--ml-line); border-radius:15px;
        background:var(--ml-surface); box-shadow:0 4px 14px var(--ml-shadow);
        animation:mb-card-in .4s ease both;
        transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;
    }
    .mb-model-card:hover { transform:translateY(-3px); border-color:var(--ml-accent); box-shadow:0 12px 27px var(--ml-shadow); }
    .mb-model-card.is-winner { border-color:#1aaa86; box-shadow:0 0 0 2px rgba(26,170,134,.13),0 6px 18px var(--ml-shadow); }
    @keyframes mb-card-in { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
    .mb-model-card-heading { display:flex; align-items:center; justify-content:space-between; gap:.5rem; margin-bottom:.8rem; }
    .mb-model-card-heading > div { display:flex; align-items:center; gap:.6rem; }
    .mb-model-card h3 { margin:0; color:var(--ml-brand); font-size:.98rem; font-weight:760; }
    .mb-rank { white-space:nowrap; padding:.27rem .55rem; color:var(--ml-muted); background:var(--ml-soft); border-radius:999px; font-size:.7rem; font-weight:700; }
    .mb-rank.rank-1 { color:var(--ml-gold-ink); background:var(--ml-gold-bg); border:1px solid var(--ml-gold-line); }
    .mb-rank.rank-2 { color:var(--ml-silver-ink); background:var(--ml-silver-bg); border:1px solid var(--ml-silver-line); }
    .mb-rank.rank-3 { color:var(--ml-bronze-ink); background:var(--ml-bronze-bg); border:1px solid var(--ml-bronze-line); }
    .mb-score-line { margin:.7rem 0; }
    .mb-score-line > div:first-child { display:flex; justify-content:space-between; gap:.5rem; color:var(--ml-muted); font-size:.76rem; }
    .mb-score-line strong { color:var(--ml-brand); font-variant-numeric:tabular-nums; }
    .mb-progress { height:6px; overflow:hidden; margin-top:.32rem; background:var(--ml-soft); border-radius:999px; }
    .mb-progress span { display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,#2386e8,#21b493); transition:width .7s cubic-bezier(.2,.8,.2,1); }
    .mb-model-foot { display:flex; justify-content:space-between; gap:.5rem; flex-wrap:wrap; padding-top:.65rem; border-top:1px solid var(--ml-line); color:var(--ml-muted); font-size:.7rem; }
    .mb-model-foot span { display:inline-flex; align-items:center; gap:.25rem; }
    .mb-model-foot strong { color:var(--ml-brand); }
    .mb-model-foot svg { color:var(--ml-accent); }
    *, *::before, *::after { box-sizing:border-box; }
    .block-container, main, main > div, main [data-testid="stVerticalBlock"] {
        min-width:0; max-width:100%;
    }
    .block-container { padding-top:4.25rem!important; }
    main [data-testid="stMarkdownContainer"] :is(h1,h2,h3,h4,p,li,td,th),
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"] {
        overflow-wrap:anywhere; word-break:normal;
    }
    .ml-topbar, .ml-brand-lockup, .ml-brand-lockup > div,
    .ml-owner, .ml-hero, .ml-hero-copy, .ml-hero-visual,
    .mb-dataset-card, .mb-dataset-card-top, .mb-model-card,
    .mb-model-card-heading, .mb-model-card-heading > div,
    .mb-model-foot, .mb-winner-banner, .mb-dataset-note {
        min-width:0;
    }
    .ml-brand-title, .ml-owner, .ml-hero h1, .ml-hero p,
    .ml-section-title, .mb-section-copy {
        max-width:100%; overflow-wrap:anywhere; word-break:normal;
    }
    .ml-brand-mark, .ml-brand-mark svg, .ml-section-title svg,
    .mb-icon, .mb-icon svg, .mb-model-icon, .mb-model-icon svg {
        flex:none;
    }
    .ml-brand-title { line-height:1.25; }
    .ml-owner { line-height:1.5; }
    .ml-hero { flex-wrap:wrap; }
    .ml-hero-copy { flex:1 1 20rem; }
    .ml-hero h1 { line-height:1.12; }
    .ml-hero-visual { flex:0 1 min(34%,350px); }
    .ml-visual-head, .ml-visual-footer { gap:.6rem; flex-wrap:wrap; }
    .ml-visual-footer strong { overflow-wrap:anywhere; }
    .mb-dataset-card {
        min-height:296px; height:auto; overflow-wrap:anywhere;
    }
    .mb-dataset-card-top { flex-wrap:wrap; gap:.5rem; }
    .mb-data-source {
        max-width:100%; white-space:normal; overflow-wrap:anywhere;
    }
    .mb-dataset-card h3 {
        min-height:2.7em; line-height:1.35; overflow-wrap:anywhere;
    }
    .mb-dataset-card p {
        min-height:3.6em; overflow-wrap:anywhere;
    }
    .mb-badges, .mb-dataset-facts { flex-wrap:wrap; }
    .mb-dataset-facts { gap:.35rem .85rem; }
    .mb-dataset-note { align-items:flex-start; overflow-wrap:anywhere; }
    .mb-winner-banner { align-items:flex-start; overflow-wrap:anywhere; }
    .mb-winner-icon { flex:none; }
    .mb-model-card-heading { align-items:flex-start; flex-wrap:wrap; }
    .mb-model-card-heading > div { flex:1 1 10rem; }
    .mb-model-card h3 {
        min-width:0; overflow-wrap:anywhere; line-height:1.35;
    }
    .mb-rank {
        max-width:100%; white-space:normal; text-align:center; overflow-wrap:anywhere;
    }
    .mb-score-line > div:first-child { align-items:baseline; }
    .mb-score-line strong, .mb-model-foot strong { overflow-wrap:anywhere; }
    .mb-model-foot { justify-content:flex-start; gap:.45rem 1rem; }
    .mb-model-foot span { min-width:0; flex-wrap:wrap; overflow-wrap:anywhere; }
    [data-testid="stMetric"] { min-width:0; height:100%; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        white-space:normal; overflow-wrap:anywhere; line-height:1.25;
    }
    [data-testid="stMetricValue"] { font-size:clamp(1.05rem,3vw,1.8rem); }
    [data-baseweb="select"], [data-baseweb="input"], [data-baseweb="textarea"] {
        width:100%; min-width:0; max-width:100%;
    }
    [data-baseweb="select"] > div, [data-baseweb="select"] [role="combobox"] {
        min-width:0; max-width:100%;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stPlotlyChart"],
    [data-testid="stVegaLiteChart"], [data-testid="stImage"], [data-testid="stImage"] img,
    [data-testid="stPyplot"], canvas, iframe {
        max-width:100%; min-width:0;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] { overflow:auto; }
    .ml-table-wrap { max-width:100%; }
    .ml-comparison-table { table-layout:fixed; }
    .ml-comparison-table th, .ml-comparison-table td {
        white-space:normal; overflow-wrap:anywhere;
    }
    .ml-report-wrap { overflow:hidden; }
    .ml-report-table {
        width:100%; min-width:0!important; table-layout:fixed;
    }
    .ml-report-table th, .ml-report-table td {
        padding:.5rem .35rem; font-size:.76rem; overflow-wrap:anywhere;
    }
    .ml-report-table th:first-child, .ml-report-table td:first-child {
        width:32%; text-align:left;
    }
    .ml-report-table th:not(:first-child), .ml-report-table td:not(:first-child) {
        width:17%;
    }
    [data-testid="stFileUploader"] section, [data-testid="stFileUploader"] small {
        max-width:100%; overflow-wrap:anywhere;
    }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
        max-width:100%; white-space:normal; overflow-wrap:anywhere;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {
        min-width:0; overflow-wrap:anywhere;
    }
    [data-testid="stButton"] button {
        min-height:2.55rem; border-radius:9px!important; font-weight:680;
        transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    [data-testid="stDownloadButton"] button {
        min-height:2.55rem; border-radius:9px!important; font-weight:680;
    }
    [data-testid="stFileUploader"] section {
        min-height:105px; padding:.8rem; border:1px dashed var(--ml-line);
        border-radius:12px; background:var(--ml-surface);
    }
    [data-testid="stDataFrame"] { border:1px solid var(--ml-line); border-radius:12px; }
    [data-testid="stExpander"] { border-radius:12px; }
    .stAlert { border-radius:12px; }
    @media (max-width:900px) {
        .ml-hero { gap:1rem; }
        .ml-hero-visual { flex-basis:30%; min-width:210px; }
    }
    @media (max-width:680px) {
        .block-container { padding:4rem 1rem 2rem!important; }
        .ml-topbar { padding:.75rem .85rem; align-items:flex-start; flex-direction:column; }
        .ml-brand-lockup { width:100%; align-items:flex-start; }
        .ml-brand-lockup > div { flex:1; }
        .ml-brand-caption { display:none; }
        .ml-owner { text-align:left; font-size:.72rem!important; }
        .ml-hero { min-height:0; padding:1.35rem; }
        .ml-hero-visual { display:none; }
        .ml-hero h1 { font-size:clamp(1.8rem,9vw,2.25rem); }
        .mb-dataset-card { min-height:0; }
        .mb-dataset-card h3, .mb-dataset-card p { min-height:0; }
        .mb-dataset-note { align-items:flex-start; }
        .mb-model-grid { grid-template-columns:1fr; }
        .mb-model-foot { gap:.45rem .8rem; }
        .mb-model-card-heading { align-items:flex-start; }
        .mb-model-card h3 { font-size:.9rem; }
        .mb-dataset-facts { gap:.35rem .7rem; }
        .mb-dataset-facts span { overflow-wrap:anywhere; }
        .ml-section-title { font-size:clamp(1.15rem,6vw,1.45rem); }
        [data-testid="stMetric"] { padding:.7rem; }
        [data-testid="stMetricLabel"] { font-size:.74rem; }
        [data-testid="stMetricValue"] { font-size:1.2rem; }
        [data-testid="stDataFrame"], [data-testid="stTable"] { width:100%; }
        .ml-report-table th, .ml-report-table td { padding:.38rem .2rem; font-size:.68rem; }
    }
    @media (min-width:681px) and (max-width:1024px) {
        .ml-hero { gap:1.25rem; padding:2rem; }
        .ml-hero-visual { flex-basis:28%; min-width:190px; }
        .mb-dataset-card { min-height:320px; }
        .mb-dataset-card h3, .mb-dataset-card p { min-height:0; }
    }
    @media (max-width:420px) {
        .block-container { padding-right:.75rem!important; padding-left:.75rem!important; }
        .ml-hero { padding:1.1rem; border-radius:16px; }
        .ml-hero-actions, .ml-hero-cta { width:100%; }
        .mb-model-card { padding:.85rem; }
        .mb-model-card-heading { gap:.65rem; }
        .mb-rank { font-size:.66rem; }
        .mb-dataset-card { padding:1rem; }
    }
    [data-testid="stHeader"], [data-testid="stToolbar"] { display:none!important; }
    .block-container { padding-top:1.1rem!important; }
    html, body, html body .stApp,
    html body [data-testid="stAppViewContainer"],
    html body [data-testid="stMain"] {
        background-color:var(--ml-page)!important;
        color:var(--ml-ink)!important;
    }
    .eval-header { background-color:var(--ml-surface)!important; }
    .eval-wordmark, .eval-hero h1 { color:var(--ml-brand)!important; }
    .eval-tagline, .eval-card-note, .eval-hero p { color:var(--ml-muted)!important; }
    .stApp [role="tab"] { color:var(--ml-muted)!important; }
    .stApp [role="tab"][aria-selected="true"] {
        color:var(--ml-brand)!important;
    }
    [data-testid="stMetric"] {
        background-color:var(--ml-surface)!important;
        border-color:var(--ml-line)!important;
    }
    @media (max-width:680px) {
        .block-container { padding:1rem 1rem 2rem!important; }
    }
    </style>
    <header class="eval-header">
      <div class="eval-lockup">
        <span class="eval-logo" aria-hidden="true">
          <svg width="35" height="35" viewBox="0 0 48 48" fill="none">
            <ellipse cx="24" cy="24" rx="18" ry="7.5" stroke="white" stroke-width="1.7"/>
            <ellipse cx="24" cy="24" rx="18" ry="7.5" transform="rotate(60 24 24)" stroke="white" stroke-width="1.7"/>
            <ellipse cx="24" cy="24" rx="18" ry="7.5" transform="rotate(120 24 24)" stroke="white" stroke-width="1.7"/>
            <circle cx="24" cy="24" r="3.2" fill="white"/>
            <circle cx="41" cy="24" r="2" fill="#A5F3FC"/>
            <circle cx="15.5" cy="9.3" r="2" fill="#C4B5FD"/>
            <circle cx="15.5" cy="38.7" r="2" fill="#BFDBFE"/>
          </svg>
        </span>
        <div><div class="eval-wordmark">EVALSPHERE</div>
        <div class="eval-tagline">__PROJECT_TAGLINE__</div></div>
      </div>
      <div class="eval-card-note">Model evaluation · benchmark analysis · dataset discovery</div>
    </header>
    <section class="eval-hero">
      <div>
        <div class="eval-kicker">Classification analytics workspace</div>
        <h1>Benchmark Smarter.<br>Decide Faster.</h1>
        <p>Discover, compare, and understand model performance in one calm, focused workspace. Start with a sample dataset or bring a labeled CSV.</p>
      </div>
      <svg class="eval-orbit" viewBox="0 0 120 120" fill="none" aria-hidden="true">
        <circle cx="60" cy="60" r="8" fill="#2563EB"/>
        <ellipse cx="60" cy="60" rx="48" ry="19" stroke="#312E81" stroke-width="3"/>
        <ellipse cx="60" cy="60" rx="48" ry="19" transform="rotate(60 60 60)" stroke="#2563EB" stroke-width="3"/>
        <ellipse cx="60" cy="60" rx="48" ry="19" transform="rotate(120 60 60)" stroke="#7C3AED" stroke-width="3"/>
        <circle cx="105" cy="60" r="5" fill="#06B6D4"/>
        <circle cx="37.5" cy="21" r="5" fill="#7C3AED"/>
        <circle cx="37.5" cy="99" r="5" fill="#2563EB"/>
      </svg>
    </section>
    """
    .replace("__ML_THEME_TOKENS__", theme_tokens)
    .replace("__PROJECT_NAME__", html.escape(PROJECT_NAME))
    .replace("__PROJECT_TAGLINE__", html.escape(PROJECT_TAGLINE))
    .replace("__PROJECT_DESCRIPTION__", html.escape(PROJECT_DESCRIPTION))
    .replace("__ICON_CHART__", lucide_icon("chart", 21))
    .replace("__ICON_SPARKLES__", lucide_icon("sparkles", 16))
    .replace("__ICON_UPLOAD__", lucide_icon("upload", 16))
    .replace("__ICON_DATABASE__", lucide_icon("database", 16)),
    unsafe_allow_html=True,
)
__, profile_column = st.columns([6, 1])
with profile_column:
    with st.popover("Profile settings"):
        st.markdown("### Workspace profile")
        st.toggle("Dark appearance", key="evalsphere_dark_mode")
        st.divider()
        st.markdown("**SRIRAM POLAMARASETTI**")
        st.caption("BITS ID · 2026ak05608")
        st.caption("Preferences are stored for this browser session.")

render_evalsphere_workspace()
