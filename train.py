#!/usr/bin/env python

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "credit_risk_dataset.csv"
MODEL_PATH = BASE_DIR / "credit_risk_rf_model.joblib"
TARGET_COLUMN = "loan_status"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25
THRESHOLD = 0.45

DROPPED_COLUMNS = [
    TARGET_COLUMN,
    "person_age",
    "cb_person_cred_hist_length",
]
IMPUTED_COLUMNS = ["person_emp_length", "loan_int_rate"]


def load_data() -> pd.DataFrame:
    """Load the dataset and remove duplicate records."""
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates().reset_index(drop=True)
    return data


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified 60/20/20 train, validation, and test splits."""
    df_full_train, df_test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=1,
        stratify=data[TARGET_COLUMN],
    )
    df_train, df_val = train_test_split(
        df_full_train,
        test_size=VALIDATION_SIZE,
        random_state=1,
        stratify=df_full_train[TARGET_COLUMN],
    )

    return tuple(
        frame.reset_index(drop=True)
        for frame in (df_train, df_val, df_test)
    )


def prepare_features(
    data: pd.DataFrame,
    imputer: SimpleImputer | None = None,
    scaler: StandardScaler | None = None,
    vectorizer: DictVectorizer | None = None,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler, DictVectorizer, list[str]]:
    """Fit preprocessing on training data or apply fitted preprocessing to other splits."""
    features = data.drop(columns=DROPPED_COLUMNS).copy()
    target = data[TARGET_COLUMN].to_numpy()

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        features[IMPUTED_COLUMNS] = imputer.fit_transform(features[IMPUTED_COLUMNS])
    else:
        features[IMPUTED_COLUMNS] = imputer.transform(features[IMPUTED_COLUMNS])

    features["interest_burden"] = (
        features["loan_amnt"] * (features["loan_int_rate"] / 100)
    )
    features["income_buffer"] = (
        features["person_income"] - features["loan_amnt"]
    )

    numerical_columns = features.select_dtypes(include="number").columns.tolist()
    numerical_data = features[numerical_columns]

    if scaler is None:
        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(numerical_data)
    else:
        scaled_numerical = scaler.transform(numerical_data)

    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    categorical_records = features[categorical_columns].to_dict(orient="records")

    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=False)
        encoded_categorical = vectorizer.fit_transform(categorical_records)
    else:
        encoded_categorical = vectorizer.transform(categorical_records)

    matrix = np.hstack((scaled_numerical, encoded_categorical))
    return (
        matrix,
        target,
        imputer,
        scaler,
        vectorizer,
        numerical_columns,
    )


def benchmark_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> pd.DataFrame:
    """Compare baseline classifiers using validation metrics."""
    models = {
        "Dummy Classifier": DummyClassifier(),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree (d=6)": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        train_probability = model.predict_proba(x_train)[:, 1]
        val_probability = model.predict_proba(x_val)[:, 1]
        val_prediction = model.predict(x_val)

        results.append(
            {
                "Model": name,
                "Train AUC": round(
                    roc_auc_score(y_train, train_probability), 4
                ),
                "Val AUC": round(
                    roc_auc_score(y_val, val_probability), 4
                ),
                "Val PR-AUC": round(
                    average_precision_score(y_val, val_probability), 4
                ),
                "Accuracy": round(
                    accuracy_score(y_val, val_prediction), 4
                ),
                "Recall (1)": round(
                    recall_score(y_val, val_prediction), 4
                ),
                "F1 (1)": round(
                    f1_score(y_val, val_prediction), 4
                ),
            }
        )

    return pd.DataFrame(results).sort_values("Val AUC", ascending=False)


def tune_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomizedSearchCV:
    """Tune a random forest with stratified cross-validation."""
    parameter_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 14, 18, 24, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample", None],
    }
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=parameter_grid,
        n_jobs=-2,
        scoring="roc_auc",
        cv=5,
        random_state=2,
        n_iter=25,
    )
    search.fit(x_train, y_train)
    return search


def print_threshold_metrics(
    y_val: np.ndarray,
    val_probability: np.ndarray,
) -> None:
    """Print recall, precision, and F1 for candidate thresholds."""
    for threshold in [0.50, 0.45, 0.40, 0.35]:
        prediction = (val_probability >= threshold).astype(int)
        print(
            f"Threshold {threshold:.2f} | "
            f"Recall: {recall_score(y_val, prediction):.4f} | "
            f"Precision: {precision_score(y_val, prediction, zero_division=0):.4f} | "
            f"F1: {f1_score(y_val, prediction, zero_division=0):.4f}"
        )


def evaluate_and_save(
    imputer: SimpleImputer,
    scaler: StandardScaler,
    vectorizer: DictVectorizer,
    numerical_columns: list[str],
    x_full_train: np.ndarray,
    y_full_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    search: RandomizedSearchCV,
) -> RandomForestClassifier:
    """Train on the full training set, evaluate once, and save the model bundle."""
    final_model = RandomForestClassifier(
        **search.best_params_,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(x_full_train, y_full_train)

    test_probability = final_model.predict_proba(x_test)[:, 1]
    test_prediction = (test_probability >= THRESHOLD).astype(int)

    print(
        f"Final Test ROC-AUC: "
        f"{roc_auc_score(y_test, test_probability):.4f}"
    )
    print(
        f"Final Test PR-AUC: "
        f"{average_precision_score(y_test, test_probability):.4f}"
    )
    print()
    print(classification_report(y_test, test_prediction, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_prediction))

    model_bundle = {
        "imputer": imputer,
        "scaler": scaler,
        "vectorizer": vectorizer,
        "numerical_columns": numerical_columns,
        "model": final_model,
        "threshold": THRESHOLD,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"\nModel bundle saved to {MODEL_PATH}")

    return final_model


def main() -> None:
    data = load_data()
    print(
        f"Loaded {len(data)} rows after removing duplicates "
        f"from {DATA_PATH.name}"
    )

    df_train, df_val, df_test = split_data(data)
    df_full_train = pd.concat(
        [df_train, df_val],
        ignore_index=True,
    )

    (
        x_train,
        y_train,
        imputer,
        scaler,
        vectorizer,
        numerical_columns,
    ) = prepare_features(df_train)
    x_val, y_val, _, _, _, _ = prepare_features(
        df_val,
        imputer,
        scaler,
        vectorizer,
    )

    results = benchmark_models(x_train, y_train, x_val, y_val)
    print("\nModel comparison:")
    print(results.to_string(index=False))

    search = tune_random_forest(x_train, y_train)
    print(f"\nBest parameters: {search.best_params_}")

    best_model = search.best_estimator_
    train_probability = best_model.predict_proba(x_train)[:, 1]
    val_probability = best_model.predict_proba(x_val)[:, 1]
    print(
        f"\nTuned Train AUC: "
        f"{roc_auc_score(y_train, train_probability):.4f} | "
        f"Tuned Val AUC: "
        f"{roc_auc_score(y_val, val_probability):.4f}"
    )
    print_threshold_metrics(y_val, val_probability)

    (
        x_full_train,
        y_full_train,
        imputer,
        scaler,
        vectorizer,
        numerical_columns,
    ) = prepare_features(df_full_train)
    x_test, y_test, _, _, _, _ = prepare_features(
        df_test,
        imputer,
        scaler,
        vectorizer,
    )

    evaluate_and_save(
        imputer,
        scaler,
        vectorizer,
        numerical_columns,
        x_full_train,
        y_full_train,
        x_test,
        y_test,
        search,
    )


if __name__ == "__main__":
    main()
