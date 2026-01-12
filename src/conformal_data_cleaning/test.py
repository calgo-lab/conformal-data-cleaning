# Full Evaluation (DROP-IN REPLACEMENT)

import glob
import os
import json

import numpy as np
import pandas as pd

from tab_err.api.high_level import create_errors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from conformal_data_cleaning.cleaner.autogluon import ConformalAutoGluonCleaner


# -------------------------------
# Load datasets
# -------------------------------
csv_files = glob.glob(os.path.join("./data", "*.csv"))
dataframes = [pd.read_csv(f) for f in csv_files]

for d in dataframes:
    print(d["target"].head())


# -------------------------------
# Target + Model helpers
# -------------------------------

def detect_task(y: pd.Series) -> str:
    """Return 'regression' or 'classification'."""
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() > 20:
            return "regression"
        else:
            return "classification"
    else:
        return "classification"


def get_model(task: str):
    return RandomForestRegressor() if task == "regression" else RandomForestClassifier()


def encode_target(y: pd.Series, task: str):
    """Encode target for classification, leave numeric for regression."""
    if task == "classification":
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        return y_enc, le
    else:
        return y.astype(float), None


# -------------------------------
# Build encoded RF model
# -------------------------------
def build_encoded_model(task: str, X: pd.DataFrame):
    """Return a preprocessing + RF pipeline."""

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = get_model(task)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


# -------------------------------
# Evaluation
# -------------------------------
def evaluate_model(model, X_test, y_test, task):
    preds = model.predict(X_test)

    if task == "classification":
        # compute F1 (macro by default; you can change to 'micro' or 'weighted' if needed)
        f1 = f1_score(y_test, preds, average='macro')
        return f1  # return "error" for optimization purposes
    else:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return rmse


def process_dataset(df, cleaner_cls):
    """Run the 3-step pipeline on a single dataframe."""

    df = df.dropna(subset=["target"]).copy()

    # --- 1. Train/validation split (clean baseline) ---
    clean_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

    X_train = clean_df.drop(columns=["target"])
    y_train_raw = clean_df["target"]

    # Detect task
    task = detect_task(y_train_raw)

    # Encode target
    y_train, y_encoder = encode_target(y_train_raw, task)

    # Build encoded RF model
    model = build_encoded_model(task, X_train)
    model.fit(X_train, y_train)

    # Prepare test data
    X_test = test_df.drop(columns=["target"])
    y_test_raw = test_df["target"]

    # Encode test target if classification
    if task == "classification":
        y_test = y_encoder.transform(y_test_raw)
    else:
        y_test = y_test_raw.astype(float)

    # -------------------------
    # A) Baseline model
    # -------------------------
    baseline_err = evaluate_model(model, X_test, y_test, task)

    # -------------------------
    # B) Error features model
    # -------------------------
    X_test_errored, _ = create_errors(X_test, error_rate=0.25)
    error_features_err = evaluate_model(model, X_test_errored, y_test, task)

    # -------------------------
    # C) Cleaned model
    # -------------------------
    model_hps = {
        "hyperparameters": {
            'RF': {}
    }}
    cleaner = cleaner_cls(confidence_level=0.999, seed=42)
    fit_cleaner = cleaner.fit(X_train, ci_ag_fit_params=model_hps)  # Use subset of models with fast parameters via a parameter: ci_ag_predictor_params
    X_test_cleaned, _ = fit_cleaner.transform(X_test_errored)

    cleaned_err = evaluate_model(model, X_test_cleaned, y_test, task)

    return {
        "baseline_f1": baseline_err,
        "error_features_f1": error_features_err,
        "cleaned_f1": cleaned_err,
        "task": task,
    }


def evaluate_all(dfs, cleaner_cls):
    results = []
    for i, df in enumerate(dfs):
        print(f"Processing dataset {i+1}/{len(dfs)} ...")
        res = process_dataset(df, cleaner_cls)
        results.append(res)
        with open(f"{i}.json", "w") as f:
            json.dump(res, f, indent=2)
    return pd.DataFrame(results)


# -------------------------------
# Execute all
# -------------------------------
results = evaluate_all(dataframes, ConformalAutoGluonCleaner)
print(results)
