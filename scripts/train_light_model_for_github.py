"""Train compressed lightweight RandomForest candidates for GitHub.

This script does not replace the production model. It writes experimental
artifacts under models/light/ and evaluation reports under reports/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
SAMPLE_SIZE = 600_000
TEST_SIZE = 0.20
COMPRESSION_LEVEL = 3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "salimoshoy_full_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "light"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_CSV_PATH = REPORTS_DIR / "light_model_github_results.csv"
REPORT_TXT_PATH = REPORTS_DIR / "light_model_github_report.txt"

NUMERIC_FEATURES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "uv_index",
    "hour",
]

CATEGORICAL_FEATURES = [
    "activity",
    "continent",
    "climate_group",
    "season_block",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "recommendation"

EXCLUDED_FEATURES = {
    "activity_score",
    "comfort_distance",
    "recommendation",
    "time",
    "city",
    "country",
    "latitude",
    "longitude",
    "timezone",
}

MODEL_SPECS = [
    {
        "name": "RF_100_depth16",
        "filename": "salimoshoy_rf_100_depth16_compressed.pkl",
        "params": {"n_estimators": 100, "max_depth": 16},
    },
    {
        "name": "RF_80_depth14",
        "filename": "salimoshoy_rf_80_depth14_compressed.pkl",
        "params": {"n_estimators": 80, "max_depth": 14},
    },
    {
        "name": "RF_60_depth14",
        "filename": "salimoshoy_rf_60_depth14_compressed.pkl",
        "params": {"n_estimators": 60, "max_depth": 14},
    },
    {
        "name": "RF_50_depth12",
        "filename": "salimoshoy_rf_50_depth12_compressed.pkl",
        "params": {"n_estimators": 50, "max_depth": 12},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train lightweight compressed RandomForest candidates."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Dataset CSV path. Default: {DATASET_PATH}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help=f"Maximum stratified training sample size. Default: {SAMPLE_SIZE}",
    )
    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="desconocido")),
            ("encoder", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_model(n_estimators: int, max_depth: int) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", classifier),
        ]
    )


def validate_dataset(df: pd.DataFrame) -> None:
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "El dataset no contiene las columnas obligatorias: "
            + ", ".join(missing_columns)
        )

    accidental_features = sorted(set(FEATURE_COLUMNS).intersection(EXCLUDED_FEATURES))
    if accidental_features:
        raise ValueError(
            "Hay columnas excluidas configuradas como features: "
            + ", ".join(accidental_features)
        )


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No existe el dataset esperado: {dataset_path}\n"
            "Coloca salimoshoy_full_dataset.csv en data/processed/ o usa --dataset."
        )

    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = pd.read_csv(dataset_path, usecols=columns)
    validate_dataset(df)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str)
    return df


def make_stratified_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("--sample-size debe ser mayor que 0.")
    if len(df) <= sample_size:
        return df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    _, sample = train_test_split(
        df,
        test_size=sample_size,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    return sample.reset_index(drop=True)


def get_recall(report: dict, class_name: str) -> float:
    value = report.get(class_name, {}).get("recall")
    return float(value) if value is not None else 0.0


def measure_load_time(model_path: Path) -> tuple[Pipeline, float]:
    start = time.perf_counter()
    loaded_model = joblib.load(model_path)
    elapsed = time.perf_counter() - start
    return loaded_model, elapsed


def measure_prediction_time(model: Pipeline, X_test: pd.DataFrame) -> float:
    prediction_sample = X_test.head(1_000)
    start = time.perf_counter()
    model.predict(prediction_sample)
    return time.perf_counter() - start


def evaluate_model(
    name: str,
    model: Pipeline,
    model_path: Path,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, train_predictions)
    accuracy_test = accuracy_score(y_test, test_predictions)
    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )
    labels = sorted(y_test.astype(str).unique())
    matrix = confusion_matrix(y_test, test_predictions, labels=labels)

    joblib.dump(model, model_path, compress=COMPRESSION_LEVEL)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    loaded_model, load_time_seconds = measure_load_time(model_path)
    prediction_time_1000_seconds = measure_prediction_time(loaded_model, X_test)

    return {
        "model": name,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "size_mb": round(model_size_mb, 2),
        "accuracy_train": round(accuracy_train, 6),
        "accuracy_test": round(accuracy_test, 6),
        "difference_train_test": round(accuracy_train - accuracy_test, 6),
        "recall_malo": round(get_recall(report, "malo"), 6),
        "recall_regular": round(get_recall(report, "regular"), 6),
        "recall_bueno": round(get_recall(report, "bueno"), 6),
        "recall_excelente": round(get_recall(report, "excelente"), 6),
        "load_time_seconds": round(load_time_seconds, 6),
        "prediction_time_1000_seconds": round(prediction_time_1000_seconds, 6),
        "github_candidate": (
            model_size_mb <= 100
            and accuracy_test >= 0.97
            and get_recall(report, "malo") >= 0.95
            and get_recall(report, "regular") >= 0.95
        ),
        "classification_report_json": json.dumps(report, ensure_ascii=False),
        "confusion_matrix_labels": json.dumps(labels, ensure_ascii=False),
        "confusion_matrix_json": json.dumps(matrix.tolist(), ensure_ascii=False),
    }


def select_best_candidate(results: list[dict]) -> dict:
    valid_candidates = [row for row in results if row["github_candidate"]]
    if valid_candidates:
        return sorted(
            valid_candidates,
            key=lambda row: (-row["accuracy_test"], row["size_mb"]),
        )[0]

    return sorted(
        results,
        key=lambda row: (
            -min(row["recall_malo"], row["recall_regular"]),
            -row["accuracy_test"],
            row["size_mb"],
        ),
    )[0]


def write_text_report(
    results: list[dict],
    best: dict,
    dataset_path: Path,
    sample_rows: int,
    train_rows: int,
    test_rows: int,
) -> None:
    candidates_under_100 = [row for row in results if row["size_mb"] <= 100]
    github_candidates = [row for row in results if row["github_candidate"]]

    lines = [
        "SalimosHoy - prueba de modelos ligeros para GitHub",
        "=" * 55,
        "",
        f"Dataset: {dataset_path}",
        f"Muestra estratificada: {sample_rows:,} filas",
        f"Train/Test split: {train_rows:,} train / {test_rows:,} test",
        f"Random state: {RANDOM_STATE}",
        f"Compresion joblib: compress={COMPRESSION_LEVEL}",
        "",
        "Tabla comparativa",
        "-" * 55,
    ]

    for row in results:
        lines.extend(
            [
                f"Modelo: {row['model']}",
                f"  Archivo: {row['model_path']}",
                f"  Peso MB: {row['size_mb']}",
                f"  Accuracy train: {row['accuracy_train']}",
                f"  Accuracy test: {row['accuracy_test']}",
                f"  Diferencia train-test: {row['difference_train_test']}",
                (
                    "  Recall por clase: "
                    f"malo={row['recall_malo']}, "
                    f"regular={row['recall_regular']}, "
                    f"bueno={row['recall_bueno']}, "
                    f"excelente={row['recall_excelente']}"
                ),
                f"  Tiempo carga s: {row['load_time_seconds']}",
                f"  Tiempo prediccion 1000 filas s: {row['prediction_time_1000_seconds']}",
                f"  Candidato GitHub: {row['github_candidate']}",
                "",
            ]
        )

    lines.extend(
        [
            "Seleccion",
            "-" * 55,
            f"Mejor candidato: {best['model']}",
            f"Peso: {best['size_mb']} MB",
            f"Accuracy test: {best['accuracy_test']}",
            (
                "Recall critico: "
                f"malo={best['recall_malo']}, regular={best['recall_regular']}"
            ),
            f"Modelos por debajo de 100 MB: {len(candidates_under_100)}",
            f"Modelos que cumplen todos los criterios: {len(github_candidates)}",
            "",
        ]
    )

    if github_candidates:
        lines.append(
            "Recomendacion final: usar el mejor candidato GitHub solo si se decide "
            "integrarlo en una tarea separada. Este script no modifica app.py."
        )
    else:
        lines.append(
            "Recomendacion final: ningun modelo cumple todos los criterios. "
            "El modelo seleccionado es el mejor balance entre peso, accuracy_test "
            "y recall de las clases malo/regular; revisar el CSV antes de integrar "
            "cualquier artefacto."
        )

    REPORT_TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cargando dataset: {dataset_path}")
    df = load_dataset(dataset_path)
    sample = make_stratified_sample(df, args.sample_size)

    X = sample[FEATURE_COLUMNS].copy()
    y = sample[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []
    for spec in MODEL_SPECS:
        name = spec["name"]
        model_path = MODELS_DIR / spec["filename"]
        print(f"\nEntrenando {name}...")
        start = time.perf_counter()
        model = build_model(**spec["params"])
        model.fit(X_train, y_train)
        train_time_seconds = time.perf_counter() - start

        row = evaluate_model(
            name,
            model,
            model_path,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        row["train_time_seconds"] = round(train_time_seconds, 6)
        results.append(row)

        print(
            f"{name}: size={row['size_mb']} MB, "
            f"accuracy_test={row['accuracy_test']}, "
            f"recall_malo={row['recall_malo']}, "
            f"recall_regular={row['recall_regular']}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    best = select_best_candidate(results)
    write_text_report(
        results,
        best,
        dataset_path,
        sample_rows=len(sample),
        train_rows=len(X_train),
        test_rows=len(X_test),
    )

    print("\nResumen guardado en:")
    print(f"- {RESULTS_CSV_PATH}")
    print(f"- {REPORT_TXT_PATH}")
    print(f"\nMejor candidato: {best['model']}")
    print(f"Peso: {best['size_mb']} MB")
    print(f"Accuracy test: {best['accuracy_test']}")
    print(f"Recall malo: {best['recall_malo']}")
    print(f"Recall regular: {best['recall_regular']}")
    print(f"Candidato GitHub: {best['github_candidate']}")


if __name__ == "__main__":
    main()
