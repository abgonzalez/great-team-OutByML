"""Script para construir una muestra pequena del dataset historico."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_builder import build_dataset, get_smart_cities


CLIMATE_COLUMNS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "uv_index",
    "activity_score",
    "comfort_distance",
]

TOP_COLUMNS = [
    "time",
    "season_block",
    "city",
    "activity",
    "recommendation",
    "activity_score",
    "temperature_2m",
    "precipitation_probability",
    "wind_speed_10m",
    "relative_humidity_2m",
    "uv_index",
]


def main():
    date_ranges = [
        ("enero", "2025-01-01", "2025-01-07"),
        ("abril", "2025-04-01", "2025-04-07"),
        ("julio", "2025-07-01", "2025-07-07"),
        ("octubre", "2025-10-01", "2025-10-07"),
    ]
    output_path = PROJECT_ROOT / "data" / "processed" / "outbyml_sample_dataset.csv"

    cities = get_smart_cities()
    datasets = []

    for season_block, start_date, end_date in date_ranges:
        print(f"\nConstruyendo bloque estacional: {season_block}")
        block_dataset = build_dataset(cities, start_date, end_date)
        if not block_dataset.empty:
            block_dataset["season_block"] = season_block
            datasets.append(block_dataset)

    if datasets:
        dataset = pd.concat(datasets, ignore_index=True)
    else:
        dataset = pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print_dataset_diagnostics(dataset)
    print(f"\nArchivo guardado en: {output_path}")


def print_dataset_diagnostics(dataset):
    """Imprime una revision diagnostica del dataset generado."""
    print("\nDataset historico de muestra creado")
    print(f"Filas totales: {len(dataset)}")
    print("\nColumnas generadas:")
    print(list(dataset.columns))

    if dataset.empty:
        print("\nEl dataset esta vacio. Revisa la descarga antes de analizar calidad.")
        return

    print("\nNulos por columna:")
    print(dataset.isna().sum())

    if "recommendation" in dataset.columns:
        recommendation_counts = dataset["recommendation"].value_counts()
        recommendation_percent = (
            dataset["recommendation"].value_counts(normalize=True) * 100
        ).round(2)
        recommendation_distribution = recommendation_counts.to_frame("conteo")
        recommendation_distribution["porcentaje"] = recommendation_percent

        print("\nDistribucion global de recommendation:")
        print(recommendation_distribution)

        if "activity" in dataset.columns:
            print("\nDistribucion de recommendation por activity:")
            print(
                dataset.pivot_table(
                    index="activity",
                    columns="recommendation",
                    values="time",
                    aggfunc="count",
                    fill_value=0,
                )
            )

        if "city" in dataset.columns:
            print("\nDistribucion de recommendation por city:")
            print(
                dataset.pivot_table(
                    index="city",
                    columns="recommendation",
                    values="time",
                    aggfunc="count",
                    fill_value=0,
                )
            )

        if "climate_group" in dataset.columns:
            print("\nDistribucion de recommendation por climate_group:")
            print(
                dataset.pivot_table(
                    index="climate_group",
                    columns="recommendation",
                    values="time",
                    aggfunc="count",
                    fill_value=0,
                )
            )

        if "season_block" in dataset.columns:
            print("\nDistribucion de recommendation por season_block:")
            print(
                dataset.pivot_table(
                    index="season_block",
                    columns="recommendation",
                    values="time",
                    aggfunc="count",
                    fill_value=0,
                )
            )

    if "activity_score" in dataset.columns:
        print("\nEstadisticas de activity_score:")
        print(
            dataset["activity_score"]
            .agg(["min", "max", "mean", "median"])
            .round(2)
        )

    available_climate_columns = [
        column for column in CLIMATE_COLUMNS if column in dataset.columns
    ]
    if "recommendation" in dataset.columns and available_climate_columns:
        print("\nPromedios climaticos por recommendation:")
        print(
            dataset.groupby("recommendation")[available_climate_columns]
            .mean()
            .round(2)
        )

    available_top_columns = [column for column in TOP_COLUMNS if column in dataset.columns]
    if "activity_score" in dataset.columns and available_top_columns:
        print("\nTop 10 filas con menor activity_score:")
        print(
            dataset.sort_values("activity_score", ascending=True)
            .head(10)[available_top_columns]
            .to_string(index=False)
        )

        print("\nTop 10 filas con mayor activity_score:")
        print(
            dataset.sort_values("activity_score", ascending=False)
            .head(10)[available_top_columns]
            .to_string(index=False)
        )

    if "recommendation" in dataset.columns:
        low_representation = (
            dataset["recommendation"].value_counts(normalize=True) * 100
        ) < 5
        if low_representation.any():
            print(
                "\nIMPORTANTE: Hay clases con baja representacion. "
                "Revisar balance antes de entrenar el modelo."
            )

    if "activity_score" in dataset.columns:
        print_threshold_sensitivity(dataset)


def classify_score_with_thresholds(score, thresholds):
    """Clasifica un score usando umbrales alternativos de diagnostico."""
    if score >= thresholds["excelente"]:
        return "excelente"
    if score >= thresholds["bueno"]:
        return "bueno"
    if score >= thresholds["regular"]:
        return "regular"
    return "malo"


def print_threshold_sensitivity(dataset):
    """Imprime como cambia la distribucion al variar umbrales."""
    threshold_configs = {
        "actual_80_60_40": {"excelente": 80, "bueno": 60, "regular": 40},
        "ajuste_85_65_45": {"excelente": 85, "bueno": 65, "regular": 45},
        "ajuste_85_70_50": {"excelente": 85, "bueno": 70, "regular": 50},
        "ajuste_90_70_50": {"excelente": 90, "bueno": 70, "regular": 50},
    }

    print("\nPrueba de sensibilidad de umbrales")
    for config_name, thresholds in threshold_configs.items():
        labels = dataset["activity_score"].apply(
            lambda score: classify_score_with_thresholds(score, thresholds)
        )
        counts = labels.value_counts()
        percentages = (labels.value_counts(normalize=True) * 100).round(2)
        distribution = counts.to_frame("conteo")
        distribution["porcentaje"] = percentages

        print(f"\nConfiguracion: {config_name}")
        print(f"Umbrales: {thresholds}")
        print(distribution)


if __name__ == "__main__":
    main()
