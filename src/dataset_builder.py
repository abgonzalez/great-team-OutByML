"""Constructor educativo de dataset historico para OutByML."""

import pandas as pd
import requests

from src.features import (
    apply_weather_safety_rules,
    calculate_activity_score,
    calculate_comfort_distance,
    classify_score,
    get_activity_settings,
    get_valid_hours,
)


HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "uv_index",
]

ARCHIVE_FALLBACK_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "shortwave_radiation",
]

ACTIVITIES = ["Pasear", "Turismo", "Deporte", "Bici", "Lavar ropa"]

SAMPLE_CITIES = [
    {
        "city": "Madrid",
        "country": "España",
        "continent": "Europa",
        "latitude": 40.4165,
        "longitude": -3.70256,
        "timezone": "Europe/Madrid",
    },
    {
        "city": "Bilbao",
        "country": "España",
        "continent": "Europa",
        "latitude": 43.26271,
        "longitude": -2.92528,
        "timezone": "Europe/Madrid",
    },
    {
        "city": "London",
        "country": "Reino Unido",
        "continent": "Europa",
        "latitude": 51.50853,
        "longitude": -0.12574,
        "timezone": "Europe/London",
    },
    {
        "city": "New York",
        "country": "Estados Unidos",
        "continent": "America",
        "latitude": 40.71427,
        "longitude": -74.00597,
        "timezone": "America/New_York",
    },
    {
        "city": "Buenos Aires",
        "country": "Argentina",
        "continent": "America",
        "latitude": -34.61315,
        "longitude": -58.37723,
        "timezone": "America/Argentina/Buenos_Aires",
    },
    {
        "city": "Caracas",
        "country": "Venezuela",
        "continent": "America",
        "latitude": 10.48801,
        "longitude": -66.87919,
        "timezone": "America/Caracas",
    },
    {
        "city": "Tokyo",
        "country": "Japon",
        "continent": "Asia",
        "latitude": 35.6895,
        "longitude": 139.69171,
        "timezone": "Asia/Tokyo",
    },
    {
        "city": "Dubai",
        "country": "Emiratos Arabes Unidos",
        "continent": "Asia",
        "latitude": 25.07725,
        "longitude": 55.30927,
        "timezone": "Asia/Dubai",
    },
    {
        "city": "Cairo",
        "country": "Egipto",
        "continent": "Africa",
        "latitude": 30.06263,
        "longitude": 31.24967,
        "timezone": "Africa/Cairo",
    },
    {
        "city": "Sydney",
        "country": "Australia",
        "continent": "Oceania",
        "latitude": -33.86785,
        "longitude": 151.20732,
        "timezone": "Australia/Sydney",
    },
]

SMART_CITY_GROUPS = {
    "templado_urbano": [
        ("Madrid", "España", "Europa", 40.4165, -3.70256, "Europe/Madrid"),
        ("Barcelona", "España", "Europa", 41.38879, 2.15899, "Europe/Madrid"),
        ("Paris", "Francia", "Europa", 48.85341, 2.3488, "Europe/Paris"),
        ("London", "Reino Unido", "Europa", 51.50853, -0.12574, "Europe/London"),
        ("Rome", "Italia", "Europa", 41.89193, 12.51133, "Europe/Rome"),
        ("Lisbon", "Portugal", "Europa", 38.71667, -9.13333, "Europe/Lisbon"),
        ("Berlin", "Alemania", "Europa", 52.52437, 13.41053, "Europe/Berlin"),
        ("Amsterdam", "Paises Bajos", "Europa", 52.37403, 4.88969, "Europe/Amsterdam"),
        ("Vienna", "Austria", "Europa", 48.20849, 16.37208, "Europe/Vienna"),
        ("Prague", "Republica Checa", "Europa", 50.08804, 14.42076, "Europe/Prague"),
    ],
    "calor_seco": [
        ("Dubai", "Emiratos Arabes Unidos", "Asia", 25.07725, 55.30927, "Asia/Dubai"),
        ("Riyadh", "Arabia Saudi", "Asia", 24.68773, 46.72185, "Asia/Riyadh"),
        ("Doha", "Qatar", "Asia", 25.28545, 51.53096, "Asia/Qatar"),
        ("Cairo", "Egipto", "Africa", 30.06263, 31.24967, "Africa/Cairo"),
        ("Marrakech", "Marruecos", "Africa", 31.63416, -7.99994, "Africa/Casablanca"),
        ("Phoenix", "Estados Unidos", "America", 33.44838, -112.07404, "America/Phoenix"),
        ("Las Vegas", "Estados Unidos", "America", 36.17497, -115.13722, "America/Los_Angeles"),
        ("Seville", "España", "Europa", 37.38283, -5.97317, "Europe/Madrid"),
        ("Baghdad", "Irak", "Asia", 33.34058, 44.40088, "Asia/Baghdad"),
        ("Kuwait City", "Kuwait", "Asia", 29.36972, 47.97833, "Asia/Kuwait"),
    ],
    "frio_fuerte": [
        ("Reykjavik", "Islandia", "Europa", 64.13548, -21.89541, "Atlantic/Reykjavik"),
        ("Oslo", "Noruega", "Europa", 59.91273, 10.74609, "Europe/Oslo"),
        ("Helsinki", "Finlandia", "Europa", 60.16952, 24.93545, "Europe/Helsinki"),
        ("Stockholm", "Suecia", "Europa", 59.33258, 18.0649, "Europe/Stockholm"),
        ("Moscow", "Rusia", "Europa", 55.75222, 37.61556, "Europe/Moscow"),
        ("Montreal", "Canada", "America", 45.50884, -73.58781, "America/Toronto"),
        ("Toronto", "Canada", "America", 43.70011, -79.4163, "America/Toronto"),
        ("Ulaanbaatar", "Mongolia", "Asia", 47.90771, 106.88324, "Asia/Ulaanbaatar"),
        ("Anchorage", "Estados Unidos", "America", 61.21806, -149.90028, "America/Anchorage"),
        ("Nuuk", "Groenlandia", "America", 64.18347, -51.72157, "America/Godthab"),
    ],
    "lluvia_humedad": [
        ("Bilbao", "España", "Europa", 43.26271, -2.92528, "Europe/Madrid"),
        ("Dublin", "Irlanda", "Europa", 53.33306, -6.24889, "Europe/Dublin"),
        ("Glasgow", "Reino Unido", "Europa", 55.86515, -4.25763, "Europe/London"),
        ("Vancouver", "Canada", "America", 49.24966, -123.11934, "America/Vancouver"),
        ("Bogota", "Colombia", "America", 4.60971, -74.08175, "America/Bogota"),
        ("Quito", "Ecuador", "America", -0.22985, -78.52495, "America/Guayaquil"),
        ("Singapore", "Singapur", "Asia", 1.28967, 103.85007, "Asia/Singapore"),
        ("Kuala Lumpur", "Malasia", "Asia", 3.1412, 101.68653, "Asia/Kuala_Lumpur"),
        ("Bangkok", "Tailandia", "Asia", 13.75398, 100.50144, "Asia/Bangkok"),
        ("Mumbai", "India", "Asia", 19.07283, 72.88261, "Asia/Kolkata"),
    ],
    "tropical_humedo": [
        ("Caracas", "Venezuela", "America", 10.48801, -66.87919, "America/Caracas"),
        ("Panama City", "Panama", "America", 8.9936, -79.51973, "America/Panama"),
        ("Havana", "Cuba", "America", 23.13302, -82.38304, "America/Havana"),
        ("Santo Domingo", "Republica Dominicana", "America", 18.47186, -69.89232, "America/Santo_Domingo"),
        ("San Juan", "Puerto Rico", "America", 18.46633, -66.10572, "America/Puerto_Rico"),
        ("Manaus", "Brasil", "America", -3.10194, -60.025, "America/Manaus"),
        ("Lagos", "Nigeria", "Africa", 6.45407, 3.39467, "Africa/Lagos"),
        ("Nairobi", "Kenia", "Africa", -1.28333, 36.81667, "Africa/Nairobi"),
        ("Jakarta", "Indonesia", "Asia", -6.21462, 106.84513, "Asia/Jakarta"),
        ("Manila", "Filipinas", "Asia", 14.6042, 120.9822, "Asia/Manila"),
    ],
    "hemisferio_sur": [
        ("Buenos Aires", "Argentina", "America", -34.61315, -58.37723, "America/Argentina/Buenos_Aires"),
        ("Santiago", "Chile", "America", -33.45694, -70.64827, "America/Santiago"),
        ("Lima", "Peru", "America", -12.04318, -77.02824, "America/Lima"),
        ("Sao Paulo", "Brasil", "America", -23.5475, -46.63611, "America/Sao_Paulo"),
        ("Cape Town", "Sudafrica", "Africa", -33.92584, 18.42322, "Africa/Johannesburg"),
        ("Johannesburg", "Sudafrica", "Africa", -26.20227, 28.04363, "Africa/Johannesburg"),
        ("Sydney", "Australia", "Oceania", -33.86785, 151.20732, "Australia/Sydney"),
        ("Melbourne", "Australia", "Oceania", -37.814, 144.96332, "Australia/Melbourne"),
        ("Auckland", "Nueva Zelanda", "Oceania", -36.86667, 174.76667, "Pacific/Auckland"),
        ("Perth", "Australia", "Oceania", -31.95224, 115.8614, "Australia/Perth"),
    ],
}

SMART_CLIMATE_GROUP_ORDER = [
    "templado_urbano",
    "calor_seco",
    "frio_fuerte",
    "lluvia_humedad",
    "tropical_humedo",
    "hemisferio_sur",
]


def get_sample_cities():
    """Devuelve una lista inicial de ciudades de prueba."""
    return SAMPLE_CITIES.copy()


def get_smart_cities():
    """Devuelve ciudades variadas por grupo climatico para construir datasets."""
    cities = []
    max_group_size = max(len(group) for group in SMART_CITY_GROUPS.values())

    for index in range(max_group_size):
        for climate_group in SMART_CLIMATE_GROUP_ORDER:
            group_cities = SMART_CITY_GROUPS[climate_group]
            if index < len(group_cities):
                city, country, continent, latitude, longitude, timezone = group_cities[index]
                cities.append(
                    {
                        "city": city,
                        "country": country,
                        "continent": continent,
                        "climate_group": climate_group,
                        "latitude": latitude,
                        "longitude": longitude,
                        "timezone": timezone,
                    }
                )

    return cities


def get_sample_smart_cities(n=24):
    """Devuelve una muestra pequena balanceada por grupo climatico."""
    all_cities = get_smart_cities()
    cities_by_group = {
        climate_group: [
            city for city in all_cities if city["climate_group"] == climate_group
        ]
        for climate_group in SMART_CLIMATE_GROUP_ORDER
    }
    base_amount = n // len(SMART_CLIMATE_GROUP_ORDER)
    extra_amount = n % len(SMART_CLIMATE_GROUP_ORDER)

    selected_cities = []
    for index, climate_group in enumerate(SMART_CLIMATE_GROUP_ORDER):
        group_limit = base_amount
        if index < extra_amount:
            group_limit += 1
        selected_cities.extend(cities_by_group[climate_group][:group_limit])

    return selected_cities[:n]


def get_historical_weather(latitude, longitude, timezone, start_date, end_date):
    """Obtiene clima historico horario desde Open-Meteo."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": timezone,
    }

    try:
        response = requests.get(HISTORICAL_WEATHER_URL, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        print(
            "No se pudo descargar el historico con todas las variables. "
            "Se intentara una consulta compatible con Historical Weather API."
        )
    except ValueError:
        print(
            "No se pudo leer el historico con todas las variables. "
            "Se intentara una consulta compatible con Historical Weather API."
        )

    fallback_params = params.copy()
    fallback_params["hourly"] = ",".join(ARCHIVE_FALLBACK_VARIABLES)

    try:
        response = requests.get(HISTORICAL_WEATHER_URL, params=fallback_params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        print("Error al consultar el clima historico. Revisa la conexion o intenta mas tarde.")
        return None
    except ValueError:
        print("Error al leer la respuesta del clima historico.")
        return None


def build_weather_dataframe(weather_data, city_info):
    """Convierte la respuesta historica en un DataFrame horario con metadatos."""
    if weather_data is None or "hourly" not in weather_data:
        return pd.DataFrame()

    try:
        df = pd.DataFrame(weather_data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df["hour"] = df["time"].dt.hour

        if "precipitation_probability" not in df.columns and "precipitation" in df.columns:
            df["precipitation_probability"] = df["precipitation"].apply(
                _precipitation_to_probability
            )
        if "uv_index" not in df.columns and "shortwave_radiation" in df.columns:
            df["uv_index"] = (df["shortwave_radiation"] / 100).clip(lower=0, upper=10)

        df["city"] = city_info["city"]
        df["country"] = city_info["country"]
        df["continent"] = city_info["continent"]
        df["climate_group"] = city_info.get("climate_group", "sin_grupo")
        df["latitude"] = city_info["latitude"]
        df["longitude"] = city_info["longitude"]
        df["timezone"] = city_info["timezone"]

        return clean_weather_dataframe(df)
    except Exception:
        print("No se pudieron procesar los datos historicos.")
        return pd.DataFrame()


def clean_weather_dataframe(df):
    """Limpia valores faltantes del clima horario historico."""
    df_clean = df.copy()
    numeric_columns = [
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "precipitation_probability",
        "wind_speed_10m",
        "wind_gusts_10m",
        "cloud_cover",
        "uv_index",
    ]

    for column in numeric_columns:
        if column not in df_clean.columns:
            df_clean[column] = pd.NA
        df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")

    df_clean["temperature_2m"] = (
        df_clean["temperature_2m"].ffill().bfill().fillna(20)
    )
    df_clean["apparent_temperature"] = df_clean["apparent_temperature"].fillna(
        df_clean["temperature_2m"]
    )
    df_clean["relative_humidity_2m"] = df_clean["relative_humidity_2m"].fillna(
        df_clean["relative_humidity_2m"].median()
    ).fillna(50)
    df_clean["precipitation_probability"] = df_clean[
        "precipitation_probability"
    ].fillna(0)
    df_clean["wind_speed_10m"] = df_clean["wind_speed_10m"].fillna(0)
    df_clean["wind_gusts_10m"] = df_clean["wind_gusts_10m"].fillna(
        df_clean["wind_speed_10m"]
    )
    df_clean["cloud_cover"] = df_clean["cloud_cover"].fillna(
        df_clean["cloud_cover"].median()
    ).fillna(50)
    df_clean["uv_index"] = df_clean["uv_index"].fillna(0)

    return df_clean


def expand_by_activity(df_weather):
    """Crea una fila por actividad y conserva solo horas validas."""
    if df_weather.empty:
        return pd.DataFrame()

    activity_frames = []

    for activity in ACTIVITIES:
        settings = get_activity_settings(activity)
        df_activity = get_valid_hours(
            df_weather,
            settings["start_hour"],
            settings["end_hour"],
        )

        if df_activity.empty:
            continue

        df_activity = df_activity.copy()
        df_activity["activity"] = activity
        df_activity["activity_score"] = df_activity.apply(
            lambda row: calculate_activity_score(row, settings),
            axis=1,
        )
        df_activity["comfort_distance"] = df_activity.apply(
            lambda row: calculate_comfort_distance(row, settings),
            axis=1,
        )
        df_activity["recommendation"] = df_activity["activity_score"].apply(classify_score)
        df_activity["recommendation"] = df_activity.apply(
            lambda row: apply_weather_safety_rules(
                row,
                activity,
                row["recommendation"],
            ),
            axis=1,
        )
        activity_frames.append(df_activity)

    if not activity_frames:
        return pd.DataFrame()

    return pd.concat(activity_frames, ignore_index=True)


def build_dataset(cities, start_date, end_date):
    """Construye un dataset historico pequeno para varias ciudades."""
    dataset_frames = []

    for city_info in cities:
        print(f"Procesando {city_info['city']}, {city_info['country']}...")
        weather_data = get_historical_weather(
            city_info["latitude"],
            city_info["longitude"],
            city_info["timezone"],
            start_date,
            end_date,
        )
        df_weather = build_weather_dataframe(weather_data, city_info)
        df_activity = expand_by_activity(df_weather)

        if not df_activity.empty:
            dataset_frames.append(df_activity)

    if not dataset_frames:
        return pd.DataFrame()

    return pd.concat(dataset_frames, ignore_index=True)


def _precipitation_to_probability(precipitation):
    """Crea una probabilidad simple a partir de milimetros historicos."""
    if pd.isna(precipitation) or precipitation <= 0:
        return 0
    if precipitation <= 0.5:
        return 35
    if precipitation <= 2:
        return 70
    return 90
