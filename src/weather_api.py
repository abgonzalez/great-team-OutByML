"""Funciones para consultar clima con Open-Meteo Forecast API."""

import pandas as pd
import requests

from src.config import DEFAULT_FORECAST_DAYS, FORECAST_URL


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


def get_weather(latitude, longitude, timezone):
    """Obtiene datos horarios de clima para una ubicacion."""
    # Forecast API entrega el clima actualizado que luego clasifica el modelo.
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "forecast_days": DEFAULT_FORECAST_DAYS,
        "hourly": ",".join(HOURLY_VARIABLES),
    }

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        # Si falla la API, la app recibe None y puede mostrar un error controlado.
        print("Error al consultar el clima. Revisa tu conexion o intenta mas tarde.")
        return None
    except ValueError:
        print("Error al leer la respuesta del clima.")
        return None


def build_weather_dataframe(weather_data):
    """Convierte la respuesta de Open-Meteo en un DataFrame horario."""
    if weather_data is None:
        return pd.DataFrame()

    try:
        if "hourly" not in weather_data:
            return pd.DataFrame()

        # Mantiene una fila por hora con las variables usadas por la app.
        df = pd.DataFrame(weather_data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df["hour"] = df["time"].dt.hour
        return df
    except Exception:
        print("No se pudieron procesar los datos climaticos.")
        return pd.DataFrame()
