"""Funciones para consultar ciudades con Open-Meteo Geocoding API."""

import requests

from src.config import DEFAULT_CITY_COUNT, DEFAULT_LANGUAGE, GEOCODING_URL


def search_city(city_name, count=DEFAULT_CITY_COUNT):
    """Busca una ciudad por nombre y devuelve las coincidencias."""
    if not city_name:
        return []

    params = {
        "name": city_name,
        "count": count,
        "language": DEFAULT_LANGUAGE,
        "format": "json",
    }

    try:
        response = requests.get(GEOCODING_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except requests.RequestException:
        print("Error al buscar la ciudad. Revisa tu conexion o intenta mas tarde.")
        return []
    except ValueError:
        print("Error al leer la respuesta de geocoding.")
        return []
