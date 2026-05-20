"""Configuracion inicial del proyecto SalimosHoy?."""

# Parametros compartidos por las llamadas a Open-Meteo.
DEFAULT_LANGUAGE = "es"
DEFAULT_FORECAST_DAYS = 1
DEFAULT_CITY_COUNT = 5

# Endpoints oficiales usados por los modulos de geocoding y forecast.
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
