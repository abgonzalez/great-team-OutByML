"""Funciones para crear features y puntuaciones climaticas de OutByML."""

import math

ACTIVITY_SETTINGS = {
    "Pasear": {
        "start_hour": 7,
        "end_hour": 22,
        "ideal_temperature": 22,
        "ideal_humidity": 50,
        "ideal_wind": 5,
        "ideal_uv": 4,
        "rain_weight": 1.0,
        "wind_weight": 1.0,
        "humidity_weight": 1.0,
        "uv_weight": 1.0,
        "temperature_weight": 1.0,
    },
    "Turismo": {
        "start_hour": 8,
        "end_hour": 20,
        "ideal_temperature": 21,
        "ideal_humidity": 50,
        "ideal_wind": 6,
        "ideal_uv": 4,
        "rain_weight": 1.2,
        "wind_weight": 1.0,
        "humidity_weight": 1.0,
        "uv_weight": 1.0,
        "temperature_weight": 1.2,
    },
    "Deporte": {
        "start_hour": 6,
        "end_hour": 21,
        "ideal_temperature": 18,
        "ideal_humidity": 45,
        "ideal_wind": 5,
        "ideal_uv": 3,
        "rain_weight": 1.0,
        "wind_weight": 1.0,
        "humidity_weight": 1.4,
        "uv_weight": 1.4,
        "temperature_weight": 1.5,
    },
    "Bici": {
        "start_hour": 7,
        "end_hour": 21,
        "ideal_temperature": 20,
        "ideal_humidity": 50,
        "ideal_wind": 4,
        "ideal_uv": 4,
        "rain_weight": 1.5,
        "wind_weight": 1.5,
        "humidity_weight": 1.0,
        "uv_weight": 1.0,
        "temperature_weight": 1.0,
    },
    "Lavar ropa": {
        "start_hour": 9,
        "end_hour": 18,
        "ideal_temperature": 24,
        "ideal_humidity": 35,
        "ideal_wind": 10,
        "ideal_uv": 5,
        "rain_weight": 2.0,
        "wind_weight": 1.2,
        "humidity_weight": 1.8,
        "uv_weight": 1.0,
        "temperature_weight": 1.0,
    },
}


def get_activity_settings(activity):
    """Devuelve la configuracion climatica para una actividad."""
    return ACTIVITY_SETTINGS.get(activity, ACTIVITY_SETTINGS["Pasear"])


def safe_value(value, default):
    """Devuelve un valor numerico seguro cuando llega None o NaN."""
    if value is None:
        return default
    try:
        if math.isnan(value):
            return default
    except TypeError:
        pass
    return value


def calculate_activity_score(row, settings):
    """Calcula una puntuacion de 0 a 100 segun clima y actividad."""
    score = 100

    temperature = safe_value(row["temperature_2m"], 20)
    rain_probability = safe_value(row["precipitation_probability"], 0)
    wind_speed = safe_value(row["wind_speed_10m"], 0)
    humidity = safe_value(row["relative_humidity_2m"], 50)
    uv_index = safe_value(row["uv_index"], 0)

    temperature_distance = abs(temperature - settings["ideal_temperature"])
    humidity_distance = abs(humidity - settings["ideal_humidity"])
    uv_distance = abs(uv_index - settings["ideal_uv"])

    if temperature_distance >= 14:
        score -= 30 * settings["temperature_weight"]
    elif temperature_distance >= 8:
        score -= 15 * settings["temperature_weight"]
    elif temperature_distance >= 5:
        score -= 6 * settings["temperature_weight"]

    if rain_probability >= 70:
        score -= 35 * settings["rain_weight"]
    elif rain_probability >= 40:
        score -= 20 * settings["rain_weight"]
    elif rain_probability >= 20:
        score -= 10 * settings["rain_weight"]

    if wind_speed >= 40:
        score -= 25 * settings["wind_weight"]
    elif wind_speed >= 25:
        score -= 12 * settings["wind_weight"]

    if humidity_distance >= 35:
        score -= 10 * settings["humidity_weight"]
    elif humidity_distance >= 25:
        score -= 5 * settings["humidity_weight"]

    if uv_distance >= 5:
        score -= 10 * settings["uv_weight"]
    elif uv_distance >= 3:
        score -= 5 * settings["uv_weight"]

    return max(round(score, 2), 0)


def calculate_comfort_distance(row, settings):
    """Calcula distancia respecto a condiciones ideales. Menor es mejor."""
    temperature = safe_value(row["temperature_2m"], 20)
    rain_probability = safe_value(row["precipitation_probability"], 0)
    wind_speed = safe_value(row["wind_speed_10m"], 0)
    humidity = safe_value(row["relative_humidity_2m"], 50)
    uv_index = safe_value(row["uv_index"], 0)

    temperature_distance = (
        abs(temperature - settings["ideal_temperature"])
        * settings["temperature_weight"]
    )
    humidity_distance = (
        abs(humidity - settings["ideal_humidity"])
        / 10
        * settings["humidity_weight"]
    )
    wind_distance = (
        abs(wind_speed - settings["ideal_wind"])
        / 5
        * settings["wind_weight"]
    )
    uv_distance = (
        abs(uv_index - settings["ideal_uv"])
        / 2
        * settings["uv_weight"]
    )
    rain_distance = (
        rain_probability
        / 20
        * settings["rain_weight"]
    )

    return (
        temperature_distance
        + humidity_distance
        + wind_distance
        + uv_distance
        + rain_distance
    )


def classify_score(score):
    """Clasifica la puntuacion en una etiqueta sencilla."""
    if score >= 85:
        return "excelente"
    if score >= 65:
        return "bueno"
    if score >= 45:
        return "regular"
    return "malo"


def apply_weather_safety_rules(row, activity, recommendation):
    """Limita la recomendacion final ante lluvia, viento o temperatura extrema."""
    rain_probability = safe_value(row.get("precipitation_probability"), 0)
    wind_speed = safe_value(row.get("wind_speed_10m"), 0)
    temperature = safe_value(row.get("temperature_2m"), 20)

    outdoor_activities = ["Pasear", "Turismo", "Deporte", "Bici"]
    wind_sensitive_activities = ["Bici", "Deporte", "Lavar ropa"]

    if activity == "Lavar ropa":
        if rain_probability >= 60:
            return "malo"
        if rain_probability >= 40:
            return "malo"
        if rain_probability >= 20 and recommendation in ["excelente", "bueno"]:
            recommendation = "regular"
    elif activity in outdoor_activities:
        if rain_probability >= 70:
            return "malo"
        if rain_probability >= 40 and recommendation in ["excelente", "bueno"]:
            recommendation = "regular"
        elif rain_probability >= 20 and recommendation == "excelente":
            recommendation = "bueno"

    if activity in wind_sensitive_activities:
        if wind_speed >= 40:
            return "malo"
        if wind_speed >= 25 and recommendation in ["excelente", "bueno"]:
            recommendation = "regular"

    if activity in outdoor_activities:
        if temperature >= 38 or temperature <= 0:
            return "malo"

    if activity == "Deporte" and temperature >= 32:
        if recommendation in ["excelente", "bueno"]:
            recommendation = "regular"

    return recommendation


def get_valid_hours(df, start_hour, end_hour):
    """Filtra un DataFrame para conservar horas dentro de un rango."""
    return df[(df["hour"] >= start_hour) & (df["hour"] <= end_hour)].copy()
