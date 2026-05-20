"""Funciones para crear features y puntuaciones climaticas de SalimosHoy?."""

import math

# Parametros climaticos por actividad: horarios validos, valores ideales
# y pesos que determinan cuanto penaliza cada variable.
ACTIVITY_SETTINGS = {
    "Pasear o hacer senderismo": {
        "start_hour": 7,
        "end_hour": 21,
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
    "Jardineria y agricultura": {
        "start_hour": 7,
        "end_hour": 19,
        "ideal_temperature": 24,
        "ideal_humidity": 55,
        "ideal_wind": 8,
        "ideal_uv": 4,
        "rain_weight": 1.3,
        "wind_weight": 1.0,
        "humidity_weight": 1.2,
        "uv_weight": 1.3,
        "temperature_weight": 1.2,
    },
    "Deportes al aire libre": {
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
    "Picnic o actividades en parque": {
        "start_hour": 9,
        "end_hour": 20,
        "ideal_temperature": 23,
        "ideal_humidity": 50,
        "ideal_wind": 6,
        "ideal_uv": 4,
        "rain_weight": 1.8,
        "wind_weight": 1.2,
        "humidity_weight": 1.0,
        "uv_weight": 1.0,
        "temperature_weight": 1.0,
    },
    "Ir al cine": {
        "start_hour": 10,
        "end_hour": 23,
        "ideal_temperature": 22,
        "ideal_humidity": 50,
        "ideal_wind": 10,
        "ideal_uv": 1,
        "rain_weight": 0.3,
        "wind_weight": 0.3,
        "humidity_weight": 0.3,
        "uv_weight": 0.3,
        "temperature_weight": 0.3,
    },
    "Ir a la playa": {
        "start_hour": 8,
        "end_hour": 20,
        "ideal_temperature": 28,
        "ideal_humidity": 55,
        "ideal_wind": 8,
        "ideal_uv": 5,
        "rain_weight": 2.0,
        "wind_weight": 1.5,
        "humidity_weight": 0.8,
        "uv_weight": 1.5,
        "temperature_weight": 1.2,
    },
}


def get_activity_settings(activity):
    """Devuelve la configuracion climatica para una actividad."""
    return ACTIVITY_SETTINGS.get(activity, ACTIVITY_SETTINGS["Pasear o hacer senderismo"])


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

    # La puntuacion parte de 100 y resta penalizaciones por alejarse del ideal.
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
    # Se usa como criterio secundario para desempatar horas con score parecido.
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
    # Esta etiqueta se usa como fallback si el modelo ML no esta disponible.
    if score >= 85:
        return "excelente"
    if score >= 65:
        return "bueno"
    if score >= 45:
        return "regular"
    return "malo"


def apply_weather_safety_rules(row, activity, recommendation):
    """Limita la recomendacion final ante lluvia, viento o temperatura extrema."""
    # Estas reglas convierten la salida inicial en una recomendacion final realista.
    rain_probability = safe_value(row.get("precipitation_probability"), 0)
    wind_speed = safe_value(row.get("wind_speed_10m"), 0)
    temperature = safe_value(row.get("temperature_2m"), 20)

    outdoor_activities = [
        "Pasear o hacer senderismo",
        "Deportes al aire libre",
        "Picnic o actividades en parque",
        "Ir a la playa",
        "Jardineria y agricultura",
    ]
    wind_sensitive_activities = [
        "Deportes al aire libre",
        "Ir a la playa",
        "Picnic o actividades en parque",
    ]

    if activity == "Ir al cine":
        # Actividad interior: el clima penaliza menos, salvo condiciones extremas.
        if rain_probability >= 95 and recommendation in ["excelente", "bueno"]:
            return "regular"
        if rain_probability >= 80 and recommendation == "excelente":
            return "bueno"
        if wind_speed >= 45 and recommendation == "excelente":
            return "bueno"
        if temperature <= -3 and recommendation == "excelente":
            return "bueno"
        if temperature >= 42 and recommendation == "excelente":
            return "bueno"
        return recommendation
    elif activity in outdoor_activities:
        # La lluvia alta domina sobre el modelo para actividades al aire libre.
        if rain_probability >= 70:
            return "malo"
        if rain_probability >= 40 and recommendation in ["excelente", "bueno"]:
            recommendation = "regular"
        elif rain_probability >= 20 and recommendation == "excelente":
            recommendation = "bueno"

    if activity in wind_sensitive_activities:
        # Viento fuerte afecta especialmente deportes, playa y picnic.
        if wind_speed >= 40:
            return "malo"
        if wind_speed >= 25 and recommendation in ["excelente", "bueno"]:
            recommendation = "regular"

    if activity in outdoor_activities:
        # Temperaturas extremas se consideran inseguras para planes exteriores.
        if temperature >= 38 or temperature <= 0:
            return "malo"

    if activity == "Deportes al aire libre" and temperature >= 32:
        if recommendation in ["excelente", "bueno"]:
            recommendation = "regular"

    if activity == "Ir a la playa":
        if temperature < 20 and recommendation in ["excelente", "bueno"]:
            recommendation = "regular"

    return recommendation


def get_valid_hours(df, start_hour, end_hour):
    """Filtra un DataFrame para conservar horas dentro de un rango."""
    return df[(df["hour"] >= start_hour) & (df["hour"] <= end_hour)].copy()
