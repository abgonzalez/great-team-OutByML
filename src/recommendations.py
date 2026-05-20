"""Funciones para recomendar horarios segun condiciones climaticas."""


def sort_recommendations(df):
    """Ordena por mejor score y luego por menor distancia de confort."""
    # El score decide la prioridad; comfort_distance desempata entre horas cercanas.
    return df.sort_values(
        ["activity_score", "comfort_distance"],
        ascending=[False, True],
    )


def get_best_hour(df_valid_hours):
    """Devuelve la mejor fila segun activity_score y comfort_distance."""
    if df_valid_hours.empty:
        return None
    # La primera fila del ordenamiento es la recomendacion principal.
    return sort_recommendations(df_valid_hours).iloc[0]


def get_top_hours(df_valid_hours, top_n=5):
    """Devuelve las mejores horas segun score y distancia de confort."""
    if df_valid_hours.empty:
        return df_valid_hours
    # Top 5 mantiene el mismo criterio que la mejor hora.
    return sort_recommendations(df_valid_hours).head(top_n)


def generate_recommendation_text(best_row, activity, city_name):
    """Genera una explicacion breve para la recomendacion principal."""
    if best_row is None:
        return "No hay horas disponibles para generar una recomendacion."

    hour = best_row.get("hour", "hora no disponible")
    score = best_row.get("activity_score", 0)
    label = best_row.get("recommendation", "sin clasificar")
    temperature = best_row.get("temperature_2m", None)
    rain = best_row.get("precipitation_probability", None)
    wind = best_row.get("wind_speed_10m", None)

    base_text = (
        f"La mejor hora para {activity} en {city_name} es a las {int(hour):02d}:00. "
        f"Las condiciones son {label} y el score es {score:.1f}/100."
    )

    # Si falta alguna variable climatica, se conserva el texto base.
    if temperature is None or rain is None or wind is None:
        return base_text

    return (
        f"{base_text} A esa hora se esperan {temperature:.1f} grados, "
        f"{rain:.0f}% de probabilidad de lluvia y viento de {wind:.1f} km/h."
    )
