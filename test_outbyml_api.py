import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import requests


def search_city(city_name, count=5):
    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city_name,
        "count": count,
        "language": "es",
        "format": "json",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        print("Error al buscar la ciudad. Revisa tu conexion o intenta mas tarde.")
        return []

    data = response.json()

    if "results" not in data:
        return []

    return data["results"]


def get_weather(latitude, longitude, timezone):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(
            [
                "temperature_2m",
                "apparent_temperature",
                "relative_humidity_2m",
                "precipitation_probability",
                "wind_speed_10m",
                "wind_gusts_10m",
                "cloud_cover",
                "uv_index",
            ]
        ),
        "forecast_days": 1,
        "timezone": timezone,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        print("Error al consultar el clima. Revisa tu conexion o intenta mas tarde.")
        return None

    return response.json()


def build_weather_dataframe(weather_data):
    if weather_data is None:
        return pd.DataFrame()

    try:
        if "hourly" not in weather_data:
            return pd.DataFrame()

        hourly_data = weather_data["hourly"]

        df = pd.DataFrame(hourly_data)
        df["time"] = pd.to_datetime(df["time"])
        df["hour"] = df["time"].dt.hour
    except Exception:
        print("No se pudieron procesar los datos climaticos.")
        return pd.DataFrame()

    return df


def get_activity_settings(activity):
    activities = {
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
            "wind_weight": 0.8,
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
            "ideal_uv": 6,
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

    return activities.get(activity, activities["Pasear o hacer senderismo"])


def choose_activity():
    activities = {
        "1": "Pasear o hacer senderismo",
        "2": "Jardineria y agricultura",
        "3": "Deportes al aire libre",
        "4": "Picnic o actividades en parque",
        "5": "Ir al cine",
        "6": "Ir a la playa",
    }

    print("\nSelecciona una actividad:")
    print("1. Pasear o hacer senderismo")
    print("2. Jardineria y agricultura")
    print("3. Deportes al aire libre")
    print("4. Picnic o actividades en parque")
    print("5. Ir al cine")
    print("6. Ir a la playa")

    selected_option = input("Actividad: ").strip()

    if selected_option not in activities:
        print("Opcion no valida. Se usara Pasear por defecto.")
        return "Pasear o hacer senderismo"

    return activities[selected_option]


def choose_city_index(results):
    selected_option = input(
        "\nSelecciona el numero de la ciudad correcta: ").strip()

    if not selected_option.isdigit():
        print("Seleccion no valida. Se usara la primera opcion por defecto.")
        return 0

    selected_index = int(selected_option)

    if selected_index < 0 or selected_index >= len(results):
        print("Seleccion no valida. Se usara la primera opcion por defecto.")
        return 0

    return selected_index


def calculate_activity_score(row, settings):
    score = 100

    temperature = row["temperature_2m"]
    rain_probability = row["precipitation_probability"]
    wind_speed = row["wind_speed_10m"]
    humidity = row["relative_humidity_2m"]
    uv_index = row["uv_index"]

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
    temperature_distance = (
        abs(row["temperature_2m"] - settings["ideal_temperature"])
        * settings["temperature_weight"]
    )
    humidity_distance = (
        abs(row["relative_humidity_2m"] - settings["ideal_humidity"])
        / 10
        * settings["humidity_weight"]
    )
    wind_distance = (
        abs(row["wind_speed_10m"] - settings["ideal_wind"])
        / 5
        * settings["wind_weight"]
    )
    uv_distance = (
        abs(row["uv_index"] - settings["ideal_uv"])
        / 2
        * settings["uv_weight"]
    )
    rain_distance = row["precipitation_probability"] / \
        20 * settings["rain_weight"]

    return (
        temperature_distance
        + humidity_distance
        + wind_distance
        + uv_distance
        + rain_distance
    )


def classify_score(score):
    if score >= 80:
        return "excelente"
    if score >= 60:
        return "bueno"
    if score >= 40:
        return "regular"
    return "malo"


def get_valid_hours(df, start_hour=7, end_hour=22):
    return df[(df["hour"] >= start_hour) & (df["hour"] <= end_hour)].copy()


def plot_weather_summary(df_valid_hours, activity, city_name):
    plt.figure(figsize=(8, 5))
    plt.plot(df_valid_hours["hour"],
             df_valid_hours["activity_score"], marker="o")
    plt.title(f"Score por hora - {activity} en {city_name}")
    plt.xlabel("Hora")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(df_valid_hours["hour"],
             df_valid_hours["temperature_2m"], marker="o")
    plt.title(f"Temperatura por hora - {city_name}")
    plt.xlabel("Hora")
    plt.ylabel("Temperatura grados C")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        df_valid_hours["hour"],
        df_valid_hours["precipitation_probability"],
        marker="o",
    )
    plt.title(f"Probabilidad de lluvia por hora - {city_name}")
    plt.xlabel("Hora")
    plt.ylabel("Probabilidad de lluvia (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(df_valid_hours["hour"],
             df_valid_hours["wind_speed_10m"], marker="o")
    plt.title(f"Viento por hora - {city_name}")
    plt.xlabel("Hora")
    plt.ylabel("Velocidad del viento km/h")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_city_map(city_name, country, latitude, longitude, output_file="outbyml_city_map.html"):
    df_city = pd.DataFrame(
        [
            {
                "city_name": city_name,
                "country": country,
                "latitude": latitude,
                "longitude": longitude,
            }
        ]
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_city,
        get_position="[longitude, latitude]",
        get_radius=25000,
        get_fill_color=[0, 128, 255, 180],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=8,
        pitch=35,
    )

    tooltip = {
        "html": (
            "<b>Ciudad:</b> {city_name}<br/>"
            "<b>Pais:</b> {country}<br/>"
            "<b>Latitud:</b> {latitude}<br/>"
            "<b>Longitud:</b> {longitude}"
        ),
        "style": {
            "backgroundColor": "white",
            "color": "black",
        },
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
    )

    deck.to_html(output_file, open_browser=True)
    print(f"Mapa generado: {output_file}")


def main():
    city_name = input("Escribe una ciudad: ")

    results = search_city(city_name)

    if not results:
        print("No se encontro la ciudad.")
        return

    print("\nResultados encontrados:")

    for i, city in enumerate(results):
        name = city.get("name", "")
        admin1 = city.get("admin1", "")
        country = city.get("country", "")
        latitude = city.get("latitude", "")
        longitude = city.get("longitude", "")

        print(f"{i}. {name}, {admin1}, {country} | lat: {latitude}, lon: {longitude}")

    selected_index = choose_city_index(results)
    selected_city = results[selected_index]

    activity = choose_activity()
    settings = get_activity_settings(activity)

    latitude = selected_city["latitude"]
    longitude = selected_city["longitude"]
    timezone = selected_city.get("timezone", "auto")

    weather_data = get_weather(latitude, longitude, timezone)
    df = build_weather_dataframe(weather_data)

    if df.empty:
        print("No hay datos climaticos disponibles para esta ubicacion.")
        return

    df["activity_score"] = df.apply(
        lambda row: calculate_activity_score(row, settings),
        axis=1,
    )
    df["recommendation"] = df["activity_score"].apply(classify_score)
    df["comfort_distance"] = df.apply(
        lambda row: calculate_comfort_distance(row, settings),
        axis=1,
    )

    df_valid_hours = get_valid_hours(
        df,
        start_hour=settings["start_hour"],
        end_hour=settings["end_hour"],
    )

    if df_valid_hours.empty:
        print("No hay horas disponibles dentro de la franja seleccionada.")
        return

    best_row = df_valid_hours.sort_values(
        ["activity_score", "comfort_distance"],
        ascending=[False, True],
    ).iloc[0]

    print("\nCiudad seleccionada:")
    print(f"{selected_city.get('name')}, {selected_city.get('country')}")
    print(f"Coordenadas: {latitude}, {longitude}")
    print(f"Timezone: {timezone}")

    print("\nActividad seleccionada:")
    print(activity)

    print("\nFranja horaria evaluada:")
    print(f"{settings['start_hour']:02d}:00 a {settings['end_hour']:02d}:00")

    print("\nMejor hora:")
    print(f"Hora: {best_row['time']}")
    print(f"Score: {best_row['activity_score']}/100")
    print(f"Distancia de confort: {best_row['comfort_distance']:.2f}")
    print(f"Recomendacion: {best_row['recommendation']}")

    print("\nTop 5 mejores horas:")
    columns_to_show = [
        "time",
        "temperature_2m",
        "apparent_temperature",
        "precipitation_probability",
        "wind_speed_10m",
        "relative_humidity_2m",
        "uv_index",
        "activity_score",
        "comfort_distance",
        "recommendation",
    ]

    print(
        df_valid_hours[columns_to_show]
        .sort_values(
            ["activity_score", "comfort_distance"],
            ascending=[False, True],
        )
        .head(5)
        .to_string(index=False)
    )

    try:
        create_city_map(
            city_name=selected_city.get("name"),
            country=selected_city.get("country"),
            latitude=latitude,
            longitude=longitude,
        )
    except Exception:
        print("No se pudo generar el mapa.")

    try:
        plot_weather_summary(
            df_valid_hours=df_valid_hours,
            activity=activity,
            city_name=selected_city.get("name"),
        )
    except Exception:
        print("No se pudieron generar los graficos.")


if __name__ == "__main__":
    main()
