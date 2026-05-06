"""Funciones para mapas interactivos con Pydeck."""

import pandas as pd
import pydeck as pdk


def create_city_deck(city_name, country, latitude, longitude):
    """Crea un objeto pydeck Deck centrado en una ciudad."""
    data = pd.DataFrame(
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
        data=data,
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

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
    )
