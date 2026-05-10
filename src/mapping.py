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

    marker_layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[longitude, latitude]",
        get_radius=1800,
        get_fill_color=[245, 158, 11, 220],
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=3,
        stroked=True,
        pickable=True,
    )

    pulse_layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[longitude, latitude]",
        get_radius=6000,
        get_fill_color=[245, 158, 11, 60],
        pickable=False,
    )

    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=11,
        pitch=45,
    )

    tooltip = {
        "html": (
            "<div style='padding:8px 12px;'>"
            "<b style='font-size:14px;'>📍 {city_name}</b><br/>"
            "<span style='color:#666;'>{country}</span><br/>"
            "<span style='font-size:11px;color:#999;'>"
            "{latitude:.4f}, {longitude:.4f}</span>"
            "</div>"
        ),
        "style": {
            "backgroundColor": "#ffffff",
            "color": "#1a2b3d",
            "borderRadius": "10px",
            "boxShadow": "0 4px 16px rgba(0,0,0,0.12)",
            "border": "1px solid #eee",
        },
    }

    return pdk.Deck(
        layers=[pulse_layer, marker_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/outdoors-v12",
    )
