"""Funciones para crear graficos interactivos con Plotly."""

import plotly.express as px


def create_score_chart(df, theme_template="plotly_white"):
    """Crea un grafico de puntuacion por hora."""
    return px.line(
        df,
        x="hour",
        y="activity_score",
        markers=True,
        title="Score por hora",
        labels={"hour": "Hora", "activity_score": "Score"},
        template=theme_template,
    )


def create_temperature_chart(df, theme_template="plotly_white"):
    """Crea un grafico de temperatura por hora."""
    return px.line(
        df,
        x="hour",
        y="temperature_2m",
        markers=True,
        title="Temperatura por hora",
        labels={"hour": "Hora", "temperature_2m": "Temperatura"},
        template=theme_template,
    )


def create_rain_chart(df, theme_template="plotly_white"):
    """Crea un grafico de probabilidad de lluvia por hora."""
    fig = px.line(
        df,
        x="hour",
        y="precipitation_probability",
        markers=True,
        title="Probabilidad de lluvia por hora",
        labels={
            "hour": "Hora",
            "precipitation_probability": "Probabilidad de lluvia (%)",
        },
        template=theme_template,
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def create_wind_chart(df, theme_template="plotly_white"):
    """Crea un grafico de viento por hora."""
    return px.line(
        df,
        x="hour",
        y="wind_speed_10m",
        markers=True,
        title="Viento por hora",
        labels={"hour": "Hora", "wind_speed_10m": "Viento km/h"},
        template=theme_template,
    )
