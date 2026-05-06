"""Aplicacion web de OutByML con Streamlit."""

import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import (
    apply_weather_safety_rules,
    calculate_activity_score,
    calculate_comfort_distance,
    classify_score,
    get_activity_settings,
    get_valid_hours,
)
from src.geocoding import search_city
from src.mapping import create_city_deck
from src.recommendations import (
    generate_recommendation_text,
    get_best_hour,
    get_top_hours,
)
from src.visualization import (
    create_rain_chart,
    create_score_chart,
    create_temperature_chart,
    create_wind_chart,
)
from src.weather_api import build_weather_dataframe, get_weather


ACTIVITIES = ["Pasear", "Turismo", "Deporte", "Bici", "Lavar ropa"]
BASE_DIR = Path(__file__).resolve().parents[1]
ML_MODEL_PATH = BASE_DIR / "models" / "outbyml_random_forest_model.pkl"
HERO_IMAGE_PATH = BASE_DIR / "assets" / "tiempo.png"
HERO_BG_PATH = BASE_DIR / "assets" / "red_neuronal.jpg"
ML_FEATURE_COLUMNS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "uv_index",
    "hour",
    "activity",
    "continent",
    "climate_group",
    "season_block",
]


st.set_page_config(
    page_title="OutByML",
    page_icon=None,
    layout="wide",
)


@st.cache_data
def image_to_base64(image_path):
    """Convierte una imagen local a base64 para incrustarla en HTML/CSS."""
    image_path = Path(image_path)
    if not image_path.exists():
        return None

    return base64.b64encode(image_path.read_bytes()).decode()


def inject_styles(theme):
    """Aplica una capa visual sencilla y responsiva."""
    hero_bg_base64 = image_to_base64(HERO_BG_PATH)

    if theme == "Oscuro":
        colors = {
            "bg": "#07111F",
            "bg_layer": "linear-gradient(180deg, #07111F 0%, #0B1220 100%)",
            "panel": "#111E30",
            "panel_soft": "#172A42",
            "border": "rgba(216, 225, 234, 0.16)",
            "text": "#F3F7FB",
            "muted": "#B8C7D6",
            "accent": "#38BDF8",
            "accent_2": "#41C7B9",
            "hero_subtitle": "#DCEBFF",
            "input_bg": "rgba(255, 255, 255, 0.06)",
            "button_text": "#061018",
            "recommendation_bg": "rgba(56, 189, 248, 0.12)",
            "recommendation_text": "#EAFBFF",
            "hero_title_shadow": (
                "0 2px 0 rgba(15, 23, 42, 0.95), "
                "0 8px 24px rgba(56, 189, 248, 0.40), "
                "0 18px 45px rgba(14, 165, 233, 0.22)"
            ),
            "hero_visual": (
                "radial-gradient(circle at 28% 22%, rgba(56, 189, 248, 0.34), transparent 24%), "
                "radial-gradient(circle at 70% 68%, rgba(147, 51, 234, 0.18), transparent 22%), "
                "linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.025))"
            ),
            "shadow": "0 18px 48px rgba(0, 0, 0, 0.28)",
        }
        hero_card_background = (
            f'linear-gradient(180deg, rgba(8,18,35,0.82), rgba(12,27,50,0.88)), '
            f'url("data:image/jpeg;base64,{hero_bg_base64}")'
            if hero_bg_base64
            else "linear-gradient(180deg, var(--panel), var(--panel-soft))"
        )
    else:
        colors = {
            "bg": "#F8FAFC",
            "bg_layer": "linear-gradient(180deg, #F6FAFD 0%, #F8FAFC 100%)",
            "panel": "#FFFFFF",
            "panel_soft": "#F1F8FE",
            "border": "#D8E1EA",
            "text": "#102033",
            "muted": "#52616B",
            "accent": "#2F80ED",
            "accent_2": "#38BDF8",
            "hero_subtitle": "#24415F",
            "input_bg": "#FFFFFF",
            "button_text": "#FFFFFF",
            "recommendation_bg": "rgba(47, 128, 237, 0.08)",
            "recommendation_text": "#102033",
            "hero_title_shadow": (
                "0 2px 0 rgba(255, 255, 255, 0.85), "
                "0 8px 24px rgba(14, 165, 233, 0.20), "
                "0 18px 45px rgba(15, 23, 42, 0.10)"
            ),
            "hero_visual": (
                "radial-gradient(circle at 30% 24%, rgba(47, 128, 237, 0.24), transparent 24%), "
                "radial-gradient(circle at 68% 72%, rgba(65, 199, 185, 0.20), transparent 20%), "
                "linear-gradient(145deg, rgba(255,255,255,0.96), rgba(224,244,255,0.82))"
            ),
            "shadow": "0 16px 38px rgba(16, 32, 51, 0.10)",
        }
        hero_card_background = (
            f'linear-gradient(180deg, rgba(248,250,252,0.88), rgba(224,242,254,0.88)), '
            f'url("data:image/jpeg;base64,{hero_bg_base64}")'
            if hero_bg_base64
            else "linear-gradient(180deg, var(--panel), var(--panel-soft))"
        )

    css = """
        <style>
            :root {
                --bg: __BG__;
                --bg-layer: __BG_LAYER__;
                --panel: __PANEL__;
                --panel-soft: __PANEL_SOFT__;
                --border: __BORDER__;
                --text: __TEXT__;
                --muted: __MUTED__;
                --accent: __ACCENT__;
                --accent-2: __ACCENT_2__;
                --input-bg: __INPUT_BG__;
                --button-text: __BUTTON_TEXT__;
                --recommendation-bg: __RECOMMENDATION_BG__;
                --recommendation-text: __RECOMMENDATION_TEXT__;
                --hero-title-shadow: __HERO_TITLE_SHADOW__;
                --hero-visual: __HERO_VISUAL__;
                --card-shadow: __SHADOW__;
            }

            .stApp {
                background: var(--bg-layer);
                color: var(--text);
            }

            .block-container {
                max-width: 1200px;
                padding-top: 1.5rem;
                padding-bottom: 3.5rem;
            }

            h1, h2, h3, p, label, span {
                color: var(--text);
            }

            .hero-card {
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: clamp(1.65rem, 4vw, 3rem);
                background: __HERO_CARD_BACKGROUND__;
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                box-shadow: var(--card-shadow);
                max-width: 100%;
                margin: 0 auto 1.2rem auto;
                text-align: center;
                overflow: hidden;
            }

            .hero-card .hero-kicker {
                display: inline-flex;
                align-items: center;
                width: fit-content;
                margin: 0 auto 0.75rem auto;
                padding: 0.38rem 0.72rem;
                border: 1px solid rgba(56, 189, 248, 0.34);
                border-radius: 999px;
                background: rgba(56, 189, 248, 0.10);
                color: var(--accent);
                font-size: 0.82rem;
                font-weight: 800;
                letter-spacing: 0.02em;
                text-transform: uppercase;
            }

            .hero-banner {
                width: 100%;
                height: 170px;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 22px;
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
                margin: 0 0 1rem 0;
            }

            .hero-card h1 {
                max-width: 860px;
                font-size: clamp(3.2rem, 8vw, 6.4rem);
                line-height: 0.9;
                margin: 0 auto 0.7rem auto;
                letter-spacing: 0;
                font-weight: 900;
                color: var(--text);
                text-shadow: var(--hero-title-shadow);
            }

            .hero-card h2 {
                max-width: 780px;
                font-size: clamp(1.15rem, 2.3vw, 1.7rem);
                font-weight: 750;
                line-height: 1.3;
                margin: 0 auto 0.75rem auto;
                color: var(--text);
            }

            .hero-card p {
                max-width: 760px;
                margin: 0 auto;
                color: var(--muted);
                font-size: 1.03rem;
                line-height: 1.65;
            }

            .hero-badges {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 0.55rem;
                margin-top: 1.05rem;
            }

            .hero-badge {
                border: 1px solid rgba(56, 189, 248, 0.26);
                border-radius: 999px;
                padding: 0.48rem 0.72rem;
                background: rgba(255, 255, 255, 0.06);
                color: var(--text);
                font-size: 0.88rem;
                font-weight: 700;
                line-height: 1;
            }

            .section-title {
                font-size: clamp(1.15rem, 2vw, 1.45rem);
                font-weight: 700;
                margin: 1.5rem 0 0.3rem 0;
                color: var(--text);
                text-align: center;
            }

            .section-subtitle {
                margin: 0 0 0.85rem 0;
                color: var(--muted);
                font-size: 0.96rem;
                line-height: 1.55;
                text-align: center;
            }

            .info-grid,
            .analysis-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.9rem;
                margin: 0.9rem 0 1.2rem 0;
            }

            .analysis-grid {
                grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
                gap: 0.85rem;
            }

            .info-card {
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1rem;
                background: linear-gradient(180deg, var(--panel), var(--panel-soft));
                box-shadow: var(--card-shadow);
            }

            .analysis-card {
                position: relative;
                overflow: hidden;
                min-height: 150px;
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem;
                background:
                    radial-gradient(circle at 82% 18%, rgba(56, 189, 248, 0.14), transparent 28%),
                    linear-gradient(180deg, var(--panel), var(--panel-soft));
                box-shadow: var(--card-shadow);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
            }

            .analysis-card:hover {
                transform: translateY(-2px);
                border-color: rgba(56, 189, 248, 0.46);
                box-shadow: 0 18px 42px rgba(16, 32, 51, 0.14);
            }

            .analysis-badge {
                width: 2.7rem;
                min-width: 2.7rem;
                height: 2.7rem;
                display: grid;
                place-items: center;
                border-radius: 999px;
                margin-bottom: 0.85rem;
                background: linear-gradient(135deg, var(--accent), var(--accent-2));
                color: var(--button-text);
                font-weight: 850;
                font-size: 0.9rem;
                line-height: 1;
                box-shadow: 0 10px 22px rgba(47, 128, 237, 0.20);
            }

            .info-step {
                width: 2rem;
                height: 2rem;
                display: grid;
                place-items: center;
                border-radius: 999px;
                margin-bottom: 0.65rem;
                background: linear-gradient(135deg, var(--accent), var(--accent-2));
                color: var(--button-text);
                font-weight: 800;
                font-size: 0.95rem;
            }

            .info-card h3,
            .analysis-card h3 {
                margin: 0 0 0.35rem 0;
                font-size: 1.02rem;
                line-height: 1.25;
                text-align: center;
            }

            .info-card p,
            .analysis-card p {
                margin: 0;
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.45;
                text-align: center;
            }

            .soft-card {
                height: 100%;
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1.15rem 1.2rem;
                background: linear-gradient(180deg, var(--panel), var(--panel-soft));
                box-shadow: var(--card-shadow);
            }

            .metric-label {
                margin: 0 0 0.35rem 0;
                color: var(--muted);
                font-size: 0.9rem;
            }

            .metric-value {
                margin: 0;
                color: var(--text);
                font-size: clamp(1.45rem, 3.2vw, 2.25rem);
                font-weight: 750;
                line-height: 1.15;
                overflow-wrap: anywhere;
            }

            .metric-caption {
                margin: 0.45rem 0 0 0;
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.35;
            }

            .recommendation-box {
                border-left: 4px solid var(--accent);
                border-radius: 18px;
                padding: 1.1rem 1.2rem;
                background:
                    linear-gradient(135deg, var(--recommendation-bg), rgba(65, 199, 185, 0.08));
                color: var(--recommendation-text);
                line-height: 1.7;
                margin: 0.65rem 0 1rem 0;
                box-shadow: var(--card-shadow);
            }

            .block-card {
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem;
                background: var(--panel);
                box-shadow: var(--card-shadow);
            }

            .stButton > button {
                border-radius: 999px;
                border: 1px solid rgba(56, 189, 248, 0.50);
                background: linear-gradient(135deg, var(--accent), var(--accent-2));
                color: var(--button-text);
                font-weight: 750;
                min-height: 2.9rem;
                box-shadow: 0 10px 24px rgba(47, 128, 237, 0.18);
                transition: transform 160ms ease, filter 160ms ease, box-shadow 160ms ease;
            }

            .stButton > button:hover {
                filter: brightness(1.03);
                transform: translateY(-1px);
                box-shadow: 0 14px 28px rgba(47, 128, 237, 0.24);
                color: #061018;
            }

            div[data-testid="stTextInput"] input,
            div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
                border-radius: 13px;
                border-color: var(--border);
                background-color: var(--input-bg);
                color: var(--text);
                min-height: 2.75rem;
            }

            div[data-testid="stTextInput"] input:focus {
                border-color: var(--accent);
                box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.22);
            }

            div[data-testid="stDataFrame"] {
                border-radius: 14px;
                overflow: hidden;
            }

            .st-key-theme_picker {
                max-width: 90px;
                margin: 0;
                padding: 0.22rem 0.35rem;
                border: 1px solid var(--border);
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.08);
                box-shadow: 0 8px 20px rgba(16, 32, 51, 0.08);
            }

            .st-key-theme_menu {
                margin: 2.5rem 0 0.5rem 0;
            }

            .st-key-theme_menu .stColumn:last-child {
                display: flex;
                justify-content: flex-end;
            }

            .st-key-theme_menu button {
                min-height: 2.1rem;
                width: 2.4rem;
                padding: 0;
                border-radius: 999px;
                border: 1px solid var(--border);
                background: var(--panel);
                color: var(--text);
                box-shadow: var(--card-shadow);
            }

            .st-key-theme_menu button:hover {
                border-color: var(--accent);
                color: var(--text);
                transform: translateY(-1px);
            }

            .st-key-theme_picker div[role="radiogroup"] {
                gap: 0.15rem;
                justify-content: flex-start;
            }

            .st-key-theme_picker label {
                margin: 0;
                padding: 0.15rem 0.35rem;
                border-radius: 999px;
                font-size: 0.95rem;
                color: var(--muted);
            }

            .st-key-theme_picker label p {
                font-size: 0.95rem;
                line-height: 1;
                margin: 0;
            }

            .st-key-theme_picker [data-testid="stWidgetLabel"] {
                display: none;
            }

            .st-key-theme_picker [data-testid="stMarkdownContainer"] p {
                font-size: 0.95rem;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-color: var(--border);
                border-radius: 20px;
                background: linear-gradient(180deg, var(--panel), var(--panel-soft));
                box-shadow: var(--card-shadow);
            }

            .st-key-control_card div[data-testid="stVerticalBlockBorderWrapper"],
            .st-key-map_panel div[data-testid="stVerticalBlockBorderWrapper"],
            .st-key-score_panel div[data-testid="stVerticalBlockBorderWrapper"],
            .st-key-chart_panel div[data-testid="stVerticalBlockBorderWrapper"],
            .st-key-top_panel div[data-testid="stVerticalBlockBorderWrapper"] {
                padding: 0.15rem;
            }

            div[data-testid="stExpander"] {
                border-color: var(--border);
                border-radius: 14px;
                background: rgba(255, 255, 255, 0.02);
            }

            @media (max-width: 768px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                    padding-top: 1rem;
                }

                .hero-card {
                    padding: 1.25rem;
                    border-radius: 16px;
                }

                .st-key-theme_menu {
                    margin: 2rem 0 0.45rem 0;
                }

                .hero-banner {
                    height: 110px;
                    border-radius: 16px;
                }

                .hero-card h1 {
                    font-size: clamp(2.7rem, 16vw, 4.2rem);
                }

                .info-grid {
                    grid-template-columns: 1fr;
                }

                .analysis-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 0.75rem;
                }

                .soft-card {
                    padding: 1rem;
                    border-radius: 14px;
                }
            }

            @media (max-width: 360px) {
                .analysis-grid {
                    grid-template-columns: 1fr;
                }
            }

            @media (min-width: 769px) and (max-width: 1050px) {
                .analysis-grid {
                    grid-template-columns: repeat(3, minmax(0, 1fr));
                }
            }
        </style>
        """

    replacements = {
        "__BG__": colors["bg"],
        "__BG_LAYER__": colors["bg_layer"],
        "__PANEL__": colors["panel"],
        "__PANEL_SOFT__": colors["panel_soft"],
        "__BORDER__": colors["border"],
        "__TEXT__": colors["text"],
        "__MUTED__": colors["muted"],
        "__ACCENT__": colors["accent"],
        "__ACCENT_2__": colors["accent_2"],
        "__INPUT_BG__": colors["input_bg"],
        "__BUTTON_TEXT__": colors["button_text"],
        "__RECOMMENDATION_BG__": colors["recommendation_bg"],
        "__RECOMMENDATION_TEXT__": colors["recommendation_text"],
        "__HERO_TITLE_SHADOW__": colors["hero_title_shadow"],
        "__HERO_CARD_BACKGROUND__": hero_card_background,
        "__HERO_VISUAL__": colors["hero_visual"],
        "__SHADOW__": colors["shadow"],
        "__HERO_SUBTITLE__": colors["hero_subtitle"],
    }
    for placeholder, value in replacements.items():
        css = css.replace(placeholder, value)

    st.markdown(css, unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "city_results": [],
        "selected_city": None,
        "analysis": None,
        "last_search": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource
def load_ml_model():
    """Carga el modelo ML entrenado una sola vez por sesion."""
    if not ML_MODEL_PATH.exists():
        return None
    try:
        model = joblib.load(ML_MODEL_PATH)
        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1
        return model
    except Exception:
        return None


def get_season_block(month):
    """Mapea el mes actual al bloque estacional usado por el modelo."""
    if month in [12, 1, 2]:
        return "enero"
    if month in [3, 4, 5]:
        return "abril"
    if month in [6, 7, 8]:
        return "julio"
    return "octubre"


def get_city_continent(selected_city):
    """Obtiene el continente si la fuente de ciudad lo trae."""
    if not selected_city:
        return "desconocido"
    return (
        selected_city.get("continent")
        or selected_city.get("continent_name")
        or "desconocido"
    )


def prepare_ml_features(df_valid_hours, activity, selected_city):
    """Prepara las columnas exactas que espera el pipeline ML."""
    features = df_valid_hours.copy()
    current_month = pd.Timestamp.now().month
    features["activity"] = activity
    features["continent"] = get_city_continent(selected_city)
    features["climate_group"] = "desconocido"
    features["season_block"] = get_season_block(current_month)
    return features[ML_FEATURE_COLUMNS].copy()


def render_theme_selector():
    """Muestra el selector visual sin afectar la logica de negocio."""
    with st.container(key="theme_picker"):
        selected_theme_icon = st.radio(
            "Tema visual",
            ["☀", "☾"],
            index=1,
            key="visual_theme_icon",
            horizontal=True,
            label_visibility="collapsed",
        )
    return "Claro" if selected_theme_icon == "☀" else "Oscuro"


def render_theme_menu():
    """Muestra el selector de tema dentro de un menu discreto."""
    with st.container(key="theme_menu"):
        _, menu_col = st.columns([0.92, 0.08])
        with menu_col:
            with st.popover("☰"):
                render_theme_selector()


def get_theme_mode():
    """Devuelve el tema activo a partir del estado del selector visual."""
    selected_theme_icon = st.session_state.get("visual_theme_icon", "☾")
    return "Claro" if selected_theme_icon == "☀" else "Oscuro"


def city_label(city):
    parts = [
        city.get("name"),
        city.get("admin1"),
        city.get("country"),
    ]
    return ", ".join(str(part) for part in parts if part)


def format_hour(value):
    try:
        return f"{int(value):02d}:00"
    except (TypeError, ValueError):
        return "No disponible"


def format_score(value):
    try:
        return f"{float(value):.1f}/100"
    except (TypeError, ValueError):
        return "No disponible"


def prepare_top_table(df_top):
    visible_columns = [
        "time",
        "activity_score",
        "recommendation_final",
        "temperature_2m",
        "precipitation_probability",
        "wind_speed_10m",
        "relative_humidity_2m",
    ]
    available_columns = [column for column in visible_columns if column in df_top.columns]
    table = df_top[available_columns].copy()

    if "time" in table.columns:
        table["time"] = pd.to_datetime(table["time"]).dt.strftime("%H:%M")
    if "activity_score" in table.columns:
        table["activity_score"] = table["activity_score"].round(1)
    if "temperature_2m" in table.columns:
        table["temperature_2m"] = table["temperature_2m"].round(1)
    if "wind_speed_10m" in table.columns:
        table["wind_speed_10m"] = table["wind_speed_10m"].round(1)
    if "relative_humidity_2m" in table.columns:
        table["relative_humidity_2m"] = table["relative_humidity_2m"].round(0).astype("Int64")
    if "precipitation_probability" in table.columns:
        table["precipitation_probability"] = (
            table["precipitation_probability"].round(0).astype("Int64")
        )

    return table.rename(
        columns={
            "time": "Hora",
            "activity_score": "Score",
            "recommendation_final": "Recomendacion final",
            "temperature_2m": "Temperatura",
            "precipitation_probability": "Lluvia",
            "wind_speed_10m": "Viento",
            "relative_humidity_2m": "Humedad",
        }
    )


def run_analysis(city, activity, decision_mode, selected_hour=None):
    latitude = city.get("latitude")
    longitude = city.get("longitude")
    timezone = city.get("timezone", "auto")

    weather_data = get_weather(latitude, longitude, timezone)
    df_weather = build_weather_dataframe(weather_data)

    if df_weather.empty:
        return None, "No se pudieron obtener datos climaticos para esta ciudad."

    settings = get_activity_settings(activity)
    df_weather = df_weather.copy()
    df_weather["activity_score"] = df_weather.apply(
        lambda row: calculate_activity_score(row, settings),
        axis=1,
    )
    df_weather["comfort_distance"] = df_weather.apply(
        lambda row: calculate_comfort_distance(row, settings),
        axis=1,
    )
    df_weather["recommendation"] = df_weather["activity_score"].apply(classify_score)

    df_valid_hours = get_valid_hours(
        df_weather,
        settings["start_hour"],
        settings["end_hour"],
    )
    if df_valid_hours.empty:
        return None, "No hay horas validas para esta actividad en el rango configurado."

    ml_model = load_ml_model()
    ml_model_available = ml_model is not None
    ml_model_missing = not ML_MODEL_PATH.exists()
    ml_error = None
    if ml_model_available:
        try:
            X_ml = prepare_ml_features(df_valid_hours, activity, city)
            df_valid_hours["recommendation_ml"] = ml_model.predict(X_ml)
        except Exception as exc:
            ml_model_available = False
            ml_error = str(exc)

    source_recommendation_column = (
        "recommendation_ml" if ml_model_available else "recommendation"
    )
    df_valid_hours["recommendation_final"] = df_valid_hours.apply(
        lambda row: apply_weather_safety_rules(
            row,
            activity,
            row[source_recommendation_column],
        ),
        axis=1,
    )

    best_row = get_best_hour(df_valid_hours)
    top_hours = get_top_hours(df_valid_hours, top_n=5)

    if best_row is None:
        return None, "No se pudo calcular una recomendacion con los datos disponibles."

    selected_hour_row = None
    if decision_mode == "Evaluar hora especifica":
        selected_rows = df_valid_hours[df_valid_hours["hour"] == selected_hour]
        if selected_rows.empty:
            return (
                None,
                "No hay datos disponibles para esa hora dentro de la franja seleccionada.",
            )
        selected_hour_row = selected_rows.iloc[0]

    return {
        "activity": activity,
        "city": city,
        "city_name": city_label(city),
        "decision_mode": decision_mode,
        "selected_hour": selected_hour,
        "selected_hour_row": selected_hour_row,
        "settings": settings,
        "weather": df_weather,
        "valid_hours": df_valid_hours,
        "best_row": best_row,
        "top_hours": top_hours,
        "ml_model_available": ml_model_available,
        "ml_model_missing": ml_model_missing,
        "ml_error": ml_error,
    }, None


def build_specific_hour_text(selected_hour_row, best_row):
    """Genera un texto humano para la hora evaluada."""
    if selected_hour_row is None:
        return "No hay datos disponibles para la hora seleccionada."

    selected_hour = selected_hour_row.get("hour")
    best_hour = best_row.get("hour") if best_row is not None else None
    recommendation = selected_hour_row.get("recommendation", "")

    if selected_hour == best_hour:
        return "La hora seleccionada coincide con la mejor opcion del dia."
    if recommendation in ["excelente", "bueno"]:
        return "La hora seleccionada es una opcion favorable para la actividad elegida."
    if recommendation == "regular":
        return "La hora seleccionada es posible, pero hay mejores alternativas durante el dia."
    if recommendation == "malo":
        return (
            "La hora seleccionada no es recomendable para esta actividad. "
            "Conviene revisar otra franja horaria."
        )
    return "La hora seleccionada se puede evaluar, pero conviene compararla con el resto del dia."


def render_section_header(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="section-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Clima + ML + Seguridad</div>
            <h1>OutByML</h1>
            <h2>Tu asistente inteligente para decidir cuando salir.</h2>
            <p>
                Consulta el clima actualizado, elige una actividad y recibe una recomendacion clara
                usando Machine Learning y reglas de seguridad climatica.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">Clima actualizado</span>
                <span class="hero-badge">Modelo ML</span>
                <span class="hero-badge">Reglas de seguridad</span>
                <span class="hero-badge">Hora especifica</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_banner():
    hero_image_base64 = image_to_base64(HERO_IMAGE_PATH)
    if not hero_image_base64:
        return

    st.markdown(
        (
            '<div class="hero-banner" '
            f'style=\'background-image: url("data:image/png;base64,{hero_image_base64}");\''
            "></div>"
        ),
        unsafe_allow_html=True,
    )


def render_how_it_works():
    render_section_header(
        "Como funciona OutByML",
        "Una lectura sencilla del clima para tomar mejores decisiones antes de salir.",
    )
    st.markdown(
        """
        <div class="info-grid">
            <div class="info-card">
                <div class="info-step">1</div>
                <h3>Consulta clima actualizado</h3>
                <p>Obtiene datos horarios recientes de temperatura, lluvia, viento y humedad.</p>
            </div>
            <div class="info-card">
                <div class="info-step">2</div>
                <h3>Evalua la actividad elegida</h3>
                <p>Compara las condiciones disponibles con lo que necesita cada actividad.</p>
            </div>
            <div class="info-card">
                <div class="info-step">3</div>
                <h3>Clasifica la mejor hora</h3>
                <p>Usa ML y reglas de seguridad para dar una recomendacion final mas realista.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_scope():
    render_section_header(
        "Que analiza",
        "Variables principales que ayudan a estimar si una hora conviene para la actividad.",
    )
    st.markdown(
        """
        <div class="analysis-grid">
            <div class="analysis-card">
                <div class="analysis-badge">T°</div>
                <h3>Temperatura</h3>
                <p>Calor o frio del ambiente.</p>
            </div>
            <div class="analysis-card">
                <div class="analysis-badge">mm</div>
                <h3>Lluvia</h3>
                <p>Riesgo de precipitacion.</p>
            </div>
            <div class="analysis-card">
                <div class="analysis-badge">km/h</div>
                <h3>Viento</h3>
                <p>Velocidad del aire.</p>
            </div>
            <div class="analysis-card">
                <div class="analysis-badge">%</div>
                <h3>Humedad</h3>
                <p>Sensacion de incomodidad.</p>
            </div>
            <div class="analysis-card">
                <div class="analysis-badge">UV</div>
                <h3>Indice UV</h3>
                <p>Exposicion solar.</p>
            </div>
            <div class="analysis-card">
                <div class="analysis-badge">h</div>
                <h3>Hora del dia</h3>
                <p>Momento disponible para salir.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="soft-card">
            <p class="metric-label">{label}</p>
            <p class="metric-value">{value}</p>
            <p class="metric-caption">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_search_area():
    render_section_header(
        "Prepara tu consulta",
        "Selecciona ciudad, actividad y modo de decision para generar una recomendacion.",
    )
    with st.container(border=True, key="control_card"):
        search_col, button_col = st.columns([4, 1])

        with search_col:
            city_name = st.text_input(
                "Ciudad",
                placeholder="Ejemplo: Madrid, Buenos Aires, Tokio",
            )

        with button_col:
            st.write("")
            search_clicked = st.button("Buscar ciudad", use_container_width=True)

        if search_clicked:
            clean_city = city_name.strip()
            if not clean_city:
                st.info("Escribe el nombre de una ciudad para empezar.")
            else:
                with st.spinner("Buscando coincidencias..."):
                    results = search_city(clean_city)
                st.session_state.city_results = results
                st.session_state.last_search = clean_city
                st.session_state.selected_city = results[0] if results else None
                st.session_state.analysis = None
                if not results:
                    st.warning("No encontre resultados para esa ciudad. Prueba con otro nombre.")

        if st.session_state.city_results:
            labels = [city_label(city) for city in st.session_state.city_results]
            selected_label = st.selectbox("Selecciona la ubicacion", labels)
            selected_index = labels.index(selected_label)
            st.session_state.selected_city = st.session_state.city_results[selected_index]
        elif st.session_state.last_search:
            st.warning("No hay resultados de ciudad guardados. Realiza una nueva busqueda.")

        activity_col, mode_col = st.columns([1, 1])
        with activity_col:
            activity = st.selectbox(
                "Actividad",
                ACTIVITIES,
                index=0,
                key="activity_selector_v2",
            )
        settings = get_activity_settings(activity)
        with mode_col:
            decision_mode = st.selectbox(
                "Modo de decision",
                ["Buscar mejor hora", "Evaluar hora especifica"],
                index=0,
            )
        selected_hour = None
        if decision_mode == "Evaluar hora especifica":
            valid_hour_options = list(range(settings["start_hour"], settings["end_hour"] + 1))
            selected_hour = st.selectbox(
                "Hora",
                valid_hour_options,
                format_func=format_hour,
            )

        if st.button("Analizar clima", type="primary", use_container_width=True):
            if not st.session_state.city_results or st.session_state.selected_city is None:
                st.warning("Primero busca y selecciona una ciudad.")
                return

            with st.spinner("Analizando clima y horarios..."):
                analysis, error = run_analysis(
                    st.session_state.selected_city,
                    activity,
                    decision_mode,
                    selected_hour,
                )

            if error:
                st.session_state.analysis = None
                if "horas validas" in error or "No hay datos disponibles" in error:
                    st.warning(error)
                else:
                    st.error(error)
            else:
                st.session_state.analysis = analysis


def render_results(theme):
    analysis = st.session_state.analysis
    if not analysis:
        return

    best_row = analysis["best_row"]
    selected_hour_row = analysis.get("selected_hour_row")
    decision_mode = analysis.get("decision_mode", "Buscar mejor hora")
    city = analysis["city"]
    activity = analysis["activity"]
    city_name = analysis["city_name"]
    df_valid_hours = analysis["valid_hours"]
    top_hours = analysis["top_hours"]
    settings = analysis["settings"]
    ml_model_available = analysis.get("ml_model_available", False)
    ml_model_missing = analysis.get("ml_model_missing", False)
    ml_error = analysis.get("ml_error")
    theme_template = "plotly_dark" if theme == "Oscuro" else "plotly_white"

    result_row = selected_hour_row if decision_mode == "Evaluar hora especifica" else best_row
    hour_label = "Hora evaluada" if decision_mode == "Evaluar hora especifica" else "Mejor hora"
    hour_caption = (
        "Hora seleccionada por el usuario."
        if decision_mode == "Evaluar hora especifica"
        else "Hora recomendada para salir."
    )
    result_hour = format_hour(result_row.get("hour"))
    score = format_score(result_row.get("activity_score"))
    recommendation = str(result_row.get("recommendation_final", "sin clasificar")).capitalize()

    if ml_model_missing:
        st.warning(
            "Modelo ML no encontrado. La app esta usando la recomendacion basada en reglas."
        )
    elif not ml_model_available:
        message = "No se pudo cargar o ejecutar el modelo ML. La app esta usando la recomendacion basada en reglas."
        if ml_error:
            message = f"{message} Detalle tecnico: {ml_error}"
        st.warning(message)

    render_section_header(
        "Resultado principal",
        "Resumen directo de la mejor opcion calculada para tu consulta.",
    )
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        render_metric_card(hour_label, result_hour, hour_caption)
    with col_b:
        render_metric_card("Score", score, "Mayor puntuacion significa mejores condiciones.")
    with col_c:
        render_metric_card("Recomendacion final", recommendation, "Clasificacion climatica por hora.")
    with col_d:
        render_metric_card("Actividad", activity, city_name)

    if decision_mode == "Evaluar hora especifica":
        text_row = result_row.copy()
        text_row["recommendation"] = result_row.get("recommendation_final")
        recommendation_text = build_specific_hour_text(text_row, best_row)
        best_alternative_text = (
            f"Mejor alternativa del dia: {format_hour(best_row.get('hour'))} "
            f"con score {format_score(best_row.get('activity_score'))}."
        )
        st.markdown(
            f'<div class="recommendation-box">{best_alternative_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        text_row = best_row.copy()
        text_row["recommendation"] = best_row.get("recommendation_final")
        recommendation_text = generate_recommendation_text(text_row, activity, city_name)

    st.markdown(
        f'<div class="recommendation-box">{recommendation_text}</div>',
        unsafe_allow_html=True,
    )

    map_col, chart_col = st.columns([1, 1.45])
    with map_col:
        with st.container(border=True, key="map_panel"):
            render_section_header("Ubicacion analizada")
            deck = create_city_deck(
                city.get("name", city_name),
                city.get("country", ""),
                city.get("latitude"),
                city.get("longitude"),
            )
            st.pydeck_chart(deck, use_container_width=True)

    with chart_col:
        with st.container(border=True, key="score_panel"):
            render_section_header("Score por hora")
            st.plotly_chart(
                create_score_chart(df_valid_hours, theme_template=theme_template),
                use_container_width=True,
            )

    with st.container(border=True, key="chart_panel"):
        render_section_header(
            "Clima por variable",
            "Consulta los factores que influyen en la recomendacion final.",
        )
        temp_tab, rain_tab, wind_tab = st.tabs(["Temperatura", "Lluvia", "Viento"])
        with temp_tab:
            st.plotly_chart(
                create_temperature_chart(df_valid_hours, theme_template=theme_template),
                use_container_width=True,
            )
        with rain_tab:
            st.plotly_chart(
                create_rain_chart(df_valid_hours, theme_template=theme_template),
                use_container_width=True,
            )
        with wind_tab:
            st.plotly_chart(
                create_wind_chart(df_valid_hours, theme_template=theme_template),
                use_container_width=True,
            )

    with st.container(border=True, key="top_panel"):
        render_section_header(
            "Top 5 mejores opciones del dia",
            "Ordenadas por score y distancia de confort, manteniendo la recomendacion final visible.",
        )
        st.dataframe(
            prepare_top_table(top_hours),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Ver detalles tecnicos"):
        st.write(
            "OutByML usa un modelo ML para clasificar condiciones climaticas, pero tambien "
            "aplica reglas finales de seguridad para evitar recomendaciones poco realistas "
            "en casos de lluvia, viento fuerte o temperatura extrema."
        )
        technical_details = {
            "Ciudad": city_name,
            "Actividad": activity,
            "Hora inicial": settings["start_hour"],
            "Hora final": settings["end_hour"],
            "Temperatura ideal": settings["ideal_temperature"],
            "Humedad ideal": settings["ideal_humidity"],
            "Viento ideal": settings["ideal_wind"],
            "UV ideal": settings["ideal_uv"],
            "Recomendacion por reglas": result_row.get("recommendation", "sin clasificar"),
            "Recomendacion final": result_row.get("recommendation_final", "sin clasificar"),
        }
        if "recommendation_ml" in result_row.index:
            technical_details["Recomendacion ML"] = result_row.get(
                "recommendation_ml",
                "sin clasificar",
            )
        st.write(technical_details)


init_session_state()
visual_theme = get_theme_mode()
inject_styles(visual_theme)
render_theme_menu()
render_hero_banner()
render_hero()
render_how_it_works()
render_analysis_scope()
render_search_area()
render_results(visual_theme)
