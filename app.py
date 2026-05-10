"""Aplicacion web de SalimosHoy? con Streamlit."""

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

ACTIVITIES = [
    "Pasear o hacer senderismo",
    "Jardineria y agricultura",
    "Deportes al aire libre",
    "Picnic o actividades en parque",
    "Ir al cine",
    "Ir a la playa",
]

CITIES = [
    'Madrid', 'Dubai', 'Reykjavik', 'Bilbao', 'Caracas',
    'Buenos Aires', 'Barcelona', 'Riyadh', 'Oslo', 'Dublin',
    'Panama City', 'Santiago', 'Paris', 'Doha', 'Helsinki', 'Glasgow',
    'Havana', 'Lima', 'London', 'Cairo', 'Stockholm', 'Vancouver',
    'Santo Domingo', 'Sao Paulo', 'Rome', 'Marrakech', 'Moscow',
    'Bogota', 'San Juan', 'Cape Town', 'Lisbon', 'Phoenix', 'Montreal',
    'Quito', 'Manaus', 'Johannesburg', 'Berlin', 'Las Vegas',
    'Toronto', 'Singapore', 'Lagos', 'Sydney', 'Amsterdam', 'Seville',
    'Ulaanbaatar', 'Kuala Lumpur', 'Nairobi', 'Melbourne', 'Vienna',
    'Baghdad', 'Anchorage', 'Bangkok', 'Jakarta', 'Auckland', 'Prague',
    'Kuwait City', 'Nuuk', 'Mumbai', 'Manila', 'Perth'
]
BASE_DIR = Path(__file__).resolve().parent
ML_MODEL_PATH = BASE_DIR / "models" / "outbyml_random_forest_model.pkl"
HERO_IMAGE_PATH = BASE_DIR / "assets" / "hero_banner.jpg"
HERO_BG_PATH = BASE_DIR / "assets" / "red_neuronal.jpg"
PAGE_BG_PATH = BASE_DIR / "assets" / "weather_bg.jpg"
ACTIVITY_IMAGES = {
    "Pasear o hacer senderismo": BASE_DIR / "assets" / "activity_senderismo.jpg",
    "Jardineria y agricultura": BASE_DIR / "assets" / "activity_jardineria.jpg",
    "Deportes al aire libre": BASE_DIR / "assets" / "activity_deportes.jpg",
    "Picnic o actividades en parque": BASE_DIR / "assets" / "activity_picnic.jpg",
    "Ir al cine": BASE_DIR / "assets" / "activity_cine.jpg",
    "Ir a la playa": BASE_DIR / "assets" / "activity_playa.jpg",
}
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
    page_title="SalimosHoy?",
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
    page_bg_base64 = image_to_base64(PAGE_BG_PATH)

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
            f'linear-gradient(180deg, #111E30, #172A42), '
            f'url("data:image/jpeg;base64,{hero_bg_base64}")'
            if hero_bg_base64
            else "linear-gradient(180deg, #111E30, #172A42)"
        )
    else:
        colors = {
            "bg": "#FFFDF8",
            "bg_layer": "linear-gradient(180deg, #FFFEF9 0%, #FFF8EC 100%)",
            "panel": "#FFFFFF",
            "panel_soft": "#FFF9EF",
            "border": "rgba(234, 206, 160, 0.45)",
            "text": "#111111",
            "muted": "#333333",
            "accent": "#F59E0B",
            "accent_2": "#38BDF8",
            "hero_subtitle": "#3D5A7A",
            "input_bg": "#FFFFFF",
            "button_text": "#FFFFFF",
            "recommendation_bg": "rgba(245, 158, 11, 0.10)",
            "recommendation_text": "#1A2B3D",
            "hero_title_shadow": (
                "0 2px 0 rgba(255, 255, 255, 0.90), "
                "0 6px 18px rgba(245, 158, 11, 0.20), "
                "0 14px 35px rgba(15, 23, 42, 0.08)"
            ),
            "hero_visual": (
                "radial-gradient(circle at 30% 24%, rgba(245, 158, 11, 0.18), transparent 26%), "
                "radial-gradient(circle at 68% 72%, rgba(56, 189, 248, 0.14), transparent 22%), "
                "linear-gradient(145deg, rgba(255,255,255,0.98), rgba(255,248,230,0.85))"
            ),
            "shadow": "0 12px 32px rgba(194, 155, 80, 0.10)",
        }
        hero_card_background = (
            f'linear-gradient(180deg, rgba(255,253,248,0.92), rgba(255,248,230,0.90)), '
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
                --page-bg-image: __PAGE_BG_IMAGE__;
            }

            .stApp {
                background: var(--bg-layer);
                background-image: var(--page-bg-image);
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                color: var(--text);
                position: relative;
            }

            .stApp::before {
                content: '';
                position: fixed;
                inset: 0;
                background: rgba(255, 255, 255, 0.45);
                pointer-events: none;
                z-index: 0;
            }

            .stApp > * {
                position: relative;
                z-index: 1;
            }

            .block-container {
                max-width: 1600px;
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
                gap: 0.5rem;
                width: fit-content;
                margin: 0 auto 0.75rem auto;
            }

            .kicker-pill {
                padding: 0.32rem 0.7rem;
                border: 1px solid rgba(56, 189, 248, 0.34);
                border-radius: 9px;
                background: rgba(56, 189, 248, 0.10);
                color: var(--accent);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.04em;
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
                max-width: 1100px;
                font-size: clamp(3.2rem, 8vw, 6.4rem);
                line-height: 0.9;
                margin: 0 auto 0.7rem auto;
                letter-spacing: 0;
                font-weight: 900;
                color: var(--text);
                text-shadow: var(--hero-title-shadow);
            }

            .hero-card h2 {
                max-width: 1000px;
                font-size: clamp(1.3rem, 2.5vw, 1.85rem);
                font-weight: 750;
                line-height: 1.3;
                margin: 0 auto 0.75rem auto;
                color: var(--muted);
            }

            .hero-card p {
                max-width: 960px;
                margin: 0 auto;
                color: var(--text);
                font-size: 1.15rem;
                line-height: 1.65;
            }

            .hero-badges {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 0.55rem;
                margin-top: 1.05rem;
            }

            .hero-btn-wrapper {
                margin-top: 1.3rem;
                text-align: center;
            }

            .hero-cta-btn {
                display: inline-block;
                padding: 0.85rem 2.8rem;
                border-radius: 999px;
                background: linear-gradient(135deg, #2DB5A0, #5CC8B5);
                color: #FFFFFF;
                font-size: 1.1rem;
                font-weight: 800;
                letter-spacing: 0.3px;
                text-decoration: none;
                box-shadow: 0 10px 28px rgba(45, 181, 160, 0.3);
                transition: transform 200ms cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 200ms ease, filter 200ms ease;
            }

            .hero-cta-btn:hover {
                transform: translateY(-3px) scale(1.02);
                box-shadow: 0 14px 36px rgba(45, 181, 160, 0.4);
                filter: brightness(1.08);
                color: #FFFFFF;
                text-decoration: none;
            }

            .hero-cta-btn:hover {
                transform: translateY(-3px) scale(1.02);
                box-shadow: 0 14px 36px rgba(47, 128, 237, 0.35);
                filter: brightness(1.08);
                color: var(--button-text);
                text-decoration: none;
            }

            .hero-badge {
                border: 1px solid rgba(56, 189, 248, 0.26);
                border-radius: 9px;
                padding: 0.48rem 0.72rem;
                background: rgba(255, 255, 255, 0.06);
                color: var(--text);
                font-size: 0.88rem;
                font-weight: 700;
                line-height: 1;
            }

            .section-title {
                font-size: clamp(1.7rem, 3vw, 2.3rem);
                font-weight: 800;
                margin: 1.5rem 0 0.3rem 0;
                color: var(--text);
                text-align: center;
            }

            .section-subtitle {
                margin: 0 0 0.85rem 0;
                color: var(--muted);
                font-size: 1.15rem;
                line-height: 1.55;
                text-align: center;
            }

            .section-subtitle {
                margin: 0 0 0.85rem 0;
                color: var(--text);
                font-size: 1.05rem;
                line-height: 1.55;
                text-align: center;
            }

            .info-grid,
            .analysis-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 1rem;
                margin: 1rem 0 1.4rem 0;
            }

            .analysis-grid {
                grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
                gap: 0.85rem;
            }

            .info-card {
                position: relative;
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1.5rem 1.2rem;
                background: linear-gradient(180deg, var(--panel), var(--panel-soft));
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.08);
                transition: transform 160ms cubic-bezier(0.4, 0, 0.2, 1), border-color 160ms ease, box-shadow 160ms ease;
                display: flex;
                flex-direction: column;
                align-items: center;
                z-index: 1;
            }

            .info-card::before {
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                border-radius: 16px;
                background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), transparent 50%);
                opacity: 0;
                transition: opacity 160ms ease;
                z-index: -1;
                pointer-events: none;
            }

            .info-card:hover {
                transform: translateY(-4px);
                border-color: var(--accent);
                box-shadow: 0 16px 32px rgba(0, 0, 0, 0.08), 0 4px 8px rgba(0, 0, 0, 0.04), 0 0 0 1px var(--accent);
            }

            .info-card:hover::before {
                opacity: 1;
            }

            .analysis-card {
                position: relative;
                overflow: hidden;
                min-height: 160px;
                border: 1px solid var(--border);
                border-top: 3px solid var(--border);
                border-radius: 18px;
                padding: 1.2rem;
                background:
                    radial-gradient(circle at 82% 18%, rgba(56, 189, 248, 0.08), transparent 40%),
                    linear-gradient(180deg, var(--panel), var(--panel-soft));
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.08);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                align-items: center;
                transition: transform 160ms cubic-bezier(0.4, 0, 0.2, 1), border-color 160ms ease, box-shadow 160ms ease;
            }

            .analysis-card:hover {
                transform: translateY(-4px);
                border-top-color: var(--accent);
                border-color: var(--border);
                box-shadow: 0 16px 32px rgba(0, 0, 0, 0.08), 0 4px 8px rgba(0, 0, 0, 0.04), 0 -1px 0 0 var(--accent);
            }

            .analysis-badge {
                width: 4.5rem;
                min-width: 4.5rem;
                height: 4.5rem;
                display: grid;
                place-items: center;
                border-radius: 999px;
                margin-bottom: 1rem;
                background: linear-gradient(135deg, #2DB5A0, #5CC8B5);
                font-size: 2.4rem;
                line-height: 1;
                border: none;
                box-shadow: 0 6px 16px rgba(45, 181, 160, 0.35);
                transition: transform 300ms cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 300ms ease;
            }

            .analysis-card:hover .analysis-badge {
                transform: scale(1.15);
                box-shadow: 0 0 0 5px rgba(45, 181, 160, 0.2), 0 10px 24px rgba(45, 181, 160, 0.4);
            }
                line-height: 1;
                box-shadow: 0 4px 12px rgba(47, 128, 237, 0.3), inset 0 2px 4px rgba(255, 255, 255, 0.2);
                transition: transform 300ms cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 300ms ease;
            }

            .info-step {
                width: 5.5rem;
                height: 5.5rem;
                display: grid;
                place-items: center;
                border-radius: 999px;
                margin-bottom: 0.85rem;
                background: linear-gradient(135deg, #2DB5A0, #5CC8B5);
                font-size: 3rem;
                line-height: 1;
                border: none;
                position: relative;
                z-index: 2;
                box-shadow: 0 6px 16px rgba(45, 181, 160, 0.35);
                transition: transform 300ms cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 300ms ease;
            }
            
            .info-card:hover .info-step {
                transform: scale(1.15);
                box-shadow: 0 0 0 5px rgba(45, 181, 160, 0.2), 0 10px 24px rgba(45, 181, 160, 0.4);
            }
            
            .info-card:hover .info-step {
                box-shadow: 0 0 0 3px rgba(47, 128, 237, 0.2), 0 6px 12px rgba(47, 128, 237, 0.3);
            }

            .info-grid {
                position: relative;
            }

            .info-grid::before {
                content: '';
                position: absolute;
                top: 3.5rem;
                left: 10%;
                right: 10%;
                height: 2px;
                background: repeating-linear-gradient(90deg, var(--border) 0, var(--border) 6px, transparent 6px, transparent 12px);
                z-index: 0;
            }

            .info-card h3,
            .analysis-card h3 {
                margin: 0 0 0.5rem 0;
                font-size: 1.2rem;
                font-weight: 800;
                line-height: 1.3;
                text-align: center;
                color: var(--text);
            }

            .info-card p,
            .analysis-card p {
                margin: 0;
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.5;
                text-align: center;
            }

            .soft-card {
                height: 100%;
                min-height: 140px;
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1.15rem 1.2rem;
                background: linear-gradient(180deg, var(--panel), var(--panel-soft));
                box-shadow: var(--card-shadow);
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }

            /* Equal-height metric cards in Streamlit columns */
            [data-testid="stHorizontalBlock"] {
                align-items: stretch;
            }
            [data-testid="stHorizontalBlock"] [data-testid="stColumn"] > div {
                height: 100%;
            }
            [data-testid="stHorizontalBlock"] [data-testid="stColumn"] > div > div {
                height: 100%;
            }

            .metric-label {
                margin: 0 0 0.35rem 0;
                color: var(--muted);
                font-size: 1rem;
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
                font-size: 1rem;
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
                border: none;
                background: linear-gradient(135deg, #2DB5A0, #5CC8B5);
                color: #FFFFFF;
                font-weight: 750;
                min-height: 2.9rem;
                box-shadow: 0 6px 18px rgba(45, 181, 160, 0.25);
                transition: transform 160ms ease, filter 160ms ease, box-shadow 160ms ease;
            }

            .stButton > button:hover {
                filter: brightness(1.06);
                transform: translateY(-2px);
                box-shadow: 0 10px 24px rgba(45, 181, 160, 0.35);
                color: #FFFFFF;
            }

            div[data-testid="stTextInput"] input,
            div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
                border-radius: 13px;
                border-color: var(--border);
                background-color: var(--input-bg);
                color: var(--text);
                min-height: 3.2rem;
                font-size: 1.25rem;
                transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
            }

            [data-testid="stWidgetLabel"] label,
            [data-testid="stWidgetLabel"] p {
                color: var(--text) !important;
                font-size: 1.2rem !important;
                font-weight: 700 !important;
            }

            .st-key-control_card p,
            .st-key-control_card span,
            .st-key-control_card div {
                font-size: 1.1rem;
                color: var(--text);
            }

            div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
                border-color: var(--accent);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
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
                max-width: 300px;
                margin: 0;
                padding: 0.3rem 0.5rem;
                border: 1px solid var(--border);
                border-radius: 9px;
                background: rgba(255, 255, 255, 0.85);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                backdrop-filter: blur(8px);
            }

            .st-key-theme_menu {
                margin: 1.5rem 0 0.5rem 0;
            }

            .st-key-theme_menu .stColumn:last-child {
                display: flex;
                justify-content: flex-end;
            }

            .st-key-theme_picker div[role="radiogroup"] {
                gap: 0.2rem;
                justify-content: center;
            }

            .st-key-theme_picker label {
                margin: 0;
                padding: 0.25rem 0.5rem;
                border-radius: 9px;
                font-size: 1.15rem;
                cursor: pointer;
                transition: background 0.2s ease, transform 0.15s ease;
            }

            .st-key-theme_picker label:hover {
                background: rgba(56, 189, 248, 0.12);
                transform: scale(1.1);
            }

            .st-key-theme_picker label p {
                font-size: 1.15rem;
                line-height: 1;
                margin: 0;
            }

            .st-key-theme_picker button[kind="resetButton"],
            .st-key-theme_picker [data-testid="stClearButton"],
            .st-key-theme_picker div[role="radiogroup"] + div {
                display: none !important;
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

            .st-key-map_panel div[data-testid="stVerticalBlockBorderWrapper"],
            .st-key-score_panel div[data-testid="stVerticalBlockBorderWrapper"],
            .st-key-chart_panel div[data-testid="stVerticalBlockBorderWrapper"],
            .st-key-top_panel div[data-testid="stVerticalBlockBorderWrapper"] {
                padding: 0.15rem;
            }

            /* --- Enhanced Control Card --- */
            .st-key-control_card div[data-testid="stVerticalBlockBorderWrapper"] {
                padding: 1.5rem;
                background: linear-gradient(180deg, var(--panel-soft), var(--panel));
                border: 1px solid var(--border);
                border-radius: 18px;
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.08), 0 2px 6px rgba(0, 0, 0, 0.04), inset 0 1px 0 rgba(255, 255, 255, 0.05);
            }

            /* Spacing and visual separation between Ciudad and Actividad/Modo */
            .st-key-control_card div[data-testid="stVerticalBlock"] > div:first-child {
                padding-bottom: 1.2rem;
                margin-bottom: 0.5rem;
                border-bottom: 1px dashed var(--border);
            }

            /* Reduce gap inside the form container slightly for better cohesion */
            .st-key-control_card div[data-testid="stVerticalBlock"] {
                gap: 1rem;
            }

            /* Better button prominence inside control card */
            .st-key-control_card .stButton > button {
                margin-top: 1rem;
                background: linear-gradient(135deg, #2DB5A0, #5CC8B5);
                border: none;
                border-radius: 999px;
                color: #FFFFFF;
                font-size: 1.05rem;
                font-weight: 800;
                letter-spacing: 0.3px;
                min-height: 3.2rem;
                box-shadow: 0 8px 24px rgba(45, 181, 160, 0.3);
                transition: transform 200ms cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 200ms ease, filter 200ms ease;
            }

            .st-key-control_card .stButton > button:hover {
                transform: translateY(-3px) scale(1.01);
                box-shadow: 0 14px 32px rgba(45, 181, 160, 0.4);
                filter: brightness(1.08);
            }

            .st-key-control_card .stButton > button:hover {
                transform: translateY(-3px) scale(1.01);
                box-shadow: 0 14px 32px rgba(47, 128, 237, 0.35), inset 0 2px 4px rgba(255, 255, 255, 0.25);
                filter: brightness(1.08);
            }

            .st-key-score_panel div[data-testid="stVerticalBlockBorderWrapper"],
            .st-key-chart_panel div[data-testid="stVerticalBlockBorderWrapper"] {
                height: 100%;
            }

            /* Align score chart with climate chart (offset for tab bar) */
            .st-key-score_panel .stPlotlyChart {
                padding-top: 49px;
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
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
                
                .info-grid::before {
                    display: none; /* Hide connector line on smaller screens when wrapping */
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

    if page_bg_base64:
        if theme == "Oscuro":
            page_bg_image = (
                f'linear-gradient(180deg, rgba(7,17,31,0.85) 0%, rgba(11,18,32,0.90) 100%), '
                f'url("data:image/jpeg;base64,{page_bg_base64}")'
            )
        else:
            page_bg_image = (
                f'linear-gradient(180deg, rgba(255,253,248,0.35) 0%, rgba(255,248,236,0.40) 100%), '
                f'url("data:image/jpeg;base64,{page_bg_base64}")'
            )
    else:
        page_bg_image = "none"

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
        "__PAGE_BG_IMAGE__": page_bg_image,
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
    with st.container(key="theme_picker"):
        selected_theme_icon = st.radio(
            "Tema visual",
            ["☀️", "🌙"],
            index=0,
            key="visual_theme_icon",
            horizontal=True,
            label_visibility="collapsed",
        )
    return "Claro" if selected_theme_icon == "☀️" else "Oscuro"


def render_theme_menu():
    with st.container(key="theme_menu"):
        _, menu_col = st.columns([0.88, 0.12])
        with menu_col:
            render_theme_selector()


def get_theme_mode():
    selected_theme_icon = st.session_state.get("visual_theme_icon", "☀️")
    return "Claro" if selected_theme_icon == "☀️" else "Oscuro"


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
                <div class="hero-kicker">
                    <span class="kicker-pill">CLIMA</span>
                    <span class="kicker-pill">IA</span>
                    <span class="kicker-pill">SEGURIDAD</span>
                </div>
                <h1>¿Salimos hoy?</h1>
                <h2>El mejor momento para tu actividad, sin adivinar.</h2>
                <p>
                    Consulta el clima en tiempo real, elige una actividad
                    y recibe una recomendación clara sobre cuándo salir,
                    basada en análisis climático y aprendizaje automático.
                </p>
                <div class="hero-badges">
                    <span class="hero-badge">🌤️ Clima en tiempo real</span>
                    <span class="hero-badge">💎 Recomendación inteligente</span>
                    <span class="hero-badge">🕐 Hora óptima</span>
                </div>
                <div class="hero-btn-wrapper">
                    <a href="#elegir-actividad" class="hero-cta-btn">Elegir actividad</a>
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
            f'style=\'background-image: url("data:image/jpeg;base64,{hero_image_base64}");\''
            "></div>"
        ),
        unsafe_allow_html=True,
    )


def render_how_it_works():
    render_section_header(
        "Como funciona SalimosHoy?",
        "Una lectura sencilla del clima para tomar mejores decisiones antes de salir.",
    )
    st.markdown(
        """
        <div class="info-grid">
            <div class="info-card">
                <div class="analysis-badge">🌤️</div>
                <h3>Consulta clima actualizado</h3>
                <p>Obtiene datos horarios recientes de temperatura, lluvia, viento y humedad.</p>
            </div>
            <div class="info-card">
                <div class="analysis-badge">🌦️</div>
                <h3>Predice el tipo de clima</h3>
                <p>Usa ML para clasificar las condiciones meteorologicas en categorias de clima.</p>
            </div>
            <div class="info-card">
                <div class="analysis-badge">🏃</div>
                <h3>Evalua la actividad elegida</h3>
                <p>Compara las condiciones disponibles con lo que necesita cada actividad.</p>
            </div>
            <div class="info-card">
                <div class="analysis-badge">🕐</div>
                <h3>Clasifica la mejor hora</h3>
                <p>Usa ML y analisis climatico para dar una recomendacion final mas realista.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ml_section():
    render_section_header(
        "Machine Learning en SalimosHoy?",
        "Un modelo entrenado con datos reales para darte recomendaciones mas precisas.",
    )
    st.markdown(
        """
        <div class="info-grid" style="grid-template-columns: repeat(3, minmax(0, 1fr));">
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">🌲</div>
                <h3>Random Forest</h3>
                <p>Modelo de clasificacion basado en multiples arboles de decision que votan en conjunto para una prediccion robusta.</p>
            </div>
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">🎯</div>
                <h3>96.6% Accuracy</h3>
                <p>El modelo acierta en la gran mayoria de casos, especialmente identificando condiciones malas con 99% de precision.</p>
            </div>
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">📊</div>
                <h3>4 Categorias</h3>
                <p>Clasifica cada hora como <strong>excelente</strong>, <strong>bueno</strong>, <strong>regular</strong> o <strong>malo</strong> segun las condiciones.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if st.button(
            "🔍 Detalles tecnicos del modelo de Machine Learning",
            use_container_width=True,
            key="go_ml_details",
        ):
            st.session_state["page"] = "ml_details"
            st.rerun()


def render_ml_details_page():
    _, btn_col = st.columns([0.85, 0.15])
    with btn_col:
        if st.button("← Volver", key="back_from_ml"):
            st.session_state["page"] = "main"
            st.rerun()

    render_section_header(
        "Machine Learning en SalimosHoy?",
        "Detalles tecnicos del modelo que impulsa las recomendaciones.",
    )

    st.markdown(
        """
        <div class="info-grid" style="grid-template-columns: repeat(3, minmax(0, 1fr));">
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">🌲</div>
                <h3>Random Forest</h3>
                <p>Modelo de clasificacion basado en multiples arboles de decision que votan en conjunto para una prediccion robusta.</p>
            </div>
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">🎯</div>
                <h3>96.6% Accuracy</h3>
                <p>El modelo acierta en la gran mayoria de casos, especialmente identificando condiciones malas con 99% de precision.</p>
            </div>
            <div class="info-card" style="text-align:center;">
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">📊</div>
                <h3>4 Categorias</h3>
                <p>Clasifica cada hora como <strong>excelente</strong>, <strong>bueno</strong>, <strong>regular</strong> o <strong>malo</strong> segun las condiciones.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.subheader("Que es Random Forest?")
    st.markdown(
        """
        Random Forest es un algoritmo de **Machine Learning supervisado** que crea
        cientos de arboles de decision, cada uno entrenado con una muestra aleatoria
        de los datos. Cuando necesita clasificar una hora, **todos los arboles votan**
        y la clase mas votada gana. Esto lo hace muy robusto y resistente al overfitting.
        """
    )

    st.markdown("---")

    st.subheader("Variables de entrada (Features)")
    st.markdown(
        """
        El modelo recibe **13 variables** para cada hora del dia:

        | Variable | Descripcion | Tipo |
        |---|---|---|
        | `temperature_2m` | Temperatura a 2 metros | Meteorologica |
        | `apparent_temperature` | Sensacion termica | Meteorologica |
        | `relative_humidity_2m` | Humedad relativa (%) | Meteorologica |
        | `precipitation_probability` | Probabilidad de lluvia (%) | Meteorologica |
        | `wind_speed_10m` | Velocidad del viento (km/h) | Meteorologica |
        | `wind_gusts_10m` | Rachas de viento (km/h) | Meteorologica |
        | `cloud_cover` | Cobertura de nubes (%) | Meteorologica |
        | `uv_index` | Indice de radiacion UV | Meteorologica |
        | `hour` | Hora del dia (0-23) | Temporal |
        | `activity` | Actividad seleccionada | Contexto |
        | `continent` | Continente de la ciudad | Geografico |
        | `climate_group` | Grupo climatico | Geografico |
        | `season_block` | Estacion del ano | Temporal |
        """
    )

    st.markdown("---")

    st.subheader("Rendimiento del modelo")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Accuracy (Train)", "97.39%")
    with metric_cols[1]:
        st.metric("Accuracy (Test)", "96.59%")
    with metric_cols[2]:
        st.metric("Mejor clase", "Malo: 99%")
    with metric_cols[3]:
        st.metric("Overfitting", "< 1%", delta="-0.8%", delta_color="normal")

    st.markdown(
        """
        **Rendimiento por clase (test):**

        | Clase | Precision | Recall | F1-Score | Soporte |
        |---|---|---|---|---|
        | Excelente | 0.93 | 0.93 | 0.93 | — |
        | Bueno | 0.93 | 0.93 | 0.93 | — |
        | Regular | 0.97 | 0.97 | 0.97 | — |
        | Malo | 0.99 | 0.99 | 0.99 | — |
        """
    )

    st.markdown("---")

    st.subheader("Comparativa de modelos")
    st.markdown(
        """
        Se evaluaron 3 modelos diferentes para encontrar el mejor clasificador:

        | Modelo | Accuracy Train | Accuracy Test | Resultado |
        |---|---|---|---|
        | ✅ **Random Forest** | 97.39% | 96.59% | **Ganador** |
        | Decision Tree | 89.50% | 89.43% | Aceptable |
        | Logistic Regression | 51.55% | 50.17% | Insuficiente |

        **Random Forest** fue el claro ganador, con la mejor accuracy y el menor
        sobreajuste (diferencia train-test de solo 0.8%).

        **Logistic Regression** no fue capaz de separar las clases correctamente,
        lo que confirma que el problema no es linealmente separable.
        """
    )

    st.markdown("---")

    st.subheader("Datos de entrenamiento")
    st.markdown(
        """
        El dataset fue generado con **datos meteorologicos reales** de la API de
        [Open-Meteo](https://open-meteo.com/) para multiples ciudades y continentes.

        **Proceso de creacion del dataset:**
        1. Se recopilaron datos horarios de clima para ciudades representativas de cada continente
        2. Se calcularon puntuaciones de actividad usando formulas de distancia al clima ideal
        3. Se aplicaron filtros climaticos (lluvia intensa, viento extremo, temperaturas adversas)
        4. Se generaron etiquetas finales: excelente, bueno, regular, malo

        Estos filtros garantizan que el modelo aprende a penalizar condiciones
        adversas, como lluvia fuerte para actividades al aire libre o temperaturas
        extremas para deporte.
        """
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
                <div class="analysis-badge">🌡️</div>
                <h3>Temperatura</h3>
                <p>Calor o frio del ambiente.</p>
            </div>
            <div class="analysis-card">
                <div class="analysis-badge">🌧️</div>
                <h3>Lluvia</h3>
                <p>Riesgo de precipitacion.</p>
            </div>
            <div class="analysis-card">
                <div class="analysis-badge">💨</div>
                <h3>Viento</h3>
                <p>Velocidad del aire.</p>
            </div>
            <div class="analysis-card">
                <div class="analysis-badge">💧</div>
                <h3>Humedad</h3>
                <p>Sensacion de incomodidad.</p>
            </div>
            <div class="analysis-card">
                <div class="analysis-badge">☀️</div>
                <h3>Indice UV</h3>
                <p>Exposicion solar.</p>
            </div>
            <div class="analysis-card">
                <div class="analysis-badge">🕐</div>
                <h3>Hora del dia</h3>
                <p>Momento disponible para salir.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if st.button(
            "🔍 Detalles tecnicos del modelo de Machine Learning",
            use_container_width=True,
            key="hero_ml_details",
        ):
            st.session_state["page"] = "ml_details"
            st.rerun()


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
    st.markdown('<div id="elegir-actividad"></div>', unsafe_allow_html=True)
    render_section_header(
        "Prepara tu consulta",
        "Selecciona ciudad, actividad y modo de decision para generar una recomendacion.",
    )
    with st.container(border=True, key="control_card"):
        # Dropdown de ciudades
        selected_city_name = st.selectbox(
            "Ciudad",
            options=sorted(CITIES),
            index=None,
            placeholder="Selecciona una ciudad",
        )
        
        # Si se selecciona una ciudad, buscarla y guardarla en session_state
        if selected_city_name:
            # Solo buscar si cambio la ciudad
            if st.session_state.get("last_selected_city") != selected_city_name:
                with st.spinner(f"Buscando {selected_city_name}..."):
                    results = search_city(selected_city_name)
                st.session_state.city_results = results
                st.session_state.selected_city = results[0] if results else None
                st.session_state.last_selected_city = selected_city_name
                st.session_state.analysis = None
                if not results:
                    st.warning(f"No se encontraron datos para {selected_city_name}.")

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

    map_col, activity_img_col = st.columns([1, 1])
    with map_col:
        with st.container(border=True, key="map_panel"):
            render_section_header("Ubicacion analizada")
            lat = city.get("latitude")
            lon = city.get("longitude")
            name = city.get("name", city_name)
            country = city.get("country", "")
            query = f"{name}, {country}".replace(" ", "+")
            st.markdown(
                f'<iframe '
                f'src="https://maps.google.com/maps?q={query}&ll={lat},{lon}&z=13&output=embed" '
                f'width="100%" height="420" style="border:0; border-radius:12px;" '
                f'allowfullscreen loading="lazy">'
                f'</iframe>',
                unsafe_allow_html=True,
            )

    with activity_img_col:
        with st.container(border=True, key="activity_img_panel"):
            render_section_header(activity)
            activity_img_path = ACTIVITY_IMAGES.get(activity)
            if activity_img_path and activity_img_path.exists():
                activity_img_b64 = image_to_base64(activity_img_path)
                if activity_img_b64:
                    st.markdown(
                        f'<img src="data:image/jpeg;base64,{activity_img_b64}" '
                        f'style="width:100%; height:420px; object-fit:cover; border-radius:12px;" '
                        f'alt="{activity}"/>',
                        unsafe_allow_html=True,
                    )

    score_col, climate_col = st.columns(2)
    with score_col:
        with st.container(border=True, key="score_panel"):
            render_section_header("Score por hora")
            st.plotly_chart(
                create_score_chart(df_valid_hours, theme_template=theme_template),
                use_container_width=True,
            )

    with climate_col:
        with st.container(border=True, key="chart_panel"):
            render_section_header("Clima por variable")
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

   
init_session_state()
visual_theme = get_theme_mode()
inject_styles(visual_theme)
render_theme_menu()

if st.session_state.get("page") == "ml_details":
    render_ml_details_page()
else:
    render_hero_banner()
    render_hero()
    render_how_it_works()
    render_analysis_scope()
    render_search_area()
    render_results(visual_theme)
