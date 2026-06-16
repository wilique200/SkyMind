# app.py — SkyMind Weather Intelligence
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from src.model_loader        import load_all_models
from src.weather_api         import fetch_forecast, check_soil_data_quality
from src.geocoding           import geocode_city, reverse_geocode, find_nearest_training_city
from src.feature_engineering import engineer_features
from src.inference           import run_all_models, run_forecast
from src.edge_cases          import (
    WeatherWarnings, handle_location_fallback,
    check_location_confidence, handle_missing_soil,
    handle_nighttime_uv, validate_model_output, assess_api_response
)
from src.descriptions        import (
    generate_main_description, generate_uv_description,
    generate_soil_description, generate_heat_stress_description,
    generate_thunderstorm_description, generate_visibility_description
)
from utils.constants         import (
    WEATHER_THEMES, WMO_EMOJIS, HEAT_STRESS_LABELS,
    UV_LEVELS, OUTDOOR_SCORE_LABELS
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyMind — AI Weather Intelligence",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═════════════════════════════════════════════════════════════════════════════
# SCENE DEFINITIONS  (gradient bg + SVG illustration + color tokens)
# ═════════════════════════════════════════════════════════════════════════════
SCENES = {
    "Clear": {
        "day": {
            "bg":     "linear-gradient(160deg, #0ea5e9 0%, #38bdf8 40%, #f59e0b 80%, #f97316 100%)",
            "scene":  """
                <circle cx="72%" cy="22%" r="70"  fill="#fde68a" opacity="0.95"/>
                <circle cx="72%" cy="22%" r="95"  fill="#fde68a" opacity="0.22"/>
                <circle cx="72%" cy="22%" r="120" fill="#fde68a" opacity="0.09"/>
                <g opacity="0.5">
                  <rect x="55"  y="420" width="14" height="160" rx="7" fill="#78350f"/>
                  <ellipse cx="62" cy="415" rx="55" ry="28" fill="#15803d" transform="rotate(-20,62,415)"/>
                  <ellipse cx="62" cy="408" rx="50" ry="24" fill="#16a34a" transform="rotate(15,62,408)"/>
                  <ellipse cx="62" cy="402" rx="44" ry="21" fill="#15803d" transform="rotate(-4,62,402)"/>
                </g>
                <g opacity="0.38">
                  <rect x="310" y="450" width="12" height="130" rx="6" fill="#78350f"/>
                  <ellipse cx="316" cy="444" rx="48" ry="25" fill="#15803d" transform="rotate(25,316,444)"/>
                  <ellipse cx="316" cy="438" rx="42" ry="21" fill="#16a34a" transform="rotate(-10,316,438)"/>
                </g>
                <ellipse cx="50%" cy="108%" rx="62%" ry="26%" fill="#f59e0b" opacity="0.22"/>
            """,
            "text":   "#1e3a5f",
            "card":   "rgba(255,255,255,0.18)",
            "accent": "#f59e0b",
        },
        "night": {
            "bg":     "linear-gradient(160deg, #0f172a 0%, #1e293b 55%, #1e3a5f 100%)",
            "scene":  """
                <circle cx="75%" cy="20%" r="55" fill="#e2e8f0" opacity="0.88"/>
                <circle cx="78%" cy="17%" r="46" fill="#1e293b" opacity="0.9"/>
                <circle cx="10%" cy="8%"  r="2"   fill="white" opacity="0.9"/>
                <circle cx="25%" cy="5%"  r="1.5" fill="white" opacity="0.7"/>
                <circle cx="40%" cy="12%" r="2.5" fill="white" opacity="0.8"/>
                <circle cx="55%" cy="6%"  r="1.5" fill="white" opacity="0.6"/>
                <circle cx="20%" cy="18%" r="1"   fill="white" opacity="0.5"/>
                <circle cx="35%" cy="3%"  r="2"   fill="white" opacity="0.75"/>
                <circle cx="62%" cy="14%" r="1.5" fill="white" opacity="0.65"/>
                <circle cx="85%" cy="8%"  r="2"   fill="white" opacity="0.5"/>
            """,
            "text":   "#e2e8f0",
            "card":   "rgba(255,255,255,0.08)",
            "accent": "#60a5fa",
        },
    },
    "Cloudy": {
        "day": {
            "bg":     "linear-gradient(160deg, #64748b 0%, #94a3b8 55%, #cbd5e1 100%)",
            "scene":  """
                <ellipse cx="28%" cy="24%" rx="125" ry="58" fill="white" opacity="0.52"/>
                <ellipse cx="22%" cy="28%" rx="95"  ry="48" fill="white" opacity="0.62"/>
                <ellipse cx="40%" cy="20%" rx="85"  ry="42" fill="white" opacity="0.48"/>
                <ellipse cx="72%" cy="17%" rx="135" ry="62" fill="white" opacity="0.42"/>
                <ellipse cx="67%" cy="22%" rx="105" ry="52" fill="white" opacity="0.52"/>
                <ellipse cx="80%" cy="14%" rx="88"  ry="44" fill="white" opacity="0.38"/>
                <circle  cx="55%" cy="30%" r="52"   fill="#fde68a" opacity="0.28"/>
            """,
            "text":   "#1e293b",
            "card":   "rgba(255,255,255,0.22)",
            "accent": "#64748b",
        },
        "night": {
            "bg":     "linear-gradient(160deg, #0f172a 0%, #1e293b 60%, #334155 100%)",
            "scene":  """
                <ellipse cx="28%" cy="24%" rx="125" ry="58" fill="#334155" opacity="0.82"/>
                <ellipse cx="72%" cy="17%" rx="135" ry="62" fill="#334155" opacity="0.72"/>
                <circle  cx="10%" cy="8%"  r="1.5" fill="white" opacity="0.4"/>
                <circle  cx="55%" cy="6%"  r="1"   fill="white" opacity="0.3"/>
                <circle  cx="85%" cy="10%" r="2"   fill="white" opacity="0.35"/>
            """,
            "text":   "#e2e8f0",
            "card":   "rgba(255,255,255,0.08)",
            "accent": "#94a3b8",
        },
    },
    "Rain": {
        "day": {
            "bg":     "linear-gradient(160deg, #1e3a5f 0%, #1e40af 45%, #1d4ed8 72%, #312e81 100%)",
            "scene":  """
                <ellipse cx="28%" cy="20%" rx="145" ry="68" fill="#1e3a5f" opacity="0.9"/>
                <ellipse cx="20%" cy="25%" rx="105" ry="55" fill="#1e293b" opacity="0.95"/>
                <ellipse cx="44%" cy="16%" rx="115" ry="58" fill="#1e3a5f" opacity="0.85"/>
                <ellipse cx="74%" cy="14%" rx="155" ry="72" fill="#1e3a5f" opacity="0.9"/>
                <ellipse cx="80%" cy="20%" rx="112" ry="60" fill="#1e293b" opacity="0.9"/>
                <path d="M 195 525 Q 195 348 382 338 Q 568 348 568 525"
                      fill="#1d4ed8" opacity="0.82"/>
                <path d="M 195 525 Q 195 348 382 338 Q 568 348 568 525"
                      fill="none" stroke="#3b82f6" stroke-width="3"/>
                <line x1="382" y1="338" x2="382" y2="595"
                      stroke="#1e293b" stroke-width="9" stroke-linecap="round"/>
                <path d="M 382 595 Q 424 616 404 646"
                      fill="none" stroke="#1e293b" stroke-width="9" stroke-linecap="round"/>
            """,
            "text":   "#e2e8f0",
            "card":   "rgba(30,58,138,0.32)",
            "accent": "#60a5fa",
        },
        "night": {
            "bg":     "linear-gradient(160deg, #0f172a 0%, #1e293b 55%, #1e3a5f 100%)",
            "scene":  """
                <ellipse cx="28%" cy="20%" rx="145" ry="68" fill="#0f172a" opacity="0.96"/>
                <ellipse cx="74%" cy="14%" rx="155" ry="72" fill="#0f172a" opacity="0.96"/>
                <path d="M 195 525 Q 195 348 382 338 Q 568 348 568 525"
                      fill="#1e3a5f" opacity="0.75"/>
                <line x1="382" y1="338" x2="382" y2="595"
                      stroke="#0f172a" stroke-width="9" stroke-linecap="round"/>
            """,
            "text":   "#e2e8f0",
            "card":   "rgba(255,255,255,0.07)",
            "accent": "#60a5fa",
        },
    },
    "Thunderstorm": {
        "day": {
            "bg":     "linear-gradient(160deg, #0f172a 0%, #1e1b4b 55%, #312e81 100%)",
            "scene":  """
                <ellipse cx="40%" cy="17%" rx="165" ry="78" fill="#1e1b4b" opacity="0.95"/>
                <ellipse cx="76%" cy="12%" rx="155" ry="72" fill="#0f172a" opacity="0.95"/>
                <path d="M 355 175 L 315 308 L 352 308 L 295 452 L 412 288 L 365 288 L 422 175 Z"
                      fill="#fde68a" opacity="0.95"/>
                <path d="M 355 175 L 315 308 L 352 308 L 295 452 L 412 288 L 365 288 L 422 175 Z"
                      fill="#fbbf24" opacity="0.45" filter="blur(10px)"/>
            """,
            "text":   "#e2e8f0",
            "card":   "rgba(49,46,129,0.32)",
            "accent": "#fbbf24",
        },
        "night": {
            "bg":     "linear-gradient(160deg, #0f172a 0%, #1e1b4b 100%)",
            "scene":  """
                <ellipse cx="40%" cy="17%" rx="165" ry="78" fill="#0f172a" opacity="0.98"/>
                <ellipse cx="76%" cy="12%" rx="155" ry="72" fill="#0f172a" opacity="0.98"/>
                <path d="M 355 175 L 315 308 L 352 308 L 295 452 L 412 288 L 365 288 L 422 175 Z"
                      fill="#fde68a" opacity="0.9"/>
            """,
            "text":   "#e2e8f0",
            "card":   "rgba(255,255,255,0.07)",
            "accent": "#fbbf24",
        },
    },
    "Snow": {
        "day": {
            "bg":     "linear-gradient(160deg, #bfdbfe 0%, #dbeafe 42%, #e2e8f0 82%, #f1f5f9 100%)",
            "scene":  """
                <ellipse cx="50%" cy="20%" rx="205" ry="82" fill="white" opacity="0.58"/>
                <text x="14%" y="35%" font-size="30" fill="white" opacity="0.65">❄</text>
                <text x="74%" y="27%" font-size="22" fill="white" opacity="0.55">❄</text>
                <text x="44%" y="14%" font-size="17" fill="white" opacity="0.5">❄</text>
                <text x="82%" y="44%" font-size="25" fill="white" opacity="0.48">❄</text>
                <ellipse cx="20%"  cy="106%" rx="30%" ry="30%" fill="white" opacity="0.38"/>
                <ellipse cx="80%"  cy="109%" rx="28%" ry="28%" fill="white" opacity="0.32"/>
                <ellipse cx="50%"  cy="111%" rx="36%" ry="30%" fill="white" opacity="0.38"/>
            """,
            "text":   "#1e3a5f",
            "card":   "rgba(255,255,255,0.28)",
            "accent": "#3b82f6",
        },
        "night": {
            "bg":     "linear-gradient(160deg, #0f172a 0%, #1e293b 62%, #334155 100%)",
            "scene":  """
                <circle cx="75%" cy="20%" r="52" fill="#e2e8f0" opacity="0.78"/>
                <circle cx="78%" cy="17%" r="43" fill="#1e293b" opacity="0.88"/>
                <text x="14%" y="34%" font-size="22" fill="white" opacity="0.45">❄</text>
                <text x="74%" y="27%" font-size="16" fill="white" opacity="0.38">❄</text>
            """,
            "text":   "#e2e8f0",
            "card":   "rgba(255,255,255,0.08)",
            "accent": "#93c5fd",
        },
    },
    "Fog": {
        "day": {
            "bg":     "linear-gradient(160deg, #94a3b8 0%, #cbd5e1 52%, #e2e8f0 100%)",
            "scene":  """
                <rect x="0" y="28%" width="100%" height="16%" fill="white" opacity="0.32"/>
                <rect x="0" y="47%" width="100%" height="13%" fill="white" opacity="0.27"/>
                <rect x="0" y="62%" width="100%" height="11%" fill="white" opacity="0.22"/>
                <circle cx="50%" cy="18%" r="62" fill="#fde68a" opacity="0.18"/>
            """,
            "text":   "#1e293b",
            "card":   "rgba(255,255,255,0.25)",
            "accent": "#64748b",
        },
        "night": {
            "bg":     "linear-gradient(160deg, #0f172a 0%, #1e293b 100%)",
            "scene":  """
                <rect x="0" y="28%" width="100%" height="16%" fill="#334155" opacity="0.48"/>
                <rect x="0" y="47%" width="100%" height="13%" fill="#334155" opacity="0.38"/>
            """,
            "text":   "#e2e8f0",
            "card":   "rgba(255,255,255,0.07)",
            "accent": "#94a3b8",
        },
    },
}

def get_scene(condition, is_day):
    tod = "day" if is_day else "night"
    return SCENES.get(condition, SCENES["Cloudy"])[tod]


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═════════════════════════════════════════════════════════════════════════════
def inject_base_css(scene):
    bg     = scene["bg"]
    text   = scene["text"]
    card   = scene["card"]
    accent = scene["accent"]

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, .stApp {{
        background: {bg} !important;
        font-family: 'Inter', sans-serif;
        color: {text} !important;
        min-height: 100vh;
    }}
    .stApp > header {{ background: transparent !important; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{
        padding-top: 0 !important;
        max-width: 1140px;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }}

    .scene-canvas {{
        position: fixed; top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none; z-index: 0; overflow: hidden;
    }}

    .sky-card {{
        background: {card};
        backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        border-radius: 24px; padding: 1.4rem 1.6rem;
        border: 1px solid rgba(255,255,255,0.22);
        box-shadow: 0 8px 40px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.24);
        color: {text}; margin-bottom: 1rem;
        position: relative; z-index: 1;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .sky-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.28);
    }}

    .metric-card {{
        background: {card};
        backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        border-radius: 20px; padding: 1.1rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 4px 24px rgba(0,0,0,0.14), inset 0 1px 0 rgba(255,255,255,0.2);
        text-align: center; color: {text}; height: 100%;
        position: relative; z-index: 1;
        transition: transform 0.18s ease;
    }}
    .metric-card:hover {{ transform: translateY(-2px); }}
    .metric-label {{
        font-size: 0.67rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.1em; opacity: 0.58; margin-bottom: 0.4rem;
    }}
    .metric-value {{ font-size: 1.52rem; font-weight: 800; line-height: 1.1; }}
    .metric-sub   {{ font-size: 0.74rem; opacity: 0.62; margin-top: 0.25rem; }}

    .hero-wrap    {{ position: relative; z-index: 1; padding: 1.8rem 0 0.4rem; }}
    .hero-location {{
        font-size: 0.82rem; font-weight: 600; opacity: 0.68;
        letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.12rem;
    }}
    .hero-date    {{ font-size: 0.76rem; opacity: 0.48; margin-bottom: 1.1rem; }}
    .hero-temp-row {{ display: flex; align-items: flex-end; gap: 0.25rem; line-height: 1; margin-bottom: 0.25rem; }}
    .hero-icon    {{ font-size: 5.5rem; line-height: 1; }}
    .hero-temp    {{
        font-size: 7rem; font-weight: 900; letter-spacing: -0.04em;
        line-height: 1; color: {text};
        text-shadow: 0 4px 24px rgba(0,0,0,0.16);
    }}
    .hero-unit    {{ font-size: 2.2rem; font-weight: 300; opacity: 0.68; padding-bottom: 0.85rem; }}
    .hero-condition {{ font-size: 1.45rem; font-weight: 300; opacity: 0.86; letter-spacing: 0.02em; }}
    .hero-feels   {{ font-size: 0.88rem; opacity: 0.58; margin-top: 0.35rem; }}
    .accent-line  {{
        width: 34px; height: 3px; border-radius: 999px;
        background: {accent}; margin: 0.45rem 0 0.85rem; opacity: 0.82;
    }}

    .sky-progress-bg {{
        background: rgba(255,255,255,0.14); border-radius: 999px;
        height: 6px; width: 100%; margin-top: 0.45rem; overflow: hidden;
    }}
    .sky-progress-fill {{
        height: 6px; border-radius: 999px;
        transition: width 0.7s cubic-bezier(0.4,0,0.2,1);
    }}

    .forecast-card {{
        background: {card};
        backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        border-radius: 18px; padding: 0.72rem 0.38rem;
        text-align: center; border: 1px solid rgba(255,255,255,0.17);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12); color: {text};
        font-size: 0.79rem; transition: transform 0.15s ease;
    }}
    .forecast-card:hover {{ transform: scale(1.04); }}
    .forecast-temp-hi {{ font-weight: 800; font-size: 0.98rem; }}
    .forecast-temp-lo {{ opacity: 0.54; font-size: 0.8rem; }}

    .tab-strip {{
        display: flex; gap: 0.45rem; overflow-x: auto;
        padding: 0.55rem 0 0.35rem; scrollbar-width: none;
        position: relative; z-index: 1;
    }}
    .tab-strip::-webkit-scrollbar {{ display: none; }}
    .tab-pill {{
        flex-shrink: 0; padding: 0.38rem 1.05rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600;
        border: 1.5px solid rgba(255,255,255,0.22);
        background: rgba(255,255,255,0.08); color: {text};
        opacity: 0.52; white-space: nowrap; transition: all 0.18s ease;
    }}
    .tab-pill.active {{
        background: rgba(255,255,255,0.24); border-color: rgba(255,255,255,0.48);
        opacity: 1; box-shadow: 0 2px 12px rgba(0,0,0,0.14);
    }}

    .add-panel {{
        background: {card};
        backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        border-radius: 20px; padding: 1.15rem 1.4rem;
        border: 1px solid rgba(255,255,255,0.22);
        box-shadow: 0 8px 32px rgba(0,0,0,0.16);
        margin-bottom: 1rem; position: relative; z-index: 1;
    }}

    .section-label {{
        font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.14em; opacity: 0.48;
        margin: 1.35rem 0 0.58rem; position: relative; z-index: 1;
    }}
    .insight-header {{
        font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; opacity: 0.52; margin-bottom: 0.55rem;
    }}
    .badge-pill {{
        display: inline-block; padding: 0.18rem 0.82rem; border-radius: 999px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em; color: white;
    }}

    .sky-warning {{
        background: rgba(251,191,36,0.13); border: 1px solid rgba(251,191,36,0.38);
        border-radius: 14px; padding: 0.58rem 1rem; font-size: 0.82rem;
        color: {text}; margin-bottom: 0.45rem; position: relative; z-index: 1;
    }}
    .sky-error {{
        background: rgba(239,68,68,0.13); border: 1px solid rgba(239,68,68,0.38);
        border-radius: 14px; padding: 0.58rem 1rem; font-size: 0.82rem;
        color: {text}; margin-bottom: 0.45rem; position: relative; z-index: 1;
    }}
    .sky-info {{
        background: rgba(96,165,250,0.11); border: 1px solid rgba(96,165,250,0.28);
        border-radius: 14px; padding: 0.58rem 1rem; font-size: 0.82rem;
        color: {text}; margin-bottom: 0.45rem; position: relative; z-index: 1;
    }}
    .sky-divider {{
        height: 1px; background: rgba(255,255,255,0.13); border: none;
        margin: 1rem 0; position: relative; z-index: 1;
    }}

    .stTextInput > div > div > input {{
        background: rgba(255,255,255,0.12) !important;
        border: 1.5px solid rgba(255,255,255,0.26) !important;
        border-radius: 50px !important; color: {text} !important;
        padding: 0.62rem 1.2rem !important; font-size: 0.94rem !important;
        font-family: 'Inter', sans-serif !important;
    }}
    .stTextInput > div > div > input::placeholder {{ color: rgba(255,255,255,0.42) !important; }}
    .stButton > button {{
        background: rgba(255,255,255,0.13) !important;
        border: 1.5px solid rgba(255,255,255,0.26) !important;
        color: {text} !important; border-radius: 50px !important;
        padding: 0.42rem 1.45rem !important; font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.18s ease !important; letter-spacing: 0.01em !important;
    }}
    .stButton > button:hover {{
        background: rgba(255,255,255,0.22) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.14) !important;
    }}
    div[data-testid="stSelectbox"] > div {{
        background: rgba(255,255,255,0.12) !important;
        border: 1.5px solid rgba(255,255,255,0.26) !important;
        border-radius: 14px !important; color: {text} !important;
    }}

    @keyframes pulse    {{ 0%,100%{{opacity:1}} 50%{{opacity:0.42}} }}
    @keyframes lightning{{ 0%,87%,100%{{opacity:0}} 89%,93%{{opacity:1}} }}
    @keyframes float    {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-8px)}} }}
    @keyframes rain-drop{{
        from{{transform:translateY(-30px);opacity:0}}
        28%{{opacity:1}}
        to{{transform:translateY(100vh);opacity:0}}
    }}
    @keyframes snow-fall{{
        from{{transform:translateY(-20px) rotate(0deg);opacity:0}}
        18%{{opacity:0.8}}
        to{{transform:translateY(100vh) rotate(360deg);opacity:0}}
    }}
    .pulse     {{ animation: pulse     2s ease-in-out infinite; }}
    .lightning {{ animation: lightning 5s ease-in-out infinite; }}
    .floaty    {{ animation: float     4s ease-in-out infinite; }}
    </style>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# WEATHER ANIMATIONS
# ═════════════════════════════════════════════════════════════════════════════
def weather_animation(condition, is_day):
    if condition in ("Rain", "Thunderstorm"):
        drops = "".join([
            f'<div style="position:fixed;left:{np.random.randint(0,100)}vw;'
            f'top:-30px;width:1.5px;height:{np.random.randint(14,26)}px;'
            f'background:rgba(147,197,253,0.5);border-radius:2px;'
            f'animation:rain-drop {np.random.uniform(0.55,1.35):.2f}s linear '
            f'{np.random.uniform(0,3):.2f}s infinite;"></div>'
            for _ in range(30)
        ])
        extra = ('<div class="lightning" style="position:fixed;top:0;left:0;'
                 'width:100%;height:100%;background:rgba(253,230,138,0.11);'
                 'pointer-events:none;z-index:0;"></div>'
                 if condition == "Thunderstorm" else "")
        st.markdown(
            f'<div style="position:fixed;top:0;left:0;width:100%;height:100%;'
            f'pointer-events:none;z-index:0;overflow:hidden">{drops}</div>{extra}',
            unsafe_allow_html=True)

    elif condition == "Snow":
        flakes = "".join([
            f'<div style="position:fixed;left:{np.random.randint(0,100)}vw;'
            f'top:-20px;font-size:{np.random.randint(10,22)}px;opacity:0.72;'
            f'animation:snow-fall {np.random.uniform(3,7):.2f}s linear '
            f'{np.random.uniform(0,5):.2f}s infinite;">❄</div>'
            for _ in range(18)
        ])
        st.markdown(
            f'<div style="position:fixed;top:0;left:0;width:100%;height:100%;'
            f'pointer-events:none;z-index:0;overflow:hidden">{flakes}</div>',
            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def progress_bar(value, max_val, color="#4fc3f7", height=6):
    pct = min(100, max(0, value / max_val * 100))
    return (f'<div class="sky-progress-bg"><div class="sky-progress-fill" '
            f'style="width:{pct:.1f}%;background:{color};height:{height}px;">'
            f'</div></div>')

def metric_card(label, value, sub=None, color=None, bar=None):
    cs  = f"color:{color};" if color else ""
    sh  = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    bh  = bar if bar else ""
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="{cs}">{value}</div>{sh}{bh}</div>')

def badge(label, color):
    return f'<span class="badge-pill" style="background:{color}">{label}</span>'

def render_scene(scene):
    st.markdown(
        f'<div class="scene-canvas"><svg width="100%" height="100%" '
        f'viewBox="0 0 760 600" preserveAspectRatio="xMidYMid slice" '
        f'xmlns="http://www.w3.org/2000/svg">{scene["scene"]}</svg></div>',
        unsafe_allow_html=True)

def hourly_strip(forecast_list, scene):
    icons = {
        "Clear":{"day":"☀️","night":"🌙"},"Cloudy":{"day":"⛅","night":"☁️"},
        "Rain":{"day":"🌧️","night":"🌧️"},"Thunderstorm":{"day":"⛈️","night":"⛈️"},
        "Snow":{"day":"❄️","night":"❄️"},"Fog":{"day":"🌫️","night":"🌫️"},
    }
    st.markdown('<div class="section-label">24-Hour Forecast</div>', unsafe_allow_html=True)
    cols = st.columns(min(12, len(forecast_list[:12])))
    for i, (col, p) in enumerate(zip(cols, forecast_list[:12])):
        time_lbl = "Now" if i == 0 else p["time"].strftime("%H:%M")
        cond  = p.get("weather_condition","Cloudy")
        tod   = "day" if p.get("is_day") else "night"
        icon  = icons.get(cond,{}).get(tod,"⛅")
        temp  = p.get("temperature",0)
        rain  = p.get("rain_probability",0)
        with col:
            st.markdown(f"""
            <div class="forecast-card">
              <div style="font-size:0.68rem;opacity:0.52;font-weight:600">{time_lbl}</div>
              <div style="font-size:1.45rem;margin:0.22rem 0">{icon}</div>
              <div class="forecast-temp-hi">{temp:.0f}°</div>
              <div style="font-size:0.66rem;color:#60a5fa;margin-top:0.12rem">
                {'💧' if rain>30 else ''}{rain:.0f}%</div>
            </div>""", unsafe_allow_html=True)

def daily_strip(forecast_list):
    df = pd.DataFrame([{
        "date":p["time"].date(),"temp":p["temperature"],
        "rain_prob":p["rain_probability"],"condition":p["weather_condition"],
    } for p in forecast_list])
    daily = df.groupby("date").agg(
        temp_hi=("temp","max"),temp_lo=("temp","min"),
        rain_prob=("rain_prob","max"),
        condition=("condition",lambda x: x.mode()[0]),
    ).reset_index()
    emap = {"Clear":"☀️","Cloudy":"⛅","Rain":"🌧️",
            "Thunderstorm":"⛈️","Snow":"❄️","Fog":"🌫️"}
    st.markdown('<div class="section-label">7-Day Forecast</div>', unsafe_allow_html=True)
    cols = st.columns(min(7,len(daily)))
    for col,(_, row) in zip(cols, daily.iterrows()):
        with col:
            st.markdown(f"""
            <div class="forecast-card">
              <div style="font-size:0.7rem;font-weight:700;opacity:0.72">
                {row['date'].strftime('%a')}</div>
              <div style="font-size:1.45rem;margin:0.28rem 0">
                {emap.get(row['condition'],'⛅')}</div>
              <div class="forecast-temp-hi">{row['temp_hi']:.0f}°</div>
              <div class="forecast-temp-lo">{row['temp_lo']:.0f}°</div>
              <div style="font-size:0.63rem;color:#60a5fa;margin-top:0.18rem">
                {'💧 ' if row['rain_prob']>30 else ''}{row['rain_prob']:.0f}%</div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
if "saved_cities"    not in st.session_state: st.session_state.saved_cities    = []
if "active_city_idx" not in st.session_state: st.session_state.active_city_idx = 0
if "show_add_city"   not in st.session_state: st.session_state.show_add_city   = False
if "geocode_results" not in st.session_state: st.session_state.geocode_results  = []

if st.session_state.saved_cities:
    st.session_state.active_city_idx = max(
        0, min(st.session_state.active_city_idx,
               len(st.session_state.saved_cities)-1))

_active = (st.session_state.saved_cities[st.session_state.active_city_idx]
           if st.session_state.saved_cities else None)
_cond  = _active["preds"]["weather_condition"] if _active and _active["preds"] else "Clear"
_isday = _active["preds"]["is_day"]            if _active and _active["preds"] else True
_scene = get_scene(_cond, _isday)

inject_base_css(_scene)
models = load_all_models("models")


# ═════════════════════════════════════════════════════════════════════════════
# HEADER NAV
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="position:relative;z-index:2;padding-top:1.35rem">',
            unsafe_allow_html=True)
col_logo, col_tabs, col_plus = st.columns([2, 6, 1])

with col_logo:
    st.markdown(f"""
    <div style="padding-top:0.18rem">
      <span style="font-size:1.42rem;font-weight:900;letter-spacing:-0.03em;
                   color:{_scene['text']}">🌤️ SkyMind</span>
    </div>""", unsafe_allow_html=True)

with col_tabs:
    cities = st.session_state.saved_cities
    if cities:
        pills = ""
        for i, c in enumerate(cities):
            active_cls = "active" if i == st.session_state.active_city_idx else ""
            short_name = c["name"].split(",")[0]
            temp_str   = f" · {c['preds']['temperature']:.0f}°" if c["preds"] else ""
            pills += (
                f'<span class="tab-pill {active_cls}">'
                f'{short_name}{temp_str}</span>'
            )
        st.markdown(f'<div class="tab-strip">{pills}</div>', unsafe_allow_html=True)
        btn_cols = st.columns(len(cities))
        for i, (col, c) in enumerate(zip(btn_cols, cities)):
            temp_s = f" {c['preds']['temperature']:.0f}°" if c["preds"] else ""
            with col:
                if st.button(f"{c['name'].split(',')[0]}{temp_s}",
                             key=f"tab_{i}", help=c["name"]):
                    st.session_state.active_city_idx = i
                    st.rerun()

with col_plus:
    st.markdown("<div style='padding-top:0.12rem;text-align:right'>",
                unsafe_allow_html=True)
    if st.button("➕", key="toggle_add", help="Add city"):
        st.session_state.show_add_city   = not st.session_state.show_add_city
        st.session_state.geocode_results = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ADD-CITY PANEL
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.show_add_city:
    st.markdown('<div class="add-panel">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.84rem;font-weight:700;'
                f'letter-spacing:0.04em;margin-bottom:0.7rem;'
                f'color:{_scene["text"]}">🔍 Add a Location</div>',
                unsafe_allow_html=True)
    col_inp, col_gps = st.columns([5, 1])
    with col_inp:
        query = st.text_input("", placeholder="City, town, or region...",
                              label_visibility="collapsed", key="city_search")
        if query and len(query) >= 2:
            results = geocode_city(query, max_results=5)
            st.session_state.geocode_results = results if results else []
            if not results:
                st.markdown('<div class="sky-warning">⚠️ No results — try a nearby major city.</div>',
                            unsafe_allow_html=True)
    with col_gps:
        if st.button("📍", key="gps_btn", help="My location"):
            st.markdown("""<script>
            navigator.geolocation.getCurrentPosition(function(p){
                window.parent.postMessage({type:"GPS",lat:p.coords.latitude,
                lon:p.coords.longitude},"*");});</script>""",
                        unsafe_allow_html=True)
            st.info("Grant location access when prompted.")

    if st.session_state.geocode_results:
        results = st.session_state.geocode_results
        options = [r["display"] for r in results]
        choice  = st.selectbox("Select:", options, key="loc_choice") \
                  if len(options) > 1 else options[0]
        chosen  = results[options.index(choice)]
        already = any(c["name"] == chosen["display"]
                      for c in st.session_state.saved_cities)
        if already:
            st.markdown('<div class="sky-info">✅ Already in your list.</div>',
                        unsafe_allow_html=True)
        else:
            if st.button(f"➕ Add {chosen['display']}", key="add_btn"):
                st.session_state.saved_cities.append({
                    "name": chosen["display"], "lat": chosen["lat"],
                    "lon": chosen["lon"], "preds": None,
                    "forecasts": None, "warnings": WeatherWarnings(),
                })
                st.session_state.active_city_idx = len(st.session_state.saved_cities)-1
                st.session_state.show_add_city   = False
                st.session_state.geocode_results = []
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# FETCH & PREDICT
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.saved_cities:
    idx  = st.session_state.active_city_idx
    city = st.session_state.saved_cities[idx]

    if city["preds"] is None:
        with st.spinner(f"🌍 Fetching {city['name']}..."):
            try:
                lat, lon, name = city["lat"], city["lon"], city["name"]
                warn   = WeatherWarnings()
                raw_df = fetch_forecast(lat, lon, days=7)

                valid, _ = assess_api_response(raw_df)
                if not valid:
                    fb_lat,fb_lon,fb_name,fb_msg = handle_location_fallback(lat,lon)
                    warn.add("warning", fb_msg)
                    raw_df = fetch_forecast(fb_lat, fb_lon, days=7)
                    st.session_state.saved_cities[idx]["name"] = fb_name

                is_far, conf_msg = check_location_confidence(lat, lon)
                if is_far: warn.add("info", conf_msg)

                engineered_df = engineer_features(raw_df)
                now_hour  = datetime.now().hour
                hour_mask = engineered_df["hour"] == now_hour
                if not hour_mask.any():
                    hour_mask = pd.Series([True]+[False]*(len(engineered_df)-1),
                                          index=engineered_df.index)
                current_row = engineered_df[hour_mask].iloc[[0]]
                is_day_now  = bool(current_row["is_day"].iloc[0])

                current_preds = run_all_models(models, current_row, is_day_now, now_hour)
                current_preds["time"] = datetime.now()

                # Sanity override: rain false positive
                raw_rain = float(current_row.get("rain", pd.Series([0])).iloc[0]
                                 if hasattr(current_row,"get") else 0)
                if raw_rain == 0 and current_preds.get("rain_probability",0) > 80:
                    current_preds["rain_probability"] = min(
                        current_preds["rain_probability"], 25.0)

                all_forecasts = run_forecast(models, raw_df)

                if not handle_missing_soil(raw_df, lat, lon):
                    warn.add("info","Soil data unavailable for this location type.")
                    current_preds["hide_soil"] = True

                st.session_state.saved_cities[idx]["preds"]     = current_preds
                st.session_state.saved_cities[idx]["forecasts"] = all_forecasts
                st.session_state.saved_cities[idx]["warnings"]  = warn
                st.rerun()
            except Exception as e:
                st.markdown(f'<div class="sky-error">❌ {str(e)}</div>',
                            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DISPLAY
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.saved_cities:
    idx   = st.session_state.active_city_idx
    city  = st.session_state.saved_cities[idx]
    p     = city.get("preds")
    name  = city["name"]
    warn  = city.get("warnings", WeatherWarnings())

    if p is None:
        st.stop()

    scene = get_scene(p["weather_condition"], p["is_day"])
    inject_base_css(scene)
    render_scene(scene)
    weather_animation(p["weather_condition"], p["is_day"])

    for w in warn.get_all():
        icon = "⚠️" if w["level"]=="warning" else "ℹ️"
        cls  = "sky-warning" if w["level"]=="warning" else "sky-info"
        st.markdown(f'<div class="{cls}">{icon} {w["message"]}</div>',
                    unsafe_allow_html=True)

    # ── HERO ──────────────────────────────────────────────────────────────
    col_hero, col_metrics = st.columns([1.15, 1])
    with col_hero:
        condition = p["weather_condition"]
        is_day    = p["is_day"]
        tod       = "day" if is_day else "night"
        icon_map  = {
            "Clear":{"day":"☀️","night":"🌙"},
            "Cloudy":{"day":"⛅","night":"☁️"},
            "Rain":{"day":"🌧️","night":"🌧️"},
            "Thunderstorm":{"day":"⛈️","night":"⛈️"},
            "Snow":{"day":"❄️","night":"🌨️"},
            "Fog":{"day":"🌫️","night":"🌫️"},
        }
        icon    = icon_map.get(condition,{}).get(tod,"⛅")
        now_str = datetime.now().strftime("%A, %d %B · %H:%M")

        st.markdown(f"""
        <div class="hero-wrap">
          <div class="hero-location">📍 {name}</div>
          <div class="hero-date">{now_str}</div>
          <div class="hero-temp-row">
            <span class="hero-icon floaty">{icon}</span>
            <span class="hero-temp">{p['temperature']:.0f}</span>
            <span class="hero-unit">°C</span>
          </div>
          <div class="hero-condition">{p['weather_label']}</div>
          <div class="hero-feels">
            Feels like <strong>{p['apparent_temperature']:.0f}°C</strong>
            &nbsp;·&nbsp; Humidity <strong>{p['humidity']:.0f}%</strong>
          </div>
          <div class="accent-line" style="background:{scene['accent']}"></div>
        </div>
        """, unsafe_allow_html=True)

        desc = generate_main_description(p, name.split(",")[0])
        st.markdown(f"""
        <div class="sky-card" style="margin-top:0.55rem">
          <div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.12em;opacity:0.48;margin-bottom:0.45rem">
            🤖 SkyMind Intelligence
          </div>
          <div style="font-size:0.91rem;line-height:1.66;opacity:0.88">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        st.markdown("<div style='padding-top:1.75rem'>", unsafe_allow_html=True)
        rain_color = (
            "#ef4444" if p["rain_probability"]>70 else
            "#f59e0b" if p["rain_probability"]>40 else scene["accent"]
        )
        m1,m2 = st.columns(2)
        with m1:
            st.markdown(metric_card("💧 Rain",f"{p['rain_probability']:.0f}%",
                color=rain_color,bar=progress_bar(p["rain_probability"],100,rain_color)),
                unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("💨 Wind",f"{p['wind_speed_kmh']} km/h",
                sub=f"{p['wind_direction']:.0f}° bearing"),unsafe_allow_html=True)

        m3,m4 = st.columns(2)
        with m3:
            st.markdown(metric_card("☀️ UV Index",f"{p['uv_index']}",
                sub=f"{p['uv_label']} {p['uv_emoji']}",color=p["uv_color"]),
                unsafe_allow_html=True)
        with m4:
            vc = ("#ef4444" if p["visibility_km"]<1 else
                  "#f59e0b" if p["visibility_km"]<4 else "#22c55e")
            st.markdown(metric_card("👁️ Visibility",f"{p['visibility_km']} km",
                sub=p["visibility_label"],color=vc),unsafe_allow_html=True)

        m5,m6 = st.columns(2)
        with m5:
            st.markdown(metric_card("🌡️ Dew Point",f"{p['dew_point']:.1f}°C"),
                unsafe_allow_html=True)
        with m6:
            st.markdown(metric_card("🔵 Pressure",f"{p['pressure']:.0f} mb"),
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── FORECASTS ─────────────────────────────────────────────────────────
    forecasts = city.get("forecasts")
    if forecasts:
        st.markdown('<hr class="sky-divider">', unsafe_allow_html=True)
        hourly_strip(forecasts[:12], scene)
        st.markdown("<div style='margin-top:0.9rem'>", unsafe_allow_html=True)
        daily_strip(forecasts)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── INSIGHTS ──────────────────────────────────────────────────────────
    st.markdown('<hr class="sky-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">✨ SkyMind Exclusive Insights</div>',
                unsafe_allow_html=True)

    u1,u2 = st.columns(2)
    with u1:
        t_prob  = p["thunderstorm_prob"]
        t_risk  = p["thunderstorm_risk"]
        t_color = ("#ef4444" if t_risk=="High" else
                   "#f59e0b" if t_risk=="Moderate" else "#22c55e")
        t_desc  = generate_thunderstorm_description(t_prob, t_risk)
        pulse_a = 'class="pulse"' if t_risk=="High" else ""
        st.markdown(f"""
        <div class="sky-card">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:0.48rem">
            <div class="insight-header">⛈️ Thunderstorm Risk</div>
            <span {pulse_a}>{badge(t_risk,t_color)}</span>
          </div>
          <div style="font-size:2.2rem;font-weight:900;color:{t_color};
                      letter-spacing:-0.02em">{t_prob:.0f}%</div>
          {progress_bar(t_prob,100,t_color,7)}
          <div style="font-size:0.82rem;margin-top:0.65rem;
                      line-height:1.6;opacity:0.8">{t_desc}</div>
        </div>""", unsafe_allow_html=True)

    with u2:
        hs_level = p["heat_stress_level"]
        hs_label = p["heat_stress_label"]
        hs_color = p["heat_stress_color"]
        hs_emoji = p["heat_stress_emoji"]
        hs_desc  = generate_heat_stress_description(
            hs_level, hs_label, p["temperature"], p["humidity"])
        hs_pulse = 'class="pulse"' if hs_level>=2 else ""
        hs_dots  = "".join([
            f'<div style="text-align:center;opacity:{"1" if i==hs_level else "0.22"};'
            f'transition:opacity 0.3s">'
            f'<div style="font-size:1.28rem">{HEAT_STRESS_LABELS[i]["emoji"]}</div>'
            f'<div style="font-size:0.58rem;margin-top:0.08rem">'
            f'{HEAT_STRESS_LABELS[i]["label"]}</div></div>'
            for i in range(4)])
        st.markdown(f"""
        <div class="sky-card">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:0.48rem">
            <div class="insight-header">🥵 Heat Stress Index</div>
            <span {hs_pulse}>{badge(f"{hs_emoji} {hs_label}",hs_color)}</span>
          </div>
          <div style="display:flex;gap:0.75rem;margin-bottom:0.55rem">{hs_dots}</div>
          <div style="font-size:0.82rem;line-height:1.6;opacity:0.8">{hs_desc}</div>
        </div>""", unsafe_allow_html=True)

    u3,u4 = st.columns(2)
    with u3:
        os_score = p["outdoor_score"]
        os_label = p["outdoor_label"]
        os_emoji = p["outdoor_emoji"]
        os_color = ("#ef4444" if os_score<20 else "#f59e0b" if os_score<40 else
                    "#eab308" if os_score<60 else "#22c55e" if os_score<80 else "#15803d")
        st.markdown(f"""
        <div class="sky-card">
          <div class="insight-header">🏃 Outdoor Activity Score</div>
          <div style="display:flex;align-items:baseline;gap:0.28rem;margin-bottom:0.28rem">
            <span style="font-size:2.9rem;font-weight:900;color:{os_color};
                         letter-spacing:-0.03em">{os_score:.0f}</span>
            <span style="opacity:0.48;font-size:0.95rem">/100</span>
            <span style="font-size:1.55rem;margin-left:0.18rem">{os_emoji}</span>
          </div>
          {progress_bar(os_score,100,os_color,8)}
          <div style="font-size:0.84rem;font-weight:700;color:{os_color};
                      margin-top:0.52rem">{os_label}</div>
        </div>""", unsafe_allow_html=True)

    with u4:
        if p.get("hide_soil"):
            st.markdown("""
            <div class="sky-card" style="opacity:0.52">
              <div class="insight-header">🌱 Soil & Agricultural Score</div>
              <div style="font-size:0.86rem;opacity:0.68;margin-top:0.38rem">
                Soil data unavailable for this location type.</div>
            </div>""", unsafe_allow_html=True)
        else:
            ss = p["soil_score"]; sl = p["soil_label"]
            sc = ("#ef4444" if ss<20 else "#f59e0b" if ss<40 else
                  "#eab308" if ss<60 else "#22c55e" if ss<80 else "#15803d")
            st.markdown(f"""
            <div class="sky-card">
              <div class="insight-header">🌱 Soil & Agricultural Score</div>
              <div style="display:flex;align-items:baseline;gap:0.28rem;margin-bottom:0.28rem">
                <span style="font-size:2.9rem;font-weight:900;color:{sc};
                             letter-spacing:-0.03em">{ss:.0f}</span>
                <span style="opacity:0.48;font-size:0.95rem">/100</span>
              </div>
              {progress_bar(ss,100,sc,8)}
              <div style="font-size:0.84rem;font-weight:700;color:{sc};
                          margin-top:0.52rem">{sl}</div>
              <div style="font-size:0.79rem;opacity:0.72;margin-top:0.32rem;
                          line-height:1.55">{generate_soil_description(ss,sl)}</div>
            </div>""", unsafe_allow_html=True)

    v1,v2 = st.columns(2)
    with v1:
        uv_desc = generate_uv_description(p["uv_index"],p["uv_label"],p["is_day"])
        st.markdown(f"""
        <div class="sky-card">
          <div class="insight-header">☀️ UV Index Detail</div>
          <div style="font-size:2.55rem;font-weight:900;color:{p['uv_color']};
                      letter-spacing:-0.02em">
            {p['uv_index']} <span style="font-size:1.35rem">{p['uv_emoji']}</span></div>
          <div style="font-size:0.82rem;opacity:0.8;margin-top:0.45rem;
                      line-height:1.6">{uv_desc}</div>
        </div>""", unsafe_allow_html=True)

    with v2:
        vis_desc = generate_visibility_description(p["visibility_km"],p["visibility_label"])
        vc = ("#ef4444" if p["visibility_km"]<1 else
              "#f59e0b" if p["visibility_km"]<4 else "#22c55e")
        st.markdown(f"""
        <div class="sky-card">
          <div class="insight-header">👁️ Visibility Detail</div>
          <div style="font-size:2.55rem;font-weight:900;color:{vc};
                      letter-spacing:-0.02em">{p['visibility_km']} km</div>
          <div style="font-size:0.82rem;opacity:0.8;margin-top:0.45rem;
                      line-height:1.6">{vis_desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── Actions ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _,c2,c3,_ = st.columns([2,1,1,2])
    with c2:
        if st.button("🔄 Refresh", key="refresh_btn"):
            st.session_state.saved_cities[idx]["preds"]     = None
            st.session_state.saved_cities[idx]["forecasts"] = None
            st.session_state.saved_cities[idx]["warnings"]  = WeatherWarnings()
            st.rerun()
    with c3:
        if st.button("🗑️ Remove", key="remove_btn"):
            st.session_state.saved_cities.pop(idx)
            st.session_state.active_city_idx = max(0, idx-1)
            st.rerun()
    st.markdown("<br><br>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# LANDING SCREEN
# ═════════════════════════════════════════════════════════════════════════════
else:
    render_scene(get_scene("Clear", True))
    st.markdown(f"""
    <div style="text-align:center;padding:5rem 1rem 3rem;position:relative;z-index:1">
      <div style="font-size:5rem;margin-bottom:0.45rem" class="floaty">🌤️</div>
      <h1 style="font-size:3rem;font-weight:900;margin:0.25rem 0;
                 letter-spacing:-0.03em;color:{_scene['text']}">SkyMind</h1>
      <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.18em;opacity:0.48;margin-bottom:1.15rem">
        AI Weather Intelligence
      </div>
      <p style="font-size:1.02rem;opacity:0.68;max-width:430px;
                margin:0 auto 1.85rem;line-height:1.72">
        Tap <strong>➕</strong> to add your first city and unlock deep weather
        insights from <strong>10 ML models</strong> trained on 2 million
        global observations.
      </p>
      <div style="display:flex;justify-content:center;gap:0.65rem;
                  flex-wrap:wrap;font-size:0.8rem;opacity:0.55">
        <span>🌡️ Temperature</span><span>🌧️ Rain</span>
        <span>⛈️ Thunderstorm</span><span>🥵 Heat Stress</span>
        <span>👁️ Visibility</span><span>🏃 Outdoor Score</span>
        <span>🌱 Soil</span><span>☀️ UV Index</span>
      </div>
    </div>""", unsafe_allow_html=True)
