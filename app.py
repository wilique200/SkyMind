# app.py — SkyMind Weather Intelligence
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from src.model_loader      import load_all_models
from src.weather_api       import fetch_forecast, check_soil_data_quality
from src.geocoding         import geocode_city, reverse_geocode, find_nearest_training_city
from src.feature_engineering import engineer_features
from src.inference         import run_all_models, run_forecast
from src.edge_cases        import (
    WeatherWarnings, handle_location_fallback,
    check_location_confidence, handle_missing_soil,
    handle_nighttime_uv, validate_model_output, assess_api_response
)
from src.descriptions      import (
    generate_main_description, generate_uv_description,
    generate_soil_description, generate_heat_stress_description,
    generate_thunderstorm_description, generate_visibility_description
)
from utils.constants       import (
    WEATHER_THEMES, WMO_EMOJIS, HEAT_STRESS_LABELS,
    UV_LEVELS, OUTDOOR_SCORE_LABELS
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SkyMind — AI Weather Intelligence",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ════════════════════════════════════════════════════════════
# DYNAMIC STYLING
# ════════════════════════════════════════════════════════════
def apply_theme(condition, is_day):
    themes = WEATHER_THEMES
    time_key = "day" if is_day else "night"
    theme = themes.get(condition, themes["Cloudy"])[time_key]
    bg    = theme["bg"]
    text  = theme["text"]
    card  = theme["card"]

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {{
        background: {bg} !important;
        font-family: 'Inter', sans-serif;
        color: {text} !important;
        min-height: 100vh;
    }}
    .stApp > header {{ background: transparent !important; }}

    /* Cards */
    .sky-card {{
        background: {card};
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        color: {text};
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }}
    .sky-card:hover {{ transform: translateY(-2px); }}

    /* Metric cards */
    .metric-card {{
        background: {card};
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.15);
        text-align: center;
        color: {text};
        height: 100%;
    }}
    .metric-label {{
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.75;
        margin-bottom: 0.3rem;
    }}
    .metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.1;
    }}
    .metric-sub {{
        font-size: 0.78rem;
        opacity: 0.7;
        margin-top: 0.2rem;
    }}

    /* Hero temperature */
    .hero-temp {{
        font-size: 6rem;
        font-weight: 800;
        line-height: 1;
        color: {text};
        text-shadow: 0 2px 20px rgba(0,0,0,0.2);
    }}
    .hero-condition {{
        font-size: 1.4rem;
        font-weight: 400;
        opacity: 0.9;
        margin-top: 0.2rem;
    }}
    .hero-location {{
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }}
    .feels-like {{
        font-size: 1rem;
        opacity: 0.75;
        margin-top: 0.3rem;
    }}

    /* Progress bar */
    .sky-progress-bg {{
        background: rgba(255,255,255,0.15);
        border-radius: 999px;
        height: 8px;
        width: 100%;
        margin-top: 0.5rem;
    }}
    .sky-progress-fill {{
        height: 8px;
        border-radius: 999px;
        transition: width 0.6s ease;
    }}

    /* Forecast strip */
    .forecast-card {{
        background: {card};
        border-radius: 14px;
        padding: 0.7rem 0.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.12);
        color: {text};
        font-size: 0.82rem;
    }}
    .forecast-temp-hi {{ font-weight: 700; font-size: 1rem; }}
    .forecast-temp-lo {{ opacity: 0.6; font-size: 0.85rem; }}

    /* Section headers */
    .section-title {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        opacity: 0.65;
        margin: 1.2rem 0 0.6rem;
    }}

    /* Search bar */
    .stTextInput > div > div > input {{
        background: rgba(255,255,255,0.15) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 50px !important;
        color: {text} !important;
        padding: 0.6rem 1.2rem !important;
        font-size: 1rem !important;
    }}
    .stTextInput > div > div > input::placeholder {{
        color: rgba(255,255,255,0.5) !important;
    }}
    .stButton > button {{
        background: rgba(255,255,255,0.18) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        color: {text} !important;
        border-radius: 50px !important;
        padding: 0.4rem 1.4rem !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }}
    .stButton > button:hover {{
        background: rgba(255,255,255,0.28) !important;
        transform: translateY(-1px) !important;
    }}

    /* Badge */
    .badge {{
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }}

    /* Hide Streamlit branding */
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.5rem !important; max-width: 1100px; }}

    /* Warning */
    .sky-warning {{
        background: rgba(251,191,36,0.15);
        border: 1px solid rgba(251,191,36,0.4);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: {text};
        margin-bottom: 0.5rem;
    }}
    .sky-error {{
        background: rgba(239,68,68,0.15);
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: {text};
        margin-bottom: 0.5rem;
    }}

    /* Animations */
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    .pulse {{ animation: pulse 2s infinite; }}

    @keyframes lightning {{
        0%, 90%, 100% {{ opacity: 0; }}
        92%, 96% {{ opacity: 1; }}
    }}
    .lightning {{ animation: lightning 4s infinite; }}
    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# UI HELPER COMPONENTS
# ════════════════════════════════════════════════════════════
def progress_bar(value, max_val, color="#4fc3f7", height=8):
    pct = min(100, max(0, value / max_val * 100))
    return f"""
    <div class="sky-progress-bg">
      <div class="sky-progress-fill"
           style="width:{pct:.0f}%;background:{color};height:{height}px;">
      </div>
    </div>"""


def metric_card(label, value, sub=None, color=None, emoji=None):
    color_style = f"color:{color};" if color else ""
    emoji_html  = f"<span style='font-size:1.4rem'>{emoji}</span><br>" if emoji else ""
    sub_html    = f"<div class='metric-sub'>{sub}</div>" if sub else ""
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      {emoji_html}
      <div class="metric-value" style="{color_style}">{value}</div>
      {sub_html}
    </div>"""


def weather_animation(condition, is_day):
    """CSS-only weather animation overlaid on hero."""
    if condition == "Rain":
        drops = "".join([
            f'<div style="position:absolute;left:{np.random.randint(0,100)}%;'
            f'top:{np.random.randint(-20,100)}%;width:2px;height:12px;'
            f'background:rgba(150,200,255,0.6);border-radius:2px;'
            f'animation:fall {np.random.uniform(0.8,1.5):.1f}s linear '
            f'{np.random.uniform(0,2):.1f}s infinite;"></div>'
            for _ in range(20)
        ])
        st.markdown(f"""
        <style>
        @keyframes fall {{from{{transform:translateY(-20px);opacity:0}}
        to{{transform:translateY(60px);opacity:1}}}}
        </style>
        <div style="position:fixed;top:0;left:0;width:100%;height:100%;
             pointer-events:none;z-index:0;overflow:hidden">{drops}</div>
        """, unsafe_allow_html=True)

    elif condition == "Snow":
        flakes = "".join([
            f'<div style="position:absolute;left:{np.random.randint(0,100)}%;'
            f'top:{np.random.randint(-10,100)}%;font-size:{np.random.randint(8,18)}px;'
            f'opacity:0.7;animation:snowfall {np.random.uniform(2,5):.1f}s linear '
            f'{np.random.uniform(0,4):.1f}s infinite;">❄</div>'
            for _ in range(15)
        ])
        st.markdown(f"""
        <style>
        @keyframes snowfall {{from{{transform:translateY(-10px) rotate(0deg);opacity:0}}
        to{{transform:translateY(80px) rotate(360deg);opacity:0.8}}}}
        </style>
        <div style="position:fixed;top:0;left:0;width:100%;height:100%;
             pointer-events:none;z-index:0;">{flakes}</div>
        """, unsafe_allow_html=True)

    elif condition == "Thunderstorm":
        st.markdown("""
        <div class="lightning" style="position:fixed;top:0;left:0;width:100%;
             height:100%;background:rgba(255,255,200,0.15);
             pointer-events:none;z-index:0;"></div>
        """, unsafe_allow_html=True)


def hourly_forecast_strip(forecast_list):
    """Render 24-hour horizontal forecast."""
    st.markdown('<div class="section-title">24-Hour Forecast</div>',
                unsafe_allow_html=True)
    cols = st.columns(min(12, len(forecast_list[:12])))
    condition_emojis = {
        "Clear": {"day": "☀️", "night": "🌙"},
        "Cloudy": {"day": "⛅", "night": "☁️"},
        "Rain": {"day": "🌧️", "night": "🌧️"},
        "Thunderstorm": {"day": "⛈️", "night": "⛈️"},
        "Snow": {"day": "❄️", "night": "❄️"},
        "Fog": {"day": "🌫️", "night": "🌫️"},
    }
    for i, (col, p) in enumerate(zip(cols, forecast_list[:12])):
        time_label = p["time"].strftime("%H:%M") if i > 0 else "Now"
        cond  = p.get("weather_condition", "Cloudy")
        tod   = "day" if p.get("is_day") else "night"
        emoji = condition_emojis.get(cond, {}).get(tod, "⛅")
        temp  = p.get("temperature", 0)
        rain  = p.get("rain_probability", 0)
        with col:
            st.markdown(f"""
            <div class="forecast-card">
              <div style="font-size:0.75rem;opacity:0.7">{time_label}</div>
              <div style="font-size:1.4rem;margin:0.2rem 0">{emoji}</div>
              <div class="forecast-temp-hi">{temp:.0f}°</div>
              <div style="font-size:0.7rem;opacity:0.6;color:#60a5fa">
                {'💧' if rain > 30 else ''}{rain:.0f}%</div>
            </div>""", unsafe_allow_html=True)


def daily_forecast_strip(forecast_list):
    """Render 7-day daily forecast grouped by day."""
    st.markdown('<div class="section-title">7-Day Forecast</div>',
                unsafe_allow_html=True)
    df = pd.DataFrame([{
        "date":      p["time"].date(),
        "temp":      p["temperature"],
        "rain_prob": p["rain_probability"],
        "condition": p["weather_condition"],
        "is_day":    p["is_day"],
    } for p in forecast_list])

    daily = df.groupby("date").agg(
        temp_hi   = ("temp", "max"),
        temp_lo   = ("temp", "min"),
        rain_prob = ("rain_prob", "max"),
        condition = ("condition", lambda x: x.mode()[0]),
    ).reset_index()

    emojis = {"Clear":"☀️","Cloudy":"⛅","Rain":"🌧️",
              "Thunderstorm":"⛈️","Snow":"❄️","Fog":"🌫️"}
    cols = st.columns(min(7, len(daily)))
    for col, (_, row) in zip(cols, daily.iterrows()):
        day_name = row["date"].strftime("%a")
        emoji = emojis.get(row["condition"], "⛅")
        with col:
            st.markdown(f"""
            <div class="forecast-card">
              <div style="font-size:0.75rem;font-weight:600;opacity:0.8">
                {day_name}</div>
              <div style="font-size:1.5rem;margin:0.3rem 0">{emoji}</div>
              <div class="forecast-temp-hi">{row['temp_hi']:.0f}°</div>
              <div class="forecast-temp-lo">{row['temp_lo']:.0f}°</div>
              <div style="font-size:0.68rem;color:#60a5fa;margin-top:0.2rem">
                {'💧 ' if row['rain_prob'] > 30 else ''}{row['rain_prob']:.0f}%
              </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ════════════════════════════════════════════════════════════
if "location_name" not in st.session_state:
    st.session_state.location_name = None
if "lat" not in st.session_state:
    st.session_state.lat = None
if "lon" not in st.session_state:
    st.session_state.lon = None
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None
if "current_preds" not in st.session_state:
    st.session_state.current_preds = None
if "all_forecasts" not in st.session_state:
    st.session_state.all_forecasts = None
if "geocode_results" not in st.session_state:
    st.session_state.geocode_results = []
if "warnings" not in st.session_state:
    st.session_state.warnings = WeatherWarnings()

# Default theme until weather loads
default_condition = st.session_state.current_preds["weather_condition"] \
    if st.session_state.current_preds else "Clear"
default_is_day = st.session_state.current_preds["is_day"] \
    if st.session_state.current_preds else True

apply_theme(default_condition, default_is_day)

# ── Load models once ─────────────────────────────────────────
models = load_all_models("models")


# ════════════════════════════════════════════════════════════
# HEADER & SEARCH
# ════════════════════════════════════════════════════════════
col_logo, col_search, col_gps = st.columns([1.5, 4, 1])

with col_logo:
    st.markdown("""
    <div style="padding-top:0.3rem">
      <span style="font-size:1.6rem;font-weight:800;letter-spacing:-0.02em">
        🌤️ SkyMind
      </span><br>
      <span style="font-size:0.7rem;opacity:0.6;letter-spacing:0.05em">
        AI WEATHER INTELLIGENCE
      </span>
    </div>""", unsafe_allow_html=True)

with col_search:
    query = st.text_input(
        "", placeholder="🔍  Search any city, town, or location...",
        label_visibility="collapsed", key="search_input"
    )
    if query and len(query) >= 2:
        results = geocode_city(query, max_results=5)
        if results:
            st.session_state.geocode_results = results
        else:
            st.session_state.geocode_results = []
            st.markdown(
                '<div class="sky-warning">⚠️ No results found. Try a nearby major city.</div>',
                unsafe_allow_html=True
            )

with col_gps:
    st.markdown("<div style='padding-top:0.3rem'>", unsafe_allow_html=True)
    if st.button("📍 My Location"):
        st.markdown("""
        <script>
        navigator.geolocation.getCurrentPosition(function(pos) {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;
            window.parent.postMessage({type:"GPS",lat:lat,lon:lon},"*");
        }, function() {
            window.parent.postMessage({type:"GPS_DENIED"},"*");
        });
        </script>
        """, unsafe_allow_html=True)
        st.info("📍 Grant location access in your browser when prompted.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Location disambiguation dropdown ─────────────────────────
if st.session_state.geocode_results:
    results = st.session_state.geocode_results
    if len(results) == 1:
        chosen = results[0]
    else:
        options = [r["display"] for r in results]
        choice  = st.selectbox(
            "Multiple locations found — select yours:",
            options, key="location_choice"
        )
        chosen = results[options.index(choice)]

    if st.button(f"🌍 Load weather for **{chosen['display']}**",
                 key="load_btn"):
        st.session_state.lat           = chosen["lat"]
        st.session_state.lon           = chosen["lon"]
        st.session_state.location_name = chosen["display"]
        st.session_state.geocode_results = []
        st.session_state.warnings      = WeatherWarnings()
        st.rerun()


# ════════════════════════════════════════════════════════════
# FETCH & PREDICT
# ════════════════════════════════════════════════════════════
if st.session_state.lat is not None and st.session_state.current_preds is None:
    lat  = st.session_state.lat
    lon  = st.session_state.lon
    name = st.session_state.location_name
    warn = st.session_state.warnings

    with st.spinner(f"🌍 Fetching forecast for {name}..."):
        try:
            raw_df = fetch_forecast(lat, lon, days=7)

            # Validate API response
            valid, issue = assess_api_response(raw_df)
            if not valid:
                # Try nearest training city fallback
                fb_lat, fb_lon, fb_name, fb_msg = handle_location_fallback(lat, lon)
                warn.add("warning", fb_msg)
                raw_df = fetch_forecast(fb_lat, fb_lon, days=7)
                st.session_state.location_name = fb_name

            # Confidence check
            is_far, conf_msg = check_location_confidence(lat, lon)
            if is_far:
                warn.add("info", conf_msg)

            # Engineer features
            engineered_df = engineer_features(raw_df)

            # Find current hour row
            now_hour = datetime.now().hour
            hour_mask = engineered_df["hour"] == now_hour
            if not hour_mask.any():
                hour_mask = pd.Series([True] + [False]*(len(engineered_df)-1),
                                      index=engineered_df.index)
            current_row = engineered_df[hour_mask].iloc[[0]]
            is_day_now  = bool(current_row["is_day"].iloc[0])

            # Run models for current conditions
            current_preds = run_all_models(
                models, current_row, is_day_now, now_hour
            )
            current_preds["time"] = datetime.now()

            # Run all forecast hours
            all_forecasts = run_forecast(models, raw_df)

            # Soil check
            if not handle_missing_soil(raw_df, lat, lon):
                warn.add("info",
                    "Soil data is unreliable for this location type. "
                    "Agricultural score is hidden.")
                current_preds["hide_soil"] = True

            st.session_state.forecast_df   = raw_df
            st.session_state.current_preds = current_preds
            st.session_state.all_forecasts = all_forecasts
            st.session_state.warnings      = warn
            st.rerun()

        except Exception as e:
            st.markdown(
                f'<div class="sky-error">❌ {str(e)}</div>',
                unsafe_allow_html=True
            )


# ════════════════════════════════════════════════════════════
# MAIN DISPLAY
# ════════════════════════════════════════════════════════════
if st.session_state.current_preds:
    p    = st.session_state.current_preds
    name = st.session_state.location_name
    warn = st.session_state.warnings

    # Re-apply theme based on actual weather
    apply_theme(p["weather_condition"], p["is_day"])
    weather_animation(p["weather_condition"], p["is_day"])

    # ── Warnings banner ──────────────────────────────────────
    for w in warn.get_all():
        icon = "⚠️" if w["level"] == "warning" else "ℹ️"
        css  = "sky-warning" if w["level"] == "warning" else "sky-warning"
        st.markdown(
            f'<div class="{css}">{icon} {w["message"]}</div>',
            unsafe_allow_html=True
        )

    # ── HERO SECTION ─────────────────────────────────────────
    col_hero, col_details = st.columns([1.1, 1])

    with col_hero:
        condition = p["weather_condition"]
        is_day    = p["is_day"]
        tod       = "day" if is_day else "night"
        emoji     = WMO_EMOJIS.get(condition, {}).get(tod, "⛅")
        now_str   = datetime.now().strftime("%A, %d %B • %H:%M")

        st.markdown(f"""
        <div class="sky-card" style="min-height:260px">
          <div class="hero-location">📍 {name}</div>
          <div class="hero-location" style="font-size:0.85rem;opacity:0.6">
            {now_str}
          </div>
          <div style="display:flex;align-items:flex-end;gap:0.5rem;
                      margin:0.8rem 0">
            <span style="font-size:5rem;line-height:1">{emoji}</span>
            <span class="hero-temp">{p['temperature']:.0f}°</span>
            <span style="font-size:2rem;opacity:0.7;padding-bottom:0.8rem">C</span>
          </div>
          <div class="hero-condition">{p['weather_label']}</div>
          <div class="feels-like">
            Feels like {p['apparent_temperature']:.0f}°C &nbsp;•&nbsp;
            Humidity {p['humidity']:.0f}%
          </div>
        </div>
        """, unsafe_allow_html=True)

        # AI Description
        desc = generate_main_description(p, name.split(",")[0])
        st.markdown(f"""
        <div class="sky-card">
          <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.1em;opacity:0.6;margin-bottom:0.5rem">
            🤖 SkyMind says
          </div>
          <div style="font-size:0.95rem;line-height:1.6">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_details:
        # Core metrics grid
        m1, m2 = st.columns(2)
        with m1:
            rain_color = "#ef4444" if p["rain_probability"] > 70 \
                else "#f59e0b" if p["rain_probability"] > 40 else "#60a5fa"
            st.markdown(metric_card(
                "💧 Rain Chance",
                f"{p['rain_probability']:.0f}%",
                progress_bar(p["rain_probability"], 100, rain_color),
                rain_color
            ) + progress_bar(p["rain_probability"], 100, rain_color),
            unsafe_allow_html=True)

        with m2:
            st.markdown(metric_card(
                "💨 Wind Speed",
                f"{p['wind_speed_kmh']} km/h",
                f"Direction: {p['wind_direction']:.0f}°"
            ), unsafe_allow_html=True)

        m3, m4 = st.columns(2)
        with m3:
            st.markdown(metric_card(
                "☀️ UV Index",
                f"{p['uv_index']} — {p['uv_label']}",
                p["uv_emoji"],
                p["uv_color"]
            ), unsafe_allow_html=True)

        with m4:
            st.markdown(metric_card(
                "👁️ Visibility",
                f"{p['visibility_km']} km",
                p["visibility_label"]
            ), unsafe_allow_html=True)

        m5, m6 = st.columns(2)
        with m5:
            st.markdown(metric_card(
                "🌡️ Dew Point",
                f"{p['dew_point']:.1f}°C"
            ), unsafe_allow_html=True)

        with m6:
            st.markdown(metric_card(
                "🔵 Pressure",
                f"{p['pressure']:.0f} mb"
            ), unsafe_allow_html=True)

    # ── 24-HOUR FORECAST ─────────────────────────────────────
    if st.session_state.all_forecasts:
        st.markdown("---")
        hourly_forecast_strip(st.session_state.all_forecasts[:12])

        # ── 7-DAY FORECAST ───────────────────────────────────
        st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
        daily_forecast_strip(st.session_state.all_forecasts)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── UNIQUE FEATURES SECTION ───────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;
                letter-spacing:0.12em;opacity:0.65;margin-bottom:0.8rem">
      ✨ SkyMind Exclusive Insights
    </div>""", unsafe_allow_html=True)

    u1, u2 = st.columns(2)

    # ── Thunderstorm Risk ────────────────────────────────────
    with u1:
        t_prob  = p["thunderstorm_prob"]
        t_risk  = p["thunderstorm_risk"]
        t_color = "#ef4444" if t_risk == "High" \
            else "#f59e0b" if t_risk == "Moderate" else "#22c55e"
        t_desc  = generate_thunderstorm_description(t_prob, t_risk)
        pulse   = ' class="pulse"' if t_risk == "High" else ""

        st.markdown(f"""
        <div class="sky-card">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:0.6rem">
            <span style="font-size:0.75rem;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.08em;
                         opacity:0.7">⛈️ Thunderstorm Risk</span>
            <span{pulse} style="background:{t_color};color:white;
                   padding:0.15rem 0.7rem;border-radius:999px;
                   font-size:0.78rem;font-weight:600">{t_risk}</span>
          </div>
          <div style="font-size:2rem;font-weight:700;color:{t_color}">
            {t_prob:.0f}%
          </div>
          {progress_bar(t_prob, 100, t_color)}
          <div style="font-size:0.85rem;margin-top:0.7rem;
                      line-height:1.5;opacity:0.85">{t_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Heat Stress ──────────────────────────────────────────
    with u2:
        hs_level = p["heat_stress_level"]
        hs_label = p["heat_stress_label"]
        hs_color = p["heat_stress_color"]
        hs_emoji = p["heat_stress_emoji"]
        hs_desc  = generate_heat_stress_description(
            hs_level, hs_label, p["temperature"], p["humidity"]
        )
        hs_pulse = ' class="pulse"' if hs_level >= 2 else ""

        st.markdown(f"""
        <div class="sky-card">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:0.6rem">
            <span style="font-size:0.75rem;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.08em;
                         opacity:0.7">🥵 Heat Stress Index</span>
            <span{hs_pulse} style="background:{hs_color};color:white;
                   padding:0.15rem 0.7rem;border-radius:999px;
                   font-size:0.78rem;font-weight:600">
              {hs_emoji} {hs_label}
            </span>
          </div>
          <div style="display:flex;gap:1rem;margin-bottom:0.5rem">
            {''.join([
              f'<div style="text-align:center;opacity:{"1" if i==hs_level else "0.3"}">'
              f'<div style="font-size:1.2rem">{HEAT_STRESS_LABELS[i]["emoji"]}</div>'
              f'<div style="font-size:0.65rem">{HEAT_STRESS_LABELS[i]["label"]}</div>'
              f'</div>'
              for i in range(4)
            ])}
          </div>
          <div style="font-size:0.85rem;line-height:1.5;opacity:0.85">
            {hs_desc}
          </div>
        </div>
        """, unsafe_allow_html=True)

    u3, u4 = st.columns(2)

    # ── Outdoor Activity Score ───────────────────────────────
    with u3:
        os_score = p["outdoor_score"]
        os_label = p["outdoor_label"]
        os_emoji = p["outdoor_emoji"]
        os_color = (
            "#ef4444" if os_score < 20 else
            "#f59e0b" if os_score < 40 else
            "#eab308" if os_score < 60 else
            "#22c55e" if os_score < 80 else "#15803d"
        )
        st.markdown(f"""
        <div class="sky-card">
          <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.08em;opacity:0.7;margin-bottom:0.6rem">
            🏃 Outdoor Activity Score
          </div>
          <div style="display:flex;align-items:baseline;gap:0.4rem">
            <span style="font-size:2.5rem;font-weight:800;color:{os_color}">
              {os_score:.0f}
            </span>
            <span style="opacity:0.6">/100</span>
            <span style="font-size:1.4rem;margin-left:0.3rem">{os_emoji}</span>
          </div>
          {progress_bar(os_score, 100, os_color, 10)}
          <div style="font-size:0.9rem;font-weight:600;color:{os_color};
                      margin-top:0.5rem">{os_label}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Soil / Agricultural Score ─────────────────────────────
    with u4:
        if p.get("hide_soil"):
            st.markdown("""
            <div class="sky-card" style="opacity:0.6">
              <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                          letter-spacing:0.08em;opacity:0.7;margin-bottom:0.6rem">
                🌱 Soil &amp; Agricultural Score
              </div>
              <div style="font-size:0.9rem;opacity:0.7">
                Soil data is not available for this location type
                (ocean or polar region).
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            ss      = p["soil_score"]
            sl      = p["soil_label"]
            ss_color = (
                "#ef4444" if ss < 20 else "#f59e0b" if ss < 40 else
                "#eab308" if ss < 60 else "#22c55e" if ss < 80 else "#15803d"
            )
            soil_desc = generate_soil_description(ss, sl)
            st.markdown(f"""
            <div class="sky-card">
              <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                          letter-spacing:0.08em;opacity:0.7;margin-bottom:0.6rem">
                🌱 Soil &amp; Agricultural Score
              </div>
              <div style="display:flex;align-items:baseline;gap:0.4rem">
                <span style="font-size:2.5rem;font-weight:800;color:{ss_color}">
                  {ss:.0f}
                </span>
                <span style="opacity:0.6">/100</span>
              </div>
              {progress_bar(ss, 100, ss_color, 10)}
              <div style="font-size:0.88rem;font-weight:600;color:{ss_color};
                          margin-top:0.5rem">{sl}</div>
              <div style="font-size:0.82rem;opacity:0.8;margin-top:0.4rem;
                          line-height:1.5">{soil_desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── UV + Visibility detail ────────────────────────────────
    v1, v2 = st.columns(2)
    with v1:
        uv_desc = generate_uv_description(
            p["uv_index"], p["uv_label"], p["is_day"]
        )
        st.markdown(f"""
        <div class="sky-card">
          <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.08em;opacity:0.7;margin-bottom:0.5rem">
            ☀️ UV Index Detail
          </div>
          <div style="font-size:2rem;font-weight:700;color:{p['uv_color']}">
            {p['uv_index']} {p['uv_emoji']}
          </div>
          <div style="font-size:0.88rem;opacity:0.85;margin-top:0.4rem;
                      line-height:1.5">{uv_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with v2:
        vis_desc = generate_visibility_description(
            p["visibility_km"], p["visibility_label"]
        )
        vis_color = (
            "#ef4444" if p["visibility_km"] < 1 else
            "#f59e0b" if p["visibility_km"] < 4 else
            "#22c55e"
        )
        st.markdown(f"""
        <div class="sky-card">
          <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.08em;opacity:0.7;margin-bottom:0.5rem">
            👁️ Visibility Detail
          </div>
          <div style="font-size:2rem;font-weight:700;color:{vis_color}">
            {p['visibility_km']} km
          </div>
          <div style="font-size:0.88rem;opacity:0.85;margin-top:0.4rem;
                      line-height:1.5">{vis_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Refresh button ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_r1, col_r2, col_r3 = st.columns([2, 1, 2])
    with col_r2:
        if st.button("🔄 Refresh"):
            st.session_state.current_preds = None
            st.session_state.all_forecasts = None
            st.session_state.forecast_df   = None
            st.rerun()

else:
    # ── LANDING SCREEN ───────────────────────────────────────
    apply_theme("Clear", True)
    st.markdown("""
    <div style="text-align:center;padding:4rem 1rem 2rem">
      <div style="font-size:4rem">🌤️</div>
      <h1 style="font-size:2.5rem;font-weight:800;margin:0.5rem 0">
        SkyMind
      </h1>
      <p style="font-size:1.1rem;opacity:0.75;max-width:480px;margin:0 auto">
        AI-powered weather intelligence. Search any location to get
        deep weather insights powered by 10 machine learning models
        trained on 2M+ global observations.
      </p>
      <div style="margin-top:2rem;display:flex;justify-content:center;
                  gap:1rem;flex-wrap:wrap;font-size:0.9rem;opacity:0.7">
        <span>🌡️ Temperature</span>
        <span>🌧️ Rain Probability</span>
        <span>⛈️ Thunderstorm Risk</span>
        <span>🥵 Heat Stress</span>
        <span>👁️ Visibility</span>
        <span>🏃 Outdoor Score</span>
        <span>🌱 Soil Conditions</span>
        <span>☀️ UV Index</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
