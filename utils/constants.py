# utils/constants.py
# All static configuration for SkyMind

import numpy as np

# ── Training cities (used for nearest-fallback logic) ────────
TRAINING_CITIES = [
    {"name": "Berlin",       "country": "Germany",       "lat": 52.52,  "lon": 13.41},
    {"name": "New York",     "country": "USA",           "lat": 40.71,  "lon": -74.01},
    {"name": "Itakpe",       "country": "Nigeria",       "lat": 7.61,   "lon": 6.27},
    {"name": "Tokyo",        "country": "Japan",         "lat": 35.68,  "lon": 139.69},
    {"name": "Riyadh",       "country": "Saudi Arabia",  "lat": 24.71,  "lon": 46.67},
    {"name": "Antarctica",   "country": "Antarctica",    "lat": -75.25, "lon": -0.07},
    {"name": "Sydney",       "country": "Australia",     "lat": -33.87, "lon": 151.21},
    {"name": "Moscow",       "country": "Russia",        "lat": 55.75,  "lon": 37.62},
    {"name": "Mexico City",  "country": "Mexico",        "lat": 19.43,  "lon": -99.13},
    {"name": "New Delhi",    "country": "India",         "lat": 28.61,  "lon": 77.20},
    {"name": "Singapore",    "country": "Singapore",     "lat": 1.29,   "lon": 103.85},
    {"name": "Dubai",        "country": "UAE",           "lat": 25.20,  "lon": 55.27},
    {"name": "Rio de Janeiro","country": "Brazil",       "lat": -22.91, "lon": -43.17},
    {"name": "Stockholm",    "country": "Sweden",        "lat": 59.33,  "lon": 18.07},
    {"name": "London",       "country": "UK",            "lat": 51.51,  "lon": -0.13},
    {"name": "Cairo",        "country": "Egypt",         "lat": 30.04,  "lon": 31.24},
    {"name": "Beijing",      "country": "China",         "lat": 39.90,  "lon": 116.41},
    {"name": "Paris",        "country": "France",        "lat": 48.85,  "lon": 2.35},
    {"name": "Nairobi",      "country": "Kenya",         "lat": -1.29,  "lon": 36.82},
    {"name": "Toronto",      "country": "Canada",        "lat": 43.65,  "lon": -79.38},
]

# ── Physical bounds for input validation ─────────────────────
FEATURE_BOUNDS = {
    "temperature_2m":          (-90,   60),
    "apparent_temperature":    (-90,   70),
    "relative_humidity_2m":    (0,    100),
    "dew_point_2m":            (-90,   35),
    "wind_speed_10m":          (0,    120),
    "wind_gusts_10m":          (0,    180),
    "pressure_msl":            (870, 1084),
    "uv_index":                (0,     20),
    "uv_index_clear_sky":      (0,     20),
    "cloud_cover":             (0,    100),
    "cloud_cover_low":         (0,    100),
    "cloud_cover_mid":         (0,    100),
    "cloud_cover_high":        (0,    100),
    "visibility":              (0,  24000),
    "precipitation":           (0,    500),
    "rain":                    (0,    500),
    "snowfall":                (0,    200),
    "shortwave_radiation":     (0,   1400),
    "direct_radiation":        (0,   1200),
    "soil_temperature_0_to_7cm":   (-30, 60),
    "soil_moisture_0_to_7cm":      (0,    1),
}

# ── Weather condition themes for dynamic UI ──────────────────
WEATHER_THEMES = {
    "Clear":        {
        "day":   {"bg": "linear-gradient(135deg, #1a6fc4 0%, #38b6ff 50%, #87ceeb 100%)",
                  "text": "#ffffff", "card": "rgba(255,255,255,0.15)"},
        "night": {"bg": "linear-gradient(135deg, #0a0a2e 0%, #1a1a5e 50%, #2d2d8f 100%)",
                  "text": "#e0e8ff", "card": "rgba(255,255,255,0.08)"},
    },
    "Cloudy":       {
        "day":   {"bg": "linear-gradient(135deg, #6b8caa 0%, #8fa8c0 50%, #b0c4d8 100%)",
                  "text": "#ffffff", "card": "rgba(255,255,255,0.15)"},
        "night": {"bg": "linear-gradient(135deg, #2c3e50 0%, #3d5166 50%, #4a6278 100%)",
                  "text": "#d0dde8", "card": "rgba(255,255,255,0.08)"},
    },
    "Rain":         {
        "day":   {"bg": "linear-gradient(135deg, #2c3e6e 0%, #3d5482 50%, #4a6596 100%)",
                  "text": "#d0e4ff", "card": "rgba(255,255,255,0.12)"},
        "night": {"bg": "linear-gradient(135deg, #0d1b2a 0%, #1a2d42 50%, #243d56 100%)",
                  "text": "#a0c4e0", "card": "rgba(255,255,255,0.08)"},
    },
    "Thunderstorm": {
        "day":   {"bg": "linear-gradient(135deg, #1a1a2e 0%, #2d2d4e 50%, #3a3a6e 100%)",
                  "text": "#c8d8ff", "card": "rgba(255,255,255,0.10)"},
        "night": {"bg": "linear-gradient(135deg, #0a0a14 0%, #14141e 50%, #1e1e2e 100%)",
                  "text": "#a0b0e0", "card": "rgba(255,255,255,0.07)"},
    },
    "Snow":         {
        "day":   {"bg": "linear-gradient(135deg, #c8dff0 0%, #ddeeff 50%, #eef6ff 100%)",
                  "text": "#1a3a5c", "card": "rgba(255,255,255,0.25)"},
        "night": {"bg": "linear-gradient(135deg, #1a2a3a 0%, #243650 50%, #2e4260 100%)",
                  "text": "#c0d8f0", "card": "rgba(255,255,255,0.10)"},
    },
    "Fog":          {
        "day":   {"bg": "linear-gradient(135deg, #8a9aaa 0%, #a0b0be 50%, #b8c8d4 100%)",
                  "text": "#ffffff", "card": "rgba(255,255,255,0.18)"},
        "night": {"bg": "linear-gradient(135deg, #2a3038 0%, #363e48 50%, #424c58 100%)",
                  "text": "#c8d4dc", "card": "rgba(255,255,255,0.10)"},
    },
}

# ── WMO weather code → condition group ───────────────────────
WMO_TO_GROUP = {
    0: "Clear",
    1: "Clear", 2: "Cloudy", 3: "Cloudy",
    45: "Fog", 48: "Fog",
    51: "Rain", 53: "Rain", 55: "Rain",
    56: "Rain", 57: "Rain",
    61: "Rain", 63: "Rain", 65: "Rain",
    71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
    80: "Rain", 81: "Rain", 82: "Rain",
    85: "Snow", 86: "Snow",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}

# ── WMO weather code → display label ─────────────────────────
WMO_LABELS = {
    0:  "Clear Sky",
    1:  "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy Fog",
    51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle",
    61: "Light Rain", 63: "Rain", 65: "Heavy Rain",
    71: "Light Snow", 73: "Snow", 75: "Heavy Snow", 77: "Snow Grains",
    80: "Rain Showers", 81: "Showers", 82: "Violent Showers",
    85: "Snow Showers", 86: "Heavy Snow Showers",
    95: "Thunderstorm", 96: "Thunderstorm with Hail",
    99: "Severe Thunderstorm",
}

# ── Weather condition emojis ─────────────────────────────────
WMO_EMOJIS = {
    "Clear":        {"day": "☀️",  "night": "🌙"},
    "Cloudy":       {"day": "⛅",  "night": "☁️"},
    "Rain":         {"day": "🌧️", "night": "🌧️"},
    "Thunderstorm": {"day": "⛈️", "night": "⛈️"},
    "Snow":         {"day": "❄️",  "night": "❄️"},
    "Fog":          {"day": "🌫️", "night": "🌫️"},
}

# ── Heat stress levels ────────────────────────────────────────
HEAT_STRESS_LABELS = {
    0: {"label": "Safe",          "color": "#2ecc71", "emoji": "✅"},
    1: {"label": "Caution",       "color": "#f39c12", "emoji": "⚠️"},
    2: {"label": "Danger",        "color": "#e74c3c", "emoji": "🔴"},
    3: {"label": "Extreme Danger","color": "#8e44ad", "emoji": "🚨"},
}

# ── UV index levels ───────────────────────────────────────────
UV_LEVELS = [
    (0,  2,  "Low",       "#4fc3f7", "🟦"),
    (3,  5,  "Moderate",  "#a5d6a7", "🟩"),
    (6,  7,  "High",      "#fff176", "🟨"),
    (8,  10, "Very High", "#ffb74d", "🟧"),
    (11, 99, "Extreme",   "#ef9a9a", "🟥"),
]

# ── Outdoor score labels ──────────────────────────────────────
OUTDOOR_SCORE_LABELS = [
    (0,  20, "Stay Indoors",    "🏠"),
    (21, 40, "Not Ideal",       "😐"),
    (41, 60, "Acceptable",      "🌤️"),
    (61, 80, "Good",            "😊"),
    (81, 100,"Perfect Outside", "🌟"),
]

# ── Confidence warning threshold (km) ────────────────────────
FAR_LOCATION_THRESHOLD_KM = 2000

# ── Open-Meteo API variables to fetch ────────────────────────
OPENMETEO_HOURLY_VARS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "dew_point_2m", "precipitation", "rain", "showers", "snowfall",
    "weather_code", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
    "cloud_cover_high", "visibility", "wind_speed_10m", "wind_direction_10m",
    "wind_gusts_10m", "pressure_msl", "surface_pressure",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "uv_index", "uv_index_clear_sky",
    "sunshine_duration", "is_day", "cape", "lifted_index",
    "convective_inhibition", "et0_fao_evapotranspiration",
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
]
