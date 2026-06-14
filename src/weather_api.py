# src/weather_api.py
# Open-Meteo API calls with full edge case handling

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from utils.constants import OPENMETEO_HOURLY_VARS, FEATURE_BOUNDS
from src.geocoding import find_nearest_training_city


def _validate_and_clip(df):
    """Clip all variables to physically valid bounds."""
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def _synthesize_missing(df):
    """
    Synthesize commonly missing convective variables
    using the same approximations as training.
    """
    # CAPE: approximate from temp × humidity
    if "cape" not in df.columns or df["cape"].isna().mean() > 0.5:
        df["cape"] = np.clip(
            df["temperature_2m"] * df["relative_humidity_2m"] / 10,
            0, 2000
        )

    # Lifted Index: approximate from temperature
    if "lifted_index" not in df.columns or df["lifted_index"].isna().mean() > 0.5:
        df["lifted_index"] = -((df["temperature_2m"] - 20) / 5)

    # Convective inhibition
    if "convective_inhibition" not in df.columns or df["convective_inhibition"].isna().mean() > 0.5:
        df["convective_inhibition"] = np.clip(
            200 - df["cape"] / 10, 0, 200
        )

    # UV index from shortwave radiation if missing
    if "uv_index" not in df.columns or df["uv_index"].isna().mean() > 0.5:
        if "shortwave_radiation" in df.columns:
            df["uv_index"] = (df["shortwave_radiation"] / 25).clip(lower=0)
        else:
            df["uv_index"] = 0.0

    if "uv_index_clear_sky" not in df.columns or df["uv_index_clear_sky"].isna().mean() > 0.5:
        if "direct_radiation" in df.columns:
            df["uv_index_clear_sky"] = (df["direct_radiation"] / 25).clip(lower=0)
        else:
            df["uv_index_clear_sky"] = df.get("uv_index", pd.Series(0.0))

    # Soil variables — fill with climate-zone defaults if missing
    soil_vars = [
        "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
        "soil_temperature_28_to_100cm", "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
    ]
    for var in soil_vars:
        if var not in df.columns or df[var].isna().mean() > 0.5:
            if "temperature" in var:
                df[var] = df["temperature_2m"] * 0.8
            else:
                df[var] = 0.25  # moderate moisture default

    # ET0 evapotranspiration
    if "et0_fao_evapotranspiration" not in df.columns or \
       df["et0_fao_evapotranspiration"].isna().mean() > 0.5:
        df["et0_fao_evapotranspiration"] = np.clip(
            df["temperature_2m"] * 0.02 + 0.5, 0, 10
        )

    # Sunshine duration
    if "sunshine_duration" not in df.columns or df["sunshine_duration"].isna().mean() > 0.5:
        df["sunshine_duration"] = df.get("is_day", pd.Series(0)) * 2400

    # Visibility
    if "visibility" not in df.columns or df["visibility"].isna().mean() > 0.5:
        # Estimate from cloud cover and humidity
        df["visibility"] = np.clip(
            24000 - df["cloud_cover"] * 200 -
            df["relative_humidity_2m"] * 100,
            1000, 24000
        )

    # Apparent temperature
    if "apparent_temperature" not in df.columns or \
       df["apparent_temperature"].isna().mean() > 0.5:
        df["apparent_temperature"] = df["temperature_2m"]

    # Heat index
    if "heat_index" not in df.columns:
        t = df["temperature_2m"]
        h = df["relative_humidity_2m"]
        df["heat_index"] = (
            -8.78469475556 +
            1.61139411 * t +
            2.33854883889 * h +
            -0.14611605 * t * h +
            -0.012308094 * t**2 +
            -0.0164248277778 * h**2 +
            0.002211732 * t**2 * h +
            0.00072546 * t * h**2 +
            -0.000003582 * t**2 * h**2
        )

    return df


def fetch_forecast(lat, lon, days=7):
    """
    Fetch forecast data from Open-Meteo for given coordinates.
    Returns a cleaned DataFrame with all required variables,
    or raises an exception with a descriptive message.

    Edge cases handled:
    - API unavailable → raises with friendly message
    - Missing variables → synthesized from available data
    - Physical bounds exceeded → clipped
    - All-NaN columns → filled with defaults
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    ",".join(OPENMETEO_HOURLY_VARS),
        "forecast_days": days,
        "timezone": "auto",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot reach Open-Meteo API. Check your internet connection."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            "Open-Meteo API timed out. Please try again in a moment."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Open-Meteo returned an error: {e}")

    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError(
            "No forecast data returned for this location. "
            "Try a nearby city."
        )

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["hour"]  = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["year"]  = df["time"].dt.year

    # Season (1=Winter, 2=Spring, 3=Summer, 4=Autumn)
    # Hemisphere-aware
    is_southern = lat < 0
    month_to_season_north = {12:1,1:1,2:1, 3:2,4:2,5:2,
                              6:3,7:3,8:3, 9:4,10:4,11:4}
    if is_southern:
        month_to_season_south = {6:1,7:1,8:1, 9:2,10:2,11:2,
                                  12:3,1:3,2:3, 3:4,4:4,5:4}
        df["season"] = df["month"].map(month_to_season_south)
    else:
        df["season"] = df["month"].map(month_to_season_north)

    df["is_night"] = 1 - df.get("is_day", pd.Series(1, index=df.index))
    df["latitude"]  = lat
    df["longitude"] = lon

    # Synthesize missing / broken variables
    df = _synthesize_missing(df)

    # Validate physical bounds
    df = _validate_and_clip(df)

    # Final NaN fill — use column median or 0
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median if not np.isnan(median) else 0.0)

    return df


def check_soil_data_quality(df):
    """
    Returns True if soil data is reliable enough to show.
    Soil data is unreliable in deep ocean / ice sheets.
    """
    soil_cols = ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"]
    for col in soil_cols:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) == 0 or vals.std() < 0.001:
                return False
    return True
