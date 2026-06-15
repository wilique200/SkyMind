# src/feature_engineering.py
# CRITICAL: Must be identical to training feature engineering.
# Any change here must also be applied to weather_model_utils.py

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    Called on Open-Meteo forecast data before model inference.
    Input df must have: hour, month, latitude, longitude,
    temperature_2m, relative_humidity_2m, dew_point_2m,
    pressure_msl, cloud_cover, is_day, is_night, season,
    wind_speed_10m, wind_direction_10m
    """
    df = df.copy()
    # Ensure rain_flag exists (required by outdoor_score model)
    if 'rain_flag' not in df.columns:
        if 'rain' in df.columns:
            df['rain_flag'] = (df['rain'] > 0).astype(int)
        else:
            df['rain_flag'] = 0
    # ── Cyclical time encoding ───────────────────────────────
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_x_month"] = df["hour"] * df["month"]

    # ── Solar position proxy ─────────────────────────────────
    df["solar_hour_sin"] = np.sin(np.pi * df["hour"] / 24) * df["is_day"]
    df["solar_hour_cos"] = np.cos(np.pi * df["hour"] / 24) * df["is_day"]

    # ── Hemisphere-aware season ──────────────────────────────
    df["hemi"] = df["latitude"].apply(lambda x: -1 if x < 0 else 1)
    df["hemi_month_sin"] = df["month_sin"] * df["hemi"]
    df["hemi_month_cos"] = df["month_cos"] * df["hemi"]

    # ── Location features ────────────────────────────────────
    df["abs_latitude"] = abs(df["latitude"])
    df["tropical"]  = (df["abs_latitude"] < 23.5).astype(int)
    df["polar"]     = (df["abs_latitude"] > 66.5).astype(int)
    df["temperate"] = (~df["tropical"].astype(bool) &
                       ~df["polar"].astype(bool)).astype(int)
    df["is_tropical_zone"] = df["tropical"]

    # ── Atmospheric derived ──────────────────────────────────
    df["dew_depression"]       = df["temperature_2m"] - df["dew_point_2m"]
    df["pressure_anomaly"]     = df["pressure_msl"] - 1013.25
    df["pressure_abs_anomaly"] = abs(df["pressure_anomaly"])

    # ── Cloud features ───────────────────────────────────────
    df["cloud_heating_block"] = (df["cloud_cover"] / 100) * df["is_day"]
    df["heavy_cloud"] = (df["cloud_cover"] > 75).astype(int)

    # ── Radiation ────────────────────────────────────────────
    if "shortwave_radiation" in df.columns:
        df["radiation_day"] = df["shortwave_radiation"] * df["is_day"]

    # ── UV ───────────────────────────────────────────────────
    if "uv_index" in df.columns:
        df["uv_index"] = df["uv_index"].clip(lower=0)
    if "uv_index_clear_sky" in df.columns:
        df["uv_index_clear_sky"] = df["uv_index_clear_sky"].clip(lower=0)

    # ── Wind direction cyclical ──────────────────────────────
    if "wind_direction_10m" in df.columns:
        df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction_10m"]))
        df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction_10m"]))

    # ── Gust ratio ───────────────────────────────────────────
    if "wind_gusts_10m" in df.columns:
        df["gust_ratio"] = df["wind_gusts_10m"] / (df["wind_speed_10m"] + 0.1)

    # ── Humidity signals ─────────────────────────────────────
    df["humidity_above_70"] = (df["relative_humidity_2m"] > 70).astype(int)
    df["humidity_above_85"] = (df["relative_humidity_2m"] > 85).astype(int)
    df["near_saturation"]   = (df["dew_depression"] < 3).astype(int)
    df["fog_index"]         = df["relative_humidity_2m"] / (
        df["dew_depression"].abs() + 0.01)
    df["night_humidity"]    = df["is_night"] * df["relative_humidity_2m"]

    # ── Location flags ───────────────────────────────────────
    df["high_latitude"]   = (df["abs_latitude"] > 50).astype(int)
    df["tropical_zone"]   = df["tropical"]

    # ── Storm signals ────────────────────────────────────────
    if "cape" in df.columns:
        cape_thresh  = df["cape"].quantile(0.75)
        li_thresh    = df["lifted_index"].quantile(0.25)
        cin_thresh   = df["convective_inhibition"].quantile(0.25)
        cloud_thresh = df["heavy_cloud"].quantile(0.70)

        df["li_unstable"]      = (df["lifted_index"] < 0).astype(int)
        df["li_very_unstable"] = (df["lifted_index"] < -4).astype(int)
        df["moist_instability"] = (
            df["relative_humidity_2m"] * (df["cape"] + 1) / 1000
        )
        df["storm_fire_signal"] = (
            (df["cape"] > cape_thresh) &
            (df["lifted_index"] < li_thresh) &
            (df["convective_inhibition"] < cin_thresh) &
            (df["heavy_cloud"] > cloud_thresh)
        ).astype(int)

    # ── Heat features ────────────────────────────────────────
    df["humid_heat"]     = df["temperature_2m"] * df["relative_humidity_2m"] / 100
    df["peak_heat_hour"] = ((df["hour"] >= 11) & (df["hour"] <= 15)).astype(int)
    df["night_heat"]     = df["is_night"] * df["temperature_2m"]
    df["humidex"]        = df["temperature_2m"] + 0.5555 * (
        6.11 * np.exp(5417.753 * (
            1/273.16 - 1/(273.16 + df["dew_point_2m"].clip(lower=-50))
        )) - 10
    )

    # ── Soil features ────────────────────────────────────────
    if "soil_moisture_0_to_7cm" in df.columns:
        df["waterlog_risk"]     = (df["soil_moisture_0_to_7cm"] > 0.45).astype(int)
        df["soil_frost"]        = (df["soil_temperature_0_to_7cm"] < 2).astype(int)
        df["growing_signal"]    = np.clip(df["temperature_2m"] - 10, 0, None)
        df["moisture_gradient"] = (
            df["soil_moisture_0_to_7cm"] - df["soil_moisture_28_to_100cm"]
        )
        df["irrigation_need"]   = (
            df["et0_fao_evapotranspiration"] -
            df["soil_moisture_0_to_7cm"] * 10
        ).clip(lower=0)
        df["root_warmth"] = np.clip(
            df["soil_temperature_7_to_28cm"], 0, 35
        ) / 35

    # ── Solar / outdoor ──────────────────────────────────────
    df["lat_uv_potential"]  = np.cos(np.deg2rad(df["latitude"]))
    df["cloud_attenuation"] = 1 - (df["cloud_cover"] / 100)
    df["temp_comfort"]      = np.clip(30 - abs(df["temperature_2m"] - 22), 0, 30)
    df["season_lat"]        = df["month_sin"] * df["latitude"]

    if "sunshine_duration" in df.columns:
        df["sunshine_hrs"] = df["sunshine_duration"] / 3600

    # ── Rain bins ────────────────────────────────────────────
    df["rain_intensity"] = pd.cut(
        df["rain"],
        bins=[-1, 0, 2.5, 7.5, 999],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # ── Afternoon flag ───────────────────────────────────────
    df["afternoon"] = ((df["hour"] >= 14) & (df["hour"] <= 18)).astype(int)

    # ── Visibility km ────────────────────────────────────────
    if "visibility" in df.columns:
        df["visibility_km"] = (df["visibility"] / 1000).clip(0, 24)

    return df
