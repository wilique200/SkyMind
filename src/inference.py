# src/inference.py
# Runs all 10 models on a single row of engineered features.

import torch
import numpy as np
from utils.constants import (
    WMO_TO_GROUP, WMO_LABELS, HEAT_STRESS_LABELS,
    UV_LEVELS, OUTDOOR_SCORE_LABELS
)


def _predict_single(bundle, row_df):
    """Run one model on one row. Returns raw numpy output."""
    if bundle is None:
        return None
    features = bundle["features"]
    # Keep only features the model was trained on
    available = [f for f in features if f in row_df.columns]
    X = row_df[available].values.astype(np.float32)
    # Fill any missing features with 0
    if len(available) < len(features):
        full = np.zeros((1, len(features)), dtype=np.float32)
        for i, f in enumerate(features):
            if f in available:
                full[0, i] = X[0, available.index(f)]
        X = full
    X = bundle["scaler"].transform(X)
    tensor = torch.tensor(X, dtype=torch.float32).to(bundle["device"])
    with torch.no_grad():
        out = bundle["model"](tensor)
    return out.cpu().numpy()


def get_uv_label(uv):
    from utils.constants import UV_LEVELS
    for lo, hi, label, color, emoji in UV_LEVELS:
        if lo <= uv <= hi:
            return label, color, emoji
    return "Extreme", "#ef9a9a", "🟥"


def get_outdoor_label(score):
    for lo, hi, label, emoji in OUTDOOR_SCORE_LABELS:
        if lo <= score <= hi:
            return label, emoji
    return "Unknown", "❓"


def run_all_models(models, row_df, is_day, hour):
    """
    Run all 10 models on a single engineered row.
    Returns a rich predictions dict ready for the UI.
    """
    preds = {}

    # ── 1. Temperature ───────────────────────────────────────
    out = _predict_single(models.get("temperature"), row_df)
    preds["temperature"] = float(out[0, 0]) if out is not None else \
        float(row_df["temperature_2m"].iloc[0])

    # ── 2. Rain Probability ──────────────────────────────────
    out = _predict_single(models.get("rain_probability"), row_df)
    if out is not None:
        prob = float(torch.sigmoid(torch.tensor(out[0, 0])).item())
    else:
        prob = float(row_df["rain"].iloc[0] > 0)
    preds["rain_probability"] = round(prob * 100, 1)

    # ── 3. Weather Condition ─────────────────────────────────
    out = _predict_single(models.get("weather_condition"), row_df)
    if out is not None:
        class_names = ["Clear", "Cloudy", "Fog", "Rain", "Snow", "Thunderstorm"]
        idx = int(np.argmax(out[0]))
        preds["weather_condition"] = class_names[idx]
    else:
        wmo = int(row_df["weather_code"].iloc[0])
        preds["weather_condition"] = WMO_TO_GROUP.get(wmo, "Cloudy")

    # WMO label for display
    wmo_code = int(row_df["weather_code"].iloc[0]) \
        if "weather_code" in row_df.columns else 0
    preds["weather_label"] = WMO_LABELS.get(wmo_code, "Unknown")
    preds["weather_code"]  = wmo_code

    # ── 4. Wind Speed ────────────────────────────────────────
    out = _predict_single(models.get("wind_speed"), row_df)
    preds["wind_speed"] = float(out[0, 0]) if out is not None else \
        float(row_df["wind_speed_10m"].iloc[0])
    preds["wind_speed_kmh"] = round(preds["wind_speed"] * 3.6, 1)
    preds["wind_direction"] = float(row_df["wind_direction_10m"].iloc[0]) \
        if "wind_direction_10m" in row_df.columns else 0.0

    # ── 5. UV Index ──────────────────────────────────────────
    if not is_day:
        preds["uv_index"] = 0.0
        preds["uv_label"] = "None (Night)"
        preds["uv_color"] = "#64748b"
        preds["uv_emoji"] = "🌙"
    else:
        out = _predict_single(models.get("uv_index"), row_df)
        uv = max(0.0, float(out[0, 0])) if out is not None else \
            float(row_df.get("uv_index", 0).iloc[0])
        label, color, emoji = get_uv_label(uv)
        preds["uv_index"] = round(uv, 1)
        preds["uv_label"] = label
        preds["uv_color"] = color
        preds["uv_emoji"] = emoji

    # ── 6. Thunderstorm Risk ─────────────────────────────────
    out = _predict_single(models.get("thunderstorm"), row_df)
    if out is not None:
        t_prob = float(torch.sigmoid(torch.tensor(out[0, 0])).item())
    else:
        t_prob = 0.0
    preds["thunderstorm_prob"] = round(t_prob * 100, 1)
    preds["thunderstorm_risk"] = (
        "High" if t_prob > 0.6 else
        "Moderate" if t_prob > 0.3 else "Low"
    )

    # ── 7. Visibility ────────────────────────────────────────
    out = _predict_single(models.get("visibility"), row_df)
    vis_km = max(0.0, min(24.0, float(out[0, 0]))) if out is not None else \
        float(row_df.get("visibility_km", 10).iloc[0])
    preds["visibility_km"] = round(vis_km, 1)
    preds["visibility_label"] = (
        "Very Poor"  if vis_km < 1  else
        "Poor"       if vis_km < 4  else
        "Moderate"   if vis_km < 10 else
        "Good"       if vis_km < 20 else "Excellent"
    )

    # ── 8. Outdoor Score ─────────────────────────────────────
    out = _predict_single(models.get("outdoor_score"), row_df)
    score = max(0.0, min(100.0, float(out[0, 0]))) if out is not None else 50.0
    preds["outdoor_score"] = round(score, 1)
    preds["outdoor_label"], preds["outdoor_emoji"] = get_outdoor_label(score)

    # ── 9. Heat Stress ───────────────────────────────────────
    out = _predict_single(models.get("heat_stress"), row_df)
    if out is not None:
        hs_idx = int(np.argmax(out[0]))
    else:
        hi = float(row_df.get("heat_index", row_df["temperature_2m"]).iloc[0])
        hs_idx = 0 if hi < 27 else 1 if hi < 32 else 2 if hi < 41 else 3
    hs_info = HEAT_STRESS_LABELS[hs_idx]
    preds["heat_stress_level"] = hs_idx
    preds["heat_stress_label"] = hs_info["label"]
    preds["heat_stress_color"] = hs_info["color"]
    preds["heat_stress_emoji"] = hs_info["emoji"]

    # ── 10. Soil Score ───────────────────────────────────────
    out = _predict_single(models.get("soil_score"), row_df)
    soil = max(0.0, min(100.0, float(out[0, 0]))) if out is not None else 50.0
    preds["soil_score"] = round(soil, 1)
    preds["soil_label"] = (
        "Very Poor"  if soil < 20 else
        "Poor"       if soil < 40 else
        "Moderate"   if soil < 60 else
        "Good"       if soil < 80 else "Excellent"
    )

    # ── Passthrough values (from API, no model needed) ───────
    preds["apparent_temperature"] = float(
        row_df.get("apparent_temperature", row_df["temperature_2m"]).iloc[0]
    )
    preds["humidity"]    = float(row_df["relative_humidity_2m"].iloc[0])
    preds["pressure"]    = float(row_df["pressure_msl"].iloc[0])
    preds["dew_point"]   = float(row_df["dew_point_2m"].iloc[0])
    preds["cloud_cover"] = float(row_df["cloud_cover"].iloc[0])
    preds["is_day"]      = bool(is_day)
    preds["hour"]        = int(hour)

    return preds


def run_forecast(models, forecast_df):
    """
    Run all models on every hour of the forecast DataFrame.
    Returns list of prediction dicts, one per hour.
    """
    from src.feature_engineering import engineer_features
    engineered = engineer_features(forecast_df)
    all_preds = []

    for i in range(len(engineered)):
        row = engineered.iloc[[i]]
        is_day = bool(row["is_day"].iloc[0])
        hour   = int(row["hour"].iloc[0])
        try:
            p = run_all_models(models, row, is_day, hour)
            p["time"] = forecast_df["time"].iloc[i]
            all_preds.append(p)
        except Exception as e:
            pass  # skip broken rows silently

    return all_preds
