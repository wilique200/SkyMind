# src/edge_cases.py
# Centralized edge case detection and handling for SkyMind

import numpy as np
from src.geocoding import (
    find_nearest_training_city,
    is_far_from_training_data,
    haversine_km
)


class WeatherWarnings:
    """Collects all warnings to display to user in the UI."""
    def __init__(self):
        self.warnings = []

    def add(self, level, message):
        """level: 'info' | 'warning' | 'error'"""
        self.warnings.append({"level": level, "message": message})

    def has_warnings(self):
        return len(self.warnings) > 0

    def get_all(self):
        return self.warnings


def handle_location_fallback(lat, lon, api_error=None):
    """
    If original location fails, find nearest training city.
    Returns: (fallback_lat, fallback_lon, fallback_name, warning_msg)
    """
    nearest = find_nearest_training_city(lat, lon)
    msg = (
        f"Data unavailable for your exact location. "
        f"Showing forecast for {nearest['name']}, {nearest['country']} "
        f"({nearest['distance_km']} km away)."
    )
    return nearest["lat"], nearest["lon"], nearest["name"], msg


def check_location_confidence(lat, lon):
    """
    Warn user if location is far from all training cities.
    Returns: (is_far: bool, warning_msg: str | None)
    """
    is_far, dist_km, nearest_name = is_far_from_training_data(lat, lon)
    if is_far:
        msg = (
            f"This location is {dist_km:.0f} km from our nearest "
            f"training region ({nearest_name}). "
            f"Predictions may be less accurate than usual."
        )
        return True, msg
    return False, None


def handle_missing_soil(df, lat, lon):
    """
    Returns True if soil model should be shown, False if hidden.
    Ocean/ice locations have unreliable soil data.
    """
    # Antarctica check
    if lat < -60:
        return False

    # Deep ocean proxy: if all soil moisture values are identical
    soil_col = "soil_moisture_0_to_7cm"
    if soil_col in df.columns:
        std = df[soil_col].std()
        if std < 0.001:
            return False

    return True


def handle_nighttime_uv(hour, is_day):
    """UV is always 0 at night — skip model entirely."""
    if not is_day:
        return 0.0
    return None  # None means: run the model


def validate_model_output(prediction, model_name, bounds):
    """
    Sanity check model outputs.
    Returns clipped prediction and a warning if out of range.
    """
    lo, hi = bounds
    if prediction < lo or prediction > hi:
        clipped = np.clip(prediction, lo, hi)
        warning = (
            f"{model_name} predicted an unusual value ({prediction:.2f}). "
            f"Clipped to valid range [{lo}, {hi}]."
        )
        return clipped, warning
    return prediction, None


def handle_ambiguous_location(results):
    """
    When geocoding returns multiple results for the same name
    (e.g. 'Paris'), return top candidates for user to choose.
    """
    if not results:
        return None, "Location not found. Please try a different search term."
    if len(results) == 1:
        return results[0], None
    # Multiple results — caller should show dropdown
    return results, None


def assess_api_response(df):
    """
    Check if API response has enough valid data.
    Returns (is_valid: bool, issue_description: str | None)
    """
    if df is None or len(df) == 0:
        return False, "Empty response from weather API."

    # Check core variables are present
    required = ["temperature_2m", "relative_humidity_2m",
                 "wind_speed_10m", "weather_code"]
    for col in required:
        if col not in df.columns:
            return False, f"Missing critical variable: {col}"
        if df[col].isna().mean() > 0.8:
            return False, f"Too many missing values in {col}"

    return True, None
