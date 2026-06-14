# src/geocoding.py
# Location search, GPS handling, nearest-city fallback

import math
import requests
from utils.constants import TRAINING_CITIES, FAR_LOCATION_THRESHOLD_KM


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance between two lat/lon points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def find_nearest_training_city(lat, lon):
    """
    Returns the closest training city to given coordinates.
    Used for fallback when Open-Meteo returns no data.
    """
    nearest = None
    min_dist = float("inf")
    for city in TRAINING_CITIES:
        dist = haversine_km(lat, lon, city["lat"], city["lon"])
        if dist < min_dist:
            min_dist = dist
            nearest = city.copy()
            nearest["distance_km"] = round(dist, 1)
    return nearest


def is_far_from_training_data(lat, lon):
    """Returns (bool, distance_km, nearest_city_name)."""
    nearest = find_nearest_training_city(lat, lon)
    far = nearest["distance_km"] > FAR_LOCATION_THRESHOLD_KM
    return far, nearest["distance_km"], nearest["name"]


def geocode_city(city_name, max_results=5):
    """
    Search for a city using Nominatim (OpenStreetMap).
    Returns list of results with name, country, lat, lon.
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": city_name,
            "format": "json",
            "limit": max_results,
            "addressdetails": 1,
        }
        headers = {"User-Agent": "SkyMind-Weather-App/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data:
            addr = item.get("address", {})
            country = addr.get("country", "")
            city = (addr.get("city") or addr.get("town") or
                    addr.get("village") or addr.get("county") or
                    item.get("display_name", "").split(",")[0])
            results.append({
                "display": f"{city}, {country}",
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
            })
        return results

    except Exception as e:
        return []


def reverse_geocode(lat, lon):
    """
    Convert lat/lon to a human-readable location name.
    Used when user grants GPS access.
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"lat": lat, "lon": lon, "format": "json"}
        headers = {"User-Agent": "SkyMind-Weather-App/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        addr = data.get("address", {})
        city = (addr.get("city") or addr.get("town") or
                addr.get("village") or addr.get("county") or "Your Location")
        country = addr.get("country", "")
        return f"{city}, {country}"
    except Exception:
        return f"{lat:.2f}°, {lon:.2f}°"
