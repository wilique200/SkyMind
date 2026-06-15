# src/descriptions.py
# Rule-based smart weather descriptions for SkyMind.
# Generates contextual, conversational text based on predictions.

from datetime import datetime


def _time_of_day(hour):
    if 5  <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 21: return "evening"
    return "night"


def _temp_feel(temp):
    if temp < 0:   return "freezing"
    if temp < 10:  return "very cold"
    if temp < 16:  return "cold"
    if temp < 21:  return "cool"
    if temp < 26:  return "pleasant"
    if temp < 31:  return "warm"
    if temp < 36:  return "hot"
    return "extremely hot"


def _humidity_desc(humidity):
    if humidity < 30: return "very dry"
    if humidity < 50: return "comfortable"
    if humidity < 70: return "slightly humid"
    if humidity < 85: return "humid"
    return "very humid"


def _wind_desc(kmh):
    if kmh < 5:   return "calm"
    if kmh < 15:  return "light breeze"
    if kmh < 30:  return "moderate wind"
    if kmh < 50:  return "strong wind"
    if kmh < 75:  return "very strong wind"
    return "storm-force wind"
      


def generate_main_description(preds, location_name):
    """
    Generate the main 2-3 sentence weather description shown
    at the top of the app under the temperature.
    """
    temp     = preds["temperature"]
    feels    = preds["apparent_temperature"]
    humidity = preds["humidity"]
    condition = preds["weather_condition"]
    wind_kmh  = preds["wind_speed_kmh"]
    rain_prob = preds["rain_probability"]
    hour      = preds["hour"]
    is_day    = preds["is_day"]

    time_str  = _time_of_day(hour)
    temp_str  = _temp_feel(temp)
    humid_str = _humidity_desc(humidity)
    wind_str  = _wind_desc(wind_kmh)
    diff      = round(feels - temp, 1)

    # Opening sentence — condition + temperature
    if condition == "Clear" and is_day:
        opening = f"It's a sunny {time_str} in {location_name}, with {temp_str} temperatures at {temp:.0f}°C."
    elif condition == "Clear" and not is_day:
        opening = f"Clear skies tonight in {location_name}. Temperatures have dropped to {temp:.0f}°C."
    elif condition == "Cloudy":
        opening = f"Cloudy skies over {location_name} this {time_str}, with temperatures around {temp:.0f}°C."
    elif condition == "Rain":
        opening = f"Rainy conditions in {location_name} this {time_str}, with temperatures at {temp:.0f}°C."
    elif condition == "Thunderstorm":
        opening = f"⚡ Thunderstorm conditions in {location_name}. Stay indoors if possible — temperatures are {temp:.0f}°C."
    elif condition == "Snow":
        opening = f"❄️ Snowfall in {location_name} this {time_str}, with temperatures below {temp:.0f}°C."
    elif condition == "Fog":
        opening = f"Foggy conditions reducing visibility in {location_name} this {time_str}. Temperature is {temp:.0f}°C."
    else:
        opening = f"Current temperature in {location_name} is {temp:.0f}°C this {time_str}."

    # Feels-like sentence
    if abs(diff) >= 2:
        direction = "warmer" if diff > 0 else "cooler"
        feels_sentence = (
            f"With {humid_str} air and {wind_str}, it feels {direction} "
            f"at {feels:.0f}°C."
        )
    else:
        feels_sentence = f"Humidity is {humid_str}, making it feel about right at {feels:.0f}°C."

    # Practical tip
    if condition == "Thunderstorm":
        tip = "⚡ Avoid open areas and tall structures. Seek solid shelter immediately."
    elif rain_prob > 70:
        tip = f"🌧️ High chance of rain ({rain_prob:.0f}%). Carry an umbrella and waterproof clothing."
    elif rain_prob > 40:
        tip = f"🌦️ There's a {rain_prob:.0f}% chance of rain — keep an umbrella handy."
    elif condition == "Snow":
        tip = "❄️ Dress in layers and watch for icy surfaces when walking or driving."
    elif condition == "Fog":
        tip = "🌫️ Drive carefully — reduced visibility. Use fog lights and slow down."
    elif preds["heat_stress_level"] >= 2 and temp >= 27:
        tip = "🥵 Dangerous heat — stay hydrated, seek shade, and avoid outdoor activity between 11am–3pm."
    elif preds["heat_stress_level"] == 1 and temp >= 22:
        tip = "☀️ It's warm out — drink water regularly and use sunscreen."
    elif temp < 0:
        tip = "🧊 Sub-zero temperatures — dress in thermal layers and protect exposed skin."
    elif temp < 10:
        tip = "🧥 It's cold — wear a warm coat and consider gloves."
    elif preds["uv_index"] >= 8:
        tip = f"☀️ Very high UV ({preds['uv_index']}) — apply SPF 50+ and wear a hat outdoors."
    elif wind_kmh > 50:
        tip = "💨 Strong winds — secure loose objects and be cautious when driving high-sided vehicles."
    elif preds["outdoor_score"] >= 80:
        tip = "🌟 Excellent conditions for outdoor activities — enjoy the weather!"
    else:
        tip = f"Current conditions are {temp_str} and {humid_str}."

    return f"{opening} {feels_sentence} {tip}"


def generate_uv_description(uv_index, uv_label, is_day):
    """Short UV advice sentence."""
    if not is_day or uv_index == 0:
        return "No UV radiation at night."
    if uv_label == "Low":
        return "UV levels are low. No protection needed for most people."
    if uv_label == "Moderate":
        return "Moderate UV. Wear sunscreen if outdoors for extended periods."
    if uv_label == "High":
        return "High UV. Apply SPF 30+ and wear a hat between 10am–4pm."
    if uv_label == "Very High":
        return "Very high UV. Stay in shade, apply SPF 50+, and wear UV-blocking sunglasses."
    return "Extreme UV. Minimize outdoor exposure. Full protection essential."


def generate_soil_description(soil_score, soil_label, irrigation_need=None):
    """Agricultural advice based on soil conditions."""
    if soil_score >= 80:
        base = "Soil conditions are excellent. Optimal moisture and temperature for plant growth."
    elif soil_score >= 60:
        base = "Good soil conditions. Most crops should thrive in current conditions."
    elif soil_score >= 40:
        base = "Moderate soil conditions. Monitor moisture levels and adjust irrigation as needed."
    elif soil_score >= 20:
        base = "Poor soil conditions. Consider irrigation and soil temperature management."
    else:
        base = "Very poor soil conditions. Crops may be stressed — immediate attention recommended."

    if irrigation_need is not None:
        if irrigation_need > 5:
            base += " 💧 Irrigation is strongly recommended today."
        elif irrigation_need > 2:
            base += " 💧 Light irrigation may be beneficial."

    return base


def generate_heat_stress_description(level, label, temp, humidity):
    """Health advice based on heat stress level."""
    if level == 0:
        return f"No heat stress risk at {temp:.0f}°C. Safe for all outdoor activities."
    if level == 1:
        return (
            f"⚠️ Caution: {temp:.0f}°C with {humidity:.0f}% humidity. "
            f"Fatigue possible with prolonged exposure. Stay hydrated."
        )
    if level == 2:
        return (
            f"🔴 Danger: Heat conditions are hazardous at {temp:.0f}°C. "
            f"Heat cramps and exhaustion likely. Avoid outdoor work during peak hours."
        )
    return (
        f"🚨 Extreme Danger: Life-threatening heat at {temp:.0f}°C. "
        f"Heat stroke is imminent without action. Seek air conditioning immediately."
    )


def generate_thunderstorm_description(prob, risk):
    """Thunderstorm risk contextual message."""
    if prob < 10:
        return "Atmospheric conditions are stable. No thunderstorm risk detected."
    if prob < 30:
        return f"Low thunderstorm risk ({prob:.0f}%). Conditions are mostly stable."
    if prob < 60:
        return (
            f"⚠️ Moderate thunderstorm risk ({prob:.0f}%). "
            f"Stay alert for changing conditions, especially in the afternoon."
        )
    return (
        f"⛈️ High thunderstorm risk ({prob:.0f}%). "
        f"Seek shelter, avoid trees and open fields. Unplug sensitive electronics."
    )


def generate_visibility_description(vis_km, label):
    """Driving and activity advice based on visibility."""
    if vis_km >= 20:
        return f"Excellent visibility ({vis_km:.0f} km). Clear conditions for all activities."
    if vis_km >= 10:
        return f"Good visibility ({vis_km:.0f} km). Normal driving conditions."
    if vis_km >= 4:
        return f"Moderate visibility ({vis_km:.1f} km). Drive carefully and use headlights."
    if vis_km >= 1:
        return (
            f"⚠️ Poor visibility ({vis_km:.1f} km). "
            f"Use fog lights, reduce speed significantly, and increase following distance."
        )
    return (
        f"🚨 Very poor visibility ({vis_km:.1f} km). "
        f"Dangerous driving conditions. Avoid travel unless essential."
    )
