"""
weather.py — Current weather display using Open-Meteo free API.

No API key required. Shows current conditions for the farmer's region.
Does NOT change recommendations — display only.

Locations: Common Rwandan/East African farming regions.
"""

import logging
import json
from typing import Optional, Dict
from urllib.request import urlopen, Request
from urllib.error import URLError

logger = logging.getLogger(__name__)

# Common farming regions with coordinates
LOCATIONS = {
    "kigali": {"name": "Kigali", "lat": -1.9403, "lon": 29.8739},
    "musanze": {"name": "Musanze", "lat": -1.4975, "lon": 29.6345},
    "huye": {"name": "Huye", "lat": -2.5966, "lon": 29.7394},
    "rubavu": {"name": "Rubavu", "lat": -1.7472, "lon": 29.2728},
    "nyagatare": {"name": "Nyagatare", "lat": -1.2988, "lon": 30.3277},
    "muhanga": {"name": "Muhanga", "lat": -2.0842, "lon": 29.7558},
    "rwamagana": {"name": "Rwamagana", "lat": -1.9494, "lon": 30.4347},
    "karongi": {"name": "Karongi", "lat": -2.0658, "lon": 29.3700},
    "rusizi": {"name": "Rusizi", "lat": -2.4840, "lon": 28.9080},
    "nairobi": {"name": "Nairobi", "lat": -1.2921, "lon": 36.8219},
    "kampala": {"name": "Kampala", "lat": 0.3476, "lon": 32.5825},
    "bujumbura": {"name": "Bujumbura", "lat": -3.3731, "lon": 29.3644},
    "dar es salaam": {"name": "Dar es Salaam", "lat": -6.7924, "lon": 39.2083},
    # Default fallback
    "default": {"name": "Kigali", "lat": -1.9403, "lon": 29.8739},
}


def _match_location(region_text: str) -> Dict:
    """Match a farmer's region text to a known location."""
    if not region_text:
        return LOCATIONS["default"]

    text = region_text.lower().strip()
    for key, loc in LOCATIONS.items():
        if key in text or loc["name"].lower() in text:
            return loc
    return LOCATIONS["default"]


def get_weather(region: str = "") -> Optional[Dict]:
    """
    Fetch current weather for a region using Open-Meteo.
    Returns dict with temperature, humidity, conditions, etc.
    Returns None on failure.
    """
    loc = _match_location(region)

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={loc['lat']}&longitude={loc['lon']}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"
        f"&timezone=Africa%2FKigali"
        f"&forecast_days=3"
    )

    try:
        req = Request(url, headers={"User-Agent": "AgiriteChat/1.0"})
        resp = urlopen(req, timeout=8)
        data = json.loads(resp.read().decode("utf-8"))

        current = data.get("current", {})
        daily = data.get("daily", {})

        # Decode weather code to description
        weather_desc = _decode_weather_code(current.get("weather_code", 0))

        result = {
            "location": loc["name"],
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation", 0),
            "wind_speed": current.get("wind_speed_10m"),
            "description": weather_desc,
            "forecast": [],
        }

        # 3-day forecast
        if daily:
            dates = daily.get("time", [])
            maxs = daily.get("temperature_2m_max", [])
            mins = daily.get("temperature_2m_min", [])
            precip = daily.get("precipitation_sum", [])
            prob = daily.get("precipitation_probability_max", [])
            for i in range(min(3, len(dates))):
                result["forecast"].append({
                    "date": dates[i] if i < len(dates) else "",
                    "max_temp": maxs[i] if i < len(maxs) else None,
                    "min_temp": mins[i] if i < len(mins) else None,
                    "precipitation": precip[i] if i < len(precip) else 0,
                    "rain_chance": prob[i] if i < len(prob) else 0,
                })

        return result

    except (URLError, json.JSONDecodeError, KeyError) as e:
        logger.warning("Weather fetch failed: %s", e)
        return None
    except Exception as e:
        logger.warning("Weather fetch unexpected error: %s", e)
        return None


def _decode_weather_code(code: int) -> str:
    """Convert WMO weather code to human-readable description."""
    codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return codes.get(code, "Unknown")
