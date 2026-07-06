"""
AI Weather Summary - Natural-language summary engine.

Generates a short (2-4 sentence) plain-language summary of:
  1. the last 7 observed days of weather, and
  2. the historical comparison for the two forecast dates across past years.

Reuses the same OpenAI API key/model as the thesis chatbot. Summaries are
cached in memory per location + calendar day so repeated clicks don't
re-bill the API.
"""

import os
import re
import json
import datetime as dt

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# In-memory cache: { cache_key: summary_text }
_summary_cache = {}


def _get_openai_api_key() -> str:
    """Resolve the API key at call time (same behaviour as the chatbot)."""
    load_dotenv(override=True)
    return os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")


def _avg(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _summarize_seven_days(seven_day):
    """Compute compact aggregates for the last 7 observed days."""
    if not seven_day:
        return "No recent daily data available."

    mins = [d.get("temperature_2m_min") for d in seven_day]
    maxs = [d.get("temperature_2m_max") for d in seven_day]
    rains = [d.get("rain_sum", 0.0) or 0.0 for d in seven_day]
    humids = [d.get("relative_humidity_2m_mean") for d in seven_day]
    winds = [d.get("wind_speed_10m_mean") for d in seven_day]

    warmest = max(seven_day, key=lambda d: d.get("temperature_2m_max", -999))
    wettest = max(seven_day, key=lambda d: d.get("rain_sum", 0.0) or 0.0)
    rainy_days = sum(1 for r in rains if r > 0.1)

    return (
        f"Dates: {seven_day[0].get('date')} to {seven_day[-1].get('date')} "
        f"({len(seven_day)} days).\n"
        f"Avg min temp: {_avg(mins):.1f} C, avg max temp: {_avg(maxs):.1f} C.\n"
        f"Warmest day: {warmest.get('date')} at {warmest.get('temperature_2m_max'):.1f} C.\n"
        f"Total rainfall: {sum(rains):.1f} mm across {rainy_days} rainy day(s); "
        f"wettest day {wettest.get('date')} with {wettest.get('rain_sum', 0.0):.1f} mm.\n"
        f"Avg humidity: {_avg(humids):.0f}%, avg wind: {_avg(winds):.1f} km/h."
    )


def _summarize_historical(label, rows):
    """Compute compact aggregates for one historical (same-date) series."""
    if not rows:
        return f"{label}: no historical data available."

    years = [r.get("year") for r in rows]
    maxs = [r.get("temperature_2m_max") for r in rows]
    mins = [r.get("temperature_2m_min") for r in rows]
    rains = [r.get("rain_sum", 0.0) or 0.0 for r in rows]
    humids = [r.get("relative_humidity_2m_mean") for r in rows]

    wettest = max(rows, key=lambda r: r.get("rain_sum", 0.0) or 0.0)
    hottest = max(rows, key=lambda r: r.get("temperature_2m_max", -999))

    return (
        f"{label} (years {min(years)}-{max(years)}, {len(rows)} years):\n"
        f"  Historical avg max temp: {_avg(maxs):.1f} C, avg min temp: {_avg(mins):.1f} C.\n"
        f"  Historical avg rainfall: {_avg(rains):.1f} mm; wettest year {wettest.get('year')} "
        f"({wettest.get('rain_sum', 0.0):.1f} mm); hottest year {hottest.get('year')} "
        f"({hottest.get('temperature_2m_max'):.1f} C).\n"
        f"  Historical avg humidity: {_avg(humids):.0f}%."
    )


def _build_data_digest(location_name, seven_day, hist1, hist2, labels):
    digest = [f"LOCATION: {location_name}", "", "LAST 7 OBSERVED DAYS:",
              _summarize_seven_days(seven_day), "", "HISTORICAL SAME-DATE COMPARISON:"]
    digest.append(_summarize_historical(labels.get("day1", "Forecast Day 1"), hist1))
    if hist2:
        digest.append(_summarize_historical(labels.get("day2", "Forecast Day 2"), hist2))
    return "\n".join(digest)


def generate_weather_summary(location_data, seven_day, hist1, hist2, labels=None):
    """
    Return {"summary": str, "cached": bool, "error": bool}.
    Caches by location (name/lat/lon) + today's date.
    """
    labels = labels or {}
    name = (location_data or {}).get("name", "Iloilo City")
    lat = (location_data or {}).get("lat", "")
    lon = (location_data or {}).get("lon", "")
    today = dt.date.today().isoformat()
    cache_key = f"{name}|{lat}|{lon}|{today}"

    if cache_key in _summary_cache:
        return {"summary": _summary_cache[cache_key], "cached": True, "error": False}

    if not seven_day:
        return {
            "summary": "There isn't enough recent weather data to summarize for this location.",
            "cached": False, "error": True,
        }

    api_key = _get_openai_api_key()
    if not api_key or api_key == "your-api-key-here":
        return {
            "summary": "The AI summary is unavailable because the OpenAI API key "
                       "hasn't been configured.",
            "cached": False, "error": True,
        }

    digest = _build_data_digest(name, seven_day, hist1, hist2, labels)

    system_prompt = (
        "You are a concise weather analyst for a localized Iloilo weather dashboard. "
        "You are given pre-computed statistics about (1) the last 7 observed days and "
        "(2) how the two upcoming forecast dates have historically behaved over past years. "
        "Write a SHORT natural-language summary of 2 to 4 sentences.\n\n"
        "STRICT RULES:\n"
        "1. Use ONLY the numbers provided. Never invent values.\n"
        "2. Start by describing the last 7 days (e.g. 'Over the past 7 days...'), then "
        "describe the historical pattern (e.g. 'Historically,...' or 'Over the years,...').\n"
        "3. Do NOT mention or predict the model's forecast; only describe observed and "
        "historical data.\n"
        "4. Plain flowing sentences. No markdown, no bullet points, no headers.\n"
        "5. Keep it factual, readable, and under about 90 words."
    )

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": digest},
            ],
            "max_tokens": 220,
            "temperature": 0.4,
        }
        response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=45)

        if response.status_code == 200:
            data = response.json()
            summary = data["choices"][0]["message"]["content"]
            summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
            _summary_cache[cache_key] = summary
            return {"summary": summary, "cached": False, "error": False}

        print(f"[ERROR] OpenAI summary error ({response.status_code}): {response.text}")
        if response.status_code == 401:
            msg = "The OpenAI API key was rejected (401). Please verify OPENAI_API_KEY."
        elif response.status_code == 429:
            msg = "OpenAI rate limit or quota reached (429). Please try again later."
        else:
            msg = f"Could not generate a summary right now (status {response.status_code})."
        return {"summary": msg, "cached": False, "error": True}

    except requests.exceptions.Timeout:
        return {"summary": "The AI summary took too long to respond. Please try again.",
                "cached": False, "error": True}
    except Exception as e:
        print(f"[ERROR] generate_weather_summary: {e}")
        return {"summary": "An unexpected error occurred while generating the summary.",
                "cached": False, "error": True}
