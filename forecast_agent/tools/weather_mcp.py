"""Open-Meteo backed weather tool that mimics an MCP-style weather integration."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from typing import Any

import pandas as pd
import requests

from .. import config, runtime

WMO_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Partly cloudy",
    2: "Partly cloudy",
    3: "Partly cloudy",
    45: "Fog",
    48: "Fog",
    51: "Drizzle (light)",
    53: "Drizzle (moderate)",
    55: "Drizzle (heavy)",
    61: "Rain (light)",
    63: "Rain (moderate)",
    65: "Rain (heavy)",
    71: "Snow",
    73: "Snow",
    75: "Snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Rain showers",
    85: "Snow showers",
    86: "Snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with hail",
}


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, default=str)


def _describe_weather(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return WMO_DESCRIPTIONS.get(code, f"Weather code {code}")


def _dominant_code(values: list[int | None]) -> int | None:
    filtered = [int(value) for value in values if value is not None]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def _parse_iso_date(raw_value: str) -> date:
    return date.fromisoformat(str(raw_value).strip())


def _resolve_weather_window(start_date: str, end_date: str, pivot_date: str | None = None) -> tuple[date, date, date | None]:
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date)
    pivot = _parse_iso_date(pivot_date) if pivot_date else None
    capped_end = min(candidate for candidate in [end, pivot, date.today()] if candidate is not None)
    if capped_end < start:
        start = capped_end
    return start, capped_end, pivot


def build_weather_for_period_payload(start_date: str, end_date: str, pivot_date: str | None = None) -> dict[str, Any]:
    """Build the JSON-ready weekly Dublin weather summary payload."""
    try:
        effective_start, effective_end, resolved_pivot = _resolve_weather_window(start_date, end_date, pivot_date)
    except ValueError as exc:
        return {
            "location": "Dublin, Ireland",
            "data_source": "Open-Meteo API unavailable",
            "api_available": False,
            "error": f"Invalid date input: {exc}",
            "forced_by_user": runtime.is_weather_forced(),
            "requested_window": {"start_date": start_date, "end_date": end_date, "pivot_date": pivot_date},
            "weeks": [],
            "severe_weeks": [],
        }

    params = {
        "latitude": config.DUBLIN_LATITUDE,
        "longitude": config.DUBLIN_LONGITUDE,
        "start_date": effective_start.isoformat(),
        "end_date": effective_end.isoformat(),
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
                "cloud_cover_mean",
            ]
        ),
        "timezone": config.DUBLIN_TIMEZONE,
    }

    try:
        response = requests.get(config.OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "location": "Dublin, Ireland",
            "data_source": "Open-Meteo API unavailable",
            "api_available": False,
            "error": str(exc),
            "forced_by_user": runtime.is_weather_forced(),
            "requested_window": {"start_date": start_date, "end_date": end_date, "pivot_date": pivot_date},
            "effective_window": {
                "start_date": effective_start.isoformat(),
                "end_date": effective_end.isoformat(),
                "pivot_date": resolved_pivot.isoformat() if resolved_pivot else None,
                "capped_to_pivot_date": bool(resolved_pivot and effective_end == resolved_pivot),
                "capped_to_today": bool(effective_end == date.today()),
            },
            "weeks": [],
            "severe_weeks": [],
        }

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    weekly_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    week_starts: dict[str, str] = {}

    _weather_codes = daily.get("weather_code") or [None] * len(dates)
    _temp_means = daily.get("temperature_2m_mean") or [None] * len(dates)
    _temp_maxs = daily.get("temperature_2m_max") or [None] * len(dates)
    _temp_mins = daily.get("temperature_2m_min") or [None] * len(dates)
    _precips = daily.get("precipitation_sum") or [None] * len(dates)
    _winds = daily.get("wind_speed_10m_max") or [None] * len(dates)
    _clouds = daily.get("cloud_cover_mean") or [None] * len(dates)

    for index, raw_day in enumerate(dates):
        day = date.fromisoformat(raw_day)
        iso_year, iso_week, _ = day.isocalendar()
        week_id = f"{iso_year}-W{iso_week:02d}"
        monday = pd.Timestamp.fromisocalendar(iso_year, iso_week, 1).strftime("%Y-%m-%d")
        week_starts[week_id] = monday
        weekly_buckets[week_id].append(
            {
                "weather_code": _weather_codes[index],
                "temperature_2m_mean": _temp_means[index],
                "temperature_2m_max": _temp_maxs[index],
                "temperature_2m_min": _temp_mins[index],
                "precipitation_sum": _precips[index],
                "wind_speed_10m_max": _winds[index],
                "cloud_cover_mean": _clouds[index],
            }
        )

    weeks: list[dict[str, Any]] = []
    severe_weeks: list[str] = []

    for week_id in sorted(weekly_buckets):
        rows = weekly_buckets[week_id]
        mean_temps = [float(item["temperature_2m_mean"]) for item in rows if item["temperature_2m_mean"] is not None]
        max_temps = [float(item["temperature_2m_max"]) for item in rows if item["temperature_2m_max"] is not None]
        min_temps = [float(item["temperature_2m_min"]) for item in rows if item["temperature_2m_min"] is not None]
        precip = [float(item["precipitation_sum"]) for item in rows if item["precipitation_sum"] is not None]
        winds = [float(item["wind_speed_10m_max"]) for item in rows if item["wind_speed_10m_max"] is not None]
        clouds = [float(item["cloud_cover_mean"]) for item in rows if item["cloud_cover_mean"] is not None]
        codes = [item["weather_code"] for item in rows]

        dominant_code = _dominant_code(codes)
        total_precip = round(sum(precip), 2) if precip else 0.0
        max_wind = round(max(winds), 2) if winds else 0.0
        severe = bool(
            (dominant_code is not None and dominant_code >= 55)
            or total_precip > 30
            or max_wind > 60
        )
        if severe:
            severe_weeks.append(week_id)

        weeks.append(
            {
                "week_id": week_id,
                "week_start": week_starts[week_id],
                "avg_temp_c": round(sum(mean_temps) / len(mean_temps), 2) if mean_temps else None,
                "avg_temp_max_c": round(sum(max_temps) / len(max_temps), 2) if max_temps else None,
                "avg_temp_min_c": round(sum(min_temps) / len(min_temps), 2) if min_temps else None,
                "total_precip_mm": total_precip,
                "max_wind_kmh": max_wind,
                "cloud_cover": round(sum(clouds) / len(clouds), 2) if clouds else None,
                "dominant_weather_code": dominant_code,
                "weather_description": _describe_weather(dominant_code),
                "severe_weather_flag": severe,
            }
        )

    return {
        "location": "Dublin, Ireland",
        "data_source": "Open-Meteo Archive API",
        "api_available": True,
        "forced_by_user": runtime.is_weather_forced(),
        "requested_window": {"start_date": start_date, "end_date": end_date, "pivot_date": pivot_date},
        "effective_window": {
            "start_date": effective_start.isoformat(),
            "end_date": effective_end.isoformat(),
            "pivot_date": resolved_pivot.isoformat() if resolved_pivot else None,
            "capped_to_pivot_date": bool(resolved_pivot and effective_end == resolved_pivot),
            "capped_to_today": bool(effective_end == date.today()),
        },
        "weeks": weeks,
        "severe_weeks": severe_weeks,
    }


def get_weather_for_period(start_date: str, end_date: str, pivot_date: str | None = None) -> str:
    """Fetch Dublin weekly weather summaries for a historical window capped at pivot date or today."""
    return _json(build_weather_for_period_payload(start_date, end_date, pivot_date=pivot_date))