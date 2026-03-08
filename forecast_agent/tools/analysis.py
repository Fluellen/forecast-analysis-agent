"""Higher-order analysis tools built on top of the CSV and weather/search tools."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..data_access import get_article_forecast_frame, get_article_links_frame, normalize_for_json, to_week_id
from .data_tools import get_article_links, get_forecast_data
from .weather_mcp import get_weather_for_period


def _loads(raw: str) -> dict[str, Any]:
    return json.loads(raw)


def correlate_weather_with_demand(art_cinv: int) -> str:
    """Correlate weekly Dublin weather with weekly article demand and flag weather-impacted weeks."""
    forecast_payload = _loads(get_forecast_data(int(art_cinv)))
    rows = forecast_payload.get("rows", [])
    if not rows:
        return json.dumps(
            {
                "weeks_joined": 0,
                "correlations": {"precipitation": None, "temperature": None, "wind": None},
                "weather_weeks": [],
                "weather_impacted_weeks": [],
                "interpretation": "No forecast history was available for this article.",
            }
        )

    start_date = rows[0].get("MVT_DATE")
    end_date = rows[-1].get("MVT_DATE")
    pivot_date = forecast_payload.get("pivot_date")
    weather_payload = _loads(get_weather_for_period(start_date, end_date, pivot_date=pivot_date))
    if not weather_payload.get("api_available"):
        return json.dumps(
            {
                "weeks_joined": 0,
                "correlations": {"precipitation": None, "temperature": None, "wind": None},
                "weather_weeks": [],
                "weather_impacted_weeks": [],
                "interpretation": "Weather context was unavailable, so no weather-demand correlation could be assessed.",
            }
        )

    demand = pd.DataFrame(rows)
    weather = pd.DataFrame(weather_payload.get("weeks", []))
    if demand.empty or weather.empty:
        return json.dumps(
            {
                "weeks_joined": 0,
                "correlations": {"precipitation": None, "temperature": None, "wind": None},
                "weather_weeks": [],
                "weather_impacted_weeks": [],
                "interpretation": "There was insufficient overlap between demand history and weather history.",
            }
        )

    demand["MVT_DATE_DT"] = pd.to_datetime(demand["MVT_DATE"], errors="coerce")
    iso = demand["MVT_DATE_DT"].dt.isocalendar()
    demand["week_id"] = [to_week_id(year, week) for year, week in zip(iso["year"], iso["week"])]
    demand["ACTUAL_DEMAND"] = pd.to_numeric(demand["ACTUAL_DEMAND"], errors="coerce")
    demand["FORECAST"] = pd.to_numeric(demand["FORECAST"], errors="coerce")
    overlap = demand.merge(weather, on="week_id", how="inner").sort_values("MVT_DATE_DT").copy()
    if overlap.empty:
        return json.dumps(
            {
                "weeks_joined": 0,
                "weeks_used_for_correlation": 0,
                "correlations": {"precipitation": None, "temperature": None, "wind": None},
                "weather_weeks": [],
                "weather_impacted_weeks": [],
                "interpretation": "There was insufficient overlap between demand history and weather history.",
            }
        )

    merged = overlap[overlap["ACTUAL_DEMAND"] > 0].copy()

    def _corr(column: str) -> float | None:
        subset = merged[["ACTUAL_DEMAND", column]].dropna()
        if len(subset) < 2:
            return None
        coefficient = subset["ACTUAL_DEMAND"].corr(subset[column])
        if pd.isna(coefficient):
            return None
        return round(float(coefficient), 3)

    mean_demand = float(merged["ACTUAL_DEMAND"].mean()) if not merged.empty else 0.0
    std_demand = float(merged["ACTUAL_DEMAND"].std(ddof=0)) if len(merged) > 1 else 0.0
    impacted: list[dict[str, Any]] = []
    threshold = mean_demand - 1.5 * std_demand

    for row in merged.itertuples():
        if bool(row.severe_weather_flag) and float(row.ACTUAL_DEMAND) < threshold:
            impacted.append(
                {
                    "week_id": row.week_id,
                    "MVT_DATE": row.MVT_DATE,
                    "ACTUAL_DEMAND": float(row.ACTUAL_DEMAND),
                    "FORECAST": float(row.FORECAST),
                    "weather_description": row.weather_description,
                    "total_precip_mm": row.total_precip_mm,
                    "severe_weather_flag": bool(row.severe_weather_flag),
                }
            )

    interpretation = (
        f"Joined {len(overlap)} overlapping demand and weather weeks. "
        f"Correlations were computed on {len(merged)} weeks with positive actual demand. "
        f"Precipitation correlation was {_corr('total_precip_mm')}, temperature correlation was {_corr('avg_temp_max_c')}, "
        f"and wind correlation was {_corr('max_wind_kmh')}. "
        f"{len(impacted)} weeks showed a severe-weather signal alongside materially weak demand."
    )

    weather_weeks = [
        {
            "week_id": row.week_id,
            "MVT_DATE": row.MVT_DATE,
            "avg_temp_c": float(row.avg_temp_c) if pd.notna(row.avg_temp_c) else None,
            "avg_temp_max_c": float(row.avg_temp_max_c) if pd.notna(row.avg_temp_max_c) else None,
            "avg_temp_min_c": float(row.avg_temp_min_c) if pd.notna(row.avg_temp_min_c) else None,
        }
        for row in overlap.itertuples()
    ]

    return json.dumps(
        {
            "weeks_joined": int(len(overlap)),
            "weeks_used_for_correlation": int(len(merged)),
            "correlations": {
                "precipitation": _corr("total_precip_mm"),
                "temperature": _corr("avg_temp_max_c"),
                "wind": _corr("max_wind_kmh"),
            },
            "weather_weeks": weather_weeks,
            "weather_impacted_weeks": impacted,
            "interpretation": interpretation,
        }
    )


def analyse_year_on_year_trend(art_cinv: int) -> str:
    """Compare current demand against the same-week historical actuals from the last three years."""
    frame = get_article_forecast_frame(int(art_cinv)).copy()
    frame = frame[frame["ACTUAL_DEMAND"] > 0].copy()

    if frame.empty:
        return json.dumps(
            {
                "weeks_analysed": 0,
                "summary": {
                    "avg_yoy_vs_nm1_pct": None,
                    "avg_yoy_vs_nm2_pct": None,
                    "avg_yoy_vs_nm3_pct": None,
                    "trend_direction": "INSUFFICIENT_DATA",
                    "weeks_below_all_historical": 0,
                    "weeks_above_all_historical": 0,
                },
                "weekly_detail": [],
                "interpretation": "No positive-demand weeks were available for year-on-year analysis.",
            }
        )

    def _pct_change(current: float, baseline: float) -> float | None:
        if baseline in (None, 0) or pd.isna(baseline):
            return None
        return round(float((current - baseline) / baseline * 100), 2)

    detail: list[dict[str, Any]] = []
    nm1_values: list[float] = []
    nm2_values: list[float] = []
    nm3_values: list[float] = []
    below_all = 0
    above_all = 0

    for row in frame.itertuples():
        current = float(row.ACTUAL_DEMAND)
        yoy1 = _pct_change(current, row.NM1)
        yoy2 = _pct_change(current, row.NM2)
        yoy3 = _pct_change(current, row.NM3)
        valid_history = [float(value) for value in [row.NM1, row.NM2, row.NM3] if pd.notna(value) and float(value) > 0]
        historical_avg = float(np.mean(valid_history)) if valid_history else None
        vs_avg = _pct_change(current, historical_avg) if historical_avg else None
        if yoy1 is not None:
            nm1_values.append(yoy1)
        if yoy2 is not None:
            nm2_values.append(yoy2)
        if yoy3 is not None:
            nm3_values.append(yoy3)
        if valid_history and all(current < value for value in valid_history):
            below_all += 1
        if valid_history and all(current > value for value in valid_history):
            above_all += 1

        detail.append(
            {
                "week_id": row.WEEK_ID,
                "MVT_DATE": normalize_for_json(row.MVT_DATE_DT),
                "ACTUAL_DEMAND": current,
                "nm1_demand_1yr_ago": float(row.NM1) if pd.notna(row.NM1) else None,
                "nm2_demand_2yr_ago": float(row.NM2) if pd.notna(row.NM2) else None,
                "nm3_demand_3yr_ago": float(row.NM3) if pd.notna(row.NM3) else None,
                "yoy_vs_nm1_pct": yoy1,
                "yoy_vs_nm2_pct": yoy2,
                "yoy_vs_nm3_pct": yoy3,
                "demand_vs_historical_avg_pct": vs_avg,
                "historically_anomalous": bool(vs_avg is not None and abs(vs_avg) > 40),
            }
        )

    avg_nm1 = round(float(np.mean(nm1_values)), 2) if nm1_values else None
    avg_nm2 = round(float(np.mean(nm2_values)), 2) if nm2_values else None
    avg_nm3 = round(float(np.mean(nm3_values)), 2) if nm3_values else None

    if avg_nm1 is None:
        trend_direction = "INSUFFICIENT_DATA"
    elif avg_nm1 > 5:
        trend_direction = "GROWING"
    elif avg_nm1 < -5:
        trend_direction = "DECLINING"
    else:
        trend_direction = "STABLE"

    interpretation = (
        f"Across {len(detail)} positive-demand weeks, demand trends appear {trend_direction.lower()}. "
        f"Average change versus last year was {avg_nm1}%, versus two years ago was {avg_nm2}%, "
        f"and versus three years ago was {avg_nm3}%. "
        f"{below_all} weeks underperformed all historical anchors, while {above_all} weeks beat them all."
    )

    return json.dumps(
        {
            "weeks_analysed": int(len(detail)),
            "summary": {
                "avg_yoy_vs_nm1_pct": avg_nm1,
                "avg_yoy_vs_nm2_pct": avg_nm2,
                "avg_yoy_vs_nm3_pct": avg_nm3,
                "trend_direction": trend_direction,
                "weeks_below_all_historical": below_all,
                "weeks_above_all_historical": above_all,
            },
            "weekly_detail": detail,
            "interpretation": interpretation,
        }
    )


def get_article_links_demand(art_cinv: int) -> str:
    """Summarize demand for linked articles to assess substitution effects and duplicate links."""
    links_payload = _loads(get_article_links(int(art_cinv)))
    links = links_payload.get("links", [])
    if not links:
        return json.dumps(
            {
                "source_cinv": int(art_cinv),
                "linked_articles": [],
                "interpretation": "No direct article links were found for this CINV.",
            }
        )

    links_frame = get_article_links_frame(int(art_cinv)).copy()
    duplicate_count = int(links_frame.duplicated(subset=["art_cinv_a", "art_cinv_b", "link_desc", "coefficient"]).sum())
    duplicate_groups: list[dict[str, Any]] = []
    if not links_frame.empty:
        grouped = (
            links_frame.groupby(["art_cinv_a", "art_cinv_b", "link_desc", "coefficient"], dropna=False)
            .size()
            .reset_index(name="row_count")
        )
        for row in grouped.itertuples(index=False):
            if int(row.row_count) <= 1:
                continue
            duplicate_groups.append(
                {
                    "source_cinv": int(row.art_cinv_a),
                    "linked_cinv": int(row.art_cinv_b),
                    "link_desc": str(row.link_desc or ""),
                    "coefficient": row.coefficient,
                    "row_count": int(row.row_count),
                    "risk_note": "The same linked article appears multiple times with the same relationship metadata, so naive substitution analysis can double count its demand history.",
                }
            )
    linked_ids = sorted(
        {
            int(candidate)
            for item in links
            for candidate in [item.get("art_cinv_a"), item.get("art_cinv_b")]
            if candidate is not None and int(candidate) != int(art_cinv)
        }
    )

    summaries: list[dict[str, Any]] = []
    for linked_id in linked_ids:
        linked_frame = get_article_forecast_frame(int(linked_id)).copy()
        clean = linked_frame[linked_frame["ACTUAL_DEMAND"] > 0].copy()
        link_row = next((item for item in links if int(item.get("art_cinv_b", -1)) == linked_id), None)
        duplicate_rows_for_link = sum(1 for item in duplicate_groups if int(item.get("linked_cinv", -1)) == linked_id)
        if clean.empty:
            summaries.append(
                {
                    "cinv": linked_id,
                    "link_desc": link_row.get("link_desc") if link_row else "",
                    "coefficient": link_row.get("coefficient") if link_row else None,
                    "duplicate_rows_detected": duplicate_rows_for_link,
                    "weeks_of_data": 0,
                    "mean_demand": None,
                    "std_demand": None,
                    "data_available": False,
                }
            )
            continue

        summaries.append(
            {
                "cinv": linked_id,
                "link_desc": link_row.get("link_desc") if link_row else "",
                "coefficient": link_row.get("coefficient") if link_row else None,
                "duplicate_rows_detected": duplicate_rows_for_link,
                "weeks_of_data": int(len(clean)),
                "mean_demand": round(float(clean["ACTUAL_DEMAND"].mean()), 2),
                "std_demand": round(float(clean["ACTUAL_DEMAND"].std(ddof=0)), 2),
                "data_available": True,
            }
        )

    if duplicate_groups:
        duplicate_summary = "; ".join(
            f"linked article {item['linked_cinv']} appears {item['row_count']} times as {item['link_desc'] or 'linked'}"
            for item in duplicate_groups
        )
        interpretation = (
            f"Warning: duplicate substitution links were detected for CINV {art_cinv}: {duplicate_summary}. "
            "This means the same linked article can be counted more than once in a naive substitution read, which may overstate its explanatory power and can help explain apparent demand dips or regime changes after the duplicated relationship becomes relevant. "
            f"After deduplicating linked article IDs, {len(summaries)} linked article summaries remain available for context."
        )
    else:
        interpretation = (
            f"Found {len(summaries)} linked article summaries for CINV {art_cinv}. "
            "No duplicate link rows were detected. Use these demand baselines to judge whether the focal article's volatility may reflect substitution behaviour."
        )

    return json.dumps(
        {
            "source_cinv": int(art_cinv),
            "linked_articles": summaries,
            "duplicate_links_detected": duplicate_count,
            "duplicate_link_groups": duplicate_groups,
            "duplicate_link_warning": bool(duplicate_groups),
            "interpretation": interpretation,
        }
    )