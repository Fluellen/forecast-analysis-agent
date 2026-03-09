"""Workflow-backed forecast analysis runtime built on Microsoft Agent Framework."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from jinja2 import Template
from rich.console import Console
from typing_extensions import Never

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    AgentResponseUpdate,
    Message,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    executor,
)
from . import config
from .cinv_resolution import resolve_analysis_request
from .events import (
    AGUIEvent,
    run_error,
    run_finished,
    run_started,
    state_snapshot,
    step_finished,
    step_started,
    text_message_chunk,
    text_message_end,
    text_message_start,
)
from .responses_client import get_responses_cleanup_hooks, get_responses_client
from .templates import EMAIL_TEMPLATE, REPORT_TEMPLATE
from .tools import (
    analyse_year_on_year_trend,
    compute_forecast_health,
    correlate_weather_with_demand,
    detect_outlier_weeks,
    detect_pre_pivot_stockout_risk,
    get_article_links,
    get_article_links_demand,
    get_article_metadata,
    get_forecast_data,
    search_article_characteristics,
    search_holiday_demand_correlation,
)

console = Console()

WORKFLOW_NAME = "forecast_analysis_workflow"
WORKFLOW_DESCRIPTION = "Workflow that analyses a forecast article, conditionally enriches weather context, and drafts a report plus email."

RESOLVE_AND_IDENTIFY_ID = "resolve_and_identify"
LOAD_CONTEXT_ID = "load_context"
RUN_STATISTICS_ID = "run_statistics"
ANALYSE_YOY_ID = "analyse_year_on_year"
ENRICH_WEATHER_ID = "enrich_weather"
SKIP_WEATHER_ID = "skip_weather"
PREPARE_SYNTHESIS_ID = "prepare_report_synthesis"
SYNTHESIS_EXECUTOR_ID = "forecast_report_synthesis"
FINALIZE_OUTPUT_ID = "finalize_analysis_output"

PUBLIC_STATE_KEY = "forecast_analysis_public_state"
PAYLOADS_STATE_KEY = "forecast_analysis_payloads"

STEP_NAMES = {
    1: "Article Identification",
    2: "Data Loading",
    3: "Statistical Analysis",
    4: "Year-on-Year Trend Analysis",
    5: "Weather Context",
    6: "Report and Email Generation",
}

EXECUTOR_TO_STEP = {
    RESOLVE_AND_IDENTIFY_ID: 1,
    LOAD_CONTEXT_ID: 2,
    RUN_STATISTICS_ID: 3,
    ANALYSE_YOY_ID: 4,
    ENRICH_WEATHER_ID: 5,
    SKIP_WEATHER_ID: 5,
    PREPARE_SYNTHESIS_ID: 6,
    SYNTHESIS_EXECUTOR_ID: 6,
    FINALIZE_OUTPUT_ID: 6,
}

STEP_FINAL_EXECUTORS = {
    RESOLVE_AND_IDENTIFY_ID,
    LOAD_CONTEXT_ID,
    RUN_STATISTICS_ID,
    ANALYSE_YOY_ID,
    ENRICH_WEATHER_ID,
    SKIP_WEATHER_ID,
    FINALIZE_OUTPUT_ID,
}

_SYNTHESIS_SYSTEM_PROMPT = """
You are an expert supply chain forecast analyst working for DFAI Managed Services.

You will receive structured forecast analysis data that has already been prepared.
Use only that structured data. Do not invent facts and do not ask follow-up questions.

Produce exactly two deliverables using this delimiter format:
<REPORT>
...markdown report...
</REPORT>
<EMAIL>
Subject: ...

Dear Client,
...body...
Kind regards,
DFAI Managed Services
</EMAIL>

The report must use exactly these section headers:
## Executive Summary
## Article Profile
## Forecast Health Assessment
## Demand Volatility & Outlier Analysis
## Year-on-Year Demand Trend (NM1 / NM2 / NM3 Analysis)
## Weather Impact Analysis
## Linked Article & Substitution Context
## Data Quality & Recommendations
## Flagged Weeks for Model Exclusion

Email requirements:
- Professional and operational in tone, max 180 words.
- One short summary paragraph.
- A Key observations block.
- A Recommended actions block.
- One short closing sentence requesting client context.
- Mention specific weeks only when supported by the analysis payload.
- If weather enrichment was skipped, state that search evidence did not justify material weather sensitivity.

Important analysis rules:
- NM1, NM2, NM3 are historical demand benchmarks, not forecasts.
- If stockout risk is detected, explicitly mention the zero-shipment streak, affected period, and baseline reduction percentage.
- Reuse the supplied reporting guidance closely when it is present.
- If duplicate linked-article rows are present, explain that they can overstate substitution evidence.
- Do not mention any data that is absent from the supplied payload.
""".strip()


@dataclass(slots=True)
class WorkflowInput:
    """Structured workflow invocation parameters."""

    request_text: str
    force_weather: bool = False


def _json_clone(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, default=str))


def _load_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        return payload if isinstance(payload, dict) else {"raw": payload}
    return {"raw": raw}


def get_cleanup_hooks() -> list[Callable[[], Any]]:
    return cast(list[Callable[[], Any]], get_responses_cleanup_hooks())


def _build_synthesis_agent():
    return get_responses_client().as_agent(
        name="ForecastReportSynthesis",
        instructions=_SYNTHESIS_SYSTEM_PROMPT,
        tools=[],
    )


def _create_initial_state(request_text: str, force_weather: bool) -> dict[str, Any]:
    return {
        "cinv": None,
        "selected_cinv": None,
        "status": "running",
        "article_name": "",
        "category": "",
        "pivot_date": "",
        "report": "",
        "email": "",
        "email_subject": "",
        "flagged_weeks": [],
        "forecast_health": {},
        "outlier_analysis": {},
        "weather_sensitivity": {},
        "weather_analysis": {},
        "weather_enrichment_recommended": False,
        "weather_enrichment_activated": False,
        "weather_weeks": [],
        "stockout_risk": {},
        "year_on_year": {},
        "link_analysis": {},
        "holiday_analysis": {},
        "steps": {str(number): "waiting" for number in STEP_NAMES},
        "force_weather": bool(force_weather),
        "output_files": {},
        "input_request_text": request_text,
        "input_resolution": {},
    }


def _normalize_request_input(raw_input: Any) -> tuple[str, bool]:
    if isinstance(raw_input, int):
        return str(raw_input), False
    if isinstance(raw_input, str):
        return raw_input.strip(), False
    if isinstance(raw_input, WorkflowInput):
        return raw_input.request_text, raw_input.force_weather

    request_text = str(getattr(raw_input, "request_text", raw_input) or "").strip()
    force_weather = bool(getattr(raw_input, "force_weather", False))
    return request_text, force_weather


def _set_workflow_state(ctx: WorkflowContext[Any, Any], state: dict[str, Any], payloads: dict[str, Any]) -> None:
    ctx.set_state(PUBLIC_STATE_KEY, state)
    ctx.set_state(PAYLOADS_STATE_KEY, payloads)


def _get_public_state(ctx: WorkflowContext[Any, Any]) -> dict[str, Any]:
    return _json_clone(cast(dict[str, Any], ctx.get_state(PUBLIC_STATE_KEY, {})))


def _get_payloads(ctx: WorkflowContext[Any, Any]) -> dict[str, Any]:
    return _json_clone(cast(dict[str, Any], ctx.get_state(PAYLOADS_STATE_KEY, {})))


async def _call_json_tool(tool: Callable[..., str], *args: Any) -> dict[str, Any]:
    raw = await asyncio.to_thread(tool, *args)
    return _load_json_object(raw)


def _build_characteristics_query(metadata_payload: dict[str, Any], art_cinv: int) -> str:
    metadata = metadata_payload.get("metadata") or []
    if metadata:
        first = metadata[0]
        article_name = str(first.get("ART_DESC") or "").strip()
        if article_name:
            return f"{article_name} supply chain seasonality demand drivers Ireland retail"
    return str(art_cinv)


def _forecast_has_holidays(forecast_payload: dict[str, Any]) -> bool:
    for row in forecast_payload.get("rows", []):
        if str(row.get("HOLIDAYS_TYPE") or "").strip():
            return True
    return False


def _apply_tool_payload(state: dict[str, Any], tool_name: str, payload: dict[str, Any]) -> None:
    if tool_name == "get_article_metadata":
        metadata = payload.get("metadata") or []
        if metadata:
            first = metadata[0]
            state["article_name"] = first.get("ART_DESC", "") or state["article_name"]
            state["category"] = first.get("ART_LEVEL1_DESC", "") or state["category"]
    elif tool_name == "search_article_characteristics":
        state["weather_sensitivity"] = payload.get("weather_sensitivity") or {}
        state["weather_enrichment_recommended"] = bool(payload.get("weather_enrichment_recommended"))
    elif tool_name == "get_forecast_data":
        state["pivot_date"] = payload.get("pivot_date") or state["pivot_date"]
    elif tool_name == "compute_forecast_health":
        state["forecast_health"] = payload
    elif tool_name == "detect_pre_pivot_stockout_risk":
        state["stockout_risk"] = payload
    elif tool_name == "detect_outlier_weeks":
        state["outlier_analysis"] = payload
        flagged: list[dict[str, Any]] = []
        for item in payload.get("outliers", [])[:12]:
            reasons: list[str] = []
            if item.get("iqr_outlier"):
                reasons.append("IQR outlier")
            z_score = item.get("z_score")
            if z_score is not None and abs(float(z_score)) > 2:
                reasons.append(f"z-score {z_score}")
            if item.get("has_promo"):
                reasons.append("promo week")
            if item.get("has_holiday"):
                reasons.append(item.get("holiday_name") or "holiday week")
            flagged.append(
                {
                    "week_id": item.get("week_id"),
                    "mvt_date": item.get("MVT_DATE"),
                    "actual_demand": item.get("ACTUAL_DEMAND"),
                    "forecast": item.get("FORECAST"),
                    "reason": ", ".join(reasons) or "statistical outlier",
                }
            )
        state["flagged_weeks"] = flagged
    elif tool_name == "analyse_year_on_year_trend":
        state["year_on_year"] = payload
    elif tool_name == "get_article_links_demand":
        state["link_analysis"] = payload
    elif tool_name == "correlate_weather_with_demand":
        state["weather_analysis"] = payload
        state["weather_enrichment_activated"] = True
        state["weather_weeks"] = payload.get("weather_weeks") or []
        existing = {item["week_id"] for item in state.get("flagged_weeks", []) if item.get("week_id")}
        for item in payload.get("weather_impacted_weeks", []):
            week_id = item.get("week_id")
            if week_id in existing:
                continue
            state.setdefault("flagged_weeks", []).append(
                {
                    "week_id": week_id,
                    "mvt_date": item.get("MVT_DATE"),
                    "actual_demand": item.get("ACTUAL_DEMAND"),
                    "forecast": item.get("FORECAST"),
                    "reason": f"severe weather: {item.get('weather_description', 'weather disruption')}",
                }
            )


def _should_run_weather(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    return bool(message.get("force_weather") or message.get("weather_enrichment_recommended"))


def _should_skip_weather(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    return not _should_run_weather(message)


def _compact_year_on_year(year_on_year: dict[str, Any]) -> dict[str, Any]:
    weekly_detail = year_on_year.get("weekly_detail") or []
    anomalous_weeks = [
        {
            "week_id": item.get("week_id"),
            "MVT_DATE": item.get("MVT_DATE"),
            "ACTUAL_DEMAND": item.get("ACTUAL_DEMAND"),
            "demand_vs_historical_avg_pct": item.get("demand_vs_historical_avg_pct"),
            "historically_anomalous": item.get("historically_anomalous"),
        }
        for item in weekly_detail
        if isinstance(item, dict) and item.get("historically_anomalous")
    ][:10]
    return {
        "weeks_analysed": year_on_year.get("weeks_analysed"),
        "summary": year_on_year.get("summary") or {},
        "interpretation": year_on_year.get("interpretation"),
        "anomalous_weeks": anomalous_weeks,
    }


def _compact_weather_analysis(weather_analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "weeks_joined": weather_analysis.get("weeks_joined"),
        "weeks_used_for_correlation": weather_analysis.get("weeks_used_for_correlation"),
        "correlations": weather_analysis.get("correlations") or {},
        "interpretation": weather_analysis.get("interpretation"),
        "weather_impacted_weeks": (weather_analysis.get("weather_impacted_weeks") or [])[:10],
    }


def _compact_link_analysis(link_analysis: dict[str, Any]) -> dict[str, Any]:
    linked_articles = []
    for item in (link_analysis.get("linked_articles") or [])[:8]:
        if not isinstance(item, dict):
            continue
        linked_articles.append(
            {
                "cinv": item.get("cinv"),
                "link_desc": item.get("link_desc"),
                "coefficient": item.get("coefficient"),
                "duplicate_rows_detected": item.get("duplicate_rows_detected"),
                "weeks_of_data": item.get("weeks_of_data"),
                "mean_demand": item.get("mean_demand"),
                "std_demand": item.get("std_demand"),
                "data_available": item.get("data_available"),
            }
        )
    return {
        "interpretation": link_analysis.get("interpretation"),
        "duplicate_links_detected": link_analysis.get("duplicate_links_detected"),
        "duplicate_link_warning": link_analysis.get("duplicate_link_warning"),
        "duplicate_link_groups": (link_analysis.get("duplicate_link_groups") or [])[:8],
        "linked_articles": linked_articles,
    }


def _compact_holiday_analysis(holiday_payload: dict[str, Any]) -> dict[str, Any]:
    assessment = _load_json_object(holiday_payload.get("demand_uplift_assessment") or {})
    observed_holidays = []
    for item in (holiday_payload.get("observed_holiday_summaries") or [])[:8]:
        if not isinstance(item, dict):
            continue
        observed_holidays.append(
            {
                "holiday_name": item.get("holiday_name"),
                "occurrence_count": item.get("occurrence_count"),
                "historical_occurrences": item.get("historical_occurrences"),
                "future_occurrences": item.get("future_occurrences"),
                "pivot_week_occurrences": item.get("pivot_week_occurrences"),
                "sample_weeks": (item.get("sample_weeks") or [])[:3],
            }
        )
    return {
        "holiday_values": (holiday_payload.get("holiday_values") or [])[:10],
        "holiday_count": holiday_payload.get("holiday_count"),
        "scope": holiday_payload.get("scope"),
        "search_available": holiday_payload.get("search_available"),
        "error": holiday_payload.get("error"),
        "note": holiday_payload.get("note"),
        "demand_uplift_assessment": {
            "classification": assessment.get("classification"),
            "positive_signals": (assessment.get("positive_signals") or [])[:6],
            "negative_signals": (assessment.get("negative_signals") or [])[:6],
            "rationale": assessment.get("rationale"),
        },
        "observed_holiday_summaries": observed_holidays,
    }


def _compact_stockout_risk(stockout_risk: dict[str, Any]) -> dict[str, Any]:
    return {
        "stockout_risk_detected": stockout_risk.get("stockout_risk_detected"),
        "affected_period": stockout_risk.get("affected_period"),
        "latest_zero_shipment_streak": stockout_risk.get("latest_zero_shipment_streak") or {},
        "baseline_reduction_pct": stockout_risk.get("baseline_reduction_pct"),
        "future_forecast_weeks": stockout_risk.get("future_forecast_weeks"),
        "future_forecast_baseline_avg": stockout_risk.get("future_forecast_baseline_avg"),
        "recommended_actions": stockout_risk.get("recommended_actions") or [],
        "reporting_guidance": stockout_risk.get("reporting_guidance"),
        "interpretation": stockout_risk.get("interpretation"),
    }


def _compact_analysis_for_prompt(state: dict[str, Any], payloads: dict[str, Any]) -> dict[str, Any]:
    metadata_payload = _load_json_object(payloads.get("article_metadata") or {})
    metadata = metadata_payload.get("metadata") or []
    article_profile = metadata[0] if metadata else {
        "ART_CINV": state.get("cinv"),
        "ART_DESC": state.get("article_name"),
        "ART_LEVEL1_DESC": state.get("category"),
    }

    search_payload = _load_json_object(payloads.get("article_characteristics") or {})
    holiday_payload = _load_json_object(payloads.get("holiday_analysis") or {})

    forecast_health = _load_json_object(state.get("forecast_health") or {})
    outlier_analysis = _load_json_object(state.get("outlier_analysis") or {})
    year_on_year = _load_json_object(state.get("year_on_year") or {})
    weather_analysis = _load_json_object(state.get("weather_analysis") or {})
    link_analysis = _load_json_object(state.get("link_analysis") or {})
    stockout_risk = _load_json_object(state.get("stockout_risk") or {})

    return {
        "article_profile": article_profile,
        "resolution": state.get("input_resolution") or {},
        "forecast_summary": {
            "cinv": state.get("cinv"),
            "pivot_date": state.get("pivot_date"),
            "force_weather": state.get("force_weather"),
        },
        "article_characteristics": {
            "query": search_payload.get("query"),
            "search_available": search_payload.get("search_available"),
            "error": search_payload.get("error"),
            "weather_enrichment_recommended": state.get("weather_enrichment_recommended"),
            "weather_sensitivity": search_payload.get("weather_sensitivity") or state.get("weather_sensitivity") or {},
        },
        "holiday_analysis": _compact_holiday_analysis(holiday_payload),
        "forecast_health": {
            key: forecast_health.get(key)
            for key in [
                "pivot_date",
                "actuals_available_through",
                "weeks_in_horizon",
                "weeks_with_actuals",
                "weeks_pending_actuals",
                "weeks_analysed",
                "mape_pct",
                "wape_pct",
                "bias_units",
                "bias_pct",
                "tracking_signal",
                "tracking_signal_flag",
                "accuracy_rating",
                "interpretation",
            ]
        },
        "stockout_risk": _compact_stockout_risk(stockout_risk),
        "outlier_analysis": {
            "interpretation": outlier_analysis.get("interpretation"),
            "outlier_count": outlier_analysis.get("outlier_count"),
            "outliers": (outlier_analysis.get("outliers") or [])[:12],
        },
        "year_on_year": _compact_year_on_year(year_on_year),
        "weather_analysis": _compact_weather_analysis(weather_analysis),
        "weather_enrichment_activated": state.get("weather_enrichment_activated"),
        "link_analysis": _compact_link_analysis(link_analysis),
        "flagged_weeks": state.get("flagged_weeks") or [],
    }


def _build_synthesis_prompt(state: dict[str, Any], payloads: dict[str, Any]) -> str:
    compact_analysis = _compact_analysis_for_prompt(state, payloads)
    analysis_json = json.dumps(compact_analysis, indent=2, ensure_ascii=False, default=str)
    return (
        "Use the structured forecast analysis data below to draft the report and email.\n\n"
        "FORECAST_ANALYSIS_DATA\n"
        f"{analysis_json}\n"
    )


def _extract_completion_state(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, list):
        return None
    for item in reversed(data):
        if isinstance(item, dict) and "steps" in item and "status" in item:
            return item
    return None


@executor(id=RESOLVE_AND_IDENTIFY_ID)
async def resolve_and_identify(raw_input: WorkflowInput, ctx: WorkflowContext[dict[str, Any]]) -> None:
    request_text, force_weather = _normalize_request_input(raw_input)
    if not request_text:
        raise ValueError("A CINV or free-form analysis request is required.")

    state = _create_initial_state(request_text, force_weather)
    resolution = await resolve_analysis_request(input_text=request_text)
    resolved_cinv = int(resolution["cinv"])

    state.update(
        {
            "cinv": resolved_cinv,
            "selected_cinv": resolved_cinv,
            "input_resolution": resolution,
            "input_request_text": resolution.get("input_text") or request_text,
        }
    )

    metadata_payload = await _call_json_tool(get_article_metadata, resolved_cinv)
    _apply_tool_payload(state, "get_article_metadata", metadata_payload)

    search_query = _build_characteristics_query(metadata_payload, resolved_cinv)
    characteristics_payload = await _call_json_tool(search_article_characteristics, search_query)
    _apply_tool_payload(state, "search_article_characteristics", characteristics_payload)

    state["steps"]["1"] = "completed"
    payloads = {
        "article_metadata": metadata_payload,
        "article_characteristics": characteristics_payload,
    }
    _set_workflow_state(ctx, state, payloads)
    await ctx.send_message(state)


@executor(id=LOAD_CONTEXT_ID)
async def load_context(state: dict[str, Any], ctx: WorkflowContext[dict[str, Any]]) -> None:
    next_state = _json_clone(state)
    payloads = _get_payloads(ctx)
    art_cinv = int(next_state["cinv"])

    forecast_payload = await _call_json_tool(get_forecast_data, art_cinv)
    links_payload = await _call_json_tool(get_article_links, art_cinv)

    _apply_tool_payload(next_state, "get_forecast_data", forecast_payload)

    payloads["forecast_data"] = forecast_payload
    payloads["article_links"] = links_payload

    if int(links_payload.get("count") or 0) > 0:
        link_analysis_payload = await _call_json_tool(get_article_links_demand, art_cinv)
        _apply_tool_payload(next_state, "get_article_links_demand", link_analysis_payload)
        payloads["link_analysis"] = link_analysis_payload

    holiday_payload: dict[str, Any] = {}
    if _forecast_has_holidays(forecast_payload):
        holiday_payload = await _call_json_tool(search_holiday_demand_correlation, art_cinv)
    next_state["holiday_analysis"] = holiday_payload
    payloads["holiday_analysis"] = holiday_payload

    next_state["steps"]["2"] = "completed"
    _set_workflow_state(ctx, next_state, payloads)
    await ctx.send_message(next_state)


@executor(id=RUN_STATISTICS_ID)
async def run_statistics(state: dict[str, Any], ctx: WorkflowContext[dict[str, Any]]) -> None:
    next_state = _json_clone(state)
    payloads = _get_payloads(ctx)
    art_cinv = int(next_state["cinv"])

    forecast_health_payload = await _call_json_tool(compute_forecast_health, art_cinv)
    stockout_payload = await _call_json_tool(detect_pre_pivot_stockout_risk, art_cinv)
    outlier_payload = await _call_json_tool(detect_outlier_weeks, art_cinv)

    _apply_tool_payload(next_state, "compute_forecast_health", forecast_health_payload)
    _apply_tool_payload(next_state, "detect_pre_pivot_stockout_risk", stockout_payload)
    _apply_tool_payload(next_state, "detect_outlier_weeks", outlier_payload)

    payloads["forecast_health"] = forecast_health_payload
    payloads["stockout_risk"] = stockout_payload
    payloads["outlier_analysis"] = outlier_payload

    next_state["steps"]["3"] = "completed"
    _set_workflow_state(ctx, next_state, payloads)
    await ctx.send_message(next_state)


@executor(id=ANALYSE_YOY_ID)
async def analyse_year_on_year(state: dict[str, Any], ctx: WorkflowContext[dict[str, Any]]) -> None:
    next_state = _json_clone(state)
    payloads = _get_payloads(ctx)
    art_cinv = int(next_state["cinv"])

    year_on_year_payload = await _call_json_tool(analyse_year_on_year_trend, art_cinv)
    _apply_tool_payload(next_state, "analyse_year_on_year_trend", year_on_year_payload)
    payloads["year_on_year"] = year_on_year_payload

    next_state["steps"]["4"] = "completed"
    _set_workflow_state(ctx, next_state, payloads)
    await ctx.send_message(next_state)


@executor(id=ENRICH_WEATHER_ID)
async def enrich_weather(state: dict[str, Any], ctx: WorkflowContext[dict[str, Any]]) -> None:
    next_state = _json_clone(state)
    payloads = _get_payloads(ctx)
    art_cinv = int(next_state["cinv"])

    weather_payload = await _call_json_tool(correlate_weather_with_demand, art_cinv)
    _apply_tool_payload(next_state, "correlate_weather_with_demand", weather_payload)
    payloads["weather_analysis"] = weather_payload

    next_state["steps"]["5"] = "completed"
    _set_workflow_state(ctx, next_state, payloads)
    await ctx.send_message(next_state)


@executor(id=SKIP_WEATHER_ID)
async def skip_weather(state: dict[str, Any], ctx: WorkflowContext[dict[str, Any]]) -> None:
    next_state = _json_clone(state)
    payloads = _get_payloads(ctx)

    next_state["weather_enrichment_activated"] = False
    next_state["weather_analysis"] = {
        "weeks_joined": 0,
        "correlations": {"precipitation": None, "temperature": None, "wind": None},
        "weather_weeks": [],
        "weather_impacted_weeks": [],
        "interpretation": "Weather enrichment was skipped because the article context and search evidence did not indicate material weather sensitivity.",
    }
    next_state["steps"]["5"] = "skipped"
    payloads["weather_analysis"] = next_state["weather_analysis"]

    _set_workflow_state(ctx, next_state, payloads)
    await ctx.send_message(next_state)


@executor(id=PREPARE_SYNTHESIS_ID)
async def prepare_report_synthesis(state: dict[str, Any], ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    next_state = _json_clone(state)
    payloads = _get_payloads(ctx)
    next_state["steps"]["6"] = "in_progress"
    _set_workflow_state(ctx, next_state, payloads)

    synthesis_prompt = _build_synthesis_prompt(next_state, payloads)
    await ctx.send_message(
        AgentExecutorRequest(messages=[Message("user", text=synthesis_prompt)], should_respond=True)
    )


@executor(id=FINALIZE_OUTPUT_ID)
async def finalize_analysis_output(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[Never, dict[str, Any]],
) -> None:
    workflow_ctx = cast(WorkflowContext[Any, Any], ctx)
    state = _get_public_state(workflow_ctx)
    full_text = getattr(response.agent_response, "text", None) or "\n".join(
        message.text or "" for message in response.agent_response.messages if getattr(message, "text", None)
    )

    report_markdown, email_text = _split_deliverables(full_text)
    report_markdown = _ensure_stockout_guidance_in_report(report_markdown, state)
    email_text = _ensure_stockout_guidance_in_email(email_text, state)

    cinv = int(state["cinv"])
    rendered_email, email_subject = _render_email(cinv, state, email_text)
    rendered_report = _render_report(cinv, state, report_markdown)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_paths = _save_outputs(f"{cinv}_{timestamp}", rendered_report, rendered_email)

    state.update(
        {
            "status": "completed",
            "report": report_markdown,
            "email": rendered_email,
            "email_subject": email_subject,
            "output_files": output_paths,
        }
    )
    state["steps"]["6"] = "completed"
    _set_workflow_state(workflow_ctx, state, _get_payloads(workflow_ctx))
    await ctx.yield_output(state)


def create_forecast_workflow() -> Workflow:
    synthesis_executor = AgentExecutor(_build_synthesis_agent(), id=SYNTHESIS_EXECUTOR_ID)
    return (
        WorkflowBuilder(
            name=WORKFLOW_NAME,
            description=WORKFLOW_DESCRIPTION,
            start_executor=resolve_and_identify,
            output_executors=[finalize_analysis_output],
        )
        .add_edge(resolve_and_identify, load_context)
        .add_edge(load_context, run_statistics)
        .add_edge(run_statistics, analyse_year_on_year)
        .add_edge(analyse_year_on_year, enrich_weather, condition=_should_run_weather)
        .add_edge(analyse_year_on_year, skip_weather, condition=_should_skip_weather)
        .add_edge(enrich_weather, prepare_report_synthesis)
        .add_edge(skip_weather, prepare_report_synthesis)
        .add_edge(prepare_report_synthesis, synthesis_executor)
        .add_edge(synthesis_executor, finalize_analysis_output)
        .build()
    )


def _build_workflow_input(request: str | int, force_weather: bool) -> WorkflowInput:
    if isinstance(request, int):
        return WorkflowInput(str(request), force_weather)
    return WorkflowInput(str(request).strip(), force_weather)


def _extract_final_state_from_events(events: list[Any]) -> dict[str, Any]:
    for event in reversed(events):
        if getattr(event, "type", None) == "executor_completed" and getattr(event, "executor_id", None) == FINALIZE_OUTPUT_ID:
            state = _extract_completion_state(getattr(event, "data", None))
            if state is not None:
                return state
    raise RuntimeError("The forecast workflow did not produce a final analysis state.")


async def run_analysis(request: str | int, *, force_weather: bool = False) -> dict[str, Any]:
    workflow = create_forecast_workflow()
    events = await workflow.run(_build_workflow_input(request, force_weather))
    return _extract_final_state_from_events(list(events))


async def run_analysis_stream(request: str | int, *, force_weather: bool = False) -> AsyncGenerator[AGUIEvent, None]:
    request_text = str(request).strip()
    run_id = str(uuid.uuid4())
    thread_id = f"forecast-{run_id}"
    state = _create_initial_state(request_text, force_weather)
    active_message_id: str | None = None
    step_status = state["steps"].copy()

    def _update_step(step_number: int, status: str) -> AGUIEvent | None:
        current = step_status.get(str(step_number))
        if current == status:
            return None
        step_status[str(step_number)] = status
        state["steps"] = step_status.copy()
        if status == "in_progress":
            return step_started(STEP_NAMES[step_number], step_number)
        summary = "Skipped" if status == "skipped" else "Completed"
        return step_finished(STEP_NAMES[step_number], step_number, summary=summary)

    try:
        console.rule(f"Forecast workflow start: {request_text}")
        yield run_started(run_id, None, thread_id=thread_id)
        yield state_snapshot(state.copy())

        workflow = create_forecast_workflow()
        async for event in workflow.run(_build_workflow_input(request_text, force_weather), stream=True):
            event_type = getattr(event, "type", None)
            executor_id = getattr(event, "executor_id", None)

            if event_type == "executor_invoked" and executor_id in EXECUTOR_TO_STEP:
                started_event = _update_step(EXECUTOR_TO_STEP[executor_id], "in_progress")
                if started_event is not None:
                    yield started_event

            elif event_type == "executor_completed":
                extracted_state = _extract_completion_state(getattr(event, "data", None))
                if extracted_state is not None:
                    state = extracted_state
                    step_status = state.get("steps", step_status).copy()
                    yield state_snapshot(state.copy())

                if executor_id in STEP_FINAL_EXECUTORS and executor_id in EXECUTOR_TO_STEP:
                    step_number = EXECUTOR_TO_STEP[executor_id]
                    status = step_status.get(str(step_number), "completed")
                    if status not in {"completed", "skipped"}:
                        status = "completed"
                    finished_event = _update_step(step_number, status)
                    if finished_event is not None:
                        yield finished_event

                if executor_id == FINALIZE_OUTPUT_ID and active_message_id is not None:
                    yield text_message_end(active_message_id)
                    active_message_id = None

            elif event_type == "output" and executor_id == SYNTHESIS_EXECUTOR_ID and isinstance(event.data, AgentResponseUpdate):
                if active_message_id is None:
                    active_message_id = str(uuid.uuid4())
                    yield text_message_start(active_message_id)
                if event.data.text:
                    yield text_message_chunk(active_message_id, event.data.text)

            elif event_type == "executor_failed":
                details = getattr(event, "details", None)
                message = getattr(details, "message", None) or str(getattr(event, "data", "Workflow executor failed"))
                raise RuntimeError(message)

        state["status"] = "completed"
        yield state_snapshot(state.copy())
        yield run_finished(
            run_id,
            state.get("cinv"),
            thread_id=thread_id,
            result={"status": "completed", **(state.get("output_files") or {})},
        )
    except Exception as exc:
        console.print_exception(show_locals=False)
        state["status"] = "error"
        state["error"] = str(exc)
        if active_message_id is not None:
            yield text_message_end(active_message_id)
        yield state_snapshot(state.copy())
        yield run_error(run_id, str(exc), code=type(exc).__name__)


def _split_deliverables(text: str) -> tuple[str, str]:
    report_match = re.search(r"<REPORT>\s*(.*?)\s*</REPORT>", text, flags=re.IGNORECASE | re.DOTALL)
    email_match = re.search(r"<EMAIL>\s*(.*?)\s*</EMAIL>", text, flags=re.IGNORECASE | re.DOTALL)
    report = report_match.group(1).strip() if report_match else text.strip()
    email = email_match.group(1).strip() if email_match else ""
    return report, email


def _get_stockout_reporting_guidance(state: dict[str, Any]) -> tuple[str, float | None]:
    stockout_risk = _load_json_object(state.get("stockout_risk") or {})
    if not stockout_risk.get("stockout_risk_detected"):
        return "", None

    guidance = str(stockout_risk.get("reporting_guidance") or "").strip()
    pct = stockout_risk.get("baseline_reduction_pct")
    try:
        baseline_pct = float(pct) if pct is not None else None
    except (TypeError, ValueError):
        baseline_pct = None
    return guidance, baseline_pct


def _text_mentions_baseline_reduction(text: str, baseline_pct: float | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if baseline_pct is None:
        return "baseline" in lowered and ("reduction" in lowered or "lower" in lowered)

    pct_variants = {
        f"{baseline_pct:.1f}%",
        f"{baseline_pct:.2f}%",
        f"{baseline_pct:g}%",
    }
    return any(variant in text for variant in pct_variants)


def _insert_after_section(markdown: str, section_header: str, block: str) -> str:
    pattern = rf"(^##\s+{re.escape(section_header)}\s*$)"
    match = re.search(pattern, markdown, flags=re.MULTILINE)
    if not match:
        return markdown.rstrip() + f"\n\n## {section_header}\n{block.strip()}\n"

    insert_at = match.end()
    return markdown[:insert_at] + f"\n{block.strip()}" + markdown[insert_at:]


def _ensure_stockout_guidance_in_report(report_markdown: str, state: dict[str, Any]) -> str:
    guidance, baseline_pct = _get_stockout_reporting_guidance(state)
    if not guidance or _text_mentions_baseline_reduction(report_markdown, baseline_pct):
        return report_markdown

    return _insert_after_section(report_markdown, "Data Quality & Recommendations", guidance)


def _extract_email_parts(email_text: str, default_subject: str) -> tuple[str, str]:
    subject = default_subject
    body = email_text.strip()
    subject_match = re.search(r"^Subject:\s*(.+)$", body, flags=re.MULTILINE)
    if subject_match:
        subject = subject_match.group(1).strip()
        body = re.sub(r"^Subject:\s*.+$", "", body, count=1, flags=re.MULTILINE).strip()

    body = re.sub(r"^Dear Client,\s*", "", body, count=1, flags=re.IGNORECASE).strip()
    body = re.sub(
        r"Kind regards,\s*DFAI Managed Services\s*$",
        "",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    return subject, body


def _compose_email_text(subject: str, body: str) -> str:
    email_body = body.strip()
    if not subject:
        return email_body
    if not email_body:
        return f"Subject: {subject}"
    return f"Subject: {subject}\n\n{email_body}"


def _sentence_case_label(value: str) -> str:
    return (value or "").replace("_", " ").strip().lower()


def _build_email_summary_line(state: dict[str, Any]) -> str:
    article_name = state.get("article_name") or f"CINV {state.get('cinv', 'unknown')}"
    forecast_health = _load_json_object(state.get("forecast_health") or {})
    year_on_year = _load_json_object(state.get("year_on_year") or {})
    yoy_summary = _load_json_object(year_on_year.get("summary") or {})
    stockout_risk = _load_json_object(state.get("stockout_risk") or {})

    fragments = [f"We completed a review of the forecast for {article_name}."]

    trend_direction = str(yoy_summary.get("trend_direction") or "").strip().upper()
    avg_yoy_vs_nm1 = yoy_summary.get("avg_yoy_vs_nm1_pct")
    if trend_direction and trend_direction != "INSUFFICIENT_DATA" and avg_yoy_vs_nm1 is not None:
        fragments.append(
            f"Demand is {trend_direction.lower()} versus last year, with an average change of {float(avg_yoy_vs_nm1):.2f}% against the NM1 benchmark."
        )

    accuracy_rating = str(forecast_health.get("accuracy_rating") or "").strip().upper()
    weeks_analysed = forecast_health.get("weeks_analysed")
    wape_pct = forecast_health.get("wape_pct")
    if accuracy_rating and accuracy_rating != "INSUFFICIENT_DATA" and weeks_analysed:
        accuracy_line = f"Forecast accuracy over {int(weeks_analysed)} post-pivot weeks is rated {accuracy_rating.lower()}."
        if wape_pct is not None:
            accuracy_line = accuracy_line[:-1] + f", with WAPE at {float(wape_pct):.2f}%."
        fragments.append(accuracy_line)

    if stockout_risk.get("stockout_risk_detected"):
        fragments.append("The main risk is a pre-pivot stockout pattern that is distorting the baseline.")

    return " ".join(fragments)


def _build_email_observation_lines(state: dict[str, Any]) -> list[str]:
    observations: list[str] = []
    stockout_risk = _load_json_object(state.get("stockout_risk") or {})
    latest_streak = _load_json_object(stockout_risk.get("latest_zero_shipment_streak") or {})
    forecast_health = _load_json_object(state.get("forecast_health") or {})
    outlier_analysis = _load_json_object(state.get("outlier_analysis") or {})
    weather_analysis = _load_json_object(state.get("weather_analysis") or {})
    link_analysis = _load_json_object(state.get("link_analysis") or {})

    if stockout_risk.get("stockout_risk_detected"):
        count = latest_streak.get("count")
        start_week = str(latest_streak.get("start_week_id") or "").strip()
        end_week = str(latest_streak.get("end_week_id") or "").strip()
        baseline_pct = stockout_risk.get("baseline_reduction_pct")
        future_forecast_weeks = stockout_risk.get("future_forecast_weeks")
        detail = f"{count} consecutive zero-shipment weeks were observed from {start_week} to {end_week}" if count and start_week and end_week else "A pre-pivot zero-shipment streak was observed"
        if baseline_pct is not None and future_forecast_weeks:
            detail += f", which likely reduced the forecast baseline by about {float(baseline_pct):.2f}% across {int(future_forecast_weeks)} forecasted weeks"
        elif baseline_pct is not None:
            detail += f", which likely reduced the forecast baseline by about {float(baseline_pct):.2f}%"
        observations.append(detail + ".")

    outliers = outlier_analysis.get("outliers") or []
    if outliers:
        top_outliers = outliers[:2]
        formatted: list[str] = []
        for item in top_outliers:
            week_id = str(item.get("week_id") or "").strip()
            direction = _sentence_case_label(str(item.get("direction") or ""))
            holiday_name = str(item.get("holiday_name") or "").strip()
            if holiday_name:
                formatted.append(f"{week_id} ({holiday_name}, {direction} demand)")
            elif week_id:
                formatted.append(f"{week_id} ({direction} demand)")
        if formatted:
            observations.append(f"Additional anomalous weeks were identified around {', '.join(formatted)}.")

    impacted_weeks = weather_analysis.get("weather_impacted_weeks") or []
    if impacted_weeks:
        sample = impacted_weeks[:2]
        impacted_labels = ", ".join(str(item.get("week_id") or "").strip() for item in sample if item.get("week_id"))
        if impacted_labels:
            observations.append(f"Weather-related demand disruption was also visible in {impacted_labels}.")

    if link_analysis.get("duplicate_link_warning"):
        observations.append("Duplicate linked-article rows were detected, so substitution evidence should be interpreted cautiously.")

    if not observations and forecast_health.get("accuracy_rating"):
        observations.append(
            f"No single structural driver dominated the review, but the forecast health assessment was rated {str(forecast_health.get('accuracy_rating')).lower()}."
        )

    return observations[:4]


def _build_email_action_lines(state: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    stockout_risk = _load_json_object(state.get("stockout_risk") or {})
    forecast_health = _load_json_object(state.get("forecast_health") or {})
    flagged_weeks = state.get("flagged_weeks") or []
    latest_streak = _load_json_object(stockout_risk.get("latest_zero_shipment_streak") or {})
    affected_period = str(stockout_risk.get("affected_period") or "").strip()
    if not affected_period:
        start_week = str(latest_streak.get("start_week_id") or "").strip()
        end_week = str(latest_streak.get("end_week_id") or "").strip()
        if start_week and end_week:
            affected_period = f"{start_week} to {end_week}"

    if stockout_risk.get("stockout_risk_detected"):
        if affected_period:
            actions.append(f"Apply xout logic for {affected_period} to rebuild the baseline without the zero-demand distortion.")
            actions.append(f"Create or extend a temporary article link for {affected_period} to stabilize the forecast baseline.")
        else:
            actions.append("Apply xout logic to rebuild the baseline without the zero-demand distortion.")
            actions.append("Create or extend a temporary article link to stabilize the forecast baseline.")
    else:
        for action in stockout_risk.get("recommended_actions") or []:
            clean_action = str(action or "").strip()
            if clean_action:
                actions.append(clean_action)

    if flagged_weeks:
        actions.append("Flag the most anomalous weeks for model exclusion so they do not distort future learning.")

    if forecast_health.get("tracking_signal_flag"):
        actions.append("Review model bias and tracking signal before the next forecast refresh.")

    if not actions:
        actions.append("Keep the current forecast under review and continue monitoring for new anomalies or supply disruptions.")

    deduped: list[str] = []
    seen: set[str] = set()
    for action in actions:
        normalized = action.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(action)
    return deduped[:3]


def _build_structured_email_body(state: dict[str, Any], fallback_body: str) -> str:
    summary_line = _build_email_summary_line(state)
    observation_lines = _build_email_observation_lines(state)
    action_lines = _build_email_action_lines(state)

    body_parts = [summary_line]

    if observation_lines:
        body_parts.append("Key observations\n- " + "\n- ".join(observation_lines))

    if action_lines:
        body_parts.append("Recommended actions\n- " + "\n- ".join(action_lines))

    closing = (
        "Please let us know if there were supply constraints, customer changes, or other business events behind these weeks, "
        "as that context will help us confirm the final recommendation."
    )
    body_parts.append(closing)

    structured_body = "\n\n".join(part.strip() for part in body_parts if part and part.strip())
    if structured_body.strip():
        return structured_body
    return fallback_body.strip()


def _email_mentions_stockout_context(email_body: str, state: dict[str, Any]) -> bool:
    lowered = (email_body or "").lower()
    if not lowered:
        return False

    if any(term in lowered for term in ("stockout", "shortage", "zero shipment", "zero-shipment")):
        return True

    stockout_risk = _load_json_object(state.get("stockout_risk") or {})
    latest_streak = _load_json_object(stockout_risk.get("latest_zero_shipment_streak") or {})
    start_week = str(latest_streak.get("start_week_id") or "").lower()
    end_week = str(latest_streak.get("end_week_id") or "").lower()
    return bool(start_week and end_week and start_week in lowered and end_week in lowered)


def _email_mentions_stockout_remediation(email_body: str) -> bool:
    lowered = (email_body or "").lower()
    return any(term in lowered for term in ("xout", "article link", "linked article"))


def _ensure_stockout_guidance_in_email(email_text: str, state: dict[str, Any]) -> str:
    guidance, baseline_pct = _get_stockout_reporting_guidance(state)
    default_subject = state.get("email_subject") or ""
    subject, email_body = _extract_email_parts(email_text, default_subject)
    if not guidance or _text_mentions_baseline_reduction(email_body, baseline_pct):
        return _compose_email_text(subject, email_body)

    stockout_risk = _load_json_object(state.get("stockout_risk") or {})
    latest_streak = _load_json_object(stockout_risk.get("latest_zero_shipment_streak") or {})
    affected_period = str(stockout_risk.get("affected_period") or "").strip()
    if not affected_period:
        start_week = str(latest_streak.get("start_week_id") or "").strip()
        end_week = str(latest_streak.get("end_week_id") or "").strip()
        if start_week and end_week:
            affected_period = f"{start_week} to {end_week}"

    future_forecast_weeks = stockout_risk.get("future_forecast_weeks")
    baseline_clause = ""
    if baseline_pct is not None:
        baseline_clause = f"This likely depressed the forecast baseline by about {baseline_pct:.2f}%"
        if future_forecast_weeks:
            baseline_clause += f" across {future_forecast_weeks} forecasted weeks"
        baseline_clause += "."

    remediation_clause = ""
    if not _email_mentions_stockout_remediation(email_body) and affected_period:
        remediation_clause = (
            f" We recommend applying xout logic or a temporary article link for the affected period {affected_period}."
        )
    elif not _email_mentions_stockout_remediation(email_body):
        remediation_clause = " We recommend applying xout logic or a temporary article link for the affected period."

    if _email_mentions_stockout_context(email_body, state):
        stockout_sentence = (baseline_clause + remediation_clause).strip()
    else:
        count = latest_streak.get("count")
        start_week = str(latest_streak.get("start_week_id") or "").strip()
        end_week = str(latest_streak.get("end_week_id") or "").strip()
        range_text = f" from {start_week} to {end_week}" if start_week and end_week else ""
        intro = (
            f"We observed {count} consecutive zero-shipment weeks{range_text}."
            if count
            else "We observed a pre-pivot zero-shipment disruption."
        )
        stockout_sentence = f"{intro} {baseline_clause}{remediation_clause}".strip()

    if not stockout_sentence:
        return _compose_email_text(subject, email_body)

    if not email_body:
        updated_body = stockout_sentence
    else:
        updated_body = f"{email_body}\n\n{stockout_sentence}"
    return _compose_email_text(subject, updated_body)


def _render_report(cinv: int, state: dict[str, Any], report_markdown: str) -> str:
    template = Template(REPORT_TEMPLATE)
    return template.render(
        article_name=state.get("article_name") or "Unknown Article",
        cinv_id=cinv,
        category=state.get("category") or "Unknown Category",
        pivot_date=state.get("pivot_date") or "Unknown",
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        analysis_content=report_markdown,
        flagged_weeks=state.get("flagged_weeks", []),
    )


def _render_email(cinv: int, state: dict[str, Any], email_text: str) -> tuple[str, str]:
    subject = state.get("email_subject") or f"Forecast Review - {state.get('article_name') or f'CINV {cinv}'}"
    subject, body = _extract_email_parts(email_text, subject)
    body = _build_structured_email_body(state, body)
    template = Template(EMAIL_TEMPLATE)
    rendered = template.render(
        article_name=state.get("article_name") or "Unknown Article",
        cinv_id=cinv,
        email_body=body,
    )
    rendered = re.sub(r"^Subject:\s*.+$", f"Subject: {subject}", rendered, count=1, flags=re.MULTILINE)
    return rendered, subject


def _save_outputs(output_prefix: str, report_text: str, email_text: str) -> dict[str, str]:
    output_dir = config.ensure_output_dir()
    report_path = output_dir / f"{output_prefix}_report.txt"
    email_path = output_dir / f"{output_prefix}_email.txt"
    report_path.write_text(report_text, encoding="utf-8")
    email_path.write_text(email_text, encoding="utf-8")
    return {"report": str(report_path), "email": str(email_path)}
