"""Main Forecast Analysis Agent built on Microsoft Agent Framework."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Template
from rich.console import Console

from agent_framework import AgentResponseUpdate
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework.openai import OpenAIResponsesClient

from . import config, runtime
from .events import (
    AGUIEvent,
    run_error,
    run_finished,
    run_started,
    state_delta,
    state_snapshot,
    step_finished,
    step_started,
    text_message_chunk,
    text_message_end,
    text_message_start,
    tool_call_args,
    tool_call_end,
    tool_call_result,
    tool_call_start,
)
from .templates import EMAIL_TEMPLATE, REPORT_TEMPLATE, SYSTEM_PROMPT, build_run_prompt
from .tools import ALL_TOOLS, detect_pre_pivot_stockout_risk

console = Console()

STEP_NAMES = {
    1: "Article Identification",
    2: "Data Loading",
    3: "Statistical Analysis",
    4: "Year-on-Year Trend Analysis",
    5: "Weather Context",
    6: "Report and Email Generation",
}

TOOL_TO_STEP = {
    "get_article_metadata": 1,
    "search_article_characteristics": 1,
    "get_forecast_data": 2,
    "get_article_links": 2,
    "get_article_links_demand": 2,
    "search_holiday_demand_correlation": 2,
    "compute_forecast_health": 3,
    "detect_pre_pivot_stockout_risk": 3,
    "detect_outlier_weeks": 3,
    "analyse_year_on_year_trend": 4,
    "correlate_weather_with_demand": 5,
}

TOOL_STATE_FIELDS = (
    "article_name",
    "category",
    "pivot_date",
    "flagged_weeks",
    "forecast_health",
    "outlier_analysis",
    "stockout_risk",
    "year_on_year",
    "link_analysis",
    "weather_sensitivity",
    "weather_analysis",
    "weather_enrichment_recommended",
    "weather_enrichment_activated",
    "weather_weeks",
)


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


def _tool_state_delta(state: dict[str, Any]) -> dict[str, Any]:
    return {field: state.get(field) for field in TOOL_STATE_FIELDS}


def _apply_tool_payload(state: dict[str, Any], tool_payloads: dict[str, Any], tool_name: str, payload: dict[str, Any]) -> None:
    tool_payloads[tool_name] = payload
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
        flagged = []
        for item in payload.get("outliers", [])[:12]:
            reasons: list[str] = []
            if item.get("iqr_outlier"):
                reasons.append("IQR outlier")
            z_score = item.get("z_score")
            if z_score is not None and abs(float(z_score)) > 2:
                reasons.append(f"z-score {z_score}")
            modified_z = item.get("modified_z_score")
            if item.get("modified_z_outlier") and modified_z is not None:
                reasons.append(f"robust z-score {modified_z}")
            rolling_ratio = item.get("rolling_median_ratio")
            if item.get("rolling_baseline_outlier") and rolling_ratio is not None:
                reasons.append(f"vs recent baseline {float(rolling_ratio):.2f}x")
            historical_ratio = item.get("historical_ratio")
            if item.get("historical_baseline_outlier") and historical_ratio is not None:
                reasons.append(f"vs NM baseline {float(historical_ratio):.2f}x")
            severity = str(item.get("severity") or "").strip().upper()
            if severity in {"HIGH", "SEVERE"}:
                reasons.append(f"{severity.lower()} volatility")
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
            if item.get("week_id") in existing:
                continue
            state.setdefault("flagged_weeks", []).append(
                {
                    "week_id": item.get("week_id"),
                    "mvt_date": item.get("MVT_DATE"),
                    "actual_demand": item.get("ACTUAL_DEMAND"),
                    "forecast": item.get("FORECAST"),
                    "reason": f"severe weather: {item.get('weather_description', 'weather disruption')}",
                }
            )
class ForecastAnalysisAgent:
    """Forecast agent wrapper that streams AG-UI events while the SDK runs the model and tools."""

    def __init__(self) -> None:
        self._client: AzureOpenAIResponsesClient | OpenAIResponsesClient | None = None
        self._agent = None
        self.last_state: dict[str, Any] | None = None

    def _build_client(self) -> AzureOpenAIResponsesClient | OpenAIResponsesClient:
        if self._client is None:
            self._client = config.build_responses_client()
        return self._client

    def _build_agent(self):
        if self._agent is None:
            client = self._build_client()
            self._agent = client.as_agent(
                name="ForecastAnalysisAgent",
                instructions=SYSTEM_PROMPT,
                tools=ALL_TOOLS,
            )
        return self._agent

    async def run_analysis(self, cinv: int, *, force_weather: bool = False) -> dict[str, Any]:
        """Run a full analysis and return the final state after consuming the stream."""
        async for _ in self.run_analysis_stream(cinv, force_weather=force_weather):
            pass
        return self.last_state or {}

    async def run_analysis_stream(self, cinv: int, *, force_weather: bool = False) -> AsyncGenerator[AGUIEvent, None]:
        """Run a full analysis and yield AG-UI-compatible stream events."""
        run_id = str(uuid.uuid4())
        thread_id = f"forecast-{cinv}"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"{cinv}_{timestamp}"

        tool_names_by_call: dict[str, str] = {}
        tool_args_by_call: dict[str, str] = {}
        tool_started_at: dict[str, float] = {}
        active_message_id: str | None = None
        text_buffer: list[str] = []
        tool_payloads: dict[str, Any] = {}
        step_state = {str(number): "waiting" for number in STEP_NAMES}
        current_step: int | None = None

        state: dict[str, Any] = {
            "cinv": int(cinv),
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
            "steps": step_state.copy(),
            "force_weather": bool(force_weather),
            "run_id": run_id,
            "thread_id": thread_id,
            "output_files": {},
        }
        self.last_state = state.copy()

        try:
            state["stockout_risk"] = _load_json_object(detect_pre_pivot_stockout_risk(cinv))
        except Exception:
            state["stockout_risk"] = {}

        def update_state(**updates: Any) -> AGUIEvent:
            state.update(updates)
            self.last_state = _json_clone(state)
            return state_delta(updates)

        async def transition_step(step_number: int) -> AsyncGenerator[AGUIEvent, None]:
            nonlocal current_step
            if current_step == step_number:
                return
            if current_step is not None and step_state[str(current_step)] != "completed":
                step_state[str(current_step)] = "completed"
                yield step_finished(STEP_NAMES[current_step], current_step)
            current_step = step_number
            if step_state[str(step_number)] != "completed":
                step_state[str(step_number)] = "in_progress"
                yield step_started(STEP_NAMES[step_number], step_number)
                yield update_state(steps=step_state.copy(), current_step=step_number)

        token = runtime.set_weather_forced(force_weather)
        try:
            console.rule(f"Forecast analysis start: CINV {cinv}")
            yield run_started(run_id, cinv, thread_id=thread_id)
            yield state_snapshot(state.copy())

            agent = self._build_agent()
            stream = agent.run(build_run_prompt(cinv, force_weather), stream=True)

            async for update in stream:
                if not isinstance(update, AgentResponseUpdate):
                    continue

                for content in update.contents:
                    content_type = getattr(content, "type", None)
                    if content_type == "function_call":
                        tool_name = getattr(content, "name", None) or "tool"
                        tool_call_id = getattr(content, "call_id", None) or str(uuid.uuid4())
                        args = getattr(content, "arguments", "")
                        if isinstance(args, dict):
                            args_text = json.dumps(args)
                            parsed_args = args
                        else:
                            args_text = str(args or "")
                            try:
                                parsed_args = json.loads(args_text) if args_text else {}
                            except json.JSONDecodeError:
                                parsed_args = {"raw": args_text}

                        step_number = TOOL_TO_STEP.get(tool_name)
                        if step_number is not None:
                            async for event in transition_step(step_number):
                                yield event

                        if tool_call_id not in tool_names_by_call:
                            tool_names_by_call[tool_call_id] = tool_name
                            tool_started_at[tool_call_id] = time.perf_counter()
                            tool_args_by_call[tool_call_id] = ""
                            console.log(f"Tool start: {tool_name}")
                            yield tool_call_start(tool_call_id, tool_name, parsed_args, parent_message_id=active_message_id)

                        previous = tool_args_by_call.get(tool_call_id, "")
                        if args_text == previous:
                            continue
                        delta = args_text[len(previous) :] if args_text.startswith(previous) else args_text
                        tool_args_by_call[tool_call_id] = args_text
                        if delta:
                            yield tool_call_args(tool_call_id, delta, tool_name)

                    elif content_type == "function_result":
                        tool_call_id = getattr(content, "call_id", None) or "unknown"
                        tool_name = tool_names_by_call.get(tool_call_id, "tool")
                        result_value = getattr(content, "result", None)
                        result_text = result_value if isinstance(result_value, str) else json.dumps(result_value, default=str)
                        duration_ms = (time.perf_counter() - tool_started_at.get(tool_call_id, time.perf_counter())) * 1000
                        console.log(f"Tool end: {tool_name} ({duration_ms:.1f} ms)")
                        yield tool_call_end(tool_call_id, tool_name, duration_ms, result_text)
                        yield tool_call_result(tool_call_id, tool_name, result_text, message_id=active_message_id or str(uuid.uuid4()))

                        payload = _load_json_object(result_text)
                        _apply_tool_payload(state, tool_payloads, tool_name, payload)
                        yield update_state(**_tool_state_delta(state))

                    elif content_type == "text":
                        chunk = getattr(content, "text", "") or ""
                        if not chunk:
                            continue
                        if current_step != 6:
                            async for event in transition_step(6):
                                yield event
                        if active_message_id is None:
                            active_message_id = str(uuid.uuid4())
                            yield text_message_start(active_message_id)
                        text_buffer.append(chunk)
                        yield text_message_chunk(active_message_id, chunk)

            final_response = await stream.get_final_response()
            final_text = getattr(final_response, "text", None) or "".join(text_buffer)
            if active_message_id is not None:
                yield text_message_end(active_message_id)

            report_markdown, email_text = _split_deliverables(final_text)
            report_markdown = _ensure_volatility_guidance_in_report(report_markdown, state)
            report_markdown = _ensure_stockout_guidance_in_report(report_markdown, state)
            email_text = _ensure_stockout_guidance_in_email(email_text, state)
            rendered_email, email_subject = _render_email(cinv, state, email_text)
            rendered_report = _render_report(cinv, state, report_markdown)
            output_paths = _save_outputs(output_prefix, rendered_report, rendered_email)

            for step_number in range(1, 7):
                if step_state[str(step_number)] != "completed":
                    if step_number == 5 and not state.get("weather_enrichment_activated"):
                        step_state[str(step_number)] = "skipped"
                    else:
                        step_state[str(step_number)] = "completed"
                    yield step_finished(STEP_NAMES[step_number], step_number)

            yield update_state(
                status="completed",
                steps=step_state.copy(),
                report=report_markdown,
                email=rendered_email,
                email_subject=email_subject,
                output_files=output_paths,
                flagged_weeks=state.get("flagged_weeks", []),
                stockout_risk=state.get("stockout_risk", {}),
            )
            yield state_snapshot(state.copy())
            console.log(f"Saved outputs to {output_paths['report']} and {output_paths['email']}")
            yield run_finished(run_id, cinv, thread_id=thread_id, result={"status": "completed", **output_paths})
        except Exception as exc:
            console.print_exception(show_locals=False)
            state["status"] = "error"
            state["error"] = str(exc)
            self.last_state = _json_clone(state)
            yield state_snapshot(state.copy())
            yield run_error(run_id, str(exc), code=type(exc).__name__)
        finally:
            runtime.reset_weather_forced(token)


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
    outlier_analysis = _load_json_object(state.get("outlier_analysis") or {})
    volatility_summary = _load_json_object(outlier_analysis.get("summary") or {})
    year_on_year = _load_json_object(state.get("year_on_year") or {})
    yoy_summary = _load_json_object(year_on_year.get("summary") or {})
    stockout_risk = _load_json_object(state.get("stockout_risk") or {})

    fragments = [f"We completed a review of the forecast for {article_name}."]

    volatility_level = str(volatility_summary.get("volatility_level") or "").strip().upper()
    outlier_count = outlier_analysis.get("count")
    cv = volatility_summary.get("coefficient_of_variation")
    if volatility_level == "HIGH" and outlier_count:
        detail = f"Shipments are highly volatile, with {int(outlier_count)} flagged outlier weeks"
        if cv is not None:
            detail += f" and a coefficient of variation of {float(cv) * 100:.1f}%"
        fragments.append(detail + ".")

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
    volatility_summary = _load_json_object(outlier_analysis.get("summary") or {})
    volatile_periods = outlier_analysis.get("volatile_periods") or []
    year_on_year = _load_json_object(state.get("year_on_year") or {})
    yoy_summary = _load_json_object(year_on_year.get("summary") or {})
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
    if str(volatility_summary.get("volatility_level") or "").upper() == "HIGH":
        high_count = volatility_summary.get("high_outlier_count")
        low_count = volatility_summary.get("low_outlier_count")
        median_change = volatility_summary.get("median_abs_weekly_change_pct")
        detail = "The shipment profile is structurally volatile"
        if high_count is not None and low_count is not None:
            detail += f", with {int(high_count)} high spikes and {int(low_count)} low anomalies"
        if median_change is not None:
            detail += f" and a median week-on-week change of {float(median_change):.1f}%"
        observations.append(detail + ".")

    if volatile_periods:
        primary_period = volatile_periods[0]
        start_week = str(primary_period.get("start_week_id") or "").strip()
        end_week = str(primary_period.get("end_week_id") or "").strip()
        outlier_count = primary_period.get("outlier_count")
        dominant_pattern = str(primary_period.get("dominant_pattern") or "").strip()
        if start_week and end_week and outlier_count:
            observations.append(
                f"The main volatile period ran from {start_week} to {end_week}, with {int(outlier_count)} flagged weeks driven by {dominant_pattern}."
            )

    if outliers:
        top_outliers = outliers[:2]
        formatted = []
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
    outlier_analysis = _load_json_object(state.get("outlier_analysis") or {})
    volatility_summary = _load_json_object(outlier_analysis.get("summary") or {})
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

    if str(volatility_summary.get("volatility_level") or "").upper() == "HIGH":
        actions.append("Review whether recurring spikes reflect missing event, promotion, or customer-order signals in the baseline model.")

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


def _build_volatility_report_block(state: dict[str, Any]) -> str:
    outlier_analysis = _load_json_object(state.get("outlier_analysis") or {})
    summary = _load_json_object(outlier_analysis.get("summary") or {})
    outliers = outlier_analysis.get("outliers") or []
    volatile_periods = outlier_analysis.get("volatile_periods") or []
    if not summary and not outliers:
        return ""

    lines: list[str] = []
    weeks_evaluated = summary.get("weeks_evaluated")
    outlier_count = outlier_analysis.get("count")
    volatility_level = str(summary.get("volatility_level") or "").strip().upper()
    cv = summary.get("coefficient_of_variation")
    median_change = summary.get("median_abs_weekly_change_pct")
    if weeks_evaluated and outlier_count is not None:
        headline = f"- Volatility assessment: {volatility_level.lower() if volatility_level else 'unclassified'} across {int(weeks_evaluated)} positive-demand weeks, with {int(outlier_count)} flagged outliers"
        if cv is not None:
            headline += f" and coefficient of variation at {float(cv) * 100:.1f}%"
        if median_change is not None:
            headline += f"; median week-on-week change was {float(median_change):.1f}%"
        lines.append(headline + ".")

    high_count = summary.get("high_outlier_count")
    low_count = summary.get("low_outlier_count")
    peak_week = summary.get("peak_spike_week_id")
    peak_units = summary.get("peak_spike_units")
    if high_count is not None and low_count is not None:
        composition = f"- Composition: {int(high_count)} high-demand spikes and {int(low_count)} low-demand troughs were detected"
        if peak_week and peak_units is not None:
            composition += f"; the largest spike was {peak_week} at {peak_units} units"
        lines.append(composition + ".")

    if volatile_periods:
        period_fragments = []
        for period in volatile_periods[:2]:
            start_week = str(period.get("start_week_id") or "").strip()
            end_week = str(period.get("end_week_id") or "").strip()
            count = period.get("outlier_count")
            pattern = str(period.get("dominant_pattern") or "").strip()
            if start_week and end_week and count:
                period_fragments.append(f"{start_week} to {end_week} ({int(count)} flagged weeks, {pattern})")
        if period_fragments:
            lines.append(f"- Volatile periods: {'; '.join(period_fragments)}.")

    if outliers:
        examples = []
        for item in outliers[:3]:
            week_id = str(item.get("week_id") or "").strip()
            actual = item.get("ACTUAL_DEMAND")
            ratio = item.get("historical_ratio")
            if week_id and actual is not None:
                example = f"{week_id} ({actual} units"
                if ratio is not None:
                    example += f", {float(ratio):.2f}x the NM baseline"
                example += ")"
                examples.append(example)
        if examples:
            lines.append(f"- Representative flagged weeks: {', '.join(examples)}.")

    return "\n".join(lines)


def _ensure_volatility_guidance_in_report(report_markdown: str, state: dict[str, Any]) -> str:
    block = _build_volatility_report_block(state)
    if not block:
        return report_markdown
    return _insert_after_section(report_markdown, "Demand Volatility & Outlier Analysis", block)


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