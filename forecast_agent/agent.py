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

from azure.identity import DefaultAzureCredential
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
    "stockout_risk",
    "weather_sensitivity",
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
    elif tool_name == "detect_pre_pivot_stockout_risk":
        state["stockout_risk"] = payload
    elif tool_name == "detect_outlier_weeks":
        flagged = []
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
    elif tool_name == "correlate_weather_with_demand":
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
        if self._client is not None:
            return self._client

        azure_openai_base_url = _normalize_openai_base_url(config.AZURE_OPENAI_ENDPOINT)
        if azure_openai_base_url and config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_MODEL_ID:
            self._client = OpenAIResponsesClient(
                base_url=azure_openai_base_url,
                api_key=config.AZURE_OPENAI_API_KEY,
                model_id=config.AZURE_OPENAI_MODEL_ID,
            )
            return self._client

        if config.OPENAI_BASE_URL and config.OPENAI_API_KEY and config.OPENAI_RESPONSES_MODEL_ID:
            self._client = OpenAIResponsesClient(
                base_url=_normalize_openai_base_url(config.OPENAI_BASE_URL),
                api_key=config.OPENAI_API_KEY,
                model_id=config.OPENAI_RESPONSES_MODEL_ID,
            )
            return self._client

        if config.AZURE_AI_PROJECT_ENDPOINT and config.AZURE_OPENAI_MODEL_ID:
            self._client = AzureOpenAIResponsesClient(
                project_endpoint=config.AZURE_AI_PROJECT_ENDPOINT,
                deployment_name=config.AZURE_OPENAI_MODEL_ID,
                credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
            )
            return self._client

        if config.OPENAI_API_KEY and config.OPENAI_RESPONSES_MODEL_ID:
            self._client = OpenAIResponsesClient(
                api_key=config.OPENAI_API_KEY,
                model_id=config.OPENAI_RESPONSES_MODEL_ID,
            )
            return self._client

        raise RuntimeError(
            "No supported model configuration is available. Configure Azure AI project variables, Azure OpenAI API key "
            "variables, or direct OpenAI responses variables."
        )

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
            "weather_sensitivity": {},
            "weather_enrichment_recommended": False,
            "weather_enrichment_activated": False,
            "weather_weeks": [],
            "stockout_risk": {},
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


def _normalize_openai_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip().strip('"').strip("'")
    if not base_url:
        return ""
    return base_url if base_url.endswith("/") else f"{base_url}/"


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


def _ensure_stockout_guidance_in_email(email_text: str, state: dict[str, Any]) -> str:
    guidance, baseline_pct = _get_stockout_reporting_guidance(state)
    if not guidance or _text_mentions_baseline_reduction(email_text, baseline_pct):
        return email_text

    stockout_sentence = guidance.replace("The recommended solution is to either ", "DFAI recommends that we either ")
    email_body = email_text.strip()
    if not email_body:
        return stockout_sentence
    return f"{email_body}\n\n{stockout_sentence}"


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
    body = email_text.strip()
    subject_match = re.search(r"^Subject:\s*(.+)$", body, flags=re.MULTILINE)
    if subject_match:
        subject = subject_match.group(1).strip()
        body = re.sub(r"^Subject:\s*.+$", "", body, count=1, flags=re.MULTILINE).strip()
    body = re.sub(r"^Dear Client,\s*", "", body, count=1).strip()
    body = re.sub(r"Kind regards,\s*DFAI Managed Services\s*$", "", body, flags=re.DOTALL).strip()
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