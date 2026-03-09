"""Streamlit frontend for the Forecast Analysis Agent."""

from __future__ import annotations

import html
import json
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from forecast_agent.data_access import get_article_links_frame, load_metadata_frame
from forecast_agent.templates import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE = f"http://127.0.0.1:{API_PORT}"
DISPLAY_TEXT_BATCH_CHARS = 1200
DISPLAY_TOOL_ARGS_BATCH_CHARS = 600
DISPLAY_RESULT_PREVIEW_CHARS = 360

TAB_NAMES = ["Chat", "Report", "Charts", "Logs", "Observability"]
STEP_LABELS = {
    "1": "Article ID",
    "2": "Data Loading",
    "3": "Statistical Analysis",
    "4": "Year-on-Year Trend",
    "5": "Weather Context",
    "6": "Report Generation",
}

CHART_COLORS = {
    "blue": "#0074E8",
    "sky": "#1EACFC",
    "mint": "#00EAC3",
    "violet": "#A933FB",
    "ink": "#121212",
    "amber": "#FFB600",
    "orange": "#FF8500",
    "cream": "#F8D18B",
}

st.set_page_config(page_title="Forecast Analysis Agent", layout="wide", initial_sidebar_state="collapsed")


# ===================================================================
# DATA LOADERS (preserved)
# ===================================================================
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast = pd.read_csv(DATA_DIR / "forecast_data.csv")
    metadata = pd.read_csv(DATA_DIR / "article_metadata.csv")
    links = pd.read_csv(DATA_DIR / "links.csv")

    forecast.columns = [column.strip() for column in forecast.columns]
    metadata.columns = [column.strip() for column in metadata.columns]
    links.columns = [column.strip() for column in links.columns]

    for frame in (forecast, metadata, links):
        if "ART_CINV" in frame.columns:
            frame["ART_CINV"] = pd.to_numeric(frame["ART_CINV"], errors="coerce").astype("Int64")

    for column in ("PIVOT_DATE", "MVT_DATE"):
        if column in forecast.columns:
            forecast[f"{column}_DT"] = pd.to_datetime(forecast[column].astype(str), format="%Y%m%d", errors="coerce")

    forecast["FORECAST_ERROR"] = pd.to_numeric(forecast.get("FORECAST_ERROR"), errors="coerce")
    forecast["FORECAST"] = pd.to_numeric(forecast.get("FORECAST"), errors="coerce")
    forecast["ACTUAL_DEMAND"] = pd.to_numeric(forecast.get("ACTUAL_DEMAND"), errors="coerce")
    forecast["NM1"] = pd.to_numeric(forecast.get("NM1"), errors="coerce")
    forecast["NM2"] = pd.to_numeric(forecast.get("NM2"), errors="coerce")
    forecast["NM3"] = pd.to_numeric(forecast.get("NM3"), errors="coerce")
    forecast_iso = forecast["MVT_DATE_DT"].dt.isocalendar()
    forecast["WEEK_ID"] = forecast_iso["year"].astype(str) + "-W" + forecast_iso["week"].astype(str).str.zfill(2)

    return forecast, metadata, links


@st.cache_data(ttl=20, show_spinner=False)
def fetch_health() -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{API_BASE}/api/health", timeout=2)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


# ===================================================================
# SESSION STATE (preserved, extended with new UI keys)
# ===================================================================
def ensure_state() -> None:
    defaults: dict[str, Any] = {
        "analysis_running": False,
        "agent_events": [],
        "trace_log": [],
        "report_md": "",
        "report_stream": "",
        "email_draft": "",
        "email_subject": "",
        "agent_state": {},
        "step_status": {str(index): "waiting" for index in range(1, 7)},
        "event_queue": queue.Queue(),
        "analysis_thread": None,
        "selected_cinv": None,
        "force_weather": False,
        "error_message": "",
        "view_mode": "Analysis",
        # New UI keys
        "active_tab": "Chat",
        "show_prompt": False,
        "expanded_cards": set(),
        "expanded_log_row": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Sync active tab from query params
    qp = st.query_params
    tab_param = qp.get("tab", "").capitalize()
    if tab_param in TAB_NAMES:
        st.session_state["active_tab"] = tab_param


def reset_analysis_state() -> None:
    st.session_state.analysis_running = False
    st.session_state.agent_events = []
    st.session_state.trace_log = []
    st.session_state.report_md = ""
    st.session_state.report_stream = ""
    st.session_state.email_draft = ""
    st.session_state.email_subject = ""
    st.session_state.agent_state = {}
    st.session_state.step_status = {str(index): "waiting" for index in range(1, 7)}
    st.session_state.event_queue = queue.Queue()
    st.session_state.analysis_thread = None
    st.session_state.error_message = ""
    st.session_state.expanded_cards = set()
    st.session_state.expanded_log_row = None


# ===================================================================
# STREAMING & EVENT HANDLING (preserved exactly)
# ===================================================================
def start_analysis(cinv: int, force_weather: bool) -> None:
    reset_analysis_state()
    st.session_state.analysis_running = True
    st.session_state.selected_cinv = cinv
    st.session_state.force_weather = force_weather
    worker = threading.Thread(target=_consume_stream, args=(st.session_state.event_queue, cinv, force_weather), daemon=True)
    st.session_state.analysis_thread = worker
    worker.start()


def _consume_stream(event_queue: queue.Queue[dict[str, Any]], cinv: int, force_weather: bool) -> None:
    payload = {"cinv": int(cinv), "force_weather": bool(force_weather)}
    try:
        with httpx.Client(timeout=None) as client:
            with client.stream("POST", f"{API_BASE}/api/run", json=payload, headers={"Accept": "text/event-stream"}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    event = json.loads(raw)
                    event_queue.put(event)
                    if event.get("type") == "STREAM_END":
                        break
    except Exception as exc:
        event_queue.put({"type": "RUN_ERROR", "error": str(exc), "code": type(exc).__name__})
        event_queue.put({"type": "STREAM_END"})


def drain_events() -> None:
    while True:
        try:
            event = st.session_state.event_queue.get_nowait()
        except queue.Empty:
            break
        handle_event(event)


def handle_event(event: dict[str, Any]) -> None:
    event_type = event.get("type", "")
    st.session_state.agent_events.append(event)

    if event_type == "RUN_STARTED":
        append_trace("system", "Run started", f"CINV {event.get('cinv', st.session_state.selected_cinv)}", "running")
    elif event_type == "STEP_STARTED":
        step_number = str(event.get("step_number"))
        st.session_state.step_status[step_number] = "in_progress"
        append_trace("step", event.get("step_name", "Step started"), "In progress", "running")
    elif event_type == "STEP_FINISHED":
        step_number = str(event.get("step_number"))
        if st.session_state.step_status.get(step_number) != "skipped":
            st.session_state.step_status[step_number] = "completed"
        append_trace("step", event.get("step_name", "Step finished"), "Completed", "done")
    elif event_type in {"TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_CHUNK"}:
        st.session_state.report_stream += event.get("delta", "")
    elif event_type == "TOOL_CALL_START":
        append_trace(
            event.get("category", "tool"),
            event.get("tool_name", "tool"),
            json.dumps(event.get("args", {}), ensure_ascii=False, indent=2),
            "running",
        )
    elif event_type == "TOOL_CALL_END":
        append_trace(
            event.get("category", "tool"),
            event.get("tool_name", "tool"),
            f"{event.get('duration_ms', 0):.1f} ms\n\n{event.get('result', '')}",
            "done",
        )
    elif event_type == "TOOL_CALL_RESULT":
        if event.get("result"):
            append_trace(event.get("category", "tool"), f"{event.get('tool_name', 'tool')} result", event.get("result", ""), "done")
    elif event_type == "STATE_SNAPSHOT":
        snapshot = event.get("state") or event.get("snapshot") or {}
        apply_state_update(snapshot)
    elif event_type == "STATE_DELTA":
        delta = event.get("delta_dict") or patches_to_dict(event.get("delta", []))
        apply_state_update(delta)
    elif event_type == "RUN_FINISHED":
        st.session_state.analysis_running = False
        append_trace("system", "Run finished", "Analysis complete and output files were written.", "done")
    elif event_type == "RUN_ERROR":
        st.session_state.analysis_running = False
        st.session_state.error_message = event.get("error") or event.get("message") or "Unknown error"
        append_trace("system", "Run failed", st.session_state.error_message, "error")
    elif event_type == "STREAM_END":
        st.session_state.analysis_running = False


def patches_to_dict(patches: Any) -> dict[str, Any]:
    if isinstance(patches, dict):
        return patches
    if not isinstance(patches, list):
        return {}
    flattened: dict[str, Any] = {}
    for patch in patches:
        if patch.get("op") != "replace":
            continue
        flattened[patch.get("path", "/").lstrip("/")] = patch.get("value")
    return flattened


def apply_state_update(delta: dict[str, Any]) -> None:
    if not delta:
        return
    st.session_state.agent_state.update(delta)
    if delta.get("steps"):
        st.session_state.step_status.update(delta["steps"])
    if "report" in delta and delta["report"]:
        st.session_state.report_md = delta["report"]
    if "email" in delta and delta["email"]:
        st.session_state.email_draft = delta["email"]
    if "email_subject" in delta and delta["email_subject"]:
        st.session_state.email_subject = delta["email_subject"]


def append_trace(category: str, title: str, detail: str, status: str) -> None:
    st.session_state.trace_log.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "category": category,
            "title": title,
            "detail": detail,
            "status": status,
        },
    )


# ===================================================================
# HELPERS (preserved)
# ===================================================================
def build_cinv_options(metadata: pd.DataFrame, forecast: pd.DataFrame) -> list[dict[str, Any]]:
    cinvs = forecast[["ART_CINV"]].dropna().drop_duplicates()
    names = metadata[["ART_CINV", "ART_DESC"]].drop_duplicates(subset=["ART_CINV"], keep="first")
    merged = cinvs.merge(names, on="ART_CINV", how="left").sort_values("ART_CINV")
    return [
        {
            "cinv": int(row.ART_CINV),
            "label": f"{int(row.ART_CINV)} | {row.ART_DESC if isinstance(row.ART_DESC, str) else 'Unknown article'}",
        }
        for row in merged.itertuples()
    ]


def _compact_preview(text: str, limit: int = DISPLAY_RESULT_PREVIEW_CHARS) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n... [{len(text) - limit} more characters hidden in live view]"


def compress_events_for_display(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compressed: list[dict[str, Any]] = []
    pending_text: dict[str, Any] | None = None
    pending_tool_args: dict[str, Any] | None = None

    def flush_text() -> None:
        nonlocal pending_text
        if pending_text is not None:
            compressed.append(pending_text)
            pending_text = None

    def flush_tool_args() -> None:
        nonlocal pending_tool_args
        if pending_tool_args is not None:
            compressed.append(pending_tool_args)
            pending_tool_args = None

    for event in events:
        event_type = event.get("type", "")

        if event_type == "TEXT_MESSAGE_CONTENT":
            delta = event.get("delta", "") or ""
            message_id = event.get("message_id") or event.get("messageId")
            if (
                pending_text is not None
                and pending_text.get("message_id") == message_id
                and len(pending_text.get("delta", "")) + len(delta) <= DISPLAY_TEXT_BATCH_CHARS
            ):
                pending_text["delta"] += delta
                pending_text["chunk_count"] += 1
            else:
                flush_tool_args()
                flush_text()
                pending_text = {
                    "type": event_type,
                    "message_id": message_id,
                    "delta": delta,
                    "chunk_count": 1,
                }
            continue

        if event_type == "TOOL_CALL_ARGS":
            delta = event.get("delta", "") or ""
            tool_call_id = event.get("tool_call_id") or event.get("toolCallId")
            if (
                pending_tool_args is not None
                and pending_tool_args.get("tool_call_id") == tool_call_id
                and len(pending_tool_args.get("delta", "")) + len(delta) <= DISPLAY_TOOL_ARGS_BATCH_CHARS
            ):
                pending_tool_args["delta"] += delta
                pending_tool_args["chunk_count"] += 1
            else:
                flush_text()
                flush_tool_args()
                pending_tool_args = {
                    "type": event_type,
                    "tool_call_id": tool_call_id,
                    "tool_name": event.get("tool_name", ""),
                    "category": event.get("category", ""),
                    "delta": delta,
                    "chunk_count": 1,
                }
            continue

        flush_text()
        flush_tool_args()

        compacted = dict(event)
        if event_type in {"TOOL_CALL_END", "TOOL_CALL_RESULT"}:
            raw_result = event.get("result") or event.get("content") or ""
            compacted["preview"] = _compact_preview(raw_result)
            compacted["result_size"] = len(str(raw_result))
            if "result" in compacted:
                compacted["result"] = compacted["preview"]
            if "content" in compacted:
                compacted["content"] = compacted["preview"]
        compressed.append(compacted)

    flush_text()
    flush_tool_args()
    return compressed


def build_event_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        event_type = event.get("type", "")
        if event_type == "TEXT_MESSAGE_CONTENT":
            message = _compact_preview(event.get("delta", ""), 140)
        elif event_type == "TOOL_CALL_ARGS":
            chunk_count = event.get("chunk_count", 1)
            suffix = f" ({chunk_count} merged chunks)" if chunk_count > 1 else ""
            message = f"{_compact_preview(event.get('delta', ''), 140)}{suffix}"
        elif event_type in {"TOOL_CALL_END", "TOOL_CALL_RESULT"}:
            size = event.get("result_size")
            size_suffix = f" [{size} chars]" if size else ""
            message = f"{_compact_preview(event.get('preview') or event.get('result') or event.get('content') or '', 140)}{size_suffix}"
        else:
            message = event.get("message") or event.get("error") or event.get("step_name") or ""
        rows.append(
            {
                "index": index,
                "type": event_type,
                "tool_name": event.get("tool_name", ""),
                "category": event.get("category", ""),
                "message": message,
                "run_id": event.get("run_id") or event.get("runId") or "",
            }
        )
    return rows


# ===================================================================
# CHART BUILDERS (updated to SymphonyAI brand colours)
# ===================================================================
def build_main_chart(
    frame: pd.DataFrame,
    flagged_weeks: list[dict[str, Any]],
    weather_weeks: list[dict[str, Any]] | None = None,
    *,
    show_temperature: bool = False,
) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    if frame.empty:
        return figure

    # Build discrete ISO week labels (YYYY-W00)
    dates = pd.to_datetime(frame["MVT_DATE_DT"])
    week_labels = dates.dt.isocalendar().apply(lambda r: f"{r['year']}-W{r['week']:02d}", axis=1)

    figure.add_trace(go.Scatter(
        x=week_labels, y=frame["FORECAST"],
        mode="lines+markers", name="Forecast",
        line=dict(color=CHART_COLORS["blue"], width=2),
        marker=dict(size=5),
    ), secondary_y=False)
    figure.add_trace(go.Scatter(
        x=week_labels, y=frame["ACTUAL_DEMAND"],
        mode="lines+markers", name="Actual Demand",
        line=dict(color=CHART_COLORS["mint"], width=2),
        marker=dict(size=5),
    ), secondary_y=False)
    figure.add_trace(go.Scatter(
        x=week_labels, y=frame["NM1"],
        mode="lines", name="NM1",
        line=dict(color=CHART_COLORS["orange"], width=2),
    ), secondary_y=False)

    promo_weeks = frame[frame.get("PROMO_TYPE").fillna("").astype(str).str.strip() != ""]
    if not promo_weeks.empty:
        promo_dates = pd.to_datetime(promo_weeks["MVT_DATE_DT"])
        promo_labels = promo_dates.dt.isocalendar().apply(lambda r: f"{r['year']}-W{r['week']:02d}", axis=1)
        figure.add_trace(go.Scatter(
            x=promo_labels, y=promo_weeks["ACTUAL_DEMAND"],
            mode="markers", name="Promo Weeks",
            marker=dict(symbol="diamond", size=10, color=CHART_COLORS["cream"]),
        ), secondary_y=False)

    holiday_weeks = frame[frame.get("HOLIDAYS_TYPE").fillna("").astype(str).str.strip() != ""]
    if not holiday_weeks.empty:
        hol_dates = pd.to_datetime(holiday_weeks["MVT_DATE_DT"])
        hol_labels = hol_dates.dt.isocalendar().apply(lambda r: f"{r['year']}-W{r['week']:02d}", axis=1)
        figure.add_trace(go.Scatter(
            x=hol_labels, y=holiday_weeks["ACTUAL_DEMAND"],
            mode="markers", name="Holiday Weeks",
            marker=dict(symbol="star", size=10, color=CHART_COLORS["violet"]),
        ), secondary_y=False)

    # Build a lookup from date -> week label for flagged weeks (outliers)
    date_to_label = dict(zip(dates, week_labels))
    outlier_labels = []
    outlier_demands = []
    outlier_hovers = []
    for item in flagged_weeks:
        if not item.get("mvt_date"):
            continue
        try:
            marker_date = pd.to_datetime(item["mvt_date"])
            label = date_to_label.get(marker_date)
            if label is None:
                iso = marker_date.isocalendar()
                label = f"{iso[0]}-W{iso[1]:02d}"
            figure.add_vline(x=label, line_width=1, line_dash="dot", line_color="red")
            demand = item.get("actual_demand")
            if demand is not None:
                outlier_labels.append(label)
                outlier_demands.append(float(demand))
                outlier_hovers.append(str(item.get("reason", "outlier")))
        except Exception:
            continue

    if outlier_labels:
        figure.add_trace(go.Scatter(
            x=outlier_labels, y=outlier_demands,
            mode="markers", name="Outlier Weeks",
            marker=dict(symbol="circle-open", size=12, color="red", line=dict(width=2, color="red")),
            text=outlier_hovers,
            hovertemplate="%{x}<br>Demand: %{y}<br>%{text}<extra></extra>",
        ), secondary_y=False)

    if show_temperature and weather_weeks:
        weather_frame = pd.DataFrame(weather_weeks).copy()
        if not weather_frame.empty and "avg_temp_c" in weather_frame.columns:
            if "MVT_DATE" in weather_frame.columns:
                weather_frame["MVT_DATE_DT"] = pd.to_datetime(weather_frame["MVT_DATE"], errors="coerce")
                weather_iso = weather_frame["MVT_DATE_DT"].dt.isocalendar()
                weather_frame["plot_week_id"] = weather_iso["year"].astype(str) + "-W" + weather_iso["week"].astype(str).str.zfill(2)
                weather_frame = weather_frame.sort_values("MVT_DATE_DT")
            else:
                weather_frame["plot_week_id"] = weather_frame["week_id"]
            weather_frame = weather_frame.dropna(subset=["avg_temp_c"])
            if not weather_frame.empty:
                figure.add_trace(
                    go.Scatter(
                        x=weather_frame["plot_week_id"],
                        y=weather_frame["avg_temp_c"],
                        mode="lines",
                        name="Weekly Temperature",
                        line=dict(color="#FDB52B", width=2, dash="dot"),
                        hovertemplate="%{x}<br>Temperature: %{y:.1f} C<extra></extra>",
                    ),
                    secondary_y=True,
                )

    # Pivot week vertical line
    if "PIVOT_DATE_DT" in frame.columns:
        pivot_dates = frame["PIVOT_DATE_DT"].dropna().unique()
        for pd_val in pivot_dates:
            pivot_dt = pd.Timestamp(pd_val)
            iso = pivot_dt.isocalendar()
            pivot_label = f"{iso[0]}-W{iso[1]:02d}"
            figure.add_vline(
                x=pivot_label, line_width=2, line_dash="dash",
                line_color="#121212",
            )
            figure.add_annotation(
                x=pivot_label, y=1, yref="paper",
                text="Pivot", showarrow=False,
                font=dict(size=11, color="#121212"),
                yshift=10,
            )

    figure.update_layout(
        template="plotly_white",
        title=dict(text="Forecast vs Actual Demand", font=dict(family="Inter, sans-serif", size=16, color="#121212")),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFBFC",
        font=dict(family="Inter, sans-serif", color="#4B5563"),
        legend=dict(orientation="v", y=1, x=1.02, xanchor="left"),
        xaxis=dict(type="category", title="ISO Week", dtick=2, tickangle=90),
        yaxis=dict(title="Demand Units"),
        margin=dict(l=24, r=140, t=56, b=24),
        height=520,
    )
    figure.update_yaxes(title_text="Weekly Temperature (C)", secondary_y=True, showgrid=False)
    return figure


def build_yoy_chart(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if frame.empty:
        return figure
    x_axis = frame["WEEK_ID"]
    figure.add_trace(go.Scatter(x=x_axis, y=frame["ACTUAL_DEMAND"], mode="lines+markers", name="Actual", line=dict(color=CHART_COLORS["mint"], width=2), marker=dict(size=5)))
    figure.add_trace(go.Scatter(x=x_axis, y=frame["NM1"], mode="lines", name="NM1", line=dict(color=CHART_COLORS["orange"], dash="solid", width=2)))
    figure.add_trace(go.Scatter(x=x_axis, y=frame["NM2"], mode="lines", name="NM2", line=dict(color=CHART_COLORS["sky"], dash="dash", width=2)))
    figure.add_trace(go.Scatter(x=x_axis, y=frame["NM3"], mode="lines", name="NM3", line=dict(color=CHART_COLORS["violet"], dash="dot", width=2)))
    figure.update_layout(
        template="plotly_white",
        title=dict(text="Year-on-Year Demand Baselines", font=dict(family="Inter, sans-serif", size=16, color="#121212")),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFBFC",
        font=dict(family="Inter, sans-serif", color="#4B5563"),
        xaxis=dict(dtick=2, tickangle=90),
        margin=dict(l=24, r=24, t=56, b=24),
        height=520,
    )
    return figure


def build_error_chart(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if frame.empty:
        return figure
    frame = frame.copy()
    if frame["FORECAST_ERROR"].isna().all():
        frame["FORECAST_ERROR"] = frame["ACTUAL_DEMAND"] - frame["FORECAST"]
    # Only show weeks where ACTUAL_DEMAND is not 0
    frame = frame[frame["ACTUAL_DEMAND"].fillna(0) != 0]
    if frame.empty:
        return figure
    figure.add_trace(go.Bar(
        x=frame["WEEK_ID"], y=frame["FORECAST_ERROR"],
        marker_color=[CHART_COLORS["mint"] if value <= 0 else CHART_COLORS["orange"] for value in frame["FORECAST_ERROR"].fillna(0)],
        name="Forecast Error",
    ))
    figure.add_hline(y=0, line_color="#9CA3AF", line_dash="dot")
    figure.update_layout(
        template="plotly_white",
        title=dict(text="Forecast Error by Week", font=dict(family="Inter, sans-serif", size=16, color="#121212")),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFBFC",
        font=dict(family="Inter, sans-serif", color="#4B5563"),
        xaxis=dict(dtick=2, tickangle=90),
        margin=dict(l=24, r=24, t=56, b=24),
        height=520,
    )
    return figure


# ===================================================================
# CSS — SymphonyAI brand
# ===================================================================
def render_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --blue: #0074E8;
            --sky: #1EACFC;
            --mint: #00EAC3;
            --violet: #A933FB;
            --ink: #121212;
            --amber: #FFB600;
            --orange: #FF8500;
            --deep-blue: #004080;
            --powder: #79B1EA;
            --cream: #F8D18B;
            --lavender: #EED6FE;
            --gray-50: #F9FAFB;
            --gray-100: #F3F4F6;
            --gray-200: #E5E7EB;
            --gray-300: #D1D5DB;
            --gray-400: #9CA3AF;
            --gray-500: #6B7280;
            --gray-600: #4B5563;
            --gray-700: #374151;
            --radius-card: 8px;
            --radius-btn: 6px;
            --radius-badge: 4px;
            --radius-input: 6px;
        }

        html, body, [class*="css"] {
            font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: var(--ink);
        }

        /* Hide default sidebar */
        section[data-testid="stSidebar"] { display: none !important; }
        button[data-testid="stSidebarCollapsedControl"] { display: none !important; }

        /* Hide Streamlit header / footer */
        header[data-testid="stHeader"] { display: none !important; }
        footer { display: none !important; }
        #MainMenu { display: none !important; }

        .stApp {
            background: #F7F8FA;
        }

        /* ---- Top navigation bar ---- */
        .top-nav {
            position: fixed;
            top: 0; left: 0; right: 0;
            z-index: 9999;
            height: 56px;
            background: #FFFFFF;
            border-bottom: 1px solid var(--gray-200);
            display: flex;
            align-items: center;
            padding: 0 32px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .top-nav .brand {
            font-weight: 700;
            font-size: 15px;
            color: var(--blue);
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .top-nav .brand .dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--mint);
            display: inline-block;
        }
        .top-nav .nav-right {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 13px;
            color: var(--gray-500);
        }
        .top-nav .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: var(--radius-badge);
            font-size: 12px;
            font-weight: 500;
        }
        .status-connected { background: #ECFDF5; color: #065F46; }
        .status-offline { background: #FEF2F2; color: #991B1B; }
        .status-running { background: #FFF8E6; color: #92400E; }

        /* ---- Tab pill strip ---- */
        .tab-strip {
            display: flex;
            gap: 4px;
            padding: 8px 0 0 0;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--gray-200);
            padding-bottom: 0;
        }
        .tab-pill {
            padding: 8px 20px;
            border-radius: var(--radius-btn) var(--radius-btn) 0 0;
            font-size: 14px;
            font-weight: 500;
            color: var(--gray-500);
            cursor: pointer;
            border: none;
            background: transparent;
            transition: all 0.15s ease;
            border-bottom: 2px solid transparent;
            font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .tab-pill:hover { color: var(--blue); background: var(--gray-50); }
        .tab-pill.active {
            color: var(--blue);
            border-bottom: 2px solid var(--blue);
            background: #FFFFFF;
            font-weight: 600;
        }

        /* ---- Cards ---- */
        .card {
            background: #FFFFFF;
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-card);
            padding: 24px;
            margin-bottom: 16px;
        }
        .card-compact {
            background: #FFFFFF;
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-card);
            padding: 16px;
            margin-bottom: 12px;
        }
        .card-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--ink);
            margin-bottom: 8px;
        }
        .card-subtitle {
            font-size: 13px;
            color: var(--gray-500);
            margin-bottom: 16px;
        }

        /* ---- Step stepper ---- */
        .stepper {
            display: flex;
            align-items: flex-start;
            gap: 0;
            margin: 24px 0 32px 0;
        }
        .step-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            flex: 1;
            position: relative;
        }
        .step-circle {
            width: 36px; height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 15px;
            font-weight: 700;
            z-index: 2;
        }
        .step-circle-waiting {
            border: 2.5px solid #E4E8EF;
            background: #FFFFFF;
            color: #9CA3AF;
        }
        .step-circle-in_progress {
            border: 2.5px solid var(--amber);
            background: #FFF8E6;
            color: var(--amber);
            animation: pulse-step 1.5s infinite;
        }
        .step-circle-completed {
            background: var(--mint);
            border: none;
            color: #FFFFFF;
        }
        .step-circle-skipped {
            background: var(--orange);
            border: none;
            color: #FFFFFF;
        }
        .step-circle-error {
            background: var(--orange);
            border: none;
            color: #FFFFFF;
        }
        .step-label {
            font-size: 13px;
            margin-top: 8px;
            text-align: center;
            max-width: 110px;
            font-weight: 500;
        }
        .step-label-waiting, .step-label-completed { color: var(--gray-500); }
        .step-label-in_progress { color: var(--ink); font-weight: 600; }
        .step-label-skipped { color: var(--orange); font-weight: 600; }
        .step-label-error { color: var(--orange); font-weight: 600; }
        .step-connector {
            position: absolute;
            top: 18px;
            left: calc(50% + 18px);
            right: calc(-50% + 18px);
            height: 3px;
            background: #E4E8EF;
            z-index: 1;
        }
        .step-connector-active {
            background: linear-gradient(90deg, var(--blue), var(--sky));
        }

        @keyframes pulse-step {
            0%, 100% { box-shadow: 0 0 0 0 rgba(255, 182, 0, 0.3); }
            50% { box-shadow: 0 0 0 8px rgba(255, 182, 0, 0); }
        }

        /* ---- Tool call cards (DevUI-style) ---- */
        .tool-card {
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-card);
            margin-bottom: 8px;
            overflow: hidden;
        }
        .tool-card-header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
        }
        .tool-card-header-call {
            background: #EFF6FF;
            border-left: 3px solid var(--blue);
            color: #1E40AF;
        }
        .tool-card-header-result {
            background: #ECFDF5;
            border-left: 3px solid var(--mint);
            color: #065F46;
        }
        .tool-card-header-error {
            background: #FEF2F2;
            border-left: 3px solid var(--orange);
            color: #991B1B;
        }
        .tool-card-icon {
            font-size: 14px;
            flex-shrink: 0;
        }
        .tool-card-name {
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 13px;
        }
        .tool-card-toggle {
            margin-left: auto;
            font-size: 12px;
            color: var(--gray-400);
        }
        .tool-card-body {
            padding: 12px 14px;
            background: #FAFBFC;
            border-top: 1px solid var(--gray-200);
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 12px;
            white-space: pre-wrap;
            word-break: break-word;
            color: var(--gray-700);
            max-height: 300px;
            overflow-y: auto;
        }
        .tool-card-duration {
            font-size: 11px;
            color: var(--gray-400);
            margin-left: 8px;
        }

        /* ---- Streaming text card ---- */
        .stream-card {
            background: #FFFFFF;
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-card);
            padding: 16px;
            margin-top: 12px;
        }
        .stream-header {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .stream-header-active { color: var(--violet); }
        .stream-header-done { color: var(--mint); }
        .streaming-cursor {
            display: inline-block;
            width: 2px;
            height: 14px;
            background: var(--violet);
            animation: blink-cursor 0.8s infinite;
            margin-left: 2px;
            vertical-align: middle;
        }
        @keyframes blink-cursor {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }
        .stream-body {
            font-size: 13px;
            line-height: 1.6;
            color: var(--gray-700);
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* ---- Banners ---- */
        .banner-success {
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
            border-radius: var(--radius-card);
            padding: 16px 20px;
            color: #065F46;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 16px;
        }
        .banner-error {
            background: #FEF2F2;
            border: 1px solid #FECACA;
            border-radius: var(--radius-card);
            padding: 16px 20px;
            color: #991B1B;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 16px;
        }
        .banner-info {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: var(--radius-card);
            padding: 16px 20px;
            color: #1E40AF;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 16px;
            border-radius: var(--radius-card);
        }

        /* ---- Metric pills ---- */
        .metric-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--gray-50);
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-badge);
            font-size: 13px;
            color: var(--gray-600);
        }
        .metric-pill strong {
            color: var(--ink);
        }

        /* ---- Endpoint reference cards ---- */
        .endpoint-card {
            display: flex;
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-card);
            overflow: hidden;
            margin-bottom: 12px;
            background: #FFFFFF;
        }
        .endpoint-strip {
            width: 4px;
            flex-shrink: 0;
        }
        .endpoint-strip-post { background: var(--blue); }
        .endpoint-strip-get { background: var(--mint); }
        .endpoint-body {
            padding: 16px;
            flex: 1;
        }
        .endpoint-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        .method-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: var(--radius-badge);
            font-size: 11px;
            font-weight: 700;
            font-family: 'SFMono-Regular', Consolas, monospace;
        }
        .method-badge-post { background: #EFF6FF; color: var(--blue); }
        .method-badge-get { background: #ECFDF5; color: #065F46; }
        .endpoint-path {
            font-family: 'SFMono-Regular', Consolas, monospace;
            font-size: 14px;
            font-weight: 500;
            color: var(--ink);
        }
        .endpoint-desc {
            font-size: 13px;
            color: var(--gray-600);
            line-height: 1.5;
        }

        /* ---- Log table ---- */
        .log-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .log-table th {
            text-align: left;
            padding: 8px 10px;
            background: var(--gray-50);
            color: var(--gray-500);
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            border-bottom: 2px solid var(--gray-200);
        }
        .log-table td {
            padding: 8px 10px;
            border-bottom: 1px solid var(--gray-100);
            color: var(--gray-700);
            vertical-align: top;
        }
        .log-table tr:hover td {
            background: var(--gray-50);
        }
        .log-table .type-badge {
            display: inline-block;
            padding: 1px 6px;
            border-radius: var(--radius-badge);
            font-size: 10px;
            font-weight: 600;
            font-family: 'SFMono-Regular', Consolas, monospace;
        }
        .type-step { background: #EFF6FF; color: #1E40AF; }
        .type-tool { background: #F5F3FF; color: #5B21B6; }
        .type-text { background: #ECFDF5; color: #065F46; }
        .type-system { background: var(--gray-100); color: var(--gray-600); }
        .type-error { background: #FEF2F2; color: #991B1B; }

        /* ---- Report ---- */
        .report-section {
            border-left: 3px solid var(--blue);
            padding: 16px 20px;
            margin-bottom: 16px;
            background: #FFFFFF;
            border-radius: 0 var(--radius-card) var(--radius-card) 0;
        }
        .report-section h3 {
            color: var(--blue);
            font-size: 16px;
            margin-bottom: 8px;
        }

        /* ---- Empty states ---- */
        .empty-state {
            text-align: center;
            padding: 64px 24px;
            color: var(--gray-400);
        }
        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        .empty-state-text {
            font-size: 15px;
            color: var(--gray-500);
        }

        /* ---- Compact running bar ---- */
        .running-bar {
            background: #FFF8E6;
            border: 1px solid #FDE68A;
            border-radius: var(--radius-card);
            padding: 12px 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
            font-size: 14px;
            font-weight: 500;
            color: #92400E;
        }
        .running-bar .spinner {
            width: 16px; height: 16px;
            border: 2px solid #FDE68A;
            border-top-color: var(--amber);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* ---- Prompt modal card ---- */
        .prompt-card {
            background: #FFFFFF;
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-card);
            padding: 24px;
            margin-bottom: 16px;
            max-height: 500px;
            overflow-y: auto;
        }
        .prompt-card pre {
            background: var(--gray-50);
            border: 1px solid var(--gray-200);
            border-radius: var(--radius-btn);
            padding: 16px;
            font-size: 12px;
            line-height: 1.6;
            color: var(--gray-700);
            white-space: pre-wrap;
            word-break: break-word;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }

        /* ---- Fix Streamlit widget spacing ---- */
        .stButton > button {
            border-radius: var(--radius-btn) !important;
            font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-weight: 500 !important;
        }
        .stSelectbox > div > div {
            border-radius: var(--radius-input) !important;
            border: 1.5px solid var(--gray-300) !important;
            background: #FFFFFF !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        }
        .stSelectbox > div > div:focus-within {
            border-color: var(--blue) !important;
            box-shadow: 0 0 0 2px rgba(0,116,232,0.15) !important;
        }
        .stTextInput > div > div > input {
            border-radius: var(--radius-input) !important;
            border: 1.5px solid var(--gray-300) !important;
            background: #FFFFFF !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: var(--blue) !important;
            box-shadow: 0 0 0 2px rgba(0,116,232,0.15) !important;
        }
        .stTextArea textarea {
            background: #FFFFFF !important;
            border: 1.5px solid var(--gray-300) !important;
            border-radius: var(--radius-input) !important;
        }
        .stCheckbox { font-size: 14px; }
        /* Vertically align weather toggle with selectbox */
        .stCheckbox { padding-top: 28px; }

        /* ---- Content spacer for fixed nav ---- */
        .nav-spacer { height: 18px; }

        /* ---- Radio tab navigation ---- */
        .st-key-tab_radio {
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
        }
        .st-key-tab_radio div[role="radiogroup"] {
            gap: 0 !important;
            border-bottom: 1px solid var(--gray-200);
            padding-bottom: 0;
            margin-bottom: 16px;
        }
        .st-key-tab_radio div[role="radiogroup"] > label {
            padding: 12px 28px !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            cursor: pointer !important;
            border-bottom: 3px solid transparent;
            border-radius: var(--radius-btn) var(--radius-btn) 0 0;
            transition: all 0.15s ease;
            letter-spacing: 0.01em;
        }
        .st-key-tab_radio div[role="radiogroup"] > label:hover {
            color: var(--blue) !important;
            background: var(--gray-50) !important;
        }
        /* Hide radio circles */
        .st-key-tab_radio div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===================================================================
# TOP NAVIGATION BAR
# ===================================================================
def render_top_nav() -> None:
    health = fetch_health()
    if st.session_state.analysis_running:
        badge_class = "status-running"
        badge_text = "Running"
        badge_dot = "&#9679;"
    elif health is not None:
        badge_class = "status-connected"
        badge_text = "Connected"
        badge_dot = "&#9679;"
    else:
        badge_class = "status-offline"
        badge_text = "Offline"
        badge_dot = "&#9679;"

    cinv = st.session_state.selected_cinv
    article_name = st.session_state.agent_state.get("article_name", "")
    context_text = ""
    if cinv:
        context_text = f"CINV {cinv}"
        if article_name:
            context_text += f" &middot; {html.escape(str(article_name))}"

    st.markdown(
        f'<div class="top-nav">'
        f'<div class="brand"><span class="dot"></span> Forecast Analysis Agent</div>'
        f'<div class="nav-right"><span style="color:#9CA3AF;">{context_text}</span>'
        f'<span class="status-badge {badge_class}">{badge_dot} {badge_text}</span></div>'
        f'</div>'
        f'<div class="nav-spacer"></div>',
        unsafe_allow_html=True,
    )


# ===================================================================
# TAB STRIP
# ===================================================================
def _on_tab_change() -> None:
    selected = st.session_state.get("tab_radio", "Chat")
    st.session_state["active_tab"] = selected
    st.query_params["tab"] = selected.lower()


def render_tab_strip() -> None:
    active = st.session_state["active_tab"]
    st.radio(
        "Navigation",
        TAB_NAMES,
        index=TAB_NAMES.index(active),
        horizontal=True,
        label_visibility="collapsed",
        key="tab_radio",
        on_change=_on_tab_change,
    )


# ===================================================================
# STEP STEPPER (horizontal)
# ===================================================================
def render_step_stepper() -> None:
    parts = ['<div class="stepper">']
    step_keys = list(STEP_LABELS.keys())
    for i, key in enumerate(step_keys):
        status = st.session_state.step_status.get(key, "waiting")
        if status == "completed":
            circle_content = "&#10003;"
        elif status == "skipped":
            circle_content = "&#10005;"
        elif status == "error":
            circle_content = "&#10005;"
        else:
            circle_content = key
        connector = ""
        if i < len(step_keys) - 1:
            next_status = st.session_state.step_status.get(step_keys[i + 1], "waiting")
            conn_cls = "step-connector-active" if status in ("completed", "skipped") and next_status != "waiting" else ""
            connector = f'<div class="step-connector {conn_cls}"></div>'
        parts.append(
            f'<div class="step-item">{connector}'
            f'<div class="step-circle step-circle-{status}">{circle_content}</div>'
            f'<div class="step-label step-label-{status}">{html.escape(STEP_LABELS[key])}</div>'
            f'</div>'
        )
    parts.append('</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


# ===================================================================
# TAB 1: CHAT
# ===================================================================
def render_chat_tab(options: list[dict[str, Any]], selected_default_index: int) -> None:
    # --- If analysis is running, show compact bar instead of full input ---
    if st.session_state.analysis_running:
        cinv = st.session_state.selected_cinv or "?"
        st.markdown(
            f'<div class="running-bar"><div class="spinner"></div> Analysing CINV {html.escape(str(cinv))}...</div>',
            unsafe_allow_html=True,
        )
        bcol1, bcol2 = st.columns([1, 6])
        with bcol1:
            if st.button("Reset", key="reset_running"):
                reset_analysis_state()
                st.rerun()
    else:
        # --- Input card ---
        st.markdown(
            '<div class="card-title">Run Forecast Analysis</div>'
            '<div class="card-subtitle">Select an article CINV and configure options before starting the agent.</div>',
            unsafe_allow_html=True,
        )

        col_select, col_weather = st.columns([3, 1])
        with col_select:
            selection = st.selectbox(
                "Article (CINV)",
                options,
                index=selected_default_index,
                format_func=lambda o: o["label"],
                key="cinv_selector",
            )
        with col_weather:
            force_weather = st.checkbox(
                "Weather enrichment",
                value=st.session_state.force_weather,
                help="Force weather analysis even if the agent does not find strong weather-sensitivity evidence.",
                key="force_weather_checkbox",
            )

        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
        with btn_col1:
            if st.button("Run Analysis", type="primary", width="stretch", key="run_btn"):
                start_analysis(selection["cinv"], force_weather)
                st.rerun()
        with btn_col2:
            if st.button("View System Prompt", width="stretch", key="prompt_btn"):
                st.session_state["show_prompt"] = not st.session_state.get("show_prompt", False)
                st.rerun()

    # --- System Prompt modal ---
    if st.session_state.get("show_prompt", False):
        st.markdown(
            f'<div class="prompt-card"><div class="card-title">Agent System Prompt</div>'
            f'<pre>{html.escape(SYSTEM_PROMPT)}</pre></div>',
            unsafe_allow_html=True,
        )

    # --- Step stepper (always visible once a run has started or is running) ---
    has_run = st.session_state.analysis_running or any(s != "waiting" for s in st.session_state.step_status.values())
    if has_run:
        render_step_stepper()

    # --- RUN FINISHED banner ---
    run_finished = any(e.get("type") == "RUN_FINISHED" for e in st.session_state.agent_events)
    run_error = bool(st.session_state.error_message)

    if run_finished and not st.session_state.analysis_running:
        st.markdown(
            '<div class="banner-success">&#10003; Analysis complete. Results are available in the Report and Charts tabs.</div>',
            unsafe_allow_html=True,
        )

    if run_error:
        st.markdown(
            f'<div class="banner-error">&#10005; {html.escape(st.session_state.error_message)}</div>',
            unsafe_allow_html=True,
        )

    # --- Tool call feed (DevUI style) ---
    tool_events = [
        e for e in st.session_state.agent_events
        if e.get("type") in {"TOOL_CALL_START", "TOOL_CALL_END", "TOOL_CALL_RESULT"}
    ]
    if tool_events:
        st.markdown("**Agent Activity**")
        for idx, evt in enumerate(tool_events):
            evt_type = evt.get("type", "")
            tool_name = evt.get("tool_name", "tool")

            if evt_type == "TOOL_CALL_START":
                label = f"\U0001f527 Function Call: {tool_name}"
                body = json.dumps(evt.get("args", {}), ensure_ascii=False, indent=2)
                lang = "json"
            elif evt_type == "TOOL_CALL_END":
                duration = evt.get("duration_ms", 0)
                dur_text = f" ({duration:.0f}ms)" if duration else ""
                label = f"\u2705 Function Result: {tool_name}{dur_text}"
                body = _compact_preview(str(evt.get("result", "")), 800)
                lang = "text"
            else:
                label = f"\U0001f4c4 Result: {tool_name}"
                body = _compact_preview(str(evt.get("result", "")), 800)
                lang = "text"

            with st.expander(label, expanded=False):
                st.code(body, language=lang)

    # --- Streaming text preview ---
    text_started = any(e.get("type") in {"TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_CHUNK"} for e in st.session_state.agent_events)
    text_done = any(e.get("type") == "TEXT_MESSAGE_END" for e in st.session_state.agent_events) or run_finished

    if text_started and st.session_state.report_stream:
        if text_done and not st.session_state.analysis_running:
            st.markdown(
                '<div class="banner banner-success">'
                '&#10003; Analysis complete. Results are available in the <strong>Report</strong> and <strong>Charts</strong> tabs.'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("View Full Report \u2192", key="goto_report"):
                st.session_state["active_tab"] = "Report"
                st.query_params["tab"] = "report"
                st.rerun()
        else:
            header_html = '<div class="stream-header stream-header-active">Generating Report...<span class="streaming-cursor"></span></div>'
            preview_text = html.escape(_compact_preview(st.session_state.report_stream, 2000))
            st.markdown(
                f'<div class="stream-card">{header_html}'
                f'<div class="stream-body">{preview_text}</div></div>',
                unsafe_allow_html=True,
            )

    # --- Empty state ---
    if not has_run and not st.session_state.get("show_prompt", False):
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">&#128640;</div>'
            '<div class="empty-state-text">Select an article and click <strong>Run Analysis</strong> to start the agent.</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ===================================================================
# TAB 2: REPORT
# ===================================================================
def render_report_tab() -> None:
    report_md = st.session_state.report_md
    email_draft = st.session_state.email_draft
    email_subject = st.session_state.email_subject
    current_state = st.session_state.agent_state
    has_content = bool(report_md or st.session_state.report_stream or email_draft)

    if not has_content:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">&#128203;</div>'
            '<div class="empty-state-text">No report yet. Run an analysis from the Chat tab to generate a report.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    left, right = st.columns([3, 1])

    with left:
        # --- Report body ---
        if report_md:
            sections = [s.strip() for s in re.split(r"\n(?=## )", report_md) if s.strip()]
            for section in sections or [report_md]:
                heading_match = re.match(r"##\s+(.+)", section)
                if heading_match:
                    st.markdown(f"### {heading_match.group(1).strip()}")
                    body = section[heading_match.end():].strip()
                else:
                    body = section
                if body:
                    st.markdown(body)
        elif st.session_state.report_stream:
            st.markdown(st.session_state.report_stream)

        # --- Flagged weeks ---
        flagged_weeks = current_state.get("flagged_weeks", [])
        if flagged_weeks:
            st.markdown("#### Flagged Weeks for Model Exclusion")
            st.dataframe(pd.DataFrame(flagged_weeks), use_container_width=True, hide_index=True)

        # --- Email draft (below report) ---
        if email_draft:
            st.markdown("---")
            st.markdown('<div class="card-title">Email Draft</div>', unsafe_allow_html=True)
            if email_subject:
                st.markdown(f"**Subject:** {email_subject}")
            display_body = re.sub(r"^Subject:\s*.+$", "", email_draft, count=1, flags=re.MULTILINE).strip()
            st.text_area("Email body", display_body, height=280, key="email_area", label_visibility="collapsed")

    with right:
        # --- Actions panel (sticky) ---
        st.markdown('<p style="font-weight:600; margin:0 0 8px;">Export</p>', unsafe_allow_html=True)

        cinv_label = st.session_state.selected_cinv or "session"
        report_download = report_md or st.session_state.report_stream
        if report_download:
            st.download_button(
                "\u2193 Download Report (.txt)",
                data=report_download,
                file_name=f"forecast_report_{cinv_label}.txt",
                use_container_width=True,
                key="dl_report",
            )
        if email_draft:
            st.download_button(
                "\u2193 Download Email Draft (.txt)",
                data=email_draft,
                file_name=f"forecast_email_{cinv_label}.txt",
                use_container_width=True,
                key="dl_email",
            )


# ===================================================================
# TAB 3: CHARTS
# ===================================================================
def render_charts_tab(forecast_df: pd.DataFrame) -> None:
    cinv = st.session_state.selected_cinv
    current_state = st.session_state.agent_state

    if cinv is None:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">&#128200;</div>'
            '<div class="empty-state-text">No article selected. Run an analysis from the Chat tab first.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    selected_frame = forecast_df[forecast_df["ART_CINV"] == cinv].sort_values("MVT_DATE_DT")
    if selected_frame.empty:
        st.warning(f"No forecast data found for CINV {cinv}.")
        return

    # Chart A
    weather_sensitivity = current_state.get("weather_sensitivity") or {}
    show_weather_temperature = bool(current_state.get("weather_enrichment_activated")) and bool(
        current_state.get("weather_enrichment_recommended")
    )
    with st.container(border=True):
        st.plotly_chart(
            build_main_chart(
                selected_frame,
                current_state.get("flagged_weeks", []),
                current_state.get("weather_weeks", []),
                show_temperature=show_weather_temperature,
            ),
            use_container_width=True,
            key="chart_main",
        )
        if show_weather_temperature and weather_sensitivity:
            st.caption(
                f"Temperature overlay enabled because weather enrichment ran and the agent classified weather sensitivity as "
                f"{weather_sensitivity.get('classification', 'weather-sensitive')}."
            )

    # Chart B
    with st.container(border=True):
        st.plotly_chart(
            build_yoy_chart(selected_frame),
            use_container_width=True,
            key="chart_yoy",
        )

    # Chart C
    with st.container(border=True):
        st.plotly_chart(
            build_error_chart(selected_frame),
            use_container_width=True,
            key="chart_error",
        )

    # Linked articles table
    cinv = st.session_state.selected_cinv
    if cinv is not None:
        links_df = get_article_links_frame(int(cinv))
        if not links_df.empty:
            meta = load_metadata_frame()[["ART_CINV", "ART_DESC"]]
            links_df = links_df.merge(
                meta, left_on="art_cinv_b", right_on="ART_CINV", how="left",
            ).drop(columns=["ART_CINV"], errors="ignore")
            links_df = links_df.rename(columns={"ART_DESC": "linked_art_desc"})
            with st.container(border=True):
                st.markdown("#### Linked Articles")
                st.dataframe(links_df, use_container_width=True, hide_index=True)


# ===================================================================
# TAB 4: LOGS
# ===================================================================
def render_logs_tab() -> None:
    if not st.session_state.agent_events:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">&#128269;</div>'
            '<div class="empty-state-text">No agent run yet. Run an analysis from the Chat tab.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    display_events = compress_events_for_display(st.session_state.agent_events)
    event_rows = build_event_rows(display_events)

    # --- Metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-pill">Trace Entries <strong>{len(st.session_state.trace_log)}</strong></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-pill">Raw Events <strong>{len(st.session_state.agent_events)}</strong></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-pill">Display Events <strong>{len(display_events)}</strong></div>', unsafe_allow_html=True)
    error_count = sum(1 for e in st.session_state.agent_events if e.get("type") == "RUN_ERROR")
    m4.markdown(f'<div class="metric-pill">Errors <strong>{error_count}</strong></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # --- Category filter ---
    filter_col, search_col, dl_col = st.columns([2, 2, 1])
    with filter_col:
        categories = ["All", "Data", "Weather", "Search", "Analysis"]
        selected_cat = st.selectbox("Category", categories, key="log_category")
    with search_col:
        search_query = st.text_input("Search events...", key="log_search", placeholder="Search events...")
    with dl_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.download_button(
            "Download Log",
            data=json.dumps(st.session_state.agent_events, indent=2, default=str),
            file_name=f"agent_events_{st.session_state.selected_cinv or 'session'}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_log",
        )

    # --- Filter rows ---
    filtered = event_rows
    if selected_cat != "All":
        cat_lower = selected_cat.lower()
        filtered = [r for r in filtered if cat_lower in r.get("category", "").lower() or cat_lower in r.get("tool_name", "").lower()]
    if search_query:
        sq = search_query.lower()
        filtered = [r for r in filtered if sq in r.get("type", "").lower() or sq in r.get("tool_name", "").lower() or sq in r.get("message", "").lower()]

    # --- Build HTML table ---
    def _type_badge(t: str) -> str:
        t_lower = t.lower()
        if "step" in t_lower:
            cls = "type-step"
        elif "tool" in t_lower:
            cls = "type-tool"
        elif "text" in t_lower:
            cls = "type-text"
        elif "error" in t_lower:
            cls = "type-error"
        else:
            cls = "type-system"
        return f'<span class="type-badge {cls}">{html.escape(t)}</span>'

    if not filtered:
        st.info("No events match the current filters.")
        return

    table_html = '<table class="log-table"><thead><tr><th>#</th><th>Type</th><th>Tool</th><th>Category</th><th>Message</th></tr></thead><tbody>'
    expanded_row = st.session_state.get("expanded_log_row")
    for row in filtered:
        idx = row["index"]
        highlight = ' style="background:#EFF6FF;"' if idx == expanded_row else ""
        table_html += f"""<tr{highlight}>
            <td>{idx}</td>
            <td>{_type_badge(row['type'])}</td>
            <td style="font-family:monospace;font-size:12px;">{html.escape(row.get('tool_name',''))}</td>
            <td>{html.escape(row.get('category',''))}</td>
            <td style="max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{html.escape(row.get('message','')[:200])}</td>
        </tr>"""
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    # --- Inspect event ---
    if filtered:
        selected_index = st.selectbox(
            "Inspect event",
            options=[r["index"] for r in filtered],
            format_func=lambda v: f"#{v} | {display_events[v].get('type', 'UNKNOWN')}",
            key="log_inspect",
        )
        st.session_state["expanded_log_row"] = selected_index
        st.code(json.dumps(display_events[selected_index], indent=2, default=str), language="json")


# ===================================================================
# TAB 5: OBSERVABILITY
# ===================================================================
def render_observability_tab() -> None:
    health = fetch_health()

    # --- Health status card ---
    with st.container(border=True):
        hdr_left, hdr_right = st.columns([4, 1])
        with hdr_left:
            st.markdown('<div class="card-title">API Status</div>', unsafe_allow_html=True)
        with hdr_right:
            if st.button("\u27F3 Refresh", key="refresh_health", use_container_width=True):
                fetch_health.clear()
                st.rerun()

    if health is None:
        st.markdown(
            f'<div class="banner-error">&#10005; Agent API unreachable at {html.escape(API_BASE)}</div>',
            unsafe_allow_html=True,
        )
    else:
        status = health.get("status", "unknown")
        warnings = health.get("config_warnings") or []
        is_heroku = health.get("is_heroku", False)

        if status == "ok" and not warnings:
            st.markdown(
                '<div class="banner-success">&#10003; All systems operational</div>',
                unsafe_allow_html=True,
            )
        elif warnings:
            warning_list = ", ".join(warnings)
            st.markdown(
                f'<div class="banner-info">&#9888; Partial degradation — {html.escape(warning_list)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="banner-success">&#10003; All systems operational</div>',
                unsafe_allow_html=True,
            )

        # Metric pills
        pill1, pill2 = st.columns(2)
        with pill1:
            heroku_val = "Yes" if is_heroku else "No"
            st.markdown(f'<div class="metric-pill">Is Heroku <strong>{heroku_val}</strong></div>', unsafe_allow_html=True)
        with pill2:
            warn_val = f"{len(warnings)} warnings" if warnings else "None"
            st.markdown(f'<div class="metric-pill">Config Warnings <strong>{warn_val}</strong></div>', unsafe_allow_html=True)

        # Raw JSON
        with st.expander("View raw health response"):
            st.code(json.dumps(health, indent=2, default=str), language="json")

    # --- Endpoint reference cards ---
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">Endpoint Reference</div>', unsafe_allow_html=True)

    # POST /api/run
    st.markdown(
        '<div class="endpoint-card">'
        '<div class="endpoint-strip endpoint-strip-post"></div>'
        '<div class="endpoint-body">'
        '<div class="endpoint-header"><span class="method-badge method-badge-post">POST</span>'
        '<span class="endpoint-path">/api/run</span></div>'
        '<div class="endpoint-desc">Start a forecast analysis run. Accepts <code>{cinv: int}</code>. Returns AG-UI SSE event stream.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # GET /api/health
    st.markdown(
        '<div class="endpoint-card">'
        '<div class="endpoint-strip endpoint-strip-get"></div>'
        '<div class="endpoint-body">'
        '<div class="endpoint-header"><span class="method-badge method-badge-get">GET</span>'
        '<span class="endpoint-path">/api/health</span></div>'
        '<div class="endpoint-desc">Health check. Returns <code>{status, config_warnings, is_heroku}</code>.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # GET /api/cinvs
    st.markdown(
        '<div class="endpoint-card">'
        '<div class="endpoint-strip endpoint-strip-get"></div>'
        '<div class="endpoint-body">'
        '<div class="endpoint-header"><span class="method-badge method-badge-get">GET</span>'
        '<span class="endpoint-path">/api/cinvs</span></div>'
        '<div class="endpoint-desc">List available CINV IDs from the forecast data CSV.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Timestamp
    st.markdown(
        f'<div style="text-align:center;color:#9CA3AF;font-size:12px;margin-top:24px;">Last checked: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>',
        unsafe_allow_html=True,
    )


# ===================================================================
# MAIN FLOW
# ===================================================================
ensure_state()
render_css()
forecast_df, metadata_df, _links_df = load_data()
options = build_cinv_options(metadata_df, forecast_df)
drain_events()

render_top_nav()
render_tab_strip()

# Resolve selected CINV default index for Chat tab
selected_default_index = 0
if st.session_state.selected_cinv is not None:
    for index, option in enumerate(options):
        if option["cinv"] == st.session_state.selected_cinv:
            selected_default_index = index
            break

# --- Route to active tab ---
active_tab = st.session_state["active_tab"]

if active_tab == "Chat":
    render_chat_tab(options, selected_default_index)
elif active_tab == "Report":
    render_report_tab()
elif active_tab == "Charts":
    render_charts_tab(forecast_df)
elif active_tab == "Logs":
    render_logs_tab()
elif active_tab == "Observability":
    render_observability_tab()

# --- Auto-refresh while streaming ---
if st.session_state.analysis_running:
    time.sleep(0.4)
    st.rerun()