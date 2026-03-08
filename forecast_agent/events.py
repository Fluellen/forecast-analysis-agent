"""AG-UI event helpers and SSE serialization utilities."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ag_ui.encoder import EventEncoder

try:
    from ag_ui.core import (
        RunErrorEvent,
        RunFinishedEvent,
        RunStartedEvent,
        StateDeltaEvent,
        StateSnapshotEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        ToolCallStartEvent,
    )
except ImportError:  # pragma: no cover
    RunErrorEvent = RunFinishedEvent = RunStartedEvent = None
    StateDeltaEvent = StateSnapshotEvent = None
    TextMessageContentEvent = TextMessageEndEvent = TextMessageStartEvent = None
    ToolCallArgsEvent = ToolCallEndEvent = ToolCallResultEvent = ToolCallStartEvent = None


class EventType(str, Enum):
    """Event types exposed by the server stream."""

    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"


@dataclass
class _FallbackEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "type": self.type,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
        }
        payload.update(self.payload)
        return payload


@dataclass
class AGUIEvent:
    """Wrapper that serializes SDK events or fallback events to SSE."""

    event: Any
    extra: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        if isinstance(self.event, _FallbackEvent):
            payload = self.event.to_payload()
            payload.update(self.extra)
            return payload

        encoded = EventEncoder().encode(self.event)
        payload = json.loads(encoded.removeprefix("data:").strip())
        payload.update(self.extra)
        return payload

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.to_payload())}\n\n"


def run_started(run_id: str, cinv: int, *, thread_id: str | None = None) -> AGUIEvent:
    thread_id = thread_id or f"cinv-{cinv}"
    if RunStartedEvent is not None:
        return AGUIEvent(
            RunStartedEvent(
                threadId=thread_id,
                runId=run_id,
            ),
            {"run_id": run_id, "cinv": cinv},
        )
    return AGUIEvent(_FallbackEvent(EventType.RUN_STARTED.value, {"run_id": run_id, "cinv": cinv}))


def run_finished(run_id: str, cinv: int, *, thread_id: str | None = None, result: Any | None = None) -> AGUIEvent:
    thread_id = thread_id or f"cinv-{cinv}"
    if RunFinishedEvent is not None:
        return AGUIEvent(
            RunFinishedEvent(
                threadId=thread_id,
                runId=run_id,
                result=result,
            ),
            {"run_id": run_id, "cinv": cinv},
        )
    return AGUIEvent(_FallbackEvent(EventType.RUN_FINISHED.value, {"run_id": run_id, "cinv": cinv}))


def run_error(run_id: str, error: str, *, code: str | None = None) -> AGUIEvent:
    if RunErrorEvent is not None:
        return AGUIEvent(
            RunErrorEvent(
                message=error,
                code=code,
            ),
            {"run_id": run_id, "error": error},
        )
    return AGUIEvent(_FallbackEvent(EventType.RUN_ERROR.value, {"run_id": run_id, "error": error}))


def text_message_start(message_id: str) -> AGUIEvent:
    if TextMessageStartEvent is not None:
        return AGUIEvent(TextMessageStartEvent(messageId=message_id, role="assistant"), {"message_id": message_id})
    return AGUIEvent(_FallbackEvent(EventType.TEXT_MESSAGE_START.value, {"message_id": message_id, "role": "assistant"}))


def text_message_chunk(message_id: str, chunk: str) -> AGUIEvent:
    if TextMessageContentEvent is not None:
        return AGUIEvent(TextMessageContentEvent(messageId=message_id, delta=chunk), {"message_id": message_id})
    return AGUIEvent(_FallbackEvent(EventType.TEXT_MESSAGE_CONTENT.value, {"message_id": message_id, "delta": chunk}))


def text_message_end(message_id: str) -> AGUIEvent:
    if TextMessageEndEvent is not None:
        return AGUIEvent(TextMessageEndEvent(messageId=message_id), {"message_id": message_id})
    return AGUIEvent(_FallbackEvent(EventType.TEXT_MESSAGE_END.value, {"message_id": message_id}))


def tool_call_start(tool_call_id: str, tool_name: str, args: dict[str, Any], *, parent_message_id: str | None = None) -> AGUIEvent:
    payload = {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "args": args,
        "category": _tool_category(tool_name),
    }
    if ToolCallStartEvent is not None:
        return AGUIEvent(
            ToolCallStartEvent(
                toolCallId=tool_call_id,
                toolCallName=tool_name,
                parentMessageId=parent_message_id,
            ),
            payload,
        )
    return AGUIEvent(_FallbackEvent(EventType.TOOL_CALL_START.value, payload))


def tool_call_args(tool_call_id: str, delta: str, tool_name: str) -> AGUIEvent:
    payload = {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "delta": delta,
        "category": _tool_category(tool_name),
    }
    if ToolCallArgsEvent is not None:
        return AGUIEvent(ToolCallArgsEvent(toolCallId=tool_call_id, delta=delta), payload)
    return AGUIEvent(_FallbackEvent(EventType.TOOL_CALL_ARGS.value, payload))


def tool_call_end(tool_call_id: str, tool_name: str, duration_ms: float, result: str) -> AGUIEvent:
    payload = {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "duration_ms": duration_ms,
        "result": result[:600],
        "category": _tool_category(tool_name),
    }
    if ToolCallEndEvent is not None:
        return AGUIEvent(ToolCallEndEvent(toolCallId=tool_call_id), payload)
    return AGUIEvent(_FallbackEvent(EventType.TOOL_CALL_END.value, payload))


def tool_call_result(tool_call_id: str, tool_name: str, result: str, *, message_id: str | None = None) -> AGUIEvent:
    message_id = message_id or str(uuid.uuid4())
    payload = {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "content": result,
        "result": result,
        "category": _tool_category(tool_name),
    }
    if ToolCallResultEvent is not None:
        return AGUIEvent(
            ToolCallResultEvent(
                messageId=message_id,
                toolCallId=tool_call_id,
                content=result,
                role="tool",
            ),
            payload,
        )
    return AGUIEvent(_FallbackEvent(EventType.TOOL_CALL_RESULT.value, payload))


def state_snapshot(state: dict[str, Any]) -> AGUIEvent:
    payload = {"state": state}
    if StateSnapshotEvent is not None:
        return AGUIEvent(StateSnapshotEvent(snapshot=state), payload)
    return AGUIEvent(_FallbackEvent(EventType.STATE_SNAPSHOT.value, payload))


def state_delta(delta: dict[str, Any]) -> AGUIEvent:
    payload = {
        "delta": delta,
        "delta_dict": delta,
    }
    if StateDeltaEvent is not None:
        patches = [{"op": "replace", "path": f"/{key}", "value": value} for key, value in delta.items()]
        return AGUIEvent(StateDeltaEvent(delta=patches), payload)
    return AGUIEvent(_FallbackEvent(EventType.STATE_DELTA.value, payload))


def step_started(step_name: str, step_number: int) -> AGUIEvent:
    return AGUIEvent(
        _FallbackEvent(
            EventType.STEP_STARTED.value,
            {
                "step_name": step_name,
                "step_number": step_number,
            },
        )
    )


def step_finished(step_name: str, step_number: int, summary: str = "") -> AGUIEvent:
    return AGUIEvent(
        _FallbackEvent(
            EventType.STEP_FINISHED.value,
            {
                "step_name": step_name,
                "step_number": step_number,
                "summary": summary,
            },
        )
    )


def _tool_category(tool_name: str) -> str:
    categories = {
        "get_article_metadata": "data",
        "get_forecast_data": "data",
        "get_article_links": "data",
        "compute_forecast_health": "data",
        "detect_outlier_weeks": "data",
        "get_weather_for_period": "weather",
        "search_article_characteristics": "search",
        "search_holiday_demand_correlation": "search",
        "correlate_weather_with_demand": "analysis",
        "analyse_year_on_year_trend": "analysis",
        "get_article_links_demand": "analysis",
    }
    return categories.get(tool_name, "data")