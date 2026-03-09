"""Resolve analysis requests into a single article CINV."""

from __future__ import annotations

import json
import re
from textwrap import dedent
from typing import Any

from agent_framework import AgentResponseUpdate

from . import config
from .data_access import list_available_articles

_RESPONSES_CLIENT = None


def _get_responses_client():
    global _RESPONSES_CLIENT
    if _RESPONSES_CLIENT is None:
        _RESPONSES_CLIENT = config.build_responses_client()
    return _RESPONSES_CLIENT


_RESOLUTION_SYSTEM_PROMPT = dedent(
    """
    You extract a single DFAI article CINV from planner requests.

    Rules:
    - Return JSON only.
    - Use this schema exactly:
      {"cinv": <int or null>, "confidence": "high"|"medium"|"low", "article_name": <string or null>, "reason": <string>}
    - If the request explicitly mentions a CINV, return that exact integer even if it is not in the catalog.
    - If the request does not mention a CINV explicitly, map the request to the best matching article from the provided catalog.
    - If the request is ambiguous or no reasonable match exists, return {"cinv": null, ...}.
    - Do not return markdown, commentary, or code fences.
    """
).strip()


def _extract_direct_cinv(text: str) -> int | None:
    stripped = (text or "").strip()
    if re.fullmatch(r"\d+", stripped):
        return int(stripped)
    return None


def _catalog_context() -> str:
    articles = list_available_articles()
    if not articles:
        return "No article catalog is available."
    return "\n".join(f"- {item['cinv']}: {item['name']}" for item in articles[:200])


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    candidate = match.group(0) if match else cleaned
    return json.loads(candidate)


async def resolve_analysis_request(*, cinv: Any = None, input_text: str | None = None) -> dict[str, Any]:
    """Resolve either a direct CINV or a free-form request into a single article CINV."""
    if cinv is not None:
        return {
            "cinv": int(cinv),
            "mode": "direct_cinv",
            "confidence": "high",
            "article_name": None,
            "reason": "The request already provided a direct CINV.",
            "input_text": str(cinv),
        }

    raw_text = (input_text or "").strip()
    if not raw_text:
        raise ValueError("Either cinv or input_text must be provided.")

    direct_cinv = _extract_direct_cinv(raw_text)
    if direct_cinv is not None:
        return {
            "cinv": direct_cinv,
            "mode": "direct_cinv",
            "confidence": "high",
            "article_name": None,
            "reason": "The request contained only a direct numeric CINV.",
            "input_text": raw_text,
        }

    client = _get_responses_client()
    agent = client.as_agent(
        name="CinvResolutionAgent",
        instructions=_RESOLUTION_SYSTEM_PROMPT,
        tools=[],
    )

    prompt = dedent(
        f"""
        Resolve the following planner request into one article CINV.

        Known article catalog:
        {_catalog_context()}

        Planner request:
        {raw_text}
        """
    ).strip()

    chunks: list[str] = []
    stream = agent.run(prompt, stream=True)
    async for update in stream:
        if not isinstance(update, AgentResponseUpdate):
            continue
        for content in update.contents:
            if getattr(content, "type", None) == "text":
                chunk = getattr(content, "text", "") or ""
                if chunk:
                    chunks.append(chunk)

    final_response = await stream.get_final_response()
    final_text = getattr(final_response, "text", None) or "".join(chunks)
    payload = _extract_json_object(final_text)

    resolved_cinv = payload.get("cinv")
    if resolved_cinv is None:
        reason = str(payload.get("reason") or "The model could not identify a single CINV from the provided text.")
        raise ValueError(f"Could not resolve a single CINV from the provided text. {reason}")

    return {
        "cinv": int(resolved_cinv),
        "mode": "llm_extracted",
        "confidence": str(payload.get("confidence") or "medium"),
        "article_name": payload.get("article_name"),
        "reason": str(payload.get("reason") or "Resolved from free-form text."),
        "input_text": raw_text,
        "raw_model_output": final_text,
    }