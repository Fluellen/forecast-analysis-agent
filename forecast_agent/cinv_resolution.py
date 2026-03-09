"""Resolve analysis requests into a single article CINV."""

from __future__ import annotations

import json
import re
from textwrap import dedent
from typing import Any

from azure.identity import DefaultAzureCredential

from agent_framework import AgentResponseUpdate
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework.openai import OpenAIResponsesClient

from . import config
from .data_access import list_available_articles

_RESPONSES_CLIENT: AzureOpenAIResponsesClient | OpenAIResponsesClient | None = None

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


def _normalize_openai_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip().strip('"').strip("'")
    if not base_url:
        return ""
    return base_url if base_url.endswith("/") else f"{base_url}/"


def _build_client() -> AzureOpenAIResponsesClient | OpenAIResponsesClient:
    global _RESPONSES_CLIENT
    if _RESPONSES_CLIENT is not None:
        return _RESPONSES_CLIENT

    azure_openai_base_url = _normalize_openai_base_url(config.AZURE_OPENAI_ENDPOINT)
    if azure_openai_base_url and config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_MODEL_ID:
        _RESPONSES_CLIENT = OpenAIResponsesClient(
            base_url=azure_openai_base_url,
            api_key=config.AZURE_OPENAI_API_KEY,
            model_id=config.AZURE_OPENAI_MODEL_ID,
        )
        return _RESPONSES_CLIENT

    if config.OPENAI_BASE_URL and config.OPENAI_API_KEY and config.OPENAI_RESPONSES_MODEL_ID:
        _RESPONSES_CLIENT = OpenAIResponsesClient(
            base_url=_normalize_openai_base_url(config.OPENAI_BASE_URL),
            api_key=config.OPENAI_API_KEY,
            model_id=config.OPENAI_RESPONSES_MODEL_ID,
        )
        return _RESPONSES_CLIENT

    if config.AZURE_AI_PROJECT_ENDPOINT and config.AZURE_OPENAI_MODEL_ID:
        _RESPONSES_CLIENT = AzureOpenAIResponsesClient(
            project_endpoint=config.AZURE_AI_PROJECT_ENDPOINT,
            deployment_name=config.AZURE_OPENAI_MODEL_ID,
            credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
        )
        return _RESPONSES_CLIENT

    if config.OPENAI_API_KEY and config.OPENAI_RESPONSES_MODEL_ID:
        _RESPONSES_CLIENT = OpenAIResponsesClient(
            api_key=config.OPENAI_API_KEY,
            model_id=config.OPENAI_RESPONSES_MODEL_ID,
        )
        return _RESPONSES_CLIENT

    raise RuntimeError(
        "No supported model configuration is available for CINV extraction. Configure Azure AI project variables, "
        "Azure OpenAI API key variables, or direct OpenAI responses variables."
    )


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

    client = _build_client()
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