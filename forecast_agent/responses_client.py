"""Shared Responses API client helpers."""

from __future__ import annotations

from typing import Any

from azure.identity import DefaultAzureCredential

from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework.openai import OpenAIResponsesClient

from . import config

_RESPONSES_CLIENT: AzureOpenAIResponsesClient | OpenAIResponsesClient | None = None
_RESPONSES_CREDENTIAL: DefaultAzureCredential | None = None


def _normalize_openai_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip().strip('"').strip("'")
    if not base_url:
        return ""
    return base_url if base_url.endswith("/") else f"{base_url}/"


def get_responses_client() -> AzureOpenAIResponsesClient | OpenAIResponsesClient:
    global _RESPONSES_CLIENT, _RESPONSES_CREDENTIAL
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
        if _RESPONSES_CREDENTIAL is None:
            _RESPONSES_CREDENTIAL = DefaultAzureCredential(exclude_interactive_browser_credential=False)
        _RESPONSES_CLIENT = AzureOpenAIResponsesClient(
            project_endpoint=config.AZURE_AI_PROJECT_ENDPOINT,
            deployment_name=config.AZURE_OPENAI_MODEL_ID,
            credential=_RESPONSES_CREDENTIAL,
        )
        return _RESPONSES_CLIENT

    if config.OPENAI_API_KEY and config.OPENAI_RESPONSES_MODEL_ID:
        _RESPONSES_CLIENT = OpenAIResponsesClient(
            api_key=config.OPENAI_API_KEY,
            model_id=config.OPENAI_RESPONSES_MODEL_ID,
        )
        return _RESPONSES_CLIENT

    raise RuntimeError(
        "No supported model configuration is available. Configure Azure AI project variables, Azure OpenAI API key "
        "variables, or direct OpenAI responses variables."
    )


def get_responses_cleanup_hooks() -> list[callable]:
    hooks: list[callable] = []
    if _RESPONSES_CREDENTIAL is not None:
        hooks.append(_RESPONSES_CREDENTIAL.close)
    return hooks
