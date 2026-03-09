"""Project configuration helpers for the forecast analysis agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

FORECAST_DATA_PATH = DATA_DIR / "forecast_data.csv"
ARTICLE_METADATA_PATH = DATA_DIR / "article_metadata.csv"
LINKS_PATH = DATA_DIR / "links.csv"

load_dotenv(ROOT_DIR / ".env", override=False)

AZURE_AI_PROJECT_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT", "")
AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_RESPONSES_MODEL_ID = os.getenv("OPENAI_RESPONSES_MODEL_ID", "") or os.getenv("OPENAI_MODEL", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

AZURE_OPENAI_MODEL_ID = AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME or AZURE_OPENAI_DEPLOYMENT_NAME or AZURE_OPENAI_MODEL

API_PORT = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
MAX_RUN_TIMEOUT_SECONDS = int(os.getenv("MAX_RUN_TIMEOUT_SECONDS", "600"))
IS_HEROKU = bool(os.getenv("DYNO"))

DUBLIN_LATITUDE = 53.3498
DUBLIN_LONGITUDE = -6.2603
DUBLIN_TIMEZONE = "Europe/London"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def normalize_openai_base_url(base_url: str) -> str:
    """Normalize an OpenAI-compatible base URL to end with a single trailing slash."""
    base_url = (base_url or "").strip().strip('"').strip("'")
    if not base_url:
        return ""
    return base_url if base_url.endswith("/") else f"{base_url}/"


def build_responses_client():
    """Build a responses-capable chat client from the active environment configuration.

    Priority order matches the documented configuration precedence:
    Azure OpenAI endpoint+key > OpenAI-compatible base URL > Azure AI project > OpenAI API key.
    Returns an ``AzureOpenAIResponsesClient`` or ``OpenAIResponsesClient`` instance.
    """
    from azure.identity import DefaultAzureCredential

    from agent_framework.azure import AzureOpenAIResponsesClient
    from agent_framework.openai import OpenAIResponsesClient

    azure_openai_base_url = normalize_openai_base_url(AZURE_OPENAI_ENDPOINT)
    if azure_openai_base_url and AZURE_OPENAI_API_KEY and AZURE_OPENAI_MODEL_ID:
        return OpenAIResponsesClient(
            base_url=azure_openai_base_url,
            api_key=AZURE_OPENAI_API_KEY,
            model_id=AZURE_OPENAI_MODEL_ID,
        )

    if OPENAI_BASE_URL and OPENAI_API_KEY and OPENAI_RESPONSES_MODEL_ID:
        return OpenAIResponsesClient(
            base_url=normalize_openai_base_url(OPENAI_BASE_URL),
            api_key=OPENAI_API_KEY,
            model_id=OPENAI_RESPONSES_MODEL_ID,
        )

    if AZURE_AI_PROJECT_ENDPOINT and AZURE_OPENAI_MODEL_ID:
        return AzureOpenAIResponsesClient(
            project_endpoint=AZURE_AI_PROJECT_ENDPOINT,
            deployment_name=AZURE_OPENAI_MODEL_ID,
            credential=DefaultAzureCredential(exclude_interactive_browser_credential=False),
        )

    if OPENAI_API_KEY and OPENAI_RESPONSES_MODEL_ID:
        return OpenAIResponsesClient(
            api_key=OPENAI_API_KEY,
            model_id=OPENAI_RESPONSES_MODEL_ID,
        )

    raise RuntimeError(
        "No supported model configuration is available. Configure Azure AI project variables, Azure OpenAI API key "
        "variables, or direct OpenAI responses variables."
    )


def ensure_output_dir() -> Path:
    """Ensure the output directory exists and return it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def validate() -> list[str]:
    """Return non-fatal configuration warnings for the API health endpoint."""
    warnings: list[str] = []

    for path in [FORECAST_DATA_PATH, ARTICLE_METADATA_PATH, LINKS_PATH]:
        if not path.exists():
            warnings.append(f"Missing data file: {path.name}")

    has_azure_project = bool(AZURE_AI_PROJECT_ENDPOINT and AZURE_OPENAI_MODEL_ID)
    has_azure_v1 = bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_MODEL_ID)
    has_openai = bool((OPENAI_BASE_URL or OPENAI_API_KEY) and OPENAI_RESPONSES_MODEL_ID)
    if not has_azure_project and not has_azure_v1 and not has_openai:
        warnings.append(
            "No supported model configuration found. Set Azure AI project variables or OpenAI responses variables."
        )

    if not TAVILY_API_KEY:
        warnings.append("TAVILY_API_KEY is not configured.")

    return warnings