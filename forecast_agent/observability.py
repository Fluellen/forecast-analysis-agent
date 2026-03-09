"""Observability bootstrap helpers for Agent Framework and OTLP backends."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from . import config

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CONFIGURED = False
_STATUS: dict[str, Any] = {
    "configured": False,
    "langfuse_configured": False,
    "langfuse_sdk_auth": False,
    "otel_traces_endpoint": "",
    "otel_service_name": "",
    "error": "",
}
_LANGFUSE_CLIENT: Any | None = None


def _has_otel_configuration() -> bool:
    return any(
        os.getenv(name)
        for name in (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
            "ENABLE_CONSOLE_EXPORTERS",
            "ENABLE_INSTRUMENTATION",
        )
    )


def _apply_langfuse_otel_defaults() -> bool:
    if not config.langfuse_configured():
        return False

    os.environ.setdefault("LANGFUSE_HOST", config.LANGFUSE_BASE_URL)
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", config.build_langfuse_traces_endpoint())
    os.environ.setdefault("OTEL_EXPORTER_OTLP_HEADERS", config.build_langfuse_auth_header())
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_HEADERS", config.build_langfuse_auth_header())
    os.environ.setdefault("OTEL_SERVICE_NAME", "forecast-analysis-agent")
    return True


def _initialize_langfuse_client() -> bool:
    global _LANGFUSE_CLIENT

    if not config.langfuse_configured():
        return False

    try:
        from langfuse import get_client

        _LANGFUSE_CLIENT = get_client()
        return bool(_LANGFUSE_CLIENT.auth_check())
    except Exception:
        logger.exception("Langfuse client authentication check failed")
        return False


def configure_observability() -> dict[str, Any]:
    """Configure Agent Framework observability once per process.

    When LANGFUSE_* variables are present, this derives the OTLP HTTP/protobuf
    traces endpoint and auth header expected by Langfuse before delegating to
    Agent Framework's standard OpenTelemetry bootstrap.
    """

    global _CONFIGURED

    with _LOCK:
        if _CONFIGURED:
            return dict(_STATUS)

        langfuse_enabled = _apply_langfuse_otel_defaults()
        should_configure = langfuse_enabled or _has_otel_configuration()

        _STATUS.update(
            {
                "langfuse_configured": langfuse_enabled,
                "langfuse_sdk_auth": False,
                "otel_traces_endpoint": os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", ""),
                "otel_service_name": os.getenv("OTEL_SERVICE_NAME", ""),
                "error": "",
            }
        )

        if not should_configure:
            return dict(_STATUS)

        try:
            from agent_framework.observability import configure_otel_providers, enable_instrumentation

            configure_otel_providers()
            enable_instrumentation(enable_sensitive_data=config.ENABLE_SENSITIVE_DATA)
            if langfuse_enabled:
                _STATUS["langfuse_sdk_auth"] = _initialize_langfuse_client()
            _CONFIGURED = True
            _STATUS["configured"] = True
        except Exception as exc:
            _STATUS["configured"] = False
            _STATUS["error"] = str(exc)
            logger.exception("Observability bootstrap failed")
            raise

        return dict(_STATUS)


def get_observability_status() -> dict[str, Any]:
    """Return a sanitized snapshot of current observability status."""
    return dict(_STATUS)


def flush_observability() -> None:
    """Flush pending OTel spans and Langfuse batches without raising."""
    try:
        from opentelemetry import trace

        tracer_provider = trace.get_tracer_provider()
        force_flush = getattr(tracer_provider, "force_flush", None)
        if callable(force_flush):
            force_flush()
    except Exception:
        logger.exception("OpenTelemetry force_flush failed")

    try:
        if _LANGFUSE_CLIENT is not None:
            _LANGFUSE_CLIENT.flush()
    except Exception:
        logger.exception("Langfuse flush failed")


def shutdown_observability() -> None:
    """Flush and shutdown observability resources without raising."""
    flush_observability()

    try:
        if _LANGFUSE_CLIENT is not None:
            _LANGFUSE_CLIENT.shutdown()
    except Exception:
        logger.exception("Langfuse shutdown failed")

    try:
        from opentelemetry import trace

        tracer_provider = trace.get_tracer_provider()
        shutdown = getattr(tracer_provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        logger.exception("OpenTelemetry shutdown failed")