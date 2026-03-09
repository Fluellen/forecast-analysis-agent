#!/usr/bin/env python3
"""Launch Microsoft Agent Framework DevUI for the forecast analysis agent."""

from __future__ import annotations

import argparse

from agent_framework.devui import register_cleanup, serve

from forecast_agent import config
from forecast_agent.agent import create_forecast_workflow, get_cleanup_hooks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Forecast Analysis Agent in Microsoft Agent Framework DevUI.")
    parser.add_argument("--port", type=int, default=config.DEVUI_PORT, help="DevUI port (default: %(default)s).")
    parser.add_argument("--host", default=config.DEVUI_HOST, help="DevUI host (default: %(default)s).")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the DevUI browser window.")
    parser.add_argument("--headless", action="store_true", help="Run the DevUI API without the web UI.")
    parser.add_argument(
        "--instrumentation",
        action="store_true",
        help="Enable DevUI OpenTelemetry instrumentation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workflow = create_forecast_workflow()
    for hook in get_cleanup_hooks():
        register_cleanup(workflow, hook)

    serve(
        entities=[workflow],
        port=args.port,
        host=args.host,
        auto_open=not args.no_open,
        ui_enabled=not args.headless,
        instrumentation_enabled=args.instrumentation,
    )


if __name__ == "__main__":
    main()