"""CLI entrypoint for running the forecast analysis agent without the web UI."""

from __future__ import annotations

import argparse
import asyncio
import json

from rich.console import Console

from forecast_agent.observability import configure_observability, shutdown_observability

configure_observability()

from forecast_agent.agent import ForecastAnalysisAgent

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DFAI Forecast Analysis Agent from the command line.")
    parser.add_argument("cinv", nargs="?", type=int, help="Article CINV identifier to analyze.")
    parser.add_argument("--cinv", dest="cinv_flag", type=int, help="Article CINV identifier to analyze.")
    parser.add_argument(
        "--force-weather",
        action="store_true",
        help="Force weather enrichment even if the agent does not find strong weather-sensitivity signals.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final state as JSON instead of the rendered report and email.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    cinv = args.cinv_flag or args.cinv
    if cinv is None:
        raise SystemExit("A CINV must be provided either positionally or via --cinv.")

    agent = ForecastAnalysisAgent()
    state = await agent.run_analysis(cinv, force_weather=args.force_weather)

    if args.json:
        console.print_json(json.dumps(state, default=str))
        return 0

    console.rule(f"Forecast Analysis Complete: CINV {cinv}")
    console.print(state.get("report", "No report generated."))
    console.rule("Email Draft")
    console.print(state.get("email", "No email draft generated."))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args))
    finally:
        shutdown_observability()


if __name__ == "__main__":
    raise SystemExit(main())