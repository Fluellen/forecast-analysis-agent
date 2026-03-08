"""Forecast Analysis Agent package."""

from __future__ import annotations

from typing import Any

__all__ = ["ForecastAnalysisAgent"]


def __getattr__(name: str) -> Any:
	if name == "ForecastAnalysisAgent":
		from .agent import ForecastAnalysisAgent

		return ForecastAnalysisAgent
	raise AttributeError(name)