"""Runtime context shared between the API layer, agent layer, and pure-Python tools."""

from __future__ import annotations

from contextvars import ContextVar, Token

_weather_forced: ContextVar[bool] = ContextVar("weather_forced", default=False)


def set_weather_forced(forced: bool) -> Token[bool]:
    """Set whether the current run should force weather enrichment."""
    return _weather_forced.set(forced)


def reset_weather_forced(token: Token[bool]) -> None:
    """Reset the weather force flag using a previously returned token."""
    _weather_forced.reset(token)


def is_weather_forced() -> bool:
    """Return whether the current run should always execute weather enrichment."""
    return _weather_forced.get()