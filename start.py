#!/usr/bin/env python3
"""Unified startup script for local use and Heroku deployment."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    is_heroku = bool(os.getenv("DYNO"))
    os.environ["API_PORT"] = str(port)

    print("Starting DFAI Forecast Agent")
    print(f"  Mode:      {'Heroku' if is_heroku else 'Local'}")
    print(f"  API port:  {port}")
    print(f"  UI:        http://0.0.0.0:{port}/")
    print(f"  Health:    http://0.0.0.0:{port}/api/health")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=not is_heroku,
        log_level="info",
    )


if __name__ == "__main__":
    main()