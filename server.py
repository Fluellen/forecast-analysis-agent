"""FastAPI application that exposes the forecast agent over AG-UI-style SSE and proxies Streamlit."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pandas as pd
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse

import forecast_agent.config as cfg
from forecast_agent.agent import ForecastAnalysisAgent
from forecast_agent.data_access import load_forecast_frame, load_metadata_frame
from forecast_agent.events import run_error

_streamlit_process: subprocess.Popen[str] | None = None


def _start_streamlit() -> None:
    global _streamlit_process
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        f"--server.port={cfg.STREAMLIT_PORT}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
    ]
    _streamlit_process = subprocess.Popen(cmd, text=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("DISABLE_STREAMLIT_AUTOSTART"):
        thread = threading.Thread(target=_start_streamlit, daemon=True)
        thread.start()
        await asyncio.sleep(3)
    yield
    if _streamlit_process and _streamlit_process.poll() is None:
        _streamlit_process.terminate()


app = FastAPI(title="DFAI Forecast Agent AG-UI Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STREAMLIT_BASE = f"http://127.0.0.1:{cfg.STREAMLIT_PORT}"
STREAMLIT_WS_BASE = f"ws://127.0.0.1:{cfg.STREAMLIT_PORT}"


@app.post("/api/run")
async def run_analysis(request: Request) -> StreamingResponse:
    body = await request.json()
    cinv = body.get("cinv")
    force_weather = bool(body.get("force_weather", body.get("use_weather", False)))
    if cinv is None:
        raise HTTPException(status_code=400, detail="cinv is required")

    async def event_generator():
        try:
            agent = ForecastAnalysisAgent()
            async for event in agent.run_analysis_stream(int(cinv), force_weather=force_weather):
                yield event.to_sse()
                await asyncio.sleep(0)
        except Exception as exc:
            yield run_error("unknown", str(exc)).to_sse()
        finally:
            yield 'data: {"type":"STREAM_END"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    warnings = cfg.validate()
    return {
        "status": "ok",
        "config_warnings": warnings,
        "agent_framework": True,
        "tavily_search": bool(cfg.TAVILY_API_KEY),
        "is_heroku": cfg.IS_HEROKU,
    }


@app.get("/api/cinvs")
async def list_cinvs() -> dict[str, list[dict[str, Any]]]:
    forecast = load_forecast_frame()[["ART_CINV"]].drop_duplicates()
    metadata = load_metadata_frame().drop_duplicates(subset=["ART_CINV"], keep="first")
    merged = forecast.merge(metadata, on="ART_CINV", how="left").sort_values("ART_CINV")
    return {
        "cinvs": [
            {
                "cinv": int(str(row.ART_CINV)),
                "name": str(row.ART_DESC) if not pd.isna(row.ART_DESC) and row.ART_DESC else "Unknown",
            }
            for row in merged.itertuples()
        ]
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_to_streamlit(path: str, request: Request) -> Response:
    url = f"{STREAMLIT_BASE}/{path}"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            upstream = await client.request(
                method=request.method,
                url=url,
                params=request.query_params,
                headers={key: value for key, value in request.headers.items() if key.lower() != "host"},
                content=await request.body(),
            )
    except httpx.ConnectError:
        return HTMLResponse("<h2>Streamlit is starting up. Refresh in a few seconds.</h2>", status_code=503)

    excluded = {"content-encoding", "transfer-encoding", "connection", "keep-alive"}
    headers = {key: value for key, value in upstream.headers.items() if key.lower() not in excluded}
    return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)


@app.websocket("/{path:path}")
async def websocket_proxy(websocket: WebSocket, path: str) -> None:
    query = websocket.url.query
    upstream_url = f"{STREAMLIT_WS_BASE}/{path}"
    if query:
        upstream_url = f"{upstream_url}?{query}"

    await websocket.accept()
    try:
        async with websockets.connect(upstream_url) as upstream:
            async def client_to_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    if message.get("text") is not None:
                        await upstream.send(message["text"])
                    elif message.get("bytes") is not None:
                        await upstream.send(message["bytes"])

            async def upstream_to_client() -> None:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    elif isinstance(message, str):
                        await websocket.send_text(message)
                    else:
                        await websocket.send_bytes(bytes(message))

            await asyncio.gather(client_to_upstream(), upstream_to_client())
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)