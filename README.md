# Forecast Analysis Agent

Forecast Analysis Agent is a planner-facing demo built on Microsoft Agent Framework. It analyses a single article CINV from local CSV data, runs forecast diagnostics, optionally adds weather context when justified, and produces a planner report plus a client-facing email.

The repository now also includes Microsoft Agent Framework DevUI integration for interactive agent testing and the OpenAI-compatible DevUI backend API.

Hosted demo: https://www.marcellmeri.online/

## What The Demo Does

- loads article metadata, weekly forecast history, and linked-article relationships
- evaluates forecast health, outliers, pre-pivot stockout risk, and NM1/NM2/NM3 comparisons
- decides whether weather enrichment is actually warranted before calling the weather tool
- streams agent reasoning, tool activity, and step progress to the UI in real time
- writes a report and email artifact to `output/` for each completed run

## High-Level Structure

```text
Browser / Streamlit UI
    |
    v
FastAPI server (SSE + Streamlit proxy)
    |
    v
Forecast Analysis Workflow
    |
    v
forecast_agent/tools
    |- CSV-backed analysis tools
    |- Tavily search tools
    '- Open-Meteo weather tool
```

## Repository Layout

```text
forecast-analysis-agent/
    app.py                  Streamlit frontend
    server.py               FastAPI API and SSE endpoint
    start.py                local entrypoint for API + UI
    run_agent.py            CLI runner
    run_devui.py            DevUI launcher for the workflow
    devui_entities/         DevUI discovery exports
    data/                   local CSV datasets
    output/                 generated reports and emails
    forecast_agent/
        agent.py            workflow orchestration and stream state
        config.py           environment and path config
        data_access.py      shared CSV loading and normalization
        templates.py        system prompt and output templates
        tools/              analysis, search, and weather tools
```

## Run The Demo Locally

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure `.env`

Copy [.env.example](.env.example) to `.env` and set one supported model configuration:

- Azure AI Foundry project endpoint plus deployment name
- Azure OpenAI endpoint plus API key plus deployment name
- OpenAI API key plus responses model id

Optional:

- `TAVILY_API_KEY` for live search enrichment
- `API_PORT` and `STREAMLIT_PORT` to override local ports

### 4. Start the app

Recommended local mode:

```powershell
python start.py
```

This starts the FastAPI backend and proxies the Streamlit UI through one process boundary.

Alternative modes:

```powershell
uvicorn server:app --reload
streamlit run app.py
```

### 5. Run from the CLI

```powershell
python run_agent.py 4685056
python run_agent.py --cinv 4685056 --force-weather --json
```

### 6. Run in Microsoft DevUI

The app exposes the same Forecast Analysis Workflow directly to Microsoft Agent Framework DevUI using the documented `serve()` and discovery patterns from the local DevUI package under `../agent-framework/python/packages/devui`.

Programmatic launch:

```powershell
python run_devui.py --port 8080
```

Then open `http://127.0.0.1:8080`.

Directory-discovery launch with the DevUI CLI:

```powershell
$env:PYTHONPATH='.'
devui .\devui_entities --port 8080
```

Useful DevUI endpoints:

- `GET /v1/entities` lists the registered entities.
- `POST /v1/responses` exposes the OpenAI-compatible Responses API.

If you want to test against the local DevUI source from this workspace instead of the published package, install it from the sibling repository:

```powershell
pip install -e ..\agent-framework\python\packages\devui
```

## API Surface

- `POST /api/run`: start a workflow run and receive `text/event-stream`
- `GET /api/health`: configuration and readiness status
- `GET /api/cinvs`: available article identifiers from local data
- `GET /v1/entities`: DevUI entity discovery endpoint when running `run_devui.py`
- `POST /v1/responses`: OpenAI-compatible DevUI Responses API when running `run_devui.py`

Example run payload:

```json
{
  "cinv": 4685056,
  "force_weather": false
}
```

## Notes

- Weather enrichment is automatic by default and only runs when the article/search evidence suggests the demand is materially weather-driven. The UI checkbox and CLI flag are force-on overrides.
- Each completed run writes `<CINV>_<timestamp>_report.txt` and `<CINV>_<timestamp>_email.txt` to `output/`.
- Heroku deployment uses [Procfile](Procfile) with `python start.py`. Prefer API-key-based Azure OpenAI configuration for hosted environments.

## Troubleshooting

- API unreachable: start the app with `python start.py` or launch the backend with `uvicorn server:app --reload`.
- DevUI import errors: run from the project root and set `PYTHONPATH=.` when using the `devui` CLI discovery mode.
- Tavily output missing: set `TAVILY_API_KEY`.
- Weather output missing: check the `Logs` view first; weather only runs automatically when the weather-sensitivity gate approves it unless `--force-weather` or the UI override is used.
