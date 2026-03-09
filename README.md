# Forecast Analysis Agent

Forecast Analysis Agent is a planner-facing demo built on Microsoft Agent Framework. It analyses a single article CINV from local CSV data, runs forecast diagnostics, optionally adds weather context when justified, and produces a planner report plus a client-facing email.

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
ForecastAnalysisAgent
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
    data/                   local CSV datasets
    output/                 generated reports and emails
    forecast_agent/
        agent.py            agent orchestration and stream state
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

## API Surface

- `POST /api/run`: start an agent run and receive `text/event-stream`
- `GET /api/health`: configuration and readiness status
- `GET /api/cinvs`: available article identifiers from local data

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
- Tavily output missing: set `TAVILY_API_KEY`.
- Weather output missing: check the `Logs` view first; weather only runs automatically when the weather-sensitivity gate approves it unless `--force-weather` or the UI override is used.
