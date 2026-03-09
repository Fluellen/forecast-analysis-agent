# Forecast Analysis Agent

Planner-facing forecast diagnostics built with Microsoft Agent Framework, FastAPI, AG-UI-style SSE, and a Streamlit terminal UI.

## What It Does

This project analyzes a selected article CINV from the CSV datasets in [data/forecast_data.csv](data/forecast_data.csv), [data/article_metadata.csv](data/article_metadata.csv), and [data/links.csv](data/links.csv). The agent:

- identifies the article and related metadata
- evaluates forecast health and outlier weeks
- detects pre-pivot zero-shipment streaks that can depress the future forecast baseline
- compares demand against NM1, NM2, and NM3 baselines
- optionally enriches the run with weather context for Dublin
- drafts a planner report and a client-facing email
- streams its reasoning trace and tool activity to the frontend in real time
- keeps a separate in-app log console with the full trace timeline and raw AG-UI event stream

## Architecture

```text
Browser
	|
	v
FastAPI server.py  ------------------------------.
	|                                              |
	|  POST /api/run (SSE)                         | reverse proxy
	v                                              |
ForecastAnalysisAgent                            |
	|                                              |
	|  Microsoft Agent Framework                   |
	v                                              |
Tool functions in forecast_agent/tools           |
	|                                              |
	+--> CSV data access                           |
	+--> Tavily search                             |
	+--> Open-Meteo weather                        |
																								 |
Streamlit app.py  <------------------------------'
```

## Project Layout

```text
forecast-analysis-agent/
	app.py
	server.py
	start.py
	run_agent.py
	requirements.txt
	Procfile
	runtime.txt
	.python-version
	.env.example
	output/
	data/
	forecast_agent/
		agent.py
		config.py
		data_access.py
		events.py
		runtime.py
		templates.py
		tools/
```

## Quick Start

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

If you are working against the sibling local framework repository instead of package releases, install these after the main requirements:

```powershell
pip install -e ..\agent-framework\python\packages\core
pip install -e ..\agent-framework\python\packages\ag-ui
```

### 3. Configure environment variables

Copy [.env.example](.env.example) to `.env` and set one supported model path:

- Azure AI Foundry project endpoint plus deployment name
- Azure OpenAI endpoint plus API key plus deployment name
- OpenAI API key plus responses model id

Optional:

- `TAVILY_API_KEY` for live article characteristic search
- `API_PORT` and `STREAMLIT_PORT` to override default ports

### 4. Run locally

Unified mode:

```powershell
python start.py
```

API only:

```powershell
uvicorn server:app --reload
```

UI only, if the API is already running:

```powershell
streamlit run app.py
```

The UI has two workspace modes in the sidebar:

- `Analysis` for charts, recent live trace, report, and email output
- `Logs` for the complete trace timeline, a compacted live event stream, filtering, and raw JSON export

CLI mode:

```powershell
python run_agent.py 4685056
python run_agent.py --cinv 4685056 --force-weather --json
```

## API Endpoints

- `POST /api/run`
	Request body:

```json
{
	"cinv": 4685056,
	"force_weather": true
}
```

	Returns `text/event-stream` with AG-UI-compatible events plus project-specific `STEP_STARTED` and `STEP_FINISHED` markers.

- `GET /api/health`
	Returns configuration warnings and environment readiness.

- `GET /api/cinvs`
	Returns available article identifiers and labels from the local CSV files.

## Tool Inventory

| Tool | Purpose |
| --- | --- |
| `get_article_metadata` | Load article master data from `article_metadata.csv` |
| `get_forecast_data` | Load weekly forecast rows from `forecast_data.csv` |
| `get_article_links` | Load linked article relationships from `links.csv` |
| `compute_forecast_health` | Compute pivot-horizon MAPE/WAPE, bias, tracking signal, and weekly forecast accuracy detail |
| `detect_pre_pivot_stockout_risk` | Detect consecutive zero-shipment weeks before the pivot, estimate baseline depression, and recommend xout or article-link remediation |
| `detect_outlier_weeks` | Flag IQR and z-score outlier weeks |
| `analyse_year_on_year_trend` | Compare demand against NM1, NM2, and NM3 baselines |
| `get_article_links_demand` | Summarize linked article demand, highlight duplicate link rows, and frame substitution context |
| `correlate_weather_with_demand` | Retrieve capped historical weather context and measure weekly demand correlation |
| `search_article_characteristics` | Pull web context via Tavily |
| `search_holiday_demand_correlation` | Run a single Tavily search for holiday or special-event demand relevance in Ireland, using observed article holiday values as context |

## Weather Approval Model

Weather enrichment is now automatic by default. The agent uses the Tavily/search result to assess whether the article is weather-sensitive and only runs the weather correlation step when that evidence suggests it is worthwhile. The UI checkbox and CLI flag now act as a force-on override, so a user can require weather enrichment even when the search evidence is weak.

## Output Files

Every completed run writes two artifacts to `output/`:

- `<CINV>_<timestamp>_report.txt`
- `<CINV>_<timestamp>_email.txt`

## Heroku Notes

- The dyno entrypoint is [Procfile](Procfile) using `python start.py`.
- The project includes both [runtime.txt](runtime.txt) and [.python-version](.python-version). Current target is Python 3.11.15.
- Heroku should use API-key-based auth rather than Azure CLI auth. Prefer:
	- `AZURE_OPENAI_ENDPOINT`
	- `AZURE_OPENAI_API_KEY`
	- `AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME`

## Troubleshooting

### The UI says the API is unreachable

Start the backend first with `python start.py` or `uvicorn server:app --reload`.

### Azure AI Foundry auth fails locally

If you use `AZURE_AI_PROJECT_ENDPOINT`, authenticate with `az login` first. For hosted environments, switch to Azure OpenAI endpoint plus API key variables instead.

### No Tavily output appears

Set `TAVILY_API_KEY`. The tool degrades cleanly when the key is absent.

### Weather data does not appear

Weather enrichment only runs automatically when the agent finds weather-sensitivity signals in the search context, unless you force it on from the UI or with `--force-weather`. Weather requests are automatically capped to the article pivot date and to today so the Open-Meteo archive API is not asked for future dates. If weather is still missing, inspect the `Logs` workspace in the UI and review the weather tool result payload.

### Streamlit assets fail behind the FastAPI proxy

Use `python start.py` so the API server launches Streamlit and proxies both HTTP and WebSocket traffic through a single process boundary.
