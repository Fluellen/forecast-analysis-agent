"""Workflow export used by DevUI directory discovery."""

from agent_framework.devui import register_cleanup

from forecast_agent.agent import create_forecast_workflow, get_cleanup_hooks

workflow = create_forecast_workflow()

for hook in get_cleanup_hooks():
    register_cleanup(workflow, hook)