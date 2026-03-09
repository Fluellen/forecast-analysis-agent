"""Tool exports for the forecast analysis agent."""

from .analysis import analyse_year_on_year_trend, correlate_weather_with_demand, get_article_links_demand
from .data_tools import (
    compute_forecast_health,
    detect_pre_pivot_stockout_risk,
    detect_outlier_weeks,
    get_article_links,
    get_article_metadata,
    get_forecast_data,
)
from .search_tool import search_article_characteristics, search_holiday_demand_correlation
from .weather_mcp import get_weather_for_period

ALL_TOOLS = [
    get_article_metadata,
    get_forecast_data,
    get_article_links,
    compute_forecast_health,
    detect_pre_pivot_stockout_risk,
    detect_outlier_weeks,
    search_article_characteristics,
    search_holiday_demand_correlation,
    correlate_weather_with_demand,
    analyse_year_on_year_trend,
    get_article_links_demand,
]

__all__ = [
    "ALL_TOOLS",
    "get_article_metadata",
    "get_forecast_data",
    "get_article_links",
    "compute_forecast_health",
    "detect_pre_pivot_stockout_risk",
    "detect_outlier_weeks",
    "get_weather_for_period",
    "search_article_characteristics",
    "search_holiday_demand_correlation",
    "correlate_weather_with_demand",
    "analyse_year_on_year_trend",
    "get_article_links_demand",
]