"""Shared data access and normalization utilities for local CSV-backed tools."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import pandas as pd

from . import config

EXPECTED_FORECAST_COLUMNS = [
    "ART_CINV",
    "PIVOT_DATE",
    "MVT_DATE",
    "ISO_YEAR",
    "ISO_WEEK",
    "FORECAST",
    "ACTUAL_DEMAND",
    "MOVING_AVG",
    "DC_CODE",
    "PROMO_TYPE",
    "HOLIDAYS_TYPE",
    "NM1",
    "NM2",
    "NM3",
]

EXPECTED_METADATA_COLUMNS = [
    "ART_CINV",
    "ART_DESC",
    "ART_LEVEL1_DESC",
    "ART_LEVEL2_DESC",
    "ART_LEVEL3_DESC",
]

EXPECTED_LINKS_COLUMNS = [
    "item_link",
    "link_desc",
    "art_cinv_a",
    "art_cinv_b",
    "start_date",
    "end_date",
    "node_code",
    "coefficient",
]


def _strip_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


@lru_cache(maxsize=1)
def load_forecast_frame() -> pd.DataFrame:
    """Load and normalize the main forecast data frame."""
    frame = _strip_columns(pd.read_csv(config.FORECAST_DATA_PATH, low_memory=False))
    for column in EXPECTED_FORECAST_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[EXPECTED_FORECAST_COLUMNS].copy()
    numeric_columns = [
        "ART_CINV",
        "PIVOT_DATE",
        "MVT_DATE",
        "ISO_YEAR",
        "ISO_WEEK",
        "FORECAST",
        "ACTUAL_DEMAND",
        "MOVING_AVG",
        "DC_CODE",
        "NM1",
        "NM2",
        "NM3",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in ["PROMO_TYPE", "HOLIDAYS_TYPE"]:
        frame[column] = frame[column].fillna("").astype(str).replace({"None": "", "nan": ""}).str.strip()

    frame["PIVOT_DATE_DT"] = parse_yyyymmdd_series(frame["PIVOT_DATE"])
    frame["MVT_DATE_DT"] = parse_yyyymmdd_series(frame["MVT_DATE"])
    iso_components = iso_week_components(frame["MVT_DATE_DT"])
    frame["ISO_YEAR"] = iso_components["ISO_YEAR"]
    frame["ISO_WEEK"] = iso_components["ISO_WEEK"]
    frame["WEEK_ID"] = iso_components["WEEK_ID"]
    return frame


@lru_cache(maxsize=1)
def load_metadata_frame() -> pd.DataFrame:
    """Load and normalize the article metadata frame."""
    frame = _strip_columns(pd.read_csv(config.ARTICLE_METADATA_PATH, low_memory=False))
    for column in EXPECTED_METADATA_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[EXPECTED_METADATA_COLUMNS].copy()
    frame["ART_CINV"] = pd.to_numeric(frame["ART_CINV"], errors="coerce")
    for column in EXPECTED_METADATA_COLUMNS[1:]:
        frame[column] = frame[column].fillna("").astype(str).replace({"None": "", "nan": ""}).str.strip()
    return frame


@lru_cache(maxsize=1)
def load_links_frame() -> pd.DataFrame:
    """Load and normalize the article links frame."""
    frame = _strip_columns(pd.read_csv(config.LINKS_PATH, low_memory=False))
    if "site_reference" in frame.columns and "coefficient" not in frame.columns:
        frame["coefficient"] = pd.NA

    for column in EXPECTED_LINKS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[[column for column in frame.columns if column in set(EXPECTED_LINKS_COLUMNS + ["site_reference"])]].copy()
    for column in ["item_link", "art_cinv_a", "art_cinv_b", "start_date", "end_date", "node_code", "coefficient"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["link_desc"] = frame["link_desc"].fillna("").astype(str).replace({"None": "", "nan": ""}).str.strip()
    return frame


def parse_yyyymmdd_series(series: pd.Series) -> pd.Series:
    """Parse integer-like YYYYMMDD values into pandas timestamps."""
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    non_null = series.dropna()
    if non_null.empty:
        return parsed
    formatted = non_null.astype(int).astype(str).str.zfill(8)
    parsed.loc[formatted.index] = pd.to_datetime(formatted, format="%Y%m%d", errors="coerce")
    return parsed


def iso_week_components(series: pd.Series) -> pd.DataFrame:
    """Derive canonical ISO year/week identifiers from a date-like series."""
    timestamps = pd.to_datetime(series, errors="coerce")
    iso = timestamps.dt.isocalendar()
    return pd.DataFrame(
        {
            "ISO_YEAR": iso["year"].astype("Int64"),
            "ISO_WEEK": iso["week"].astype("Int64"),
            "WEEK_ID": [to_week_id(year, week) for year, week in zip(iso["year"], iso["week"])],
        },
        index=series.index,
    )


def to_week_id(iso_year: Any, iso_week: Any) -> str | None:
    """Build a stable ISO week identifier."""
    if pd.isna(iso_year) or pd.isna(iso_week):
        return None
    return f"{int(iso_year)}-W{int(iso_week):02d}"


def normalize_for_json(value: Any) -> Any:
    """Convert pandas and numpy values into JSON-safe Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    return value


def frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to normalized JSON-style records."""
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({str(key): normalize_for_json(value) for key, value in row.items()})
    return records


def get_article_metadata_frame(art_cinv: int) -> pd.DataFrame:
    """Return metadata rows for an article."""
    frame = load_metadata_frame()
    return frame[frame["ART_CINV"] == art_cinv].copy()


def get_article_forecast_frame(art_cinv: int) -> pd.DataFrame:
    """Return normalized forecast rows for an article sorted by movement date."""
    frame = load_forecast_frame()
    result = frame[frame["ART_CINV"] == art_cinv].copy()
    return result.sort_values(["MVT_DATE", "PIVOT_DATE"], ascending=[True, True])


def get_article_links_frame(art_cinv: int) -> pd.DataFrame:
    """Return direct link rows for an article."""
    frame = load_links_frame()
    return frame[frame["art_cinv_a"] == art_cinv].copy()