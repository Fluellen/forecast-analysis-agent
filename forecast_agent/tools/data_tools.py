"""Core CSV-backed tools for article metadata and forecast diagnostics."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy import stats

from ..data_access import (
    frame_to_records,
    get_article_forecast_frame,
    get_article_links_frame,
    get_article_metadata_frame,
    normalize_for_json,
    normalize_for_json,
)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, default=str)


def get_article_metadata(art_cinv: int) -> str:
    """Load article_metadata.csv and return all metadata for the given ART_CINV as JSON."""
    frame = get_article_metadata_frame(int(art_cinv))
    records = frame_to_records(frame)
    return _json(
        {
            "art_cinv": int(art_cinv),
            "count": len(records),
            "metadata": records,
        }
    )


def get_forecast_data(art_cinv: int) -> str:
    """Load forecast_data.csv filtered to the given ART_CINV and return sorted weekly history as JSON."""
    frame = get_article_forecast_frame(int(art_cinv)).copy()
    pivot_date = None
    if not frame.empty:
        frame["PIVOT_DATE"] = frame["PIVOT_DATE_DT"].dt.strftime("%Y-%m-%d")
        frame["MVT_DATE"] = frame["MVT_DATE_DT"].dt.strftime("%Y-%m-%d")
        if frame["PIVOT_DATE"].notna().any():
            pivot_date = str(frame["PIVOT_DATE"].dropna().iloc[-1])
    records = frame_to_records(
        frame[
            [
                "ART_CINV",
                "PIVOT_DATE",
                "MVT_DATE",
                "ISO_YEAR",
                "ISO_WEEK",
                "WEEK_ID",
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
        ]
        if not frame.empty
        else frame
    )

    return _json(
        {
            "art_cinv": int(art_cinv),
            "count": len(records),
            "pivot_date": pivot_date,
            "date_range": {
                "start": records[0]["MVT_DATE"] if records else None,
                "end": records[-1]["MVT_DATE"] if records else None,
            },
            "rows": records,
        }
    )


def get_article_links(art_cinv: int) -> str:
    """Load links.csv rows where art_cinv_a equals the given article and return them as JSON."""
    frame = get_article_links_frame(int(art_cinv)).copy()
    records = frame_to_records(frame)
    return _json(
        {
            "art_cinv": int(art_cinv),
            "links": records,
            "count": len(records),
        }
    )


def compute_forecast_health(art_cinv: int) -> str:
    """Compute forecast-horizon accuracy, bias, tracking-signal, and weekly error detail for an article."""
    frame = get_article_forecast_frame(int(art_cinv)).copy()

    if frame.empty:
        return _json(
            {
                "art_cinv": int(art_cinv),
                "pivot_date": None,
                "actuals_available_through": None,
                "weeks_before_pivot_excluded": 0,
                "weeks_in_horizon": 0,
                "weeks_with_actuals": 0,
                "weeks_pending_actuals": 0,
                "weeks_analysed": 0,
                "mape_pct": None,
                "wape_pct": None,
                "bias_units": None,
                "bias_pct": None,
                "tracking_signal": None,
                "tracking_signal_flag": False,
                "accuracy_rating": "INSUFFICIENT_DATA",
                "weekly_metrics": [],
                "note": "No forecast rows were available for this article.",
            }
        )

    pivot_ts = None
    if "PIVOT_DATE_DT" in frame.columns and frame["PIVOT_DATE_DT"].notna().any():
        pivot_ts = frame["PIVOT_DATE_DT"].dropna().iloc[-1]

    horizon = frame.copy()
    pre_pivot_excluded = 0
    if pivot_ts is not None and "MVT_DATE_DT" in horizon.columns:
        pre_pivot_excluded = int((horizon["MVT_DATE_DT"] < pivot_ts).sum())
        horizon = horizon[horizon["MVT_DATE_DT"] >= pivot_ts].copy()

    if horizon.empty:
        return _json(
            {
                "art_cinv": int(art_cinv),
                "pivot_date": normalize_for_json(pivot_ts),
                "actuals_available_through": None,
                "weeks_before_pivot_excluded": pre_pivot_excluded,
                "weeks_in_horizon": 0,
                "weeks_with_actuals": 0,
                "weeks_pending_actuals": 0,
                "weeks_analysed": 0,
                "mape_pct": None,
                "wape_pct": None,
                "bias_units": None,
                "bias_pct": None,
                "tracking_signal": None,
                "tracking_signal_flag": False,
                "accuracy_rating": "INSUFFICIENT_DATA",
                "weekly_metrics": [],
                "note": "No forecast-horizon weeks were available on or after the pivot date.",
            }
        )

    horizon = horizon.sort_values("MVT_DATE_DT").copy()
    last_observed_actual_ts = None
    if (horizon["ACTUAL_DEMAND"] > 0).any():
        last_observed_actual_ts = horizon.loc[horizon["ACTUAL_DEMAND"] > 0, "MVT_DATE_DT"].max()

    actual_available = horizon["ACTUAL_DEMAND"].notna().copy()
    if last_observed_actual_ts is not None:
        actual_available &= ~((horizon["ACTUAL_DEMAND"] == 0) & (horizon["MVT_DATE_DT"] > last_observed_actual_ts))

    actual_known = horizon[actual_available].copy()
    pending_actuals = horizon[~actual_available].copy()
    pct_eligible = actual_known[actual_known["ACTUAL_DEMAND"] > 0].copy()

    weekly_metrics: list[dict[str, Any]] = []
    for row in horizon.itertuples():
        actual_value = None if normalize_for_json(row.ACTUAL_DEMAND) is None else float(row.ACTUAL_DEMAND)
        forecast_value = None if normalize_for_json(row.FORECAST) is None else float(row.FORECAST)
        error_units = None
        abs_error_units = None
        abs_pct_error = None
        accuracy_pct = None
        actual_status = "pending"

        if actual_value is not None:
            actual_status = "available"

        if actual_value == 0 and last_observed_actual_ts is not None and row.MVT_DATE_DT > last_observed_actual_ts:
            actual_status = "pending"

        if actual_value is not None and forecast_value is None:
            actual_status = "forecast_missing"

        if actual_status != "pending" and actual_value is not None and forecast_value is not None:
            error_units = forecast_value - actual_value
            abs_error_units = abs(error_units)
            if actual_value > 0:
                abs_pct_error = abs_error_units / actual_value * 100
                accuracy_pct = max(0.0, 100.0 - abs_pct_error)
            elif actual_value == 0:
                actual_status = "available_zero_actual"

        weekly_metrics.append(
            {
                "week_id": row.WEEK_ID,
                "MVT_DATE": normalize_for_json(row.MVT_DATE_DT),
                "PIVOT_DATE": normalize_for_json(row.PIVOT_DATE_DT),
                "FORECAST": forecast_value,
                "ACTUAL_DEMAND": actual_value,
                "actual_status": actual_status,
                "error_units": round(error_units, 2) if error_units is not None else None,
                "abs_error_units": round(abs_error_units, 2) if abs_error_units is not None else None,
                "abs_pct_error": round(abs_pct_error, 2) if abs_pct_error is not None else None,
                "accuracy_pct": round(accuracy_pct, 2) if accuracy_pct is not None else None,
                "promo_type": str(row.PROMO_TYPE).strip(),
                "holiday_type": str(row.HOLIDAYS_TYPE).strip(),
            }
        )

    if actual_known.empty:
        return _json(
            {
                "art_cinv": int(art_cinv),
                "pivot_date": normalize_for_json(pivot_ts),
                "actuals_available_through": normalize_for_json(last_observed_actual_ts),
                "weeks_before_pivot_excluded": pre_pivot_excluded,
                "weeks_in_horizon": int(len(horizon)),
                "weeks_with_actuals": 0,
                "weeks_pending_actuals": int(len(pending_actuals)),
                "weeks_analysed": 0,
                "mape_pct": None,
                "wape_pct": None,
                "bias_units": None,
                "bias_pct": None,
                "tracking_signal": None,
                "tracking_signal_flag": False,
                "accuracy_rating": "INSUFFICIENT_DATA",
                "weekly_metrics": weekly_metrics,
                "note": "No forecast-horizon weeks have actual demand yet.",
            }
        )

    errors = actual_known["FORECAST"] - actual_known["ACTUAL_DEMAND"]
    abs_errors = np.abs(errors)
    abs_pct_errors = np.abs(pct_eligible["FORECAST"] - pct_eligible["ACTUAL_DEMAND"]) / pct_eligible["ACTUAL_DEMAND"]
    mape = float(abs_pct_errors.mean() * 100) if not abs_pct_errors.empty else None
    bias = float(errors.mean()) if not errors.empty else None
    mean_actual = float(actual_known["ACTUAL_DEMAND"].mean()) if not actual_known.empty else None
    bias_pct = float((bias / mean_actual) * 100) if bias is not None and mean_actual not in (None, 0) else None
    mad = float(abs_errors.mean()) if not errors.empty else None
    rsfe = float(errors.sum()) if not errors.empty else None
    tracking_signal = float(rsfe / mad) if mad not in (None, 0) and rsfe is not None else None
    total_actual = float(actual_known["ACTUAL_DEMAND"].sum()) if not actual_known.empty else None
    wape = float(abs_errors.sum() / total_actual * 100) if total_actual not in (None, 0) else None

    if mape is None:
        rating = "INSUFFICIENT_DATA"
    elif mape < 15:
        rating = "EXCELLENT"
    elif mape < 25:
        rating = "GOOD"
    elif mape < 40:
        rating = "FAIR"
    else:
        rating = "POOR"

    return _json(
        {
            "art_cinv": int(art_cinv),
            "pivot_date": normalize_for_json(pivot_ts),
            "actuals_available_through": normalize_for_json(last_observed_actual_ts),
            "weeks_before_pivot_excluded": pre_pivot_excluded,
            "weeks_in_horizon": int(len(horizon)),
            "weeks_with_actuals": int(len(actual_known)),
            "weeks_pending_actuals": int(len(pending_actuals)),
            "weeks_analysed": int(len(actual_known)),
            "mape_pct": round(mape, 2) if mape is not None else None,
            "wape_pct": round(wape, 2) if wape is not None else None,
            "bias_units": round(bias, 2) if bias is not None else None,
            "bias_pct": round(bias_pct, 2) if bias_pct is not None else None,
            "tracking_signal": round(tracking_signal, 2) if tracking_signal is not None else None,
            "tracking_signal_flag": bool(tracking_signal is not None and abs(tracking_signal) > 6),
            "accuracy_rating": rating,
            "weekly_metrics": weekly_metrics,
            "note": (
                "Metrics are calculated only for weeks on or after the pivot date. "
                "Weeks with missing actual demand, plus trailing zero-demand placeholder weeks beyond the last observed actual week, remain listed in weekly_metrics but are excluded from aggregate accuracy calculations."
            ),
        }
    )


def detect_outlier_weeks(art_cinv: int) -> str:
    """Detect statistically unusual demand weeks for an article using IQR and z-score methods."""
    frame = get_article_forecast_frame(int(art_cinv)).copy()
    clean = frame[frame["ACTUAL_DEMAND"] > 0].copy()

    if clean.empty:
        return _json({"art_cinv": int(art_cinv), "count": 0, "outliers": []})

    q1 = clean["ACTUAL_DEMAND"].quantile(0.25)
    q3 = clean["ACTUAL_DEMAND"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    std = clean["ACTUAL_DEMAND"].std(ddof=0)
    mean = clean["ACTUAL_DEMAND"].mean()
    if std and not np.isclose(std, 0):
        clean["z_score"] = stats.zscore(clean["ACTUAL_DEMAND"], nan_policy="omit")
    else:
        clean["z_score"] = 0.0

    clean["iqr_outlier"] = (clean["ACTUAL_DEMAND"] < lower) | (clean["ACTUAL_DEMAND"] > upper)
    clean["z_outlier"] = clean["z_score"].abs() > 2
    outliers = clean[clean["iqr_outlier"] | clean["z_outlier"]].copy()

    records: list[dict[str, Any]] = []
    for row in outliers.sort_values("z_score", key=lambda column: column.abs(), ascending=False).itertuples():
        records.append(
            {
                "week_id": row.WEEK_ID,
                "MVT_DATE": normalize_for_json(row.MVT_DATE_DT),
                "ACTUAL_DEMAND": normalize_for_json(row.ACTUAL_DEMAND),
                "FORECAST": normalize_for_json(row.FORECAST),
                "direction": "HIGH" if float(row.ACTUAL_DEMAND) >= mean else "LOW",
                "z_score": round(float(row.z_score), 3),
                "iqr_outlier": bool(row.iqr_outlier),
                "has_promo": bool(str(row.PROMO_TYPE).strip()),
                "has_holiday": bool(str(row.HOLIDAYS_TYPE).strip()),
                "holiday_name": str(row.HOLIDAYS_TYPE).strip(),
            }
        )

    return _json(
        {
            "art_cinv": int(art_cinv),
            "count": len(records),
            "outliers": records,
        }
    )