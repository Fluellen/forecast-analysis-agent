"""Core CSV-backed tools for article metadata and forecast diagnostics."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..data_access import (
    frame_to_records,
    get_article_forecast_frame,
    get_article_links_frame,
    get_article_metadata_frame,
    normalize_for_json,
)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, default=str)


def _clean_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "nan", "null"} else text


FORECAST_EXPORT_COLUMNS = [
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


def build_article_metadata_payload(art_cinv: int) -> dict[str, Any]:
    """Build the JSON-ready metadata payload for an article."""
    frame = get_article_metadata_frame(int(art_cinv))
    records = frame_to_records(frame)
    return {
        "art_cinv": int(art_cinv),
        "count": len(records),
        "metadata": records,
    }


def build_forecast_data_payload(art_cinv: int) -> dict[str, Any]:
    """Build the JSON-ready forecast history payload for an article."""
    frame = get_article_forecast_frame(int(art_cinv)).copy()
    pivot_date = None
    if not frame.empty:
        frame["PIVOT_DATE"] = frame["PIVOT_DATE_DT"].dt.strftime("%Y-%m-%d")
        frame["MVT_DATE"] = frame["MVT_DATE_DT"].dt.strftime("%Y-%m-%d")
        if frame["PIVOT_DATE"].notna().any():
            pivot_date = str(frame["PIVOT_DATE"].dropna().iloc[-1])

    records = frame_to_records(frame[FORECAST_EXPORT_COLUMNS])
    return {
        "art_cinv": int(art_cinv),
        "count": len(records),
        "pivot_date": pivot_date,
        "date_range": {
            "start": records[0]["MVT_DATE"] if records else None,
            "end": records[-1]["MVT_DATE"] if records else None,
        },
        "rows": records,
    }


def build_article_links_payload(art_cinv: int) -> dict[str, Any]:
    """Build the JSON-ready direct-links payload for an article."""
    frame = get_article_links_frame(int(art_cinv)).copy()
    records = frame_to_records(frame)
    return {
        "art_cinv": int(art_cinv),
        "links": records,
        "count": len(records),
    }


def get_article_metadata(art_cinv: int) -> str:
    """Load article_metadata.csv and return all metadata for the given ART_CINV as JSON."""
    return _json(build_article_metadata_payload(int(art_cinv)))


def get_forecast_data(art_cinv: int) -> str:
    """Load forecast_data.csv filtered to the given ART_CINV and return sorted weekly history as JSON."""
    return _json(build_forecast_data_payload(int(art_cinv)))


def get_article_links(art_cinv: int) -> str:
    """Load links.csv rows where art_cinv_a equals the given article and return them as JSON."""
    return _json(build_article_links_payload(int(art_cinv)))


def _split_week_id(week_id: str | None) -> tuple[int | None, int | None]:
    if not week_id or "-W" not in week_id:
        return None, None
    raw_year, raw_week = week_id.split("-W", maxsplit=1)
    try:
        return int(raw_year), int(raw_week)
    except ValueError:
        return None, None


def _format_week_span(start_week_id: str | None, end_week_id: str | None) -> str | None:
    if not start_week_id or not end_week_id:
        return None
    if start_week_id == end_week_id:
        return start_week_id
    start_year, start_week = _split_week_id(start_week_id)
    end_year, end_week = _split_week_id(end_week_id)
    if start_year is not None and end_year is not None and start_year == end_year and start_week is not None and end_week is not None:
        return f"weeks {start_week}-{end_week} of {start_year}"
    return f"{start_week_id} to {end_week_id}"


def detect_pre_pivot_stockout_risk(art_cinv: int) -> str:
    """Detect zero-shipment streaks before the pivot that can depress the future forecast baseline."""
    frame = get_article_forecast_frame(int(art_cinv)).copy()

    if frame.empty:
        return _json(
            {
                "art_cinv": int(art_cinv),
                "stockout_risk_detected": False,
                "analysis_window_weeks": 0,
                "zero_shipment_weeks_in_window": 0,
                "latest_zero_shipment_streak": None,
                "baseline_reduction_pct": None,
                "interpretation": "No forecast rows were available, so no pre-pivot stockout assessment could be made.",
                "recommended_actions": [],
            }
        )

    pivot_candidates = frame.get("PIVOT_DATE_DT")
    if pivot_candidates is None or pivot_candidates.dropna().empty:
        return _json(
            {
                "art_cinv": int(art_cinv),
                "stockout_risk_detected": False,
                "analysis_window_weeks": 0,
                "zero_shipment_weeks_in_window": 0,
                "latest_zero_shipment_streak": None,
                "baseline_reduction_pct": None,
                "interpretation": "The article has no pivot date, so no pre-pivot stockout assessment could be made.",
                "recommended_actions": [],
            }
        )

    pivot_ts = pivot_candidates.dropna().iloc[-1]
    pre_pivot = frame[frame["MVT_DATE_DT"] < pivot_ts].sort_values("MVT_DATE_DT").copy()
    if pre_pivot.empty:
        return _json(
            {
                "art_cinv": int(art_cinv),
                "stockout_risk_detected": False,
                "analysis_window_weeks": 0,
                "zero_shipment_weeks_in_window": 0,
                "latest_zero_shipment_streak": None,
                "baseline_reduction_pct": None,
                "interpretation": "There are no pre-pivot weeks available for stockout assessment.",
                "recommended_actions": [],
            }
        )

    analysis_window = pre_pivot.tail(13).copy()
    analysis_window["ACTUAL_DEMAND"] = pd.to_numeric(analysis_window["ACTUAL_DEMAND"], errors="coerce").fillna(0.0)
    analysis_window["FORECAST"] = pd.to_numeric(analysis_window["FORECAST"], errors="coerce")
    analysis_window["MOVING_AVG"] = pd.to_numeric(analysis_window["MOVING_AVG"], errors="coerce")

    zero_mask = analysis_window["ACTUAL_DEMAND"].eq(0)
    zero_shipment_weeks = analysis_window.loc[zero_mask, "WEEK_ID"].astype(str).tolist()

    streak_groups = zero_mask.ne(zero_mask.shift(fill_value=False)).cumsum()
    streaks: list[dict[str, Any]] = []
    for _, streak_frame in analysis_window.groupby(streak_groups):
        if not bool(streak_frame["ACTUAL_DEMAND"].eq(0).all()):
            continue
        first_row = streak_frame.iloc[0]
        last_row = streak_frame.iloc[-1]
        streaks.append(
            {
                "count": int(len(streak_frame)),
                "start_week_id": str(first_row["WEEK_ID"]),
                "end_week_id": str(last_row["WEEK_ID"]),
                "start_date": normalize_for_json(first_row["MVT_DATE_DT"]),
                "end_date": normalize_for_json(last_row["MVT_DATE_DT"]),
                "week_ids": streak_frame["WEEK_ID"].astype(str).tolist(),
            }
        )

    latest_streak = streaks[-1] if streaks else None
    recent_non_zero = analysis_window.loc[analysis_window["ACTUAL_DEMAND"] > 0, "ACTUAL_DEMAND"]
    observed_avg = float(analysis_window["ACTUAL_DEMAND"].mean()) if not analysis_window.empty else None
    xout_replacement_units = float(recent_non_zero.mean()) if not recent_non_zero.empty else None
    xout_adjusted_avg = None
    if xout_replacement_units is not None:
        xout_adjusted_series = analysis_window["ACTUAL_DEMAND"].where(~zero_mask, xout_replacement_units)
        xout_adjusted_avg = float(xout_adjusted_series.mean())

    future_horizon = frame[frame["MVT_DATE_DT"] >= pivot_ts].sort_values("MVT_DATE_DT").copy()
    pivot_row = future_horizon.head(1)
    pivot_forecast = None
    pivot_moving_avg = None
    if not pivot_row.empty:
        pivot_forecast = normalize_for_json(pd.to_numeric(pivot_row.iloc[0]["FORECAST"], errors="coerce"))
        pivot_moving_avg = normalize_for_json(pd.to_numeric(pivot_row.iloc[0]["MOVING_AVG"], errors="coerce"))

    future_forecast_values = pd.to_numeric(future_horizon["FORECAST"], errors="coerce").dropna()
    future_forecast_weeks = int(len(future_forecast_values))
    future_forecast_baseline_avg = float(future_forecast_values.mean()) if not future_forecast_values.empty else None
    future_forecast_baseline_total = float(future_forecast_values.sum()) if not future_forecast_values.empty else None

    baseline_reduction_pct = None
    if xout_adjusted_avg not in (None, 0) and future_forecast_baseline_avg is not None:
        baseline_reduction_pct = max(
            0.0,
            float((xout_adjusted_avg - future_forecast_baseline_avg) / xout_adjusted_avg * 100),
        )

    stockout_risk_detected = bool(latest_streak and latest_streak["count"] >= 2 and baseline_reduction_pct is not None and baseline_reduction_pct >= 5)
    affected_period = _format_week_span(
        latest_streak.get("start_week_id") if latest_streak else None,
        latest_streak.get("end_week_id") if latest_streak else None,
    )
    existing_link_rows = int(len(get_article_links_frame(int(art_cinv))))

    recommended_actions: list[str] = []
    if stockout_risk_detected and latest_streak is not None:
        if xout_replacement_units is not None:
            recommended_actions.append(
                f"Apply xout logic to replace the zero-shipment weeks in {affected_period or latest_streak['start_week_id']} with an estimated demand of about {xout_replacement_units:.0f} units per week when rebuilding the baseline."
            )
        recommended_actions.append(
            f"Create or extend an article link for {affected_period or latest_streak['start_week_id']} so the baseline can borrow demand from a similar article during the affected period."
        )

    if stockout_risk_detected and latest_streak is not None:
        adjusted_baseline = float(xout_adjusted_avg) if xout_adjusted_avg is not None else 0.0
        future_baseline_avg = float(future_forecast_baseline_avg) if future_forecast_baseline_avg is not None else 0.0
        interpretation = (
            f"Article CINV {art_cinv} shows {latest_streak['count']} consecutive pre-pivot weeks with zero shipments from "
            f"{latest_streak['start_week_id']} to {latest_streak['end_week_id']}. This pattern is more consistent with a shortage, stockout, or underlying data issue than a true collapse in demand. "
            f"Using a 13-week pre-pivot baseline window, replacing those zero weeks with xout demand estimates implies an adjusted baseline of about {adjusted_baseline:.2f} units, versus an average future forecast baseline of {future_baseline_avg:.2f} units across {future_forecast_weeks} forecasted weeks. "
            f"That means the future baseline may be artificially lowered by about {baseline_reduction_pct:.2f}%."
        )
    elif latest_streak is not None:
        interpretation = (
            f"Article CINV {art_cinv} contains a pre-pivot zero-shipment streak, but the measured baseline distortion did not exceed the alert threshold. "
            f"The latest streak covered {latest_streak['count']} weeks from {latest_streak['start_week_id']} to {latest_streak['end_week_id']}."
        )
    else:
        interpretation = f"No material pre-pivot zero-shipment streaks were detected for article CINV {art_cinv}."

    reporting_guidance = None
    if stockout_risk_detected and latest_streak is not None and baseline_reduction_pct is not None:
        reporting_guidance = (
            f"For article CINV {art_cinv}, I identified {latest_streak['count']} consecutive weeks of zero shipments before the pivot "
            f"({latest_streak['start_week_id']} to {latest_streak['end_week_id']}). This pattern suggests a stockout, shortage, or underlying data issue, "
            f"which artificially lowers the average future forecast baseline by approximately {baseline_reduction_pct:.2f}% across {future_forecast_weeks} forecasted weeks. "
            f"The recommended solution is to either apply the xout logic or create an article link for the affected period {affected_period}."
        )

    return _json(
        {
            "art_cinv": int(art_cinv),
            "pivot_date": normalize_for_json(pivot_ts),
            "analysis_window_weeks": int(len(analysis_window)),
            "zero_shipment_weeks_in_window": int(zero_mask.sum()),
            "zero_shipment_week_ids": zero_shipment_weeks,
            "latest_zero_shipment_streak": latest_streak,
            "stockout_risk_detected": stockout_risk_detected,
            "affected_period": affected_period,
            "observed_window_avg_demand": round(observed_avg, 2) if observed_avg is not None else None,
            "xout_replacement_units": round(xout_replacement_units, 2) if xout_replacement_units is not None else None,
            "xout_adjusted_window_avg_demand": round(xout_adjusted_avg, 2) if xout_adjusted_avg is not None else None,
            "pivot_forecast_baseline": round(float(pivot_forecast), 2) if pivot_forecast is not None else None,
            "pivot_moving_avg_baseline": round(float(pivot_moving_avg), 2) if pivot_moving_avg is not None else None,
            "future_forecast_weeks": future_forecast_weeks,
            "future_forecast_baseline_avg": round(future_forecast_baseline_avg, 2) if future_forecast_baseline_avg is not None else None,
            "future_forecast_baseline_total": round(future_forecast_baseline_total, 2) if future_forecast_baseline_total is not None else None,
            "baseline_reduction_pct": round(baseline_reduction_pct, 2) if baseline_reduction_pct is not None else None,
            "existing_link_rows": existing_link_rows,
            "possible_causes": ["stockout_or_shortage", "underlying_data_issue"] if latest_streak is not None else [],
            "recommended_actions": recommended_actions,
            "reporting_guidance": reporting_guidance,
            "interpretation": interpretation,
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
    """Detect statistically unusual demand weeks for an article.

    The baseline keeps the historical IQR and z-score rules for compatibility, then
    adds more robust signals for genuinely volatile series where repeated spikes can
    inflate the global variance and hide structurally anomalous weeks.
    """
    frame = get_article_forecast_frame(int(art_cinv)).copy()
    clean = frame[frame["ACTUAL_DEMAND"] > 0].sort_values("MVT_DATE_DT").copy()

    if clean.empty:
        return _json(
            {
                "art_cinv": int(art_cinv),
                "count": 0,
                "outliers": [],
                "summary": {
                    "weeks_evaluated": 0,
                    "volatility_level": "INSUFFICIENT_DATA",
                },
                "volatile_periods": [],
            }
        )

    actual = clean["ACTUAL_DEMAND"].astype(float)
    q1 = actual.quantile(0.25)
    q3 = actual.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    std = float(actual.std(ddof=0)) if len(actual) > 1 else 0.0
    mean = float(actual.mean())
    median = float(actual.median())
    cv = float(std / mean) if mean and not np.isclose(mean, 0.0) else 0.0
    abs_pct_change = actual.pct_change().abs().replace([np.inf, -np.inf], np.nan)
    median_abs_pct_change = float(abs_pct_change.dropna().median()) if abs_pct_change.notna().any() else 0.0

    if std and not np.isclose(std, 0):
        clean["z_score"] = stats.zscore(actual, nan_policy="omit")
    else:
        clean["z_score"] = 0.0

    mad = float((actual - median).abs().median()) if len(actual) > 1 else 0.0
    if mad and not np.isclose(mad, 0.0):
        clean["modified_z_score"] = 0.6745 * (actual - median) / mad
    else:
        clean["modified_z_score"] = 0.0

    clean["rolling_median_baseline"] = actual.rolling(6, min_periods=4).median().shift(1)
    clean["rolling_median_ratio"] = np.where(
        clean["rolling_median_baseline"].notna() & ~np.isclose(clean["rolling_median_baseline"], 0.0),
        actual / clean["rolling_median_baseline"],
        np.nan,
    )

    historical_baseline = clean[["NM1", "NM2", "NM3"]].replace(0, np.nan).mean(axis=1)
    clean["historical_baseline"] = historical_baseline
    clean["historical_ratio"] = np.where(
        historical_baseline.notna() & ~np.isclose(historical_baseline, 0.0),
        actual / historical_baseline,
        np.nan,
    )

    clean["iqr_outlier"] = (actual < lower) | (actual > upper)
    clean["z_outlier"] = clean["z_score"].abs() > 2
    clean["modified_z_outlier"] = clean["modified_z_score"].abs() > 3.5

    high_volatility_series = cv >= 0.4 or median_abs_pct_change >= 0.45
    clean["rolling_baseline_outlier"] = (
        clean["rolling_median_ratio"].notna()
        & ((clean["rolling_median_ratio"] >= 1.9) | (clean["rolling_median_ratio"] <= 0.53))
        & clean["modified_z_outlier"]
    )
    clean["historical_baseline_outlier"] = (
        clean["historical_ratio"].notna()
        & ((clean["historical_ratio"] >= 1.9) | (clean["historical_ratio"] <= 0.53))
        & clean["modified_z_outlier"]
    )
    clean["enhanced_outlier"] = bool(high_volatility_series) & (
        clean["rolling_baseline_outlier"] | clean["historical_baseline_outlier"]
    )

    recovery_suppressed_count = 0
    recovery_context = json.loads(detect_pre_pivot_stockout_risk(int(art_cinv)))
    latest_zero_streak = recovery_context.get("latest_zero_shipment_streak") or {}
    recovery_baseline_raw = recovery_context.get("xout_replacement_units")
    recovery_baseline = None
    try:
        recovery_baseline = float(recovery_baseline_raw) if recovery_baseline_raw is not None else None
    except (TypeError, ValueError):
        recovery_baseline = None

    recovery_mask = pd.Series(False, index=clean.index)
    zero_streak_end = pd.to_datetime(latest_zero_streak.get("end_date"), errors="coerce")
    if pd.notna(zero_streak_end) and recovery_baseline and not np.isclose(recovery_baseline, 0.0):
        recovery_window_end = zero_streak_end + pd.Timedelta(days=42)
        recovery_mask = (
            clean["MVT_DATE_DT"].gt(zero_streak_end)
            & clean["MVT_DATE_DT"].le(recovery_window_end)
            & actual.ge(recovery_baseline * 0.9)
            & actual.le(recovery_baseline * 1.15)
            & actual.lt(mean)
            & ~clean["rolling_baseline_outlier"]
            & ~clean["historical_baseline_outlier"]
        )

    outliers = clean[
        clean["iqr_outlier"]
        | clean["z_outlier"]
        | clean["enhanced_outlier"]
    ].copy()
    if recovery_mask.any():
        recovery_suppressed_count = int(outliers.index.isin(clean.index[recovery_mask]).sum())
        if recovery_suppressed_count:
            outliers = outliers.loc[~outliers.index.isin(clean.index[recovery_mask])].copy()

    detection_notes = [
        "Base detection uses global IQR fences and z-score thresholds for backward compatibility.",
    ]
    if high_volatility_series:
        detection_notes.append(
            "High-volatility enhancement activated because repeated spikes widened the global distribution. "
            "Robust median/MAD, recent rolling-baseline ratios, and same-week NM history are used to recover concealed spikes."
        )
    if recovery_suppressed_count:
        detection_notes.append(
            f"Suppressed {recovery_suppressed_count} low-demand recovery week(s) immediately after the latest zero-shipment streak because demand had already returned close to the xout recovery baseline."
        )

    def _outlier_methods(row: Any) -> list[str]:
        methods: list[str] = []
        if bool(row.iqr_outlier):
            methods.append("iqr")
        if bool(row.z_outlier):
            methods.append("z_score")
        if bool(row.modified_z_outlier):
            methods.append("modified_z_score")
        if bool(row.rolling_baseline_outlier):
            methods.append("recent_baseline")
        if bool(row.historical_baseline_outlier):
            methods.append("historical_baseline")
        return methods

    def _severity(methods: list[str], modified_z_score: float) -> str:
        if len(methods) >= 3 or abs(modified_z_score) >= 6:
            return "SEVERE"
        if len(methods) >= 2 or abs(modified_z_score) >= 4:
            return "HIGH"
        return "MODERATE"

    records: list[dict[str, Any]] = []
    for row in outliers.sort_values("modified_z_score", key=lambda column: column.abs(), ascending=False).itertuples():
        holiday_name = _clean_label(row.HOLIDAYS_TYPE)
        promo_name = _clean_label(row.PROMO_TYPE)
        methods = _outlier_methods(row)
        records.append(
            {
                "week_id": row.WEEK_ID,
                "MVT_DATE": normalize_for_json(row.MVT_DATE_DT),
                "ACTUAL_DEMAND": normalize_for_json(row.ACTUAL_DEMAND),
                "FORECAST": normalize_for_json(row.FORECAST),
                "direction": "HIGH" if float(row.ACTUAL_DEMAND) >= mean else "LOW",
                "z_score": round(float(row.z_score), 3),
                "modified_z_score": round(float(row.modified_z_score), 3),
                "iqr_outlier": bool(row.iqr_outlier),
                "z_outlier": bool(row.z_outlier),
                "modified_z_outlier": bool(row.modified_z_outlier),
                "rolling_median_ratio": round(float(row.rolling_median_ratio), 3) if pd.notna(row.rolling_median_ratio) else None,
                "rolling_baseline_outlier": bool(row.rolling_baseline_outlier),
                "historical_ratio": round(float(row.historical_ratio), 3) if pd.notna(row.historical_ratio) else None,
                "historical_baseline_outlier": bool(row.historical_baseline_outlier),
                "outlier_methods": methods,
                "severity": _severity(methods, float(row.modified_z_score)),
                "has_promo": bool(promo_name),
                "promo_name": promo_name,
                "has_holiday": bool(holiday_name),
                "holiday_name": holiday_name,
            }
        )

    high_outliers = [item for item in records if item.get("direction") == "HIGH"]
    low_outliers = [item for item in records if item.get("direction") == "LOW"]
    max_spike = max(high_outliers, key=lambda item: float(item.get("ACTUAL_DEMAND") or 0), default=None)

    grouped_periods: list[list[dict[str, Any]]] = []
    for item in sorted(records, key=lambda candidate: str(candidate.get("MVT_DATE") or "")):
        if not grouped_periods:
            grouped_periods.append([item])
            continue
        previous = grouped_periods[-1][-1]
        previous_ts = pd.to_datetime(previous.get("MVT_DATE"), errors="coerce")
        current_ts = pd.to_datetime(item.get("MVT_DATE"), errors="coerce")
        if pd.notna(previous_ts) and pd.notna(current_ts) and (current_ts - previous_ts).days <= 35:
            grouped_periods[-1].append(item)
        else:
            grouped_periods.append([item])

    volatile_periods: list[dict[str, Any]] = []
    for group in grouped_periods:
        if not group:
            continue
        high_count = sum(1 for item in group if item.get("direction") == "HIGH")
        low_count = len(group) - high_count
        dominant_pattern = "repeating spikes" if high_count > low_count else "repeating troughs" if low_count > high_count else "mixed volatility"
        volatile_periods.append(
            {
                "start_week_id": group[0].get("week_id"),
                "end_week_id": group[-1].get("week_id"),
                "start_date": group[0].get("MVT_DATE"),
                "end_date": group[-1].get("MVT_DATE"),
                "outlier_count": len(group),
                "high_outlier_count": high_count,
                "low_outlier_count": low_count,
                "dominant_pattern": dominant_pattern,
                "representative_weeks": [item.get("week_id") for item in group[:4] if item.get("week_id")],
            }
        )

    if len(records) >= max(8, int(len(clean) * 0.18)) or (cv >= 0.4 and len(records) >= max(4, int(len(clean) * 0.05))):
        volatility_level = "HIGH"
    elif cv >= 0.25 or len(records) >= max(5, int(len(clean) * 0.1)):
        volatility_level = "MODERATE"
    else:
        volatility_level = "LOW"

    summary = {
        "weeks_evaluated": int(len(clean)),
        "outlier_share_pct": round(float(len(records) / len(clean) * 100), 2),
        "high_outlier_count": int(len(high_outliers)),
        "low_outlier_count": int(len(low_outliers)),
        "mean_demand": round(mean, 2),
        "median_demand": round(median, 2),
        "std_demand": round(std, 2),
        "coefficient_of_variation": round(cv, 3),
        "median_abs_weekly_change_pct": round(median_abs_pct_change * 100, 2),
        "volatility_level": volatility_level,
        "high_volatility_series": bool(high_volatility_series),
        "peak_spike_week_id": max_spike.get("week_id") if max_spike else None,
        "peak_spike_units": max_spike.get("ACTUAL_DEMAND") if max_spike else None,
    }

    return _json(
        {
            "art_cinv": int(art_cinv),
            "count": len(records),
            "outliers": records,
            "summary": summary,
            "volatile_periods": volatile_periods,
            "detection_notes": detection_notes,
        }
    )