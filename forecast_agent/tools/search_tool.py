"""Tavily-backed market and demand characteristic search tool."""

from __future__ import annotations

import asyncio
import json
import re
from functools import lru_cache
from textwrap import dedent
from typing import Any

from agent_framework import AgentResponseUpdate
from tavily import TavilyClient

from .. import config
from ..data_access import get_article_forecast_frame, get_article_metadata_frame, normalize_for_json
from ..responses_client import get_responses_client

SEARCH_SUFFIX = "supply chain seasonality demand drivers Ireland retail"

PRODUCT_WEATHER_SIGNALS = {
    "ice cube": 4,
    "ice cubes": 4,
    "ice cream": 4,
    "barbecue": 4,
    "bbq": 4,
    "grill": 3,
    "fan": 4,
    "heater": 4,
    "umbrella": 4,
    "raincoat": 4,
    "sunscreen": 4,
    "sun cream": 4,
    "de-icer": 4,
    "deicer": 4,
    "soft drink": 3,
    "soda": 2,
    "water": 2,
    "beer": 2,
    "soup": 2,
    "tea": 1,
    "coffee": 1,
}

NON_WEATHER_STAPLE_SIGNALS = {
    "toilet tissue": 5,
    "toilet paper": 5,
    "tissue paper": 4,
    "paper towel": 4,
    "paper towels": 4,
    "kitchen roll": 4,
    "napkin": 3,
    "napkins": 3,
    "household paper": 4,
    "detergent": 3,
    "laundry": 2,
    "cleaning": 2,
    "soap": 2,
    "shampoo": 2,
    "toothpaste": 2,
}

EVIDENCE_WEATHER_SIGNALS = {
    "weather": 2,
    "temperature": 2,
    "heatwave": 3,
    "hot weather": 3,
    "warm weather": 2,
    "cold weather": 3,
    "cold spell": 3,
    "rain": 2,
    "rainfall": 2,
    "snow": 3,
    "storm": 3,
    "wind": 2,
    "sunny": 2,
    "summer": 1,
    "winter": 1,
}

WEATHER_DEMAND_DRIVER_SIGNALS = {
    "weather-sensitive": 4,
    "weather sensitive": 4,
    "temperature-sensitive": 4,
    "temperature sensitive": 4,
    "hot weather increases demand": 5,
    "warm weather increases demand": 5,
    "cold weather increases demand": 5,
    "demand rises in hot weather": 5,
    "demand rises during hot weather": 5,
    "demand rises in warm weather": 5,
    "demand falls in cold weather": 4,
    "demand increases in summer": 4,
    "summer demand spike": 4,
    "heatwave demand": 4,
    "rain drives demand": 4,
    "weather-driven demand": 5,
    "weather driven demand": 5,
    "temperature affects demand": 4,
    "temperature-driven demand": 4,
}

HOLIDAY_DEMAND_POSITIVE_SIGNALS = {
    "increase": 2,
    "increased demand": 3,
    "higher demand": 3,
    "uplift": 3,
    "boost": 2,
    "spike": 3,
    "peak": 2,
    "seasonal": 1,
    "festive": 2,
    "celebration": 1,
    "party": 1,
    "hosting": 1,
    "gift": 1,
    "stock up": 2,
    "baking": 1,
    "entertaining": 1,
}

HOLIDAY_DEMAND_NEGATIVE_SIGNALS = {
    "no impact": 3,
    "little impact": 2,
    "not affected": 3,
    "unrelated": 2,
    "not seasonal": 2,
    "decline": 2,
    "lower demand": 3,
}

_WEATHER_DECISION_SYSTEM_PROMPT = dedent(
        """
        You decide whether weather enrichment should run for a retail demand forecast analysis.

        Return JSON only using this schema exactly:
        {
            "schema_version": "weather_decision_v2",
            "classification": "likely" | "possible" | "unlikely",
            "recommended": <true|false>,
            "confidence": "high" | "medium" | "low",
            "demand_effect": "positive" | "negative" | "mixed" | "none" | "unclear",
            "evidence_summary": <string>,
            "rationale": <string>,
            "matched_signals": [<string>, ...],
            "blocking_signals": [<string>, ...],
            "supporting_evidence": [<string>, ...]
        }

        Requirements:
        - All fields are required.
        - Arrays must contain short plain-text strings only.
        - `recommended` may be true only when classification is "likely".
        - If `recommended` is true, confidence must be "high" or "medium", demand_effect cannot be "none" or "unclear", and supporting_evidence must not be empty.
        - If evidence is mixed or weak, use classification "possible" and recommended false.
        - If evidence is absent or points away from weather-driven demand, use classification "unlikely" and recommended false.

        Decision rules:
        - Recommend weather enrichment only when customer demand itself is materially affected by weather.
        - Do not recommend weather enrichment for staple or household products when the evidence only shows generic seasonality, storms, or operational disruption.
        - Generic mentions of weather, seasons, or disruption are insufficient without a plausible demand effect.
        - Use only the supplied context and evidence. Do not invent facts.
        - Do not wrap the JSON in markdown.
        """
).strip()

_WEATHER_DECISION_SCHEMA_VERSION = "weather_decision_v2"
_WEATHER_CLASSIFICATIONS = {"likely", "possible", "unlikely"}
_WEATHER_CONFIDENCE_LEVELS = {"high", "medium", "low"}
_WEATHER_DEMAND_EFFECTS = {"positive", "negative", "mixed", "none", "unclear"}


def _extract_cinv_candidates(query: str) -> list[int]:
    return [int(match) for match in re.findall(r"\b\d{5,}\b", query or "")]


def _clean_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).strip()


def _collect_signal_hits(text: str, signal_weights: dict[str, int]) -> list[tuple[str, int]]:
    lowered = (text or "").lower()
    hits: list[tuple[str, int]] = []
    for signal, weight in signal_weights.items():
        if signal in lowered:
            hits.append((signal, weight))
    return hits


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    candidate = match.group(0) if match else cleaned
    payload = json.loads(candidate)
    return payload if isinstance(payload, dict) else {"raw": payload}


def _truncate_text(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)].rstrip() + "..."


def _compact_sources_for_prompt(sources: list[dict[str, Any]], *, max_items: int = 5) -> list[dict[str, str]]:
    compact_sources: list[dict[str, str]] = []
    for item in sources[:max_items]:
        compact_sources.append(
            {
                "title": _truncate_text(str(item.get("title", "")), 160),
                "url": _truncate_text(str(item.get("url", "")), 200),
                "snippet": _truncate_text(str(item.get("snippet", "")), 320),
            }
        )
    return compact_sources


def _collect_unique_holidays(art_cinv: int) -> tuple[list[str], list[dict[str, Any]], str | None]:
    frame = get_article_forecast_frame(int(art_cinv)).copy()
    if frame.empty:
        return [], [], None

    pivot_ts = None
    if "PIVOT_DATE_DT" in frame.columns and frame["PIVOT_DATE_DT"].notna().any():
        pivot_ts = frame["PIVOT_DATE_DT"].dropna().iloc[-1]

    holiday_weeks: dict[str, list[dict[str, Any]]] = {}
    for row in frame.itertuples():
        holiday_name = str(getattr(row, "HOLIDAYS_TYPE", "") or "").strip()
        if not holiday_name:
            continue

        mvt_date = getattr(row, "MVT_DATE_DT", None)
        relation_to_pivot = "historical"
        if pivot_ts is not None and mvt_date is not None:
            if mvt_date > pivot_ts:
                relation_to_pivot = "future"
            elif mvt_date == pivot_ts:
                relation_to_pivot = "pivot_week"

        holiday_weeks.setdefault(holiday_name, []).append(
            {
                "week_id": getattr(row, "WEEK_ID", None),
                "MVT_DATE": normalize_for_json(mvt_date),
                "relation_to_pivot": relation_to_pivot,
            }
        )

    holiday_summaries: list[dict[str, Any]] = []
    for holiday_name, occurrences in holiday_weeks.items():
        holiday_summaries.append(
            {
                "holiday_name": holiday_name,
                "occurrence_count": len(occurrences),
                "historical_occurrences": sum(1 for week in occurrences if week.get("relation_to_pivot") == "historical"),
                "future_occurrences": sum(1 for week in occurrences if week.get("relation_to_pivot") == "future"),
                "pivot_week_occurrences": sum(1 for week in occurrences if week.get("relation_to_pivot") == "pivot_week"),
                "first_occurrence": occurrences[0].get("MVT_DATE") if occurrences else None,
                "last_occurrence": occurrences[-1].get("MVT_DATE") if occurrences else None,
                "sample_weeks": occurrences[:3],
            }
        )

    holiday_summaries.sort(key=lambda item: item["holiday_name"])
    return [item["holiday_name"] for item in holiday_summaries], holiday_summaries, normalize_for_json(pivot_ts)


def _run_tavily_search(query: str, *, max_results: int = 5) -> tuple[dict[str, Any] | None, str | None]:
    if not config.TAVILY_API_KEY:
        return None, "TAVILY_API_KEY is not configured."

    try:
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        result = client.search(
            query=query,
            search_depth="advanced",
            topic="general",
            max_results=max_results,
            include_answer="advanced",
            include_raw_content=False,
            timeout=45,
        )
        return result, None
    except Exception as exc:
        return None, str(exc)


def _extract_sources(result: dict[str, Any], *, max_results: int = 5) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for item in result.get("results", [])[:max_results]:
        sources.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "") or item.get("snippet", ""),
            }
        )
    return sources


def _assess_holiday_demand_relevance(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_text = " ".join(
        [answer or ""] + [f"{item.get('title', '')} {item.get('snippet', '')}".strip() for item in sources]
    ).lower()
    positive_hits = _collect_signal_hits(evidence_text, HOLIDAY_DEMAND_POSITIVE_SIGNALS)
    negative_hits = _collect_signal_hits(evidence_text, HOLIDAY_DEMAND_NEGATIVE_SIGNALS)
    positive_score = sum(weight for _, weight in positive_hits)
    negative_score = sum(weight for _, weight in negative_hits)
    net_score = positive_score - negative_score

    if positive_score >= 5 and net_score >= 3:
        classification = "likely_increase"
        rationale = "Search evidence suggests this holiday is associated with stronger demand for the article category or use case."
    elif positive_score >= 2 and net_score > 0:
        classification = "possible_increase"
        rationale = "Search evidence suggests a plausible holiday-related uplift, but the link is not strong."
    elif negative_score >= 3 and net_score <= 0:
        classification = "unlikely_increase"
        rationale = "Search evidence suggests limited or no clear holiday-related uplift for this article."
    else:
        classification = "unclear"
        rationale = "The search evidence is mixed or too weak to establish a clear holiday-demand relationship."

    return {
        "classification": classification,
        "positive_signals": [signal for signal, _ in positive_hits[:6]],
        "negative_signals": [signal for signal, _ in negative_hits[:6]],
        "rationale": rationale,
    }


def _assess_weather_sensitivity(
    original_query: str,
    rewrite_info: dict[str, Any],
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    article_context = " ".join(
        part for part in [rewrite_info.get("article_name", ""), rewrite_info.get("category_text", ""), original_query] if part
    )
    evidence_text = " ".join(
        [answer or ""]
        + [f"{item.get('title', '')} {item.get('snippet', '')}".strip() for item in sources]
    )

    product_hits = _collect_signal_hits(article_context, PRODUCT_WEATHER_SIGNALS)
    staple_hits = _collect_signal_hits(article_context, NON_WEATHER_STAPLE_SIGNALS)
    evidence_hits = _collect_signal_hits(evidence_text, EVIDENCE_WEATHER_SIGNALS)
    demand_driver_hits = _collect_signal_hits(evidence_text, WEATHER_DEMAND_DRIVER_SIGNALS)

    merged_hits: dict[str, int] = {}
    for signal, weight in product_hits + evidence_hits + demand_driver_hits:
        merged_hits[signal] = max(weight, merged_hits.get(signal, 0))

    product_score = sum(weight for _, weight in product_hits)
    staple_score = sum(weight for _, weight in staple_hits)
    generic_evidence_score = sum(weight for _, weight in evidence_hits)
    demand_driver_score = sum(weight for _, weight in demand_driver_hits)
    score = sum(merged_hits.values())
    strong_product_signal = any(weight >= 4 for _, weight in product_hits)
    direct_weather_demand_evidence = demand_driver_score >= 4
    meaningful_weather_evidence = generic_evidence_score >= 4
    blocked_as_staple = staple_score >= 4 and not strong_product_signal and not direct_weather_demand_evidence

    if blocked_as_staple:
        classification = "unlikely"
        recommended = False
        rationale = (
            "Article context points to a staple household product, and the available search evidence does not show that customer demand itself is materially weather-driven. "
            "Generic weather or storm mentions are not enough to justify weather enrichment."
        )
    elif strong_product_signal and staple_score == 0:
        classification = "likely"
        recommended = True
        rationale = "Article context strongly suggests that demand is materially weather-sensitive."
    elif direct_weather_demand_evidence and (product_score >= 2 or meaningful_weather_evidence):
        classification = "likely"
        recommended = True
        rationale = "Search evidence explicitly links demand changes to weather conditions, so weather enrichment is warranted."
    elif demand_driver_score >= 2 or (product_score >= 2 and generic_evidence_score >= 2):
        classification = "possible"
        recommended = False
        rationale = (
            "Some weather-linked signals were found, but the evidence is not strong enough to show that demand is materially weather-driven."
        )
    else:
        classification = "unlikely"
        recommended = False
        rationale = "No strong evidence of material weather sensitivity was found in the available article context."

    matched_signals = [signal for signal, _ in sorted(merged_hits.items(), key=lambda item: (-item[1], item[0]))[:6]]
    return {
        "classification": classification,
        "recommended": recommended,
        "schema_version": _WEATHER_DECISION_SCHEMA_VERSION,
        "decision_source": "heuristic_fallback",
        "confidence": "medium",
        "demand_effect": "unclear",
        "evidence_summary": rationale,
        "score": score,
        "product_score": product_score,
        "evidence_score": generic_evidence_score,
        "demand_driver_score": demand_driver_score,
        "staple_score": staple_score,
        "matched_signals": matched_signals,
        "blocking_signals": [signal for signal, _ in staple_hits[:6]],
        "supporting_evidence": matched_signals[:4],
        "rationale": rationale,
    }


def _normalize_llm_weather_sensitivity(payload: dict[str, Any]) -> dict[str, Any]:
    required_fields = {
        "schema_version",
        "classification",
        "recommended",
        "confidence",
        "demand_effect",
        "evidence_summary",
        "rationale",
        "matched_signals",
        "blocking_signals",
        "supporting_evidence",
    }
    missing_fields = sorted(required_fields.difference(payload))
    if missing_fields:
        raise ValueError(f"Weather decision payload is missing required fields: {', '.join(missing_fields)}")

    schema_version = str(payload["schema_version"] or "").strip()
    if schema_version != _WEATHER_DECISION_SCHEMA_VERSION:
        raise ValueError(f"Unexpected weather decision schema version: {schema_version or '<empty>'}")

    classification = str(payload["classification"] or "").strip().lower()
    if classification not in _WEATHER_CLASSIFICATIONS:
        raise ValueError(f"Invalid weather decision classification: {classification or '<empty>'}")

    recommended = payload["recommended"]
    if not isinstance(recommended, bool):
        raise ValueError("Weather decision recommended field must be a boolean")

    confidence = str(payload["confidence"] or "").strip().lower()
    if confidence not in _WEATHER_CONFIDENCE_LEVELS:
        raise ValueError(f"Invalid weather decision confidence: {confidence or '<empty>'}")

    demand_effect = str(payload["demand_effect"] or "").strip().lower()
    if demand_effect not in _WEATHER_DEMAND_EFFECTS:
        raise ValueError(f"Invalid weather decision demand_effect: {demand_effect or '<empty>'}")

    evidence_summary = str(payload["evidence_summary"] or "").strip()
    if not evidence_summary:
        raise ValueError("Weather decision evidence_summary must be a non-empty string")

    rationale = str(payload["rationale"] or "").strip()
    if not rationale:
        raise ValueError("Weather decision rationale must be a non-empty string")

    def _normalize_string_list(field_name: str) -> list[str]:
        raw_value = payload[field_name]
        if not isinstance(raw_value, list):
            raise ValueError(f"Weather decision {field_name} must be an array")
        normalized_items: list[str] = []
        for item in raw_value:
            if not isinstance(item, str):
                raise ValueError(f"Weather decision {field_name} must contain strings only")
            normalized_item = item.strip()
            if normalized_item:
                normalized_items.append(normalized_item)
        return normalized_items[:6]

    matched_signals = _normalize_string_list("matched_signals")
    blocking_signals = _normalize_string_list("blocking_signals")
    supporting_evidence = _normalize_string_list("supporting_evidence")

    if recommended and classification != "likely":
        raise ValueError("Weather decision recommended=true requires classification='likely'")
    if classification in {"possible", "unlikely"} and recommended:
        raise ValueError("Weather decision classification='possible' or 'unlikely' cannot recommend enrichment")
    if recommended and confidence == "low":
        raise ValueError("Weather decision recommended=true requires medium or high confidence")
    if recommended and demand_effect in {"none", "unclear"}:
        raise ValueError("Weather decision recommended=true requires a concrete demand_effect")
    if recommended and not supporting_evidence:
        raise ValueError("Weather decision recommended=true requires supporting_evidence entries")

    return {
        "classification": classification,
        "recommended": recommended,
        "schema_version": schema_version,
        "decision_source": "llm",
        "confidence": confidence,
        "demand_effect": demand_effect,
        "evidence_summary": evidence_summary,
        "score": None,
        "product_score": None,
        "evidence_score": None,
        "demand_driver_score": None,
        "staple_score": None,
        "matched_signals": matched_signals,
        "blocking_signals": blocking_signals,
        "supporting_evidence": supporting_evidence,
        "rationale": rationale,
    }


async def _assess_weather_sensitivity_llm(
    original_query: str,
    rewrite_info: dict[str, Any],
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    client = get_responses_client()
    agent = client.as_agent(
        name="WeatherSensitivityDecision",
        instructions=_WEATHER_DECISION_SYSTEM_PROMPT,
        tools=[],
    )

    prompt = dedent(
        f"""
        Decide whether weather enrichment is warranted for this article.

        ARTICLE_CONTEXT:
        {json.dumps({
            "original_query": _truncate_text(original_query, 200),
            "article_name": rewrite_info.get("article_name", ""),
            "category_text": rewrite_info.get("category_text", ""),
            "rewritten_from_cinv": bool(rewrite_info.get("rewritten_from_cinv")),
            "source_cinv": rewrite_info.get("source_cinv"),
        }, ensure_ascii=True)}

        SEARCH_EVIDENCE:
        {json.dumps({
            "answer_excerpt": _truncate_text(answer, 1200),
            "sources": _compact_sources_for_prompt(sources),
        }, ensure_ascii=True)}
        """
    ).strip()

    chunks: list[str] = []
    stream = agent.run(prompt, stream=True)
    async for update in stream:
        if not isinstance(update, AgentResponseUpdate):
            continue
        for content in update.contents:
            if getattr(content, "type", None) != "text":
                continue
            chunk = getattr(content, "text", "") or ""
            if chunk:
                chunks.append(chunk)

    final_response = await stream.get_final_response()
    final_text = getattr(final_response, "text", None) or "".join(chunks)
    return _normalize_llm_weather_sensitivity(_extract_json_object(final_text))


def _decide_weather_sensitivity(
    original_query: str,
    rewrite_info: dict[str, Any],
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        return asyncio.run(_assess_weather_sensitivity_llm(original_query, rewrite_info, answer, sources))
    except Exception as exc:
        fallback = _assess_weather_sensitivity(original_query, rewrite_info, answer, sources)
        fallback["llm_error"] = str(exc)
        return fallback


@lru_cache(maxsize=128)
def _article_search_context(art_cinv: int) -> dict[str, str]:
    frame = get_article_metadata_frame(int(art_cinv))
    if frame.empty:
        return {}
    first = frame.iloc[0]
    article_name = _clean_query_text(str(first.get("ART_DESC") or ""))
    category_parts = [
        _clean_query_text(str(first.get("ART_LEVEL1_DESC") or "")),
        _clean_query_text(str(first.get("ART_LEVEL2_DESC") or "")),
        _clean_query_text(str(first.get("ART_LEVEL3_DESC") or "")),
    ]
    category_text = " ".join(part for part in category_parts if part)
    return {"article_name": article_name, "category_text": category_text}


def _normalize_search_query(query: str) -> tuple[str, dict[str, Any]]:
    original_query = _clean_query_text(query)
    candidates = _extract_cinv_candidates(original_query)

    for candidate in candidates:
        context = _article_search_context(candidate)
        article_name = context.get("article_name")
        if not article_name:
            continue
        category_text = context.get("category_text", "")
        normalized_query = " ".join(part for part in [article_name, category_text, SEARCH_SUFFIX] if part)
        return _clean_query_text(normalized_query), {
            "rewritten_from_cinv": True,
            "source_cinv": candidate,
            "article_name": article_name,
            "category_text": category_text,
        }

    return original_query, {
        "rewritten_from_cinv": False,
        "source_cinv": None,
        "article_name": "",
        "category_text": "",
    }


def search_article_characteristics(query: str) -> str:
    """Search the web for demand drivers, seasonality, and supply-chain characteristics for an article."""
    normalized_query, rewrite_info = _normalize_search_query(query)
    weather_sensitivity = _decide_weather_sensitivity(query, rewrite_info, "", [])

    if not config.TAVILY_API_KEY:
        return json.dumps(
            {
                "query": normalized_query,
                "original_query": query,
                "answer": "",
                "sources": [],
                "search_available": False,
                "error": "TAVILY_API_KEY is not configured.",
                "weather_enrichment_recommended": weather_sensitivity["recommended"],
                "weather_sensitivity": weather_sensitivity,
                **rewrite_info,
            }
        )

    result, error = _run_tavily_search(normalized_query, max_results=5)
    if error:
        return json.dumps(
            {
                "query": normalized_query,
                "original_query": query,
                "answer": "",
                "sources": [],
                "search_available": False,
                "error": error,
                "weather_enrichment_recommended": weather_sensitivity["recommended"],
                "weather_sensitivity": weather_sensitivity,
                **rewrite_info,
            }
        )

    assert result is not None
    sources = _extract_sources(result, max_results=5)

    weather_sensitivity = _decide_weather_sensitivity(query, rewrite_info, result.get("answer", "") or "", sources)

    return json.dumps(
        {
            "query": normalized_query,
            "original_query": query,
            "answer": result.get("answer", "") or "",
            "sources": sources,
            "search_available": True,
            "weather_enrichment_recommended": weather_sensitivity["recommended"],
            "weather_sensitivity": weather_sensitivity,
            **rewrite_info,
        }
    )


def search_holiday_demand_correlation(art_cinv: int) -> str:
    """Search whether holidays across the article history are associated with increased demand for the article."""
    context = _article_search_context(int(art_cinv))
    article_name = context.get("article_name", "")
    category_text = context.get("category_text", "")
    holidays, holiday_summaries, pivot_date = _collect_unique_holidays(int(art_cinv))

    query_parts = [
        article_name,
        category_text,
        "Ireland retail holiday special event demand uplift seasonality correlation",
    ]
    if holidays:
        query_parts.append("observed events " + ", ".join(holidays[:8]))
    query = _clean_query_text(" ".join(part for part in query_parts if part))

    result, error = _run_tavily_search(query, max_results=5)
    answer = ""
    sources: list[dict[str, Any]] = []
    if result is not None:
        answer = result.get("answer", "") or ""
        sources = _extract_sources(result, max_results=5)

    if not holidays:
        return json.dumps(
            {
                "art_cinv": int(art_cinv),
                "article_name": article_name,
                "category_text": category_text,
                "pivot_date": pivot_date,
                "query": query,
                "holiday_values": [],
                "holiday_count": 0,
                "observed_holiday_summaries": [],
                "scope": "full_article_history",
                "answer": answer,
                "sources": sources,
                "search_available": result is not None,
                "error": error,
                "demand_uplift_assessment": _assess_holiday_demand_relevance(answer, sources),
                "note": "No holiday values were present in the article's available weekly history, so the search only assessed general holiday or special-event relevance.",
            }
        )

    return json.dumps(
        {
            "art_cinv": int(art_cinv),
            "article_name": article_name,
            "category_text": category_text,
            "pivot_date": pivot_date,
            "query": query,
            "holiday_values": holidays,
            "holiday_count": len(holidays),
            "observed_holiday_summaries": holiday_summaries,
            "scope": "full_article_history",
            "answer": answer,
            "sources": sources,
            "search_available": result is not None,
            "error": error,
            "demand_uplift_assessment": _assess_holiday_demand_relevance(answer, sources),
        }
    )