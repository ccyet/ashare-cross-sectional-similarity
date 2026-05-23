from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pandas as pd

from ashare_cross_section_similarity.review import ReviewResult


class ReviewAIFormatError(ValueError):
    pass


@dataclass(frozen=True)
class ReviewAIResult:
    review: str
    analysis: str
    critique: str
    evidence_refs: tuple[str, ...]
    disclaimer: str
    raw: str


def build_review_ai_evidence(
    result: ReviewResult,
    comparisons: pd.DataFrame | None = None,
    *,
    stock_names: dict[str, str] | None = None,
    warnings: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    names = stock_names or {}
    symbol = result.symbol
    return {
        "target": {
            "symbol": symbol,
            "name": names.get(symbol, ""),
            "start": _date_text(result.start),
            "end": _date_text(result.end),
            "bars": int(len(result.window)),
        },
        "overview": _json_safe_mapping(result.overview),
        "segments": _frame_records(result.main_segments),
        "comparisons": _frame_records(comparisons if comparisons is not None else pd.DataFrame()),
        "warnings": [str(item) for item in warnings if str(item).strip()],
        "limits": [
            "只基于本地行情、相似度和对比统计，不读取新闻或基本面。",
            "输出仅用于研究复盘，不构成投资建议。",
        ],
    }


def build_review_ai_messages(evidence: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "你是A股走势复盘助手。必须只基于用户提供的JSON证据做复盘、分析、锐评，"
        "不得编造新闻、基本面、资金流或未提供的数据。"
        "输出必须是严格JSON对象，字段只能包含：review、analysis、critique、evidence_refs、disclaimer。"
        "review写结构化复盘；analysis写数据分析；critique写锐评和反证；"
        "evidence_refs列出引用的证据字段，例如 segments[0] 或 comparisons[0]。"
    )
    user = json.dumps(evidence, ensure_ascii=False, default=str)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_review_ai_result(raw: str) -> ReviewAIResult:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReviewAIFormatError(f"模型输出不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewAIFormatError("模型输出必须是 JSON 对象。")
    missing = [field for field in ("review", "analysis", "critique") if not str(payload.get(field, "")).strip()]
    if missing:
        raise ReviewAIFormatError(f"模型输出缺少必要字段：{', '.join(missing)}。")
    refs = payload.get("evidence_refs", [])
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list):
        raise ReviewAIFormatError("evidence_refs 必须是字符串数组。")
    return ReviewAIResult(
        review=str(payload["review"]).strip(),
        analysis=str(payload["analysis"]).strip(),
        critique=str(payload["critique"]).strip(),
        evidence_refs=tuple(str(item).strip() for item in refs if str(item).strip()),
        disclaimer=str(payload.get("disclaimer") or "仅用于研究复盘，不构成投资建议。").strip(),
        raw=raw,
    )


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [_json_safe_mapping(row) for row in frame.to_dict(orient="records")]


def _json_safe_mapping(values: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in values.items():
        if pd.isna(value):
            safe[str(key)] = None
        elif isinstance(value, pd.Timestamp):
            safe[str(key)] = _date_text(value)
        else:
            safe[str(key)] = value.item() if hasattr(value, "item") else value
    return safe


def _date_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")
