from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd

from ashare_cross_section_similarity.review import ReviewResult


class ReviewAIFormatError(ValueError):
    pass


@dataclass(frozen=True)
class ReviewAIScriptCard:
    title: str
    body: str
    grade: str = ""
    tomorrow_check: str = ""


@dataclass(frozen=True)
class ReviewAIResult:
    review: str
    analysis: str
    critique: str
    evidence_refs: tuple[str, ...]
    disclaimer: str
    raw: str
    script_cards: tuple[ReviewAIScriptCard, ...] = ()


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
        "输出必须是严格JSON对象，字段只能包含：review、analysis、critique、script_cards、evidence_refs、disclaimer。"
        "不得输出 Markdown、解释性文字或 JSON 以外的任何内容。"
        "必须统一按《A股多股/ETF排序锐评框架》组织：市场总环境、排序总表、逐个锐评、关键转折点复盘、明日验证。"
        "排序不按代码顺序，按指数环境、相对强弱、回撤控制、关键转折点和A股语境综合排序。"
        "review写研究端内容：先给排序表，再给关键转折点，最后给明日验证。"
        "analysis写数据分析：解释指数阶段、超额收益、最大回撤、上涨K占比和转折位置。"
        "critique写视频端脚本：不按代码顺序，按市场地位排序；"
        "等级顺序固定为夯爆了 > 人上人 > 立棍单打 > 刷子 > 混子 > NPC > 拉完了。"
        "每个标的三句话封顶：一句定性、一句数据、一句结局；不预测，视频端不得写明天。"
        "script_cards必须是数组，每个元素包含title、body、grade、tomorrow_check；"
        "grade必须使用夯爆了/人上人/立棍单打/刷子/混子/NPC/拉完了之一；"
        "tomorrow_check字段沿用字段名但内容必须写当前结局或状态，不得写明天。"
        "每个结论都必须能对应 evidence_refs 中的证据字段，"
        "evidence_refs必须非空，例如 segments[0] 或 comparisons[0]。"
    )
    user = json.dumps(evidence, ensure_ascii=False, default=str)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_review_ai_result(raw: str, evidence: dict[str, Any] | None = None) -> ReviewAIResult:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReviewAIFormatError(f"模型输出不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewAIFormatError("模型输出必须是 JSON 对象。")
    review = _required_text(payload, "review")
    analysis = _required_text(payload, "analysis")
    critique = _required_text(payload, "critique")
    script_cards = _script_cards(payload.get("script_cards"))
    disclaimer = _optional_text(payload, "disclaimer") or "仅用于研究复盘，不构成投资建议。"
    refs = _evidence_refs(payload.get("evidence_refs"))
    if evidence is not None:
        _validate_evidence_refs(refs, evidence)
    return ReviewAIResult(
        review=review,
        analysis=analysis,
        critique=critique,
        evidence_refs=refs,
        disclaimer=disclaimer,
        raw=raw,
        script_cards=script_cards,
    )


def _required_text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ReviewAIFormatError(f"{field} 必须是非空字符串。")
    text = value.strip()
    if not text:
        raise ReviewAIFormatError(f"模型输出缺少必要字段：{field}。")
    return text


def _optional_text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ReviewAIFormatError(f"{field} 必须是字符串。")
    return value.strip()


def _evidence_refs(refs: object) -> tuple[str, ...]:
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list):
        raise ReviewAIFormatError("evidence_refs 必须是字符串数组。")
    if not all(isinstance(item, str) for item in refs):
        raise ReviewAIFormatError("evidence_refs 必须是字符串数组。")
    cleaned = tuple(item.strip() for item in refs if item.strip())
    if not cleaned:
        raise ReviewAIFormatError("evidence_refs 必须至少包含一个证据引用。")
    return cleaned


def _script_cards(value: object) -> tuple[ReviewAIScriptCard, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ReviewAIFormatError("script_cards 必须是对象数组。")
    cards: list[ReviewAIScriptCard] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ReviewAIFormatError(f"script_cards[{index}] 必须是对象。")
        title = _required_text(item, "title")
        body = _required_text(item, "body")
        cards.append(
            ReviewAIScriptCard(
                title=title,
                body=body,
                grade=_optional_text(item, "grade"),
                tomorrow_check=_optional_text(item, "tomorrow_check"),
            )
        )
    return tuple(cards)


def _validate_evidence_refs(refs: tuple[str, ...], evidence: dict[str, Any]) -> None:
    for ref in refs:
        if not _evidence_ref_exists(ref, evidence):
            raise ReviewAIFormatError(f"evidence_refs 包含不存在的证据引用：{ref}")


def _evidence_ref_exists(ref: str, evidence: dict[str, Any]) -> bool:
    current: Any = evidence
    for part in ref.split("."):
        if not part:
            return False
        key, index = _parse_ref_part(part)
        if key is None:
            return False
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
        if index is not None:
            if not isinstance(current, list) or index >= len(current):
                return False
            current = current[index]
    return True


def _parse_ref_part(part: str) -> tuple[str | None, int | None]:
    if "[" not in part and "]" not in part:
        return (part, None) if part else (None, None)
    if not part.endswith("]") or part.count("[") != 1 or part.count("]") != 1:
        return None, None
    key, raw_index = part[:-1].split("[", 1)
    if not key or not raw_index.isdigit():
        return None, None
    return key, int(raw_index)


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [_json_safe_mapping(row) for row in frame.to_dict(orient="records")]


def _json_safe_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(value) for key, value in values.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist())
    if isinstance(value, pd.Timestamp):
        return _date_text(value)
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def _date_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")
