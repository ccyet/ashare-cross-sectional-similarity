from __future__ import annotations

from typing import Protocol

import pandas as pd

from ashare_cross_section_similarity.llm_client import LLMClient, LLMConfig
from ashare_cross_section_similarity.review import ReviewResult, rank_review_results
from ashare_cross_section_similarity.review_ai import (
    ReviewAIResult,
    build_review_ai_evidence,
    build_review_ai_messages,
    parse_review_ai_result,
)


class AIChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


def build_desktop_review_ai_evidence(
    results: list[ReviewResult] | tuple[ReviewResult, ...],
    *,
    warnings: list[str] | tuple[str, ...] = (),
) -> dict[str, object]:
    valid = [result for result in results if not result.window.empty]
    if not valid:
        return {
            "mode": "empty",
            "targets": [result.symbol for result in results],
            "warnings": [str(item) for item in warnings],
            "limits": ["没有可复盘行情时不得编造结论。"],
        }
    if len(valid) == 1:
        return build_review_ai_evidence(valid[0], pd.DataFrame(), warnings=warnings)
    ranking = rank_review_results(valid, pd.DataFrame())
    return {
        "mode": "multi_stock",
        "targets": [result.symbol for result in valid],
        "rankings": ranking.to_dict(orient="records"),
        "warnings": [str(item) for item in warnings if str(item).strip()],
        "limits": [
            "只基于本地行情和排序统计，不读取新闻或基本面。",
            "输出仅用于研究复盘，不构成投资建议。",
        ],
    }


def run_review_ai(
    results: list[ReviewResult] | tuple[ReviewResult, ...],
    config: LLMConfig,
    *,
    client: AIChatClient | None = None,
) -> ReviewAIResult:
    evidence = build_desktop_review_ai_evidence(results)
    ai_client = client or LLMClient(config)
    raw = ai_client.chat(build_review_ai_messages(evidence))
    return parse_review_ai_result(raw, evidence=evidence)
