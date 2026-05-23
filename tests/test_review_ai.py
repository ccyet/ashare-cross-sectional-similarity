from __future__ import annotations

import json

import pandas as pd
import pytest

from ashare_cross_section_similarity.review import ReviewConfig, analyze_price_review, build_comparison_stats
from ashare_cross_section_similarity.review_ai import (
    ReviewAIFormatError,
    build_review_ai_evidence,
    build_review_ai_messages,
    parse_review_ai_result,
)


def _bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "stock_code": [symbol] * len(closes),
            "open": closes,
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.98 for value in closes],
            "close": closes,
            "volume": [100] * len(closes),
            "amount": [1000] * len(closes),
        }
    )


def test_build_review_ai_evidence_keeps_review_analysis_critique_inputs() -> None:
    target = _bars("000001.SZ", [10, 11, 12, 11, 13])
    result = analyze_price_review(target, ReviewConfig(symbol="000001.SZ", start="2024-01-01", end="2024-01-05"))
    comparisons = pd.DataFrame([build_comparison_stats(target, _bars("000300.SH", [10, 10.5, 10.4, 10.6, 10.7]), "沪深300")])

    evidence = build_review_ai_evidence(result, comparisons, stock_names={"000001.SZ": "平安银行"}, warnings=["样例风险"])

    assert evidence["target"]["symbol"] == "000001.SZ"
    assert evidence["target"]["name"] == "平安银行"
    assert evidence["overview"]["return"] == result.overview["return"]
    assert evidence["segments"]
    assert evidence["comparisons"][0]["标的"] == "沪深300"
    assert evidence["warnings"] == ["样例风险"]


def test_build_review_ai_messages_require_json_contract() -> None:
    messages = build_review_ai_messages({"target": {"symbol": "000001.SZ"}, "warnings": []})

    assert messages[0]["role"] == "system"
    assert "JSON" in messages[0]["content"]
    assert "review" in messages[0]["content"]
    assert "analysis" in messages[0]["content"]
    assert "critique" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "000001.SZ" in messages[1]["content"]


def test_parse_review_ai_result_accepts_required_fields() -> None:
    raw = json.dumps(
        {
            "review": "复盘内容",
            "analysis": "分析内容",
            "critique": "锐评内容",
            "evidence_refs": ["segments[0]", "comparisons[0]"],
            "disclaimer": "仅供研究复盘",
        },
        ensure_ascii=False,
    )

    result = parse_review_ai_result(raw)

    assert result.review == "复盘内容"
    assert result.analysis == "分析内容"
    assert result.critique == "锐评内容"
    assert result.evidence_refs == ("segments[0]", "comparisons[0]")


def test_parse_review_ai_result_rejects_missing_fields() -> None:
    with pytest.raises(ReviewAIFormatError, match="critique"):
        parse_review_ai_result('{"review":"ok","analysis":"ok"}')
