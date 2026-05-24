from __future__ import annotations

import json

import pandas as pd

from ashare_cross_section_similarity.desktop.ai import build_desktop_review_ai_evidence, run_review_ai
from ashare_cross_section_similarity.llm_client import LLMConfig
from ashare_cross_section_similarity.review import ReviewConfig, analyze_price_review


class _FakeAIClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return json.dumps(
            {
                "review": "复盘",
                "analysis": "分析",
                "critique": "锐评",
                "script_cards": [{"title": "300750.SZ", "body": "强。", "grade": "人上人", "tomorrow_check": ""}],
                "evidence_refs": ["target"],
                "disclaimer": "仅用于研究。",
            },
            ensure_ascii=False,
        )


def _bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "stock_code": symbol,
            "open": closes,
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.98 for value in closes],
            "close": closes,
            "volume": [1000 + index for index in range(len(closes))],
            "amount": [10_000 + index * 100 for index in range(len(closes))],
        }
    )


def test_desktop_review_ai_evidence_supports_single_and_multi_review() -> None:
    first = analyze_price_review(
        _bars("300750.SZ", [10, 11, 12, 13, 14]),
        ReviewConfig(symbol="300750.SZ", start="2024-01-01", end="2024-01-05"),
    )
    second = analyze_price_review(
        _bars("600519.SH", [30, 29, 28, 27, 26]),
        ReviewConfig(symbol="600519.SH", start="2024-01-01", end="2024-01-05"),
    )

    single = build_desktop_review_ai_evidence([first])
    multi = build_desktop_review_ai_evidence([first, second])

    assert single["target"]["symbol"] == "300750.SZ"
    assert multi["mode"] == "multi_stock"
    assert multi["targets"] == ["300750.SZ", "600519.SH"]
    assert [row["代码"] for row in multi["rankings"]] == ["300750.SZ", "600519.SH"]


def test_run_review_ai_uses_original_review_ai_prompt_contract() -> None:
    result = analyze_price_review(
        _bars("300750.SZ", [10, 11, 12, 13, 14]),
        ReviewConfig(symbol="300750.SZ", start="2024-01-01", end="2024-01-05"),
    )
    client = _FakeAIClient()

    ai_result = run_review_ai([result], LLMConfig(api_key="sk-test"), client=client)

    assert ai_result.review == "复盘"
    assert ai_result.script_cards[0].grade == "人上人"
    assert "严格JSON对象" in client.messages[0]["content"]
