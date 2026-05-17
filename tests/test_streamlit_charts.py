from __future__ import annotations

import pandas as pd

from ashare_cross_section_similarity.similarity import CrossSectionSearchResult
from streamlit_app import _lightweight_chart_series


def _bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "stock_code": [symbol] * len(closes),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1] * len(closes),
            "amount": [1] * len(closes),
        }
    )


def test_lightweight_chart_series_uses_time_value_points() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12]),
            _bars("000002.SZ", [20, 22, 24]),
        ],
        ignore_index=True,
    )
    result = CrossSectionSearchResult(
        target_symbol="000001.SZ",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-03"),
        window_size=3,
        results=pd.DataFrame({"symbol": ["000002.SZ"]}),
        skipped=pd.DataFrame(),
    )

    series = _lightweight_chart_series(bars, result)

    assert series[0]["title"] == "000001.SZ（目标）"
    assert series[0]["data"] == [
        {"time": "2024-01-01", "value": 100.0},
        {"time": "2024-01-02", "value": 110.0},
        {"time": "2024-01-03", "value": 120.0},
    ]
    assert series[1]["title"] == "000002.SZ"
    assert series[1]["data"][1] == {"time": "2024-01-02", "value": 110.0}
