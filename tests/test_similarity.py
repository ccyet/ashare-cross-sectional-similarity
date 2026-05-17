from __future__ import annotations

import pandas as pd

from ashare_cross_section_similarity.similarity import CrossSectionSearchConfig, search_cross_section


def _bars(symbol: str, closes: list[float], amounts: list[float] | None = None) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    amounts = amounts or [1000.0 + index * 10 for index in range(len(closes))]
    return pd.DataFrame(
        {
            "date": dates,
            "stock_code": symbol,
            "open": closes,
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.98 for value in closes],
            "close": closes,
            "volume": [100.0 + index for index in range(len(closes))],
            "amount": amounts,
        }
    )


def test_search_ranks_same_time_universe_by_target_interval_shape() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12, 11, 13]),
            _bars("000002.SZ", [20, 22, 24, 22, 26]),
            _bars("000003.SZ", [10, 9, 8, 7, 6]),
        ],
        ignore_index=True,
    )

    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="000001.SZ",
            universe_symbols=("000002.SZ", "000003.SZ"),
            start="2024-01-01",
            end="2024-01-05",
            top_n=2,
        ),
    )

    assert result.target_symbol == "000001.SZ"
    assert result.window_size == 5
    assert result.results["symbol"].tolist() == ["000002.SZ", "000003.SZ"]
    assert result.results["综合相似度"].iloc[0] > result.results["综合相似度"].iloc[1]
    assert result.results["路径相似度"].between(0, 1).all()


def test_search_excludes_target_symbol_and_reports_skip_reasons() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12, 13, 14]),
            _bars("000002.SZ", [10, 11]),
        ],
        ignore_index=True,
    )

    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="000001.SZ",
            universe_symbols=("000001.SZ", "000002.SZ"),
            start="2024-01-01",
            end="2024-01-05",
            min_coverage=0.8,
        ),
    )

    assert result.results.empty
    assert result.skipped["symbol"].tolist() == ["000002.SZ"]
    assert "区间数据不足" in result.skipped["原因"].iloc[0]


def test_search_uses_same_window_semantics_after_grouped_filtering() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [99, 10, 11, 12, 11, 13, 98]),
            _bars("000002.SZ", [88, 20, 22, 24, 22, 26, 87]),
            _bars("000003.SZ", [77, 10, 9, 8, 7, 6, 76]),
            _bars("000004.SZ", [66, 10, 11, 65, 64, 63, 62]),
        ],
        ignore_index=True,
    )

    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="000001.SZ",
            universe_symbols=("000002.SZ", "000003.SZ", "000004.SZ"),
            start="2024-01-02",
            end="2024-01-06",
            top_n=3,
            min_coverage=1.0,
        ),
    )

    assert result.window_size == 5
    assert result.results["symbol"].tolist() == ["000002.SZ", "000004.SZ", "000003.SZ"]
    assert result.results["区间开始"].tolist() == [pd.Timestamp("2024-01-02")] * 3
    assert result.results["区间结束"].tolist() == [pd.Timestamp("2024-01-06")] * 3
