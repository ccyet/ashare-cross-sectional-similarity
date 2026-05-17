from __future__ import annotations

import pandas as pd

from ashare_cross_section_similarity.history import (
    HistorySearchConfig,
    _filter_nearby_history_windows,
    search_history,
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
            "volume": [100.0 + index for index in range(len(closes))],
            "amount": [1000.0 + index * 10 for index in range(len(closes))],
        }
    )


def test_search_history_finds_same_symbol_prior_similar_windows() -> None:
    bars = _bars(
        "000001.SZ",
        [
            10, 11, 12, 11, 13,
            20, 19, 18, 17, 16,
            30, 33, 36, 33, 39,
            40, 41, 42, 43, 44,
        ],
    )

    result = search_history(
        bars,
        HistorySearchConfig(
            symbol="000001.SZ",
            as_of="2024-01-15",
            window_size=5,
            forward_windows=(2,),
            top_n=2,
            exclusion_bars=0,
            nearby_gap_days=0,
        ),
    )

    assert result.symbol == "000001.SZ"
    assert result.current_window["date"].min() == pd.Timestamp("2024-01-11")
    assert result.results["窗口开始"].tolist()[0] == pd.Timestamp("2024-01-01")
    assert result.historical_windows[0]["date"].min() == pd.Timestamp("2024-01-01")
    assert result.results["综合相似度"].iloc[0] > result.results["综合相似度"].iloc[1]
    assert "t_plus_2_return" in result.results.columns


def test_search_history_does_not_use_data_after_as_of_for_matching() -> None:
    bars = _bars(
        "000001.SZ",
        [
            10, 11, 12, 11, 13,
            20, 19, 18, 17, 16,
            30, 33, 36, 33, 39,
            30, 33, 36, 33, 39,
        ],
    )

    result = search_history(
        bars,
        HistorySearchConfig(
            symbol="000001.SZ",
            as_of="2024-01-15",
            window_size=5,
            forward_windows=(2,),
            top_n=10,
            exclusion_bars=0,
        ),
    )

    assert result.results["窗口开始"].max() < pd.Timestamp("2024-01-11")


def test_nearby_history_windows_keep_most_similar_sample() -> None:
    frame = pd.DataFrame(
        [
            {
                "窗口开始": pd.Timestamp("2016-12-17"),
                "窗口结束": pd.Timestamp("2017-01-08"),
                "综合相似度": 0.95,
            },
            {
                "窗口开始": pd.Timestamp("2016-12-15"),
                "窗口结束": pd.Timestamp("2017-01-06"),
                "综合相似度": 0.90,
            },
            {
                "窗口开始": pd.Timestamp("2017-03-01"),
                "窗口结束": pd.Timestamp("2017-03-20"),
                "综合相似度": 0.80,
            },
        ]
    )

    filtered = _filter_nearby_history_windows(frame, top_n=3, min_gap_days=20)

    assert filtered["窗口开始"].tolist() == [
        pd.Timestamp("2016-12-17"),
        pd.Timestamp("2017-03-01"),
    ]
