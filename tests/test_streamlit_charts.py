from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from ashare_cross_section_similarity.similarity import CrossSectionSearchResult
from streamlit_app import (
    _download_symbols_with_progress,
    _format_results,
    _forward_stats_load_end,
    _lightweight_kline_series,
    _pin_symbol_row,
)


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


def test_lightweight_kline_series_uses_top_six_ohlc_points() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12]),
            *[_bars(f"00000{index}.SZ", [20 + index, 22 + index, 24 + index]) for index in range(2, 9)],
        ],
        ignore_index=True,
    )
    result = CrossSectionSearchResult(
        target_symbol="000001.SZ",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-03"),
        window_size=3,
        results=pd.DataFrame({"symbol": [f"00000{index}.SZ" for index in range(2, 9)]}),
        skipped=pd.DataFrame(),
    )

    series = _lightweight_kline_series(bars, result)

    assert series[0]["title"] == "000001.SZ（目标）"
    assert series[0]["data"] == [
        {"time": "2024-01-01", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
        {"time": "2024-01-02", "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0},
        {"time": "2024-01-03", "open": 12.0, "high": 12.0, "low": 12.0, "close": 12.0},
    ]
    assert series[1]["title"] == "000002.SZ"
    assert series[1]["data"][1] == {"time": "2024-01-02", "open": 24.0, "high": 24.0, "low": 24.0, "close": 24.0}
    assert [item["title"] for item in series] == [
        "000001.SZ（目标）",
        "000002.SZ",
        "000003.SZ",
        "000004.SZ",
        "000005.SZ",
        "000006.SZ",
        "000007.SZ",
    ]


def test_format_results_formats_forward_returns_as_percentages() -> None:
    formatted = _format_results(
        pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "区间开始": [pd.Timestamp("2024-01-01")],
                "区间结束": [pd.Timestamp("2024-01-05")],
                "综合相似度": [0.81234],
                "区间收益": [0.1234],
                "t_plus_3_return": [0.0567],
            }
        )
    )

    assert formatted["综合相似度"].iloc[0] == "81.23%"
    assert formatted["区间收益"].iloc[0] == "12.34%"
    assert formatted["后3根收益"].iloc[0] == "5.67%"
    assert formatted["区间开始"].iloc[0] == "2024-01-01"


def test_forward_stats_load_end_extends_historical_window() -> None:
    assert _forward_stats_load_end("2024-01-01", today=pd.Timestamp("2024-02-20")) == "2024-02-15"


def test_pin_symbol_row_keeps_target_visible_at_top() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000503.SZ", "000852.SH", "000543.SZ"],
            "status": ["available", "available", "missing_file"],
        }
    )

    pinned = _pin_symbol_row(frame, "000852.sh", limit=2)

    assert pinned["symbol"].tolist() == ["000852.SH", "000503.SZ"]


def test_download_symbols_with_progress_downloads_each_symbol(monkeypatch) -> None:
    calls: list[tuple[tuple[str, ...], str]] = []

    def fake_update_local_bars(**kwargs: object) -> pd.DataFrame:
        calls.append((tuple(kwargs["symbols"]), str(kwargs["end"])))
        symbol = kwargs["symbols"][0]
        return pd.DataFrame(
            [{"symbol": symbol, "status": "delegated", "rows": 0, "new_rows": 0, "message": "ok"}]
        )

    def fake_data_check(**kwargs: object) -> pd.DataFrame:
        symbol = kwargs["symbols"][0]
        if symbol == "000001.SZ":
            return pd.DataFrame(
                [{"symbol": symbol, "status": "missing_file", "rows": 0, "start": None, "end": None, "message": "本地 parquet 不存在"}]
            )
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "status": "available",
                    "rows": 20,
                    "start": pd.Timestamp("2024-01-01"),
                    "end": pd.Timestamp("2024-01-31"),
                    "message": "",
                }
            ]
        )

    monkeypatch.setattr("streamlit_app.update_local_bars", fake_update_local_bars)
    monkeypatch.setattr("streamlit_app.data_check", fake_data_check)
    progress: list[tuple[int, int, str, str]] = []

    result = _download_symbols_with_progress(
        symbols=["000001.SZ", "000001.SZ", "000002.SZ"],
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-31",
        trend_repo=Path("/tmp/trend"),
        data_root=Path("/tmp/data"),
        provider="",
        download_engine="trend",
        progress_callback=lambda completed, total, symbol, status: progress.append((completed, total, symbol, status)),
    )

    assert calls == [(("000001.SZ",), "2024-01-31"), (("000002.SZ",), "2024-01-31")]
    assert progress == [
        (0, 2, "000001.SZ", "running"),
        (1, 2, "000001.SZ", "missing_file"),
        (1, 2, "000002.SZ", "running"),
        (2, 2, "000002.SZ", "available"),
    ]
    assert_frame_equal(
        result,
        pd.DataFrame(
            [
                {"symbol": "000001.SZ", "status": "missing_file", "rows": 0, "start": None, "end": None, "message": "下载命令执行后仍缺本地 parquet；本地 parquet 不存在"},
                {
                    "symbol": "000002.SZ",
                    "status": "available",
                    "rows": 20,
                    "start": pd.Timestamp("2024-01-01"),
                    "end": pd.Timestamp("2024-01-31"),
                    "message": "",
                },
            ]
        ),
    )
