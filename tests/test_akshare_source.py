from __future__ import annotations

import pandas as pd
import pytest

from ashare_cross_section_similarity.akshare_source import fetch_akshare_bars


class _FakeAkShare:
    def __init__(self) -> None:
        self.stock_calls: list[dict[str, object]] = []
        self.index_calls: list[dict[str, object]] = []

    def stock_zh_a_hist(self, **kwargs: object) -> pd.DataFrame:
        self.stock_calls.append(kwargs)
        return pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "开盘": [10.0],
                "最高": [11.0],
                "最低": [9.5],
                "收盘": [10.5],
                "成交量": [1000.0],
                "成交额": [10000.0],
            }
        )

    def stock_zh_index_daily_em(self, **kwargs: object) -> pd.DataFrame:
        self.index_calls.append(kwargs)
        return pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "volume": [100.0],
                "amount": [1000.0],
            }
        )


def test_fetch_akshare_bars_normalizes_stock_payload() -> None:
    fake = _FakeAkShare()

    out = fetch_akshare_bars(
        symbols=("600519.SH",),
        start="2024-01-01",
        end="2024-01-31",
        timeframe="1d",
        adjust="qfq",
        ak_client=fake,
    )

    assert fake.stock_calls == [
        {
            "symbol": "600519",
            "period": "daily",
            "start_date": "20240101",
            "end_date": "20240131",
            "adjust": "qfq",
        }
    ]
    assert out.to_dict("records") == [
        {
            "date": pd.Timestamp("2024-01-02"),
            "stock_code": "600519.SH",
            "open": 10.0,
            "high": 11.0,
            "low": 9.5,
            "close": 10.5,
            "volume": 1000.0,
            "amount": 10000.0,
        }
    ]


def test_fetch_akshare_bars_uses_index_endpoint_for_index_proxy() -> None:
    fake = _FakeAkShare()

    out = fetch_akshare_bars(
        symbols=("399006.SZ",),
        start="2024-01-01",
        end="2024-01-31",
        ak_client=fake,
    )

    assert fake.index_calls == [{"symbol": "399006", "start_date": "20240101", "end_date": "20240131"}]
    assert out["stock_code"].tolist() == ["399006.SZ"]


def test_fetch_akshare_bars_rejects_intraday_timeframe() -> None:
    with pytest.raises(ValueError, match="分钟线请使用 TDX"):
        fetch_akshare_bars(
            symbols=("600519.SH",),
            start="2024-01-01",
            end="2024-01-31",
            timeframe="30m",
            ak_client=_FakeAkShare(),
        )
