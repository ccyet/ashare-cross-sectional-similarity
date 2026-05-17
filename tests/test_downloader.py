from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_cross_section_similarity.downloader import (
    AkshareBarsClient,
    DownloadRequest,
    data_check,
    infer_asset_type,
    normalize_akshare_bars,
    update_local_bars,
)


class FakeClient(AkshareBarsClient):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[DownloadRequest] = []

    def fetch(self, request: DownloadRequest) -> pd.DataFrame:
        self.calls.append(request)
        return self.frame.copy()


def test_normalize_akshare_bars_accepts_chinese_daily_columns() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-01", "2024-01-02"],
            "开盘": [10, 11],
            "最高": [11, 12],
            "最低": [9, 10],
            "收盘": [10.5, 11.5],
            "成交量": [100, 110],
            "成交额": [1000, 1200],
        }
    )

    out = normalize_akshare_bars(raw, "000001.SZ", "2024-01-01", "2024-01-02")

    assert out["stock_code"].tolist() == ["000001.SZ", "000001.SZ"]
    assert out["close"].tolist() == [10.5, 11.5]
    assert out["amount"].tolist() == [1000.0, 1200.0]


def test_update_local_bars_fetches_and_merges_parquet(tmp_path: Path) -> None:
    existing_root = tmp_path / "market" / "daily" / "qfq"
    existing_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "stock_code": ["000001.SZ"],
            "open": [9],
            "high": [10],
            "low": [8],
            "close": [9.5],
            "volume": [90],
            "amount": [900],
        }
    ).to_parquet(existing_root / "000001.SZ.parquet", index=False)
    client = FakeClient(
        pd.DataFrame(
            {
                "日期": ["2024-01-01", "2024-01-02"],
                "开盘": [10, 11],
                "最高": [11, 12],
                "最低": [9, 10],
                "收盘": [10.5, 11.5],
                "成交量": [100, 110],
                "成交额": [1000, 1200],
            }
        )
    )

    result = update_local_bars(
        symbols=("000001.SZ",),
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-02",
        client=client,
    )

    stored = pd.read_parquet(existing_root / "000001.SZ.parquet")
    assert result["status"].tolist() == ["success"]
    assert stored["close"].tolist() == [10.5, 11.5]
    assert client.calls[0].symbol == "000001.SZ"


def test_data_check_reports_missing_and_available_symbols(tmp_path: Path) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [10, 20],
            "amount": [100, 200],
        }
    ).to_parquet(qfq / "000001.SZ.parquet", index=False)

    out = data_check(
        symbols=("000001.SZ", "000002.SZ"),
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-02",
    )

    assert out[["symbol", "status", "rows"]].to_dict("records") == [
        {"symbol": "000001.SZ", "status": "available", "rows": 2},
        {"symbol": "000002.SZ", "status": "missing_file", "rows": 0},
    ]


def test_infer_asset_type_distinguishes_stock_index_and_etf() -> None:
    assert infer_asset_type("000001.SZ") == "stock"
    assert infer_asset_type("000300.SH") == "index"
    assert infer_asset_type("399006.SZ") == "index"
    assert infer_asset_type("510300.SH") == "etf"


def test_minute_index_download_fails_explicitly() -> None:
    client = AkshareBarsClient()
    request = DownloadRequest(
        symbol="000300.SH",
        timeframe="30m",
        start="2024-01-01",
        end="2024-01-02",
    )

    try:
        client._fetch_minute(object(), "000300.SH", "000300", request)
    except ValueError as exc:
        assert "指数分钟线" in str(exc)
    else:
        raise AssertionError("指数分钟线应显式失败")
