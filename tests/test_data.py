from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_cross_section_similarity.data import load_local_bars, resolve_timeframe_root


def test_resolve_timeframe_root_maps_daily_root_to_intraday_root(tmp_path: Path) -> None:
    daily_root = tmp_path / "market" / "daily"

    assert resolve_timeframe_root(daily_root, "1d") == daily_root
    assert resolve_timeframe_root(daily_root, "30m") == tmp_path / "market" / "30m"


def test_load_local_bars_reads_only_requested_symbols_and_timeframe(tmp_path: Path) -> None:
    daily_qfq = tmp_path / "market" / "daily" / "qfq"
    minute_qfq = tmp_path / "market" / "30m" / "qfq"
    daily_qfq.mkdir(parents=True)
    minute_qfq.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [10, 20],
            "amount": [100, 200],
        }
    ).to_parquet(daily_qfq / "000001.SZ.parquet", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-01 10:00:00", "2024-01-01 10:30:00"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "open": [10, 11],
            "high": [10, 11],
            "low": [10, 11],
            "close": [10, 11],
            "volume": [100, 110],
            "amount": [1000, 1100],
        }
    ).to_parquet(minute_qfq / "000001.SZ.parquet", index=False)

    out = load_local_bars(
        data_root=tmp_path / "market" / "daily",
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2024-01-01",
        end="2024-01-01 23:59:59",
    )

    assert out["close"].tolist() == [10.0, 11.0]
    assert out["stock_code"].tolist() == ["000001.SZ", "000001.SZ"]
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-01 10:00:00")


def test_load_local_bars_date_only_end_includes_full_intraday_session(tmp_path: Path) -> None:
    qfq = tmp_path / "market" / "30m" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01 10:00:00", "2024-01-01 14:30:00"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "open": [10, 11],
            "high": [10, 11],
            "low": [10, 11],
            "close": [10, 11],
            "volume": [100, 110],
            "amount": [1000, 1100],
        }
    ).to_parquet(qfq / "000001.SZ.parquet", index=False)

    out = load_local_bars(
        data_root=tmp_path / "market" / "daily",
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2024-01-01",
        end="2024-01-01",
    )

    assert out["close"].tolist() == [10.0, 11.0]


def test_load_local_bars_accepts_stock_code_column_without_optional_liquidity(
    tmp_path: Path,
) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "stock_code": ["000001"],
            "open": [10],
            "high": [11],
            "low": [9],
            "close": [10.5],
            "extra": ["ignored"],
        }
    ).to_parquet(qfq / "000001.SZ.parquet", index=False)

    out = load_local_bars(
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2024-01-01",
        end="2024-01-01",
    )

    assert out["stock_code"].tolist() == ["000001.SZ"]
    assert out["close"].tolist() == [10.5]
    assert out["volume"].isna().all()
    assert out["amount"].isna().all()


def test_load_local_bars_accepts_symbol_column(tmp_path: Path) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "symbol": ["600519.SH"],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [10],
            "amount": [100],
        }
    ).to_parquet(qfq / "600519.SH.parquet", index=False)

    out = load_local_bars(
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        symbols=("600519.SH",),
        start="2024-01-01",
        end="2024-01-01",
    )

    assert out["stock_code"].tolist() == ["600519.SH"]
