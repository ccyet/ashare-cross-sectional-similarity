from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_cross_section_similarity.data import (
    import_price_frame,
    load_local_bars,
    load_symbol_name_map,
    resolve_timeframe_root,
    update_symbol_name_map,
)


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


def test_load_local_bars_deduplicates_requested_symbols(tmp_path: Path) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
        }
    ).to_parquet(qfq / "000001.SZ.parquet", index=False)

    out = load_local_bars(
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        symbols=("000001.sz", "000001.SZ"),
        start="2024-01-01",
        end="2024-01-02",
    )

    assert out["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert out["stock_code"].tolist() == ["000001.SZ", "000001.SZ"]


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


def test_import_price_frame_writes_multi_symbol_parquet(tmp_path: Path) -> None:
    status = import_price_frame(
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        frame=pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
                "symbol": ["000001.sz", "000001.SZ", "600519.SH"],
                "open": [10, 11, 20],
                "high": [12, 13, 21],
                "low": [9, 10, 19],
                "close": [11, 12, 20.5],
                "volume": [100, 110, 200],
            }
        ),
        source_name="upload.csv",
    )

    assert status["symbol"].tolist() == ["000001.SZ", "600519.SH"]
    assert status["status"].tolist() == ["imported", "imported"]

    loaded = load_local_bars(
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        symbols=("000001.SZ", "600519.SH"),
        start="2024-01-01",
        end="2024-01-02",
    )

    assert loaded["stock_code"].tolist() == ["000001.SZ", "000001.SZ", "600519.SH"]
    assert loaded["close"].tolist() == [11.0, 12.0, 20.5]


def test_import_price_frame_persists_uploaded_symbol_names(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"

    import_price_frame(
        data_root=data_root,
        timeframe="1d",
        adjust="qfq",
        frame=pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01"],
                "symbol": ["880081.SH", "600519.SH"],
                "name": ["通达信趋势", "贵州茅台"],
                "open": [10, 20],
                "high": [11, 21],
                "low": [9, 19],
                "close": [10.5, 20.5],
            }
        ),
        source_name="upload.csv",
    )

    assert load_symbol_name_map(data_root, ("880081.SH", "600519.SH")) == {
        "880081.SH": "通达信趋势",
        "600519.SH": "贵州茅台",
    }


def test_update_symbol_name_map_merges_sector_index_names(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"

    update_symbol_name_map(data_root, {"880081.SH": "通达信趋势"}, source="tdx_sector_index")
    update_symbol_name_map(data_root, {"880082.SH": "板块趋势"}, source="tdx_sector_index")

    assert load_symbol_name_map(data_root, ("880081.SH", "880082.SH")) == {
        "880081.SH": "通达信趋势",
        "880082.SH": "板块趋势",
    }


def test_import_price_frame_uses_fallback_symbol_for_single_symbol_file(tmp_path: Path) -> None:
    import_price_frame(
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        frame=pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "open": [10],
                "high": [11],
                "low": [9],
                "close": [10.5],
            }
        ),
        fallback_symbol="000001",
    )

    loaded = load_local_bars(
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2024-01-01",
        end="2024-01-01",
    )

    assert loaded["stock_code"].tolist() == ["000001.SZ"]
    assert loaded["close"].tolist() == [10.5]


def test_import_price_frame_requires_symbol_column_or_fallback(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "open": [10],
            "high": [11],
            "low": [9],
            "close": [10.5],
        }
    )

    try:
        import_price_frame(
            data_root=tmp_path / "market" / "daily",
            timeframe="1d",
            adjust="qfq",
            frame=frame,
        )
    except ValueError as exc:
        assert "symbol" in str(exc)
    else:
        raise AssertionError("expected ValueError")
